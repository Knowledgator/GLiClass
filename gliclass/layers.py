# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team and Knowledgator.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.activations import ACT2FN

from .config import GLiClassModelConfig

class LstmSeq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(LstmSeq2SeqEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x, mask, hidden=None):
        # Packing the input sequence
        lengths = mask.sum(dim=1).cpu()
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing packed sequence through LSTM
        packed_output, hidden = self.lstm(packed_x, hidden)

        # Unpacking the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output

class FeaturesProjector(nn.Module):
    def __init__(self, config: GLiClassModelConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.encoder_config.hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.dropout = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.hidden_size, config.encoder_config.hidden_size, bias=True)

    def forward(self, features):
        hidden_states = self.linear_1(features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class BiEncoderProjector(nn.Module):
    def __init__(self, config: GLiClassModelConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.label_model_config.hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, config.encoder_config.hidden_size, bias=True)

    def forward(self, features):
        hidden_states = self.linear_1(features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # StableDropout only calls this function when training.
        train = True
        # TODO: We should check if the opset_version being used to export
        # is > 12 here, but there's no good way to do that. As-is, if the
        # opset_version < 12, export will fail with a CheckerError.
        # Once https://github.com/pytorch/pytorch/issues/78391 is fixed, do something like:
        # if opset_version < 12:
        #   return torch.onnx.symbolic_opset9.dropout(g, input, dropout_p, train)
        return symbolic_opset12.dropout(g, input, dropout_p, train)
    
# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_output))

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.cross_attn(query, key, value, attn_mask=mask)
        return self.norm(query + self.dropout(attn_output))

class Fuser(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionBlock(d_model, num_heads, dropout),
                CrossAttentionBlock(d_model, num_heads, dropout)
            ])
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, query_mask=None, key_mask=None):
        if query_mask is not None and key_mask is not None:
            self_attn_mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
            cross_attn_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
        else:
            self_attn_mask = None
            cross_attn_mask = None

        value = self.fc(key)

        for self_attn, cross_attn in self.layers:
            query = self_attn(query, mask=self_attn_mask)
            query = cross_attn(query, key, value, mask=cross_attn_mask)

        return query

class LayerwiseAttention(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        
        # Squeeze operation
        self.squeeze = nn.Linear(hidden_size, 1)
        
        # Excitation operation
        self.W1 = nn.Linear(num_layers, num_layers // 2)
        self.W2 = nn.Linear(num_layers // 2, num_layers)
        
        # Final projection
        self.output_projection = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, encoder_outputs):
        # encoder_outputs is a list of tensors, each of shape [B, L, D]
        B, L, D = encoder_outputs[0].shape
        
        # Concatenate all layers
        U = torch.stack(encoder_outputs, dim=1)  # [B, K, L, D]
        
        # Squeeze operation
        Z = self.squeeze(U).squeeze(-1)  # [B, K, L]
        Z = Z.mean(dim=2)  # [B, K]
        
        # Excitation operation
        s = self.W2(F.relu(self.W1(Z)))  # [B, K]
        s = torch.sigmoid(s)  # [B, K]
        
        # Apply attention weights
        U_weighted = U * s.unsqueeze(-1).unsqueeze(-1)  # [B, K, L, D]
        
        # Sum across layers
        U_sum = U_weighted.sum(dim=1)  # [B, L, D]
        
        # Final projection
        output = self.output_projection(U_sum)  # [B, L, output_size]
        
        return output