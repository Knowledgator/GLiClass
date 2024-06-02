from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import (logging)
from transformers.models.auto import AutoModel
from .config import GLiClassModelConfig
from .layers import FeaturesProjector, LstmSeq2SeqEncoder, StableDropout
from .poolings import POOLING2OBJECT
from .scorers import SCORER2OBJECT

logger = logging.get_logger(__name__)


class GLiClassPreTrainedModel(PreTrainedModel):
    config_class = GLiClassModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.encoder_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.normal_(param.data, mean=0.0, std=std)
                elif 'bias' in name:
                    param.data.zero_()
    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


class GLiClassModel(GLiClassPreTrainedModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained = False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)
        if from_pretrained:
            self.encoder_model = AutoModel.from_pretrained(
                config.encoder_model_name
            )
        else:
            self.encoder_model = AutoModel.from_config(
                config.encoder_config
            )
        
        self.text_projector = FeaturesProjector(config)
        self.classes_projector = FeaturesProjector(config)
        
        if config.pooling_strategy not in POOLING2OBJECT:
            raise NotImplementedError(f"{config.pooling_strategy} is not implemented pooling type.")
        else:
            self.pooler = POOLING2OBJECT[config.pooling_strategy]()

        if config.pooling_strategy not in POOLING2OBJECT:
            raise NotImplementedError(f"{config.scorer_type} is not implemented. Choose one of this: 'dot', 'weighted-dot'")
        else:
            self.scorer = SCORER2OBJECT[config.scorer_type](config.hidden_size)

        if config.use_lstm:
            self.lstm = LstmSeq2SeqEncoder(config.hidden_size, config.hidden_size, bidirectional=True)
        
        drop_out = getattr(config.encoder_config, "cls_dropout", None)
        drop_out = self.config.encoder_config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.num_labels = -1
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder_model.set_input_embeddings(value)

    def tie_weights(self):
        return self.encoder_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.encoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.encoder_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    

    def _extract_class_features(self, token_embeds, input_ids, attention_mask):
        batch_size, sequence_length, embed_dim = token_embeds.shape

        class_token_mask = input_ids == self.config.class_token_index
        num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

        max_embed_dim = num_class_tokens.max()
        aranged_class_idx = torch.arange(max_embed_dim, 
                                            dtype=attention_mask.dtype, 
                                            device=token_embeds.device).expand(batch_size, -1)
        
        batch_indices, target_class_idx = torch.where(aranged_class_idx<num_class_tokens)
        _, class_indices = torch.where(class_token_mask)
        # class_indices+=1

        classes_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
        )
        classes_embedding_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=token_embeds.device
        )

        classes_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
        classes_embedding_mask[batch_indices, target_class_idx] = 1
        
        return classes_embedding, classes_embedding_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        encoder_layer = outputs[0]

        if self.config.use_lstm:
            encoder_layer = self.lstm(encoder_layer, attention_mask)
        
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)

        classes_embedding, classes_embedding_mask = self._extract_class_features(encoder_layer, 
                                                                        input_ids, attention_mask)
        classes_embedding = self.classes_projector(classes_embedding)

        logits = self.scorer(pooled_output, classes_embedding)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                all_losses = loss_fct(logits, labels)
                all_losses = all_losses * classes_embedding_mask.float()
                loss = all_losses.mean()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
