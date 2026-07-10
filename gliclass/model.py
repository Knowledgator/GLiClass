import os
import warnings
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
import transformers
from torch import nn
from packaging import version
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import SequenceClassifierOutput

# Import initialization module (transformers 5.0+) or fallback to torch.nn.init
try:
    from transformers import initialization as init
except ImportError:
    # transformers < 5.0 doesn't have this module, use torch.nn.init instead
    from torch.nn import init
from .utils import MissedPackageException, is_module_available
from .config import GLiClassModelConfig
from .layers import FeaturesProjector, BiEncoderProjector, LayerwiseAttention, LstmSeq2SeqEncoder
from .scorers import SCORER2OBJECT
from .poolings import POOLING2OBJECT
from .loss_functions import focal_loss_with_logits, sequence_contrastive_loss

IS_LLM2VEC = is_module_available("llm2vec")
IS_PEFT = is_module_available("peft")
IS_TURBOT5 = is_module_available("turbot5")
IS_FLASHDEBERTA = is_module_available("flashdeberta")

logger = logging.get_logger(__name__)

if IS_LLM2VEC:
    from llm2vec.models import GemmaBiModel, LlamaBiModel, Qwen2BiModel, MistralBiModel

    DECODER_MODEL_MAPPING = {
        "MistralConfig": MistralBiModel,
        "LlamaConfig": LlamaBiModel,
        "GemmaConfig": GemmaBiModel,
        "Qwen2Config": Qwen2BiModel,
    }
else:
    DECODER_MODEL_MAPPING = {}

if IS_TURBOT5:
    from turbot5.model.modeling import T5EncoderModel as FlashT5EncoderModel
from transformers import T5EncoderModel, UMT5EncoderModel

if IS_FLASHDEBERTA:
    from flashdeberta import FlashDebertaV2Model
from transformers import DebertaV2Model

if IS_PEFT:
    from peft import LoraConfig, get_peft_model


@dataclass
class GLiClassOutput(SequenceClassifierOutput):
    text_embeddings: torch.Tensor | None = None
    class_embeddings: torch.Tensor | None = None


class GLiClassPreTrainedModel(PreTrainedModel):
    config_class = GLiClassModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = False
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]

    def _initialize_weights(self, module, is_remote_code: bool = False):
        """
        Initialize weights if not already initialized.

        This method is called by transformers 5.0+ during post_init().
        It uses the _is_hf_initialized flag to prevent reinitializing weights
        that were already loaded from a checkpoint.

        For transformers 4.x, this method is not called, maintaining backward compatibility.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        self._init_weights(module)
        module._is_hf_initialized = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.encoder_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            init.normal_(module.class_embedding, mean=0.0, std=std)

        if hasattr(module, "segment_embeddings"):
            init.normal_(module.segment_embeddings.weight, mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    init.normal_(param, mean=0.0, std=std)
                elif "bias" in name:
                    init.zeros_(param)


class GLiClassBaseModel(nn.Module):  # ):
    def __init__(self, config: GLiClassModelConfig, device="cpu", **kwargs):
        super().__init__()
        self.config = config
        self.text_projector = FeaturesProjector(config)
        self.classes_projector = FeaturesProjector(config)

        if config.pooling_strategy not in POOLING2OBJECT:
            raise NotImplementedError(f"{config.pooling_strategy} is not implemented pooling type.")
        else:
            self.pooler = POOLING2OBJECT[config.pooling_strategy]()

        if config.pooling_strategy not in POOLING2OBJECT:
            raise NotImplementedError(
                f"{config.scorer_type} is not implemented. Choose one of this: 'dot', 'weighted-dot'"
            )
        else:
            self.scorer = SCORER2OBJECT[config.scorer_type](
                config.hidden_size,
                num_heads=config.scorer_num_heads,
                scorer_mlp_hidden_size=config.scorer_mlp_hidden_size,
                attn_dropout=config.scorer_attn_dropout,
            )

        if config.use_lstm:
            self.lstm = LstmSeq2SeqEncoder(config.hidden_size, config.hidden_size // 2, bidirectional=True)

        if config.squeeze_layers:
            self.layer_wise_attention = LayerwiseAttention(
                config.encoder_config.num_hidden_layers, config.encoder_config.hidden_size
            )

        drop_out = getattr(config, "dropout", 0.0)
        # self.dropout = StableDropout(drop_out)
        self.dropout = nn.Dropout(drop_out)

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.epsilon = 1e-8
        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.num_labels = -1

        self.device = torch.device(device)

    def _extract_class_features(self, token_embeds, input_ids, attention_mask, max_num_classes=None):
        batch_size, _sequence_length, embed_dim = token_embeds.shape

        class_token_mask = input_ids == self.config.class_token_index
        num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

        # max_num_classes from caller (CPU int) avoids GPU→CPU sync via .item()
        max_embed_dim = max_num_classes if max_num_classes is not None else self.config.max_num_classes

        # Get class token pooling method from config (default to "first" for backward compatibility)
        class_token_pooling = getattr(self.config, "class_token_pooling", "first")

        if class_token_pooling == "average":
            # Average all tokens belonging to each class label
            classes_embedding, classes_embedding_mask = self._extract_class_features_averaged(
                token_embeds,
                input_ids,
                attention_mask,
                class_token_mask,
                num_class_tokens,
                max_embed_dim,
                batch_size,
                embed_dim,
            )
        else:
            # Original behavior: use only the class token (or token after it)
            classes_embedding, classes_embedding_mask = self._extract_class_features_first(
                token_embeds,
                input_ids,
                attention_mask,
                class_token_mask,
                num_class_tokens,
                max_embed_dim,
                batch_size,
                embed_dim,
            )

        # Text features extraction
        if self.config.extract_text_features:
            text_token_mask = input_ids == self.config.text_token_index
            text_token_indices = text_token_mask.int().argmax(dim=-1)  # (batch,)
            max_text_length = input_ids.shape[-1]  # static, no GPU→CPU sync

            # (batch, max_text_length): source position in token_embeds for each target slot
            aranged_target_idx = (
                torch.arange(max_text_length, device=token_embeds.device).unsqueeze(0).expand(batch_size, -1)
            )
            valid_mask = aranged_target_idx < (input_ids.shape[-1] - text_token_indices).unsqueeze(1)

            source_indices = (text_token_indices.unsqueeze(1) + aranged_target_idx).clamp(max=input_ids.shape[-1] - 1)
            batch_arange = torch.arange(batch_size, device=token_embeds.device).unsqueeze(1)

            # Gather then zero-out invalid positions — no nonzero/scatter needed
            text_tokens_embeddings = token_embeds[batch_arange, source_indices] * valid_mask.unsqueeze(-1).to(
                token_embeds.dtype
            )
            text_tokens_mask = attention_mask[batch_arange, source_indices] * valid_mask
        else:
            text_tokens_embeddings = token_embeds
            text_tokens_mask = attention_mask
        return classes_embedding, classes_embedding_mask, text_tokens_embeddings, text_tokens_mask

    def _extract_class_features_first(
        self,
        token_embeds,
        input_ids,
        attention_mask,
        class_token_mask,
        num_class_tokens,
        max_embed_dim,
        batch_size,
        embed_dim,
    ):
        """Extract only the class token embedding (or token after it). Fully vectorized."""
        class_cum = class_token_mask.long().cumsum(dim=-1)  # (batch, seq)
        k_range = torch.arange(max_embed_dim, device=token_embeds.device).view(1, -1, 1)

        # select_mask[b, k, s] = True at the position of the k-th class token
        select_mask = class_token_mask.unsqueeze(1) & ((class_cum.unsqueeze(1) - 1) == k_range)

        if not self.config.embed_class_token:
            # Shift right by 1: select the token immediately after each class token
            shifted = torch.zeros_like(select_mask)
            shifted[:, :, 1:] = select_mask[:, :, :-1]
            select_mask = shifted

        classes_embedding = torch.einsum("bks,bsd->bkd", select_mask.to(token_embeds.dtype), token_embeds)

        arange_k = torch.arange(max_embed_dim, device=token_embeds.device).unsqueeze(0)
        classes_embedding_mask = (arange_k < num_class_tokens).to(attention_mask.dtype)

        return classes_embedding, classes_embedding_mask

    def _extract_class_features_averaged(
        self,
        token_embeds,
        input_ids,
        attention_mask,
        class_token_mask,
        num_class_tokens,
        max_embed_dim,
        batch_size,
        embed_dim,
    ):
        """Average all tokens belonging to each class label. Fully vectorized."""
        # class_cum[b, s] = cumulative count of class tokens up to position s
        class_cum = class_token_mask.long().cumsum(dim=-1)  # (batch, seq)

        if self.config.extract_text_features:
            text_token_mask = input_ids == self.config.text_token_index
        else:
            text_token_mask = torch.zeros_like(class_token_mask)
        # text_cum[b, s] >= 1 at and after the text token → use as exclusion boundary
        text_cum = text_token_mask.long().cumsum(dim=-1)  # (batch, seq)

        # span_mask[b, k, s] = True if token s belongs to the span of class k
        k_range = torch.arange(max_embed_dim, device=token_embeds.device).view(1, -1, 1)
        span_mask = (
            (class_cum.unsqueeze(1) == (k_range + 1))  # in the span of class k
            & (text_cum.unsqueeze(1) == 0)  # before the text boundary
            & attention_mask.unsqueeze(1).bool()  # real token (not padding)
        )
        if not self.config.embed_class_token:
            span_mask = span_mask & ~class_token_mask.unsqueeze(1)

        span_float = span_mask.to(token_embeds.dtype)  # (batch, max_embed_dim, seq)
        class_counts = span_float.sum(dim=-1, keepdim=True).clamp(min=1)
        classes_embedding = torch.einsum("bks,bsd->bkd", span_float, token_embeds) / class_counts

        arange_k = torch.arange(max_embed_dim, device=token_embeds.device).unsqueeze(0)
        classes_embedding_mask = (arange_k < num_class_tokens).to(attention_mask.dtype)

        return classes_embedding, classes_embedding_mask

    def get_loss(self, logits, labels, classes_embedding=None, classes_embedding_mask=None):
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
                all_losses = focal_loss_with_logits(
                    logits,
                    labels,
                    self.config.focal_loss_alpha,
                    self.config.focal_loss_gamma,
                    self.config.focal_loss_reduction,
                )
                if classes_embedding_mask is not None:
                    all_losses = all_losses * classes_embedding_mask.float()
                loss = all_losses.mean()

            if self.config.contrastive_loss_coef > 0 and classes_embedding is not None:
                contrastive_loss = sequence_contrastive_loss(classes_embedding, classes_embedding_mask)
                loss = loss + contrastive_loss * self.config.contrastive_loss_coef
        return loss


class GLiClassUniEncoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        config_name = config.encoder_config.__class__.__name__

        model_kwargs = {}
        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(
                    f"The llm2vec package must be installed to use this decoder model: {config_name}"
                )
            else:
                print("Loading decoder model using LLM2Vec...")
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            decoder = True
        elif config_name in {"T5Config", "MT5Config", "UMT5Config"}:
            decoder = False
            turbot5_type = os.environ.get("TURBOT5_ATTN_TYPE", "")
            if turbot5_type and IS_TURBOT5:
                ModelClass = FlashT5EncoderModel
                model_kwargs = {"attention_type": turbot5_type}
            elif config_name == "UMT5Config":
                ModelClass = UMT5EncoderModel
            else:
                ModelClass = T5EncoderModel
        elif config_name in {"DebertaV2Config"}:
            decoder = False
            if os.environ.get("USE_FLASHDEBERTA", "") and IS_FLASHDEBERTA:
                print("Using FlashDeberta backend.")
                ModelClass = FlashDebertaV2Model
            else:
                ModelClass = DebertaV2Model

        else:
            decoder = False
            ModelClass = AutoModel

        if from_pretrained:
            self.encoder_model = ModelClass.from_pretrained(config.encoder_model_name, **model_kwargs)
        elif decoder:
            self.encoder_model = ModelClass(config.encoder_config)
        elif config_name in {"T5Config", "MT5Config", "UMT5Config", "DebertaV2Config"}:
            self.encoder_model = ModelClass._from_config(config.encoder_config)
        else:
            self.encoder_model = ModelClass.from_config(config.encoder_config)

        if config.vocab_size is not None and hasattr(self.encoder_model, "resize_token_embeddings"):
            current_vocab = self.encoder_model.config.vocab_size
            if current_vocab != config.vocab_size:
                self.encoder_model.resize_token_embeddings(config.vocab_size)

        adapter_config_file = Path(config.encoder_model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(
                    "Adapter configs were detected, if you want to apply them you need to install peft package.",
                    stacklevel=2,
                )
            else:
                adapter_config = LoraConfig.from_pretrained(config.encoder_model_name)
                self.encoder_model = get_peft_model(self.encoder_model, adapter_config)

        if config.use_segment_embeddings:
            self.segment_embeddings = nn.Embedding(3, config.encoder_config.hidden_size)
            nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=config.initializer_range)

    def _create_segment_ids(self, input_ids):
        batch_size, _seq_length = input_ids.shape
        segment_ids = torch.zeros_like(input_ids)  # Default: segment 0 (labels)

        # Find example token positions
        example_token_mask = input_ids == self.config.example_token_index
        example_token_indices = example_token_mask.int().argmin(dim=-1)
        has_example = example_token_mask.any(dim=-1)

        text_token_mask = input_ids == self.config.text_token_index
        text_token_indices = text_token_mask.int().argmax(dim=-1)

        for batch_idx in range(batch_size):
            text_start = text_token_indices[batch_idx].item()

            # If examples exist, assign segment 1 to example section
            if has_example[batch_idx]:
                example_start = example_token_indices[batch_idx].item()
                segment_ids[batch_idx, text_start:example_start] = 1
                segment_ids[batch_idx, example_start:] = 2
            else:
                segment_ids[batch_idx, text_start:] = 1

        return segment_ids

    def process_encoder_output(self, input_ids, attention_mask, encoder_layer, labels=None, max_num_classes=None):
        classes_embedding, classes_embedding_mask, text_token_embeddings, text_mask = self._extract_class_features(
            encoder_layer, input_ids, attention_mask, max_num_classes
        )
        if self.config.use_lstm:
            text_token_embeddings = self.lstm(text_token_embeddings, text_mask)

        pooled_output = self.pooler(text_token_embeddings)
        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)
        if self.config.normalize_features:
            pooled_output = pooled_output / (pooled_output.norm(p=2, dim=-1, keepdim=True) + self.epsilon)

        classes_embedding = self.classes_projector(classes_embedding)
        if self.config.normalize_features:
            classes_embedding = classes_embedding / (classes_embedding.norm(p=2, dim=-1, keepdim=True) + self.epsilon)

        logits = self.scorer(pooled_output, classes_embedding, text_mask=text_mask)

        if self.config.normalize_features:
            logits = logits * self.logit_scale.to(classes_embedding.device)

        loss = self.get_loss(logits, labels, classes_embedding, classes_embedding_mask)
        return (logits, loss, pooled_output, classes_embedding)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
        return_dict: bool | None = None,
        max_num_classes: int | None = None,
        **kwargs,
    ) -> Tuple | GLiClassOutput:
        r"""
        Labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.squeeze_layers or self.config.layer_wise:
            output_hidden_states = True
            return_dict = True

        if self.config.use_segment_embeddings:
            embedding_layer = self.encoder_model.get_input_embeddings()
            token_embeds = embedding_layer(input_ids)

            segment_ids = self._create_segment_ids(input_ids)
            segment_embeds = self.segment_embeddings(segment_ids)

            inputs_embeds = token_embeds + segment_embeds

            outputs = self.encoder_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            outputs = self.encoder_model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if self.config.layer_wise and labels is not None:
            hidden_states = outputs.hidden_states
            loss = 0
            for encoder_layer in hidden_states:
                logits, layer_loss, pooled_output, classes_embedding = self.process_encoder_output(
                    input_ids, attention_mask, encoder_layer, labels, max_num_classes
                )
                loss += layer_loss
        else:
            if self.config.encoder_layer_id == -1:
                if self.config.squeeze_layers:
                    encoder_layer = self.layer_wise_attention(outputs.hidden_states)
                else:
                    encoder_layer = outputs[0]
            else:
                encoder_layer = outputs.hidden_states[self.config.encoder_layer_id]
            logits, loss, pooled_output, classes_embedding = self.process_encoder_output(
                input_ids, attention_mask, encoder_layer, labels, max_num_classes
            )

        if not return_dict:
            output = (logits, *outputs[1:])
            return ((loss, *output)) if loss is not None else output

        return GLiClassOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            text_embeddings=pooled_output if output_text_embeddings else None,
            class_embeddings=classes_embedding if output_class_embeddings else None,
        )


class GLiClassEncoderDecoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        if not config.encoder_config.is_encoder_decoder:
            raise ValueError("You need to choose encoder-decoder model as a backbone.")

        if from_pretrained:
            self.encoder_decoder_model = AutoModel.from_pretrained(config.encoder_model_name)
        else:
            self.encoder_decoder_model = AutoModel.from_config(config.encoder_config)

    @staticmethod
    def _make_bidirectional_4d_mask(attention_mask_2d, dtype):
        """Convert a 2D padding mask into a 4D bidirectional attention mask.

        When a 4D mask is passed to the decoder, the model uses it as-is
        without applying its default causal pattern, enabling bidirectional
        self-attention in the decoder.

        Args:
            attention_mask_2d: (batch_size, seq_length) with 1 for real tokens, 0 for padding.
            dtype: The dtype of the model (needed for the min-value fill).

        Returns:
            4D mask of shape (batch_size, 1, seq_length, seq_length).
            Values are 0.0 for attended positions and a large negative value for masked positions.
        """
        batch_size, seq_length = attention_mask_2d.shape
        # (batch_size, 1, 1, seq_length) - masks out padding columns
        padding_mask = (1.0 - attention_mask_2d.to(dtype))[:, None, None, :] * torch.finfo(dtype).min
        return padding_mask.expand(batch_size, 1, seq_length, seq_length)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        class_input_ids: torch.Tensor | None = None,
        class_attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
        return_dict: bool | None = True,
        **kwargs,
    ) -> Tuple | SequenceClassifierOutput:
        r"""
        Labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Build a 4D bidirectional mask for the decoder so it attends to
        # all non-padding positions instead of using causal masking.
        decoder_4d_mask = None
        if class_attention_mask is not None:
            model_dtype = next(self.encoder_decoder_model.parameters()).dtype
            decoder_4d_mask = self._make_bidirectional_4d_mask(class_attention_mask, model_dtype)

        outputs = self.encoder_decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=class_input_ids,
            decoder_attention_mask=decoder_4d_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        text_token_embeddings = outputs.encoder_last_hidden_state
        decoder_token_embeddings = outputs.last_hidden_state
        classes_embedding, classes_embedding_mask, _, _ = self._extract_class_features(
            decoder_token_embeddings, class_input_ids, class_attention_mask
        )

        if self.config.use_lstm:
            text_token_embeddings = self.lstm(text_token_embeddings, attention_mask)

        pooled_output = self.pooler(text_token_embeddings)
        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)
        if self.config.normalize_features:
            pooled_output = nn.functional.normalize(pooled_output, p=2, dim=-1, eps=self.epsilon)

        classes_embedding = self.classes_projector(classes_embedding)
        if self.config.normalize_features:
            classes_embedding = nn.functional.normalize(classes_embedding, p=2, dim=-1, eps=self.epsilon)

        logits = self.scorer(pooled_output, classes_embedding)

        if self.config.normalize_features:
            logits = logits * self.logit_scale.to(classes_embedding.device)

        loss = self.get_loss(logits, labels, classes_embedding, classes_embedding_mask)

        if not return_dict:
            output = (logits, *outputs[1:])
            return ((loss, *output)) if loss is not None else output

        return GLiClassOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.decoder_hidden_states,
            attentions=outputs.decoder_attentions,
            text_embeddings=pooled_output if output_text_embeddings else None,
            class_embeddings=classes_embedding if output_class_embeddings else None,
        )


class GLiClassEncoderDecoderCLS(GLiClassBaseModel):
    """Encoder-decoder architecture where labels go to the encoder and text goes to the decoder.

    Class features are extracted from encoder output using _extract_class_features().
    Text features are extracted from the last non-padding token of the decoder output.
    """

    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        if not config.encoder_config.is_encoder_decoder:
            raise ValueError("You need to choose encoder-decoder model as a backbone.")

        if from_pretrained:
            self.encoder_decoder_model = AutoModel.from_pretrained(config.encoder_model_name)
        else:
            self.encoder_decoder_model = AutoModel.from_config(config.encoder_config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        class_input_ids: torch.Tensor | None = None,
        class_attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
        return_dict: bool | None = True,
        **kwargs,
    ) -> Tuple | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Labels → encoder, Text → decoder
        outputs = self.encoder_decoder_model(
            input_ids=class_input_ids,
            attention_mask=class_attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # Class features from encoder output
        encoder_token_embeddings = outputs.encoder_last_hidden_state
        classes_embedding, classes_embedding_mask, _, _ = self._extract_class_features(
            encoder_token_embeddings, class_input_ids, class_attention_mask
        )

        # Text features from decoder's last non-padding token
        decoder_output = outputs.last_hidden_state
        batch_size = decoder_output.shape[0]
        last_non_pad_idx = attention_mask.sum(dim=1) - 1
        pooled_output = decoder_output[torch.arange(batch_size, device=decoder_output.device), last_non_pad_idx]

        pooled_output = self.text_projector(pooled_output)
        pooled_output = self.dropout(pooled_output)
        if self.config.normalize_features:
            pooled_output = nn.functional.normalize(pooled_output, p=2, dim=-1, eps=self.epsilon)

        classes_embedding = self.classes_projector(classes_embedding)
        if self.config.normalize_features:
            classes_embedding = nn.functional.normalize(classes_embedding, p=2, dim=-1, eps=self.epsilon)

        logits = self.scorer(pooled_output, classes_embedding)

        if self.config.normalize_features:
            logits = logits * self.logit_scale.to(classes_embedding.device)

        loss = self.get_loss(logits, labels, classes_embedding, classes_embedding_mask)

        if not return_dict:
            output = (logits, *outputs[1:])
            return ((loss, *output)) if loss is not None else output

        return GLiClassOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.decoder_hidden_states,
            attentions=outputs.decoder_attentions,
            text_embeddings=pooled_output if output_text_embeddings else None,
            class_embeddings=classes_embedding if output_class_embeddings else None,
        )


class GLiClassBiEncoder(GLiClassBaseModel):
    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config)
        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder model name to use it as a backbone.")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        if config.label_model_config is None:
            if config.label_model_name is None:
                raise ValueError("You need to specify label model name to use it as a backbone.")
            config.label_model_config = AutoConfig.from_pretrained(config.label_model_name)

        def initialize_encoder(configs, model_name, from_pretrained):
            if from_pretrained:
                return AutoModel.from_pretrained(model_name)
            else:
                return AutoModel.from_config(configs)

        self.encoder_model = initialize_encoder(config.encoder_config, config.encoder_model_name, from_pretrained)
        self.label_encoder = initialize_encoder(config.label_model_config, config.label_model_name, from_pretrained)
        self.biencoder_projector = BiEncoderProjector(config)

    def pool_outputs(self, encoder_outputs):
        text_embeddings = self.pooler(encoder_outputs[0])
        text_embeddings = self.text_projector(text_embeddings)
        text_embeddings = self.dropout(text_embeddings)
        if self.config.normalize_features:
            text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=-1, eps=self.epsilon)
        return text_embeddings

    def encode_text(self, input_ids, attention_mask, adapter_ids=None):
        encoder_kwargs = {}
        if adapter_ids is not None:
            encoder_kwargs["adapter_ids"] = adapter_ids
        outputs = self.encoder_model(
            input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1),
            **encoder_kwargs,
        )
        text_embeddings = self.pool_outputs(outputs)
        return text_embeddings

    def encode_classes(self, class_input_ids, class_attention_mask, labels_mask=None):
        batch_size = class_input_ids.shape[0]
        num_classes = class_input_ids.shape[1]
        if labels_mask is not None:
            batch_indices, indices = torch.where(labels_mask == 1)
            selected_input_ids = class_input_ids[batch_indices, indices]
            selected_attention_mask = class_attention_mask[batch_indices, indices]

            outputs = self.label_encoder(selected_input_ids, attention_mask=selected_attention_mask)
            class_embeddings_filtered = self.pooler(outputs[0])

            class_embeddings = torch.zeros(
                batch_size,
                num_classes,
                class_embeddings_filtered.shape[-1],
                dtype=class_embeddings_filtered.dtype,
                device=class_embeddings_filtered.device,
            )

            class_embeddings[batch_indices, indices] = class_embeddings_filtered
        else:
            class_input_ids = class_input_ids.view(-1, class_input_ids.shape[-1])
            class_attention_mask = class_attention_mask.view(-1, class_input_ids.shape[-1])
            outputs = self.label_encoder(class_input_ids, attention_mask=class_attention_mask)
            class_embeddings = self.pooler(outputs[0])
            class_embeddings = class_embeddings.reshape(batch_size, num_classes, -1)
        class_embeddings = self.biencoder_projector(class_embeddings)
        class_embeddings = self.classes_projector(class_embeddings)
        if self.config.normalize_features:
            class_embeddings = nn.functional.normalize(class_embeddings, p=2, dim=-1, eps=self.epsilon)
        return class_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        class_input_ids: torch.Tensor | None = None,
        class_attention_mask: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
        return_dict: bool | None = None,
        adapter_ids: list[str] | None = None,
        **kwargs,
    ) -> Tuple | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_embeddings = self.encode_text(input_ids, attention_mask, adapter_ids=adapter_ids)
        class_embeddings = self.encode_classes(class_input_ids, class_attention_mask, labels_mask)
        logits = self.scorer(text_embeddings, class_embeddings) * self.logit_scale.to(class_embeddings.device)

        if labels_mask is not None:
            logits = torch.where(labels_mask == 0, -1e3, logits)

        loss = self.get_loss(logits, labels, classes_embedding_mask=labels_mask)

        if not return_dict:
            output = (logits,)
            return ((loss, *output)) if loss is not None else output

        return GLiClassOutput(
            loss=loss,
            logits=logits,
            text_embeddings=text_embeddings if output_text_embeddings else None,
            class_embeddings=class_embeddings if output_class_embeddings else None,
        )


class GLiClassBiEncoderFused(GLiClassBiEncoder):
    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__(config, from_pretrained)

    def encode_text(self, input_ids, attention_mask, class_embeddings, labels_mask, adapter_ids=None):
        embedding_layer = self.encoder_model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        class_token_mask = input_ids == self.config.class_token_index
        batch_indices, class_token_indices = torch.where(class_token_mask)

        labels_batch_indices, labels_indices = torch.where(labels_mask == 1)

        selected_class_embeddings = class_embeddings[labels_batch_indices, labels_indices]

        inputs_embeds[batch_indices, class_token_indices] = selected_class_embeddings
        encoder_kwargs = {}
        if adapter_ids is not None:
            encoder_kwargs["adapter_ids"] = adapter_ids
        encoder_outputs = self.encoder_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.squeeze(1),
            **encoder_kwargs,
        )

        post_class_embeddings = torch.zeros_like(class_embeddings)
        post_class_embeddings[labels_batch_indices, labels_indices] = encoder_outputs[0][
            batch_indices, class_token_indices
        ]
        return encoder_outputs, post_class_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        class_input_ids: torch.Tensor | None = None,
        class_attention_mask: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
        return_dict: bool | None = None,
        adapter_ids: list[str] | None = None,
        **kwargs,
    ) -> Tuple | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        raw_class_embeddings = self.encode_classes(class_input_ids, class_attention_mask, labels_mask)

        encoder_outputs, class_embeddings = self.encode_text(
            input_ids, attention_mask, raw_class_embeddings, labels_mask, adapter_ids=adapter_ids
        )

        text_embeddings = self.pool_outputs(encoder_outputs)

        logits = self.scorer(text_embeddings, class_embeddings) * self.logit_scale.to(class_embeddings.device)

        if labels_mask is not None:
            logits = torch.where(labels_mask == 0, -1e3, logits)

        loss = self.get_loss(logits, labels, classes_embedding_mask=labels_mask)

        if not return_dict:
            output = (logits,)
            return ((loss, *output)) if loss is not None else output

        return GLiClassOutput(
            loss=loss,
            logits=logits,
            text_embeddings=text_embeddings if output_text_embeddings else None,
            class_embeddings=class_embeddings if output_class_embeddings else None,
        )


class GLiClassDecoderKV(nn.Module):
    """
    Decoder-KV architecture with dynamic KV cache for streaming classification.

    Sequence format: [prompt][examples]text<<SEP>>label1<<LABEL>>label2<<LABEL>>...<<SEP>>

    Cached part: [prompt][examples]
    New part each time: text<<SEP>>labels...<<SEP>>

    Flow:
    1. Decoder backbone (Qwen3) processes full sequence with past_key_values
    2. Update KV cache ONLY with [prompt][examples]text part (before labels <<SEP>>)
    3. Hidden states → DecoderKVScorer (bidirectional encoder + extraction + MLP)
    """

    def __init__(self, config: GLiClassModelConfig, from_pretrained=False):
        super().__init__()
        self.config = config

        if config.encoder_config is None:
            if config.encoder_model_name is None:
                raise ValueError("You need to specify encoder_model_name for decoder backbone (Qwen3).")
            config.encoder_config = AutoConfig.from_pretrained(config.encoder_model_name)

        config_name = config.encoder_config.__class__.__name__

        if config_name == "Qwen3_5TextConfig":
            from transformers.models.qwen3_5 import Qwen3_5TextModel

            ModelClass = Qwen3_5TextModel
        elif config_name == "Qwen3_5Config":
            from transformers.models.qwen3_5 import Qwen3_5TextModel

            ModelClass = Qwen3_5TextModel
            config.encoder_config = config.encoder_config.text_config
        elif config_name == "Qwen3Config":
            from transformers import Qwen3Model

            ModelClass = Qwen3Model
        else:
            raise ValueError(f"decoder-kv architecture requires Qwen3 or Qwen3.5. Got: {config_name}")

        if from_pretrained:
            self.decoder_model = ModelClass.from_pretrained(config.encoder_model_name)
        else:
            self.decoder_model = ModelClass(config.encoder_config)

        if config.vocab_size is not None and hasattr(self.decoder_model, "resize_token_embeddings"):
            current_vocab = self.decoder_model.config.vocab_size
            if current_vocab != config.vocab_size:
                self.decoder_model.resize_token_embeddings(config.vocab_size)

        from .scorers import DecoderKVScorer

        self.scorer = DecoderKVScorer(config)

        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.num_labels = -1

        self.sep_token_id = config.sep_token_index

    def get_loss(self, logits, labels):
        loss = None
        if labels is not None:
            # Sequence truncation may cut label tokens → align labels to logits
            num_labels = logits.shape[-1]
            if labels.shape[-1] != num_labels:
                labels = labels[:, :num_labels]

            if self.config.problem_type == "multi_label_classification":
                from .loss_functions import focal_loss_with_logits

                reduction = self.config.focal_loss_reduction or "none"
                all_losses = focal_loss_with_logits(
                    logits,
                    labels,
                    self.config.focal_loss_alpha,
                    self.config.focal_loss_gamma,
                    reduction,
                )
                loss = all_losses.mean()
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                raise NotImplementedError(f"{self.config.problem_type} is not implemented.")
        return loss

    def _extract_label_section(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract label section hidden states from full-sequence decoder output.

        Training sequence: [prompt][examples<<SEP>>]text<<SEP>>label1<<LABEL>>...<<SEP>>
        Scorer receives:   label1<<LABEL>>...<<SEP>>   (matches classify() inference format)

        Start pos = first token after last <<SEP>> before first <<LABEL>>.
        End pos = last real token (inclusive, determined by attention_mask).

        Returns:
            padded_hidden: (batch, max_label_len, hidden_size)
            padded_ids:    (batch, max_label_len)
            label_mask:    (batch, max_label_len)
        """
        batch_size, _, hidden_size = hidden_states.shape
        sep_id = self.sep_token_id
        label_id = self.config.class_token_index

        slices_h, slices_ids = [], []

        for i in range(batch_size):
            real_len = int(attention_mask[i].sum().item())
            ids_i = input_ids[i, :real_len]

            label_pos = (ids_i == label_id).nonzero(as_tuple=False)
            sep_pos = (ids_i == sep_id).nonzero(as_tuple=False)

            if label_pos.numel() == 0 or sep_pos.numel() == 0:
                # Fallback: use everything up to real_len
                slices_h.append(hidden_states[i, :real_len])
                slices_ids.append(ids_i)
                continue

            first_label = label_pos[0].item()
            # last <<SEP>> strictly before first <<LABEL>>
            seps_before = sep_pos[sep_pos < first_label]

            if seps_before.numel() == 0:
                slices_h.append(hidden_states[i, :real_len])
                slices_ids.append(ids_i)
                continue

            # start right after that SEP → first token of "label1<<LABEL>>...<<SEP>>"
            start = int(seps_before[-1].item()) + 1
            slices_h.append(hidden_states[i, start:real_len])
            slices_ids.append(ids_i[start:real_len])

        max_len = max(s.shape[0] for s in slices_h)
        padded_hidden = hidden_states.new_zeros(batch_size, max_len, hidden_size)
        padded_ids = input_ids.new_zeros(batch_size, max_len)
        label_mask = attention_mask.new_zeros(batch_size, max_len)

        for i, (h, ids) in enumerate(zip(slices_h, slices_ids)):
            length = h.shape[0]
            padded_hidden[i, :length] = h
            padded_ids[i, :length] = ids
            label_mask[i, :length] = 1

        return padded_hidden, padded_ids, label_mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: Full sequence [prompt][examples]text<<SEP>>labels...<<SEP>>
            attention_mask: Attention mask
            past_key_values: Cached KV for [prompt][examples] (optional)
            labels: Classification labels
            return_dict: Return dict output
            use_cache: Whether to return updated cache

        Returns:
            GLiClassOutput with logits, loss, and optionally past_key_values
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder_outputs = self.decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        hidden_states = decoder_outputs.last_hidden_state

        label_hidden, label_ids, label_mask = self._extract_label_section(hidden_states, input_ids, attention_mask)

        logits = self.scorer(
            hidden_states=label_hidden,
            input_ids=label_ids,
            attention_mask=label_mask,
        )

        if labels is not None:
            self.num_labels = logits.shape[-1]

        loss = self.get_loss(logits, labels)

        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (decoder_outputs.past_key_values,)
            return (loss, *output) if loss is not None else output

        return GLiClassOutput(
            loss=loss,
            logits=logits,
            hidden_states=decoder_outputs.hidden_states if hasattr(decoder_outputs, "hidden_states") else None,
            attentions=decoder_outputs.attentions if hasattr(decoder_outputs, "attentions") else None,
        )


class GLiClassModel(GLiClassPreTrainedModel):
    def __init__(self, config, from_pretrained=False):
        super().__init__(config)
        if config.architecture_type == "uni-encoder":
            self.model = GLiClassUniEncoder(config, from_pretrained)
        elif config.architecture_type == "bi-encoder":
            self.model = GLiClassBiEncoder(config, from_pretrained)
        elif config.architecture_type == "bi-encoder-fused":
            self.model = GLiClassBiEncoderFused(config, from_pretrained)
        elif config.architecture_type == "encoder-decoder":
            self.model = GLiClassEncoderDecoder(config, from_pretrained)
        elif config.architecture_type == "encoder-decoder-cls":
            self.model = GLiClassEncoderDecoderCLS(config, from_pretrained)
        elif config.architecture_type == "decoder-kv":
            self.model = GLiClassDecoderKV(config, from_pretrained)
        self.post_init()

    def get_input_embeddings(self):
        if self.config.architecture_type in {"uni-encoder"}:
            return self.model.encoder_model.get_input_embeddings()
        elif self.config.architecture_type in {"encoder-decoder", "encoder-decoder-cls"}:
            return self.model.encoder_decoder_model.get_input_embeddings()
        else:
            raise NotImplementedError("Getting input embeddings is not implemented for bi-encoder architecture")

    def set_input_embeddings(self, value):
        if self.config.architecture_type in {"uni-encoder"}:
            self.model.encoder_model.set_input_embeddings(value)
            return None
        elif self.config.architecture_type in {"encoder-decoder", "encoder-decoder-cls"}:
            self.model.encoder_decoder_model.set_input_embeddings(value)
        elif self.config.architecture_type in {"bi-encoder", "bi-encoder-fused"}:
            self.model.encoder_model.set_input_embeddings(value)
        else:
            raise NotImplementedError("Setting input embeddings is not implemented for bi-encoder architecture")

    def tie_weights(self, recompute_mapping=True, missing_keys=None):
        """
        Tie model weights for architectures that share parameters.

        This method handles:
        - Version compatibility between transformers v4 and v5
        - Different GLiClass architecture types
        - Special handling for T5/MT5 models in transformers v5+ where encoder.embed_tokens
          may be incorrectly initialized instead of being tied to shared.weight

        Args:
            recompute_mapping: Whether to recompute weight mapping (transformers v5+)
            missing_keys: Keys that are missing from checkpoint (transformers v5+)
        """
        # Get encoder model based on architecture type
        encoder_model = None
        if self.config.architecture_type in {"uni-encoder"}:
            encoder_model = self.model.encoder_model
        elif self.config.architecture_type in {"encoder-decoder", "encoder-decoder-cls"}:
            encoder_model = self.model.encoder_decoder_model
        elif self.config.architecture_type in {"bi-encoder", "bi-encoder-fused"}:
            encoder_model = self.model.encoder_model
        elif self.config.architecture_type == "decoder-kv":
            encoder_model = self.model.decoder_model
        else:
            raise NotImplementedError("Tie weights is not implemented for this architecture type")

        # Call base tie_weights with version-appropriate parameters
        if version.parse(transformers.__version__) >= version.parse("5.0.0"):
            result = encoder_model.tie_weights(recompute_mapping=recompute_mapping, missing_keys=missing_keys)
        else:
            result = encoder_model.tie_weights()

        # Fix for T5/MT5/UMT5 models in transformers v5+
        # In v5, if encoder.embed_tokens.weight is missing from checkpoint, it gets randomly
        # initialized instead of being tied to shared.weight. We explicitly ensure proper tying.
        if (
            encoder_model is not None
            and hasattr(encoder_model, "shared")
            and hasattr(encoder_model, "encoder")
            and hasattr(encoder_model.encoder, "embed_tokens")
        ):
            shared_weight = encoder_model.shared.weight
            embed_weight = encoder_model.encoder.embed_tokens.weight

            # Only tie if they're not already the same tensor
            if shared_weight is not embed_weight:
                encoder_model.encoder.embed_tokens.weight = shared_weight
                if version.parse(transformers.__version__) >= version.parse("5.0.0"):
                    logger.info(
                        "Applied transformers v5 compatibility fix: tied encoder.embed_tokens.weight "
                        "to shared.weight for T5-based model"
                    )

        return result

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of=None) -> nn.Embedding:
        if self.config.architecture_type in {"uni-encoder"}:
            model_embeds = self.model.encoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        elif self.config.architecture_type in {"encoder-decoder", "encoder-decoder-cls"}:
            model_embeds = self.model.encoder_decoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        elif self.config.architecture_type in {"bi-encoder-fused"}:
            model_embeds = self.model.encoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        elif self.config.architecture_type == "decoder-kv":
            model_embeds = self.model.decoder_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        else:
            raise NotImplementedError("Resizing is not implemented for bi-encoder architecture")
        self.config.encoder_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        if kwargs.get("adapter_ids") is None:
            kwargs.pop("adapter_ids", None)
        outputs = self.model(*args, **kwargs)
        return outputs
