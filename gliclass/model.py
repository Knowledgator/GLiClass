import os
import warnings
from typing import Any, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
import transformers
from torch import nn
from packaging import version
from transformers import AutoModel, AutoConfig, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
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
from .calibration import build_calibrator
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
    past_key_values: Any | None = None
    recurrent_logits: Tuple[torch.Tensor, ...] | None = None
    recurrent_losses: torch.Tensor | None = None
    recurrent_entropies: torch.Tensor | None = None
    recurrent_num_steps: torch.Tensor | None = None
    # (batch, num_labels), 1 for real labels; set whenever text/class embeddings are requested
    class_mask: torch.Tensor | None = None
    # Set when the model has a calibrator: logits are then calibrated,
    # inverse_temperatures * uncalibrated_logits + calibration_biases
    uncalibrated_logits: torch.Tensor | None = None
    inverse_temperatures: torch.Tensor | None = None
    calibration_biases: torch.Tensor | None = None


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
                reduction = self.config.focal_loss_reduction or "none"
                all_losses = focal_loss_with_logits(
                    logits,
                    labels,
                    self.config.focal_loss_alpha,
                    self.config.focal_loss_gamma,
                    reduction,
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
        return (logits, loss, pooled_output, classes_embedding, classes_embedding_mask)

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
                logits, layer_loss, pooled_output, classes_embedding, classes_mask = self.process_encoder_output(
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
            logits, loss, pooled_output, classes_embedding, classes_mask = self.process_encoder_output(
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
            class_mask=classes_mask,
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
            class_mask=classes_embedding_mask,
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
            class_mask=classes_embedding_mask,
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
            class_mask=labels_mask if labels_mask is not None else torch.ones_like(logits, dtype=torch.long),
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
            class_mask=labels_mask if labels_mask is not None else torch.ones_like(logits, dtype=torch.long),
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
        # EmbeddingGemma 2 is a bidirectional encoder without a KV cache: the whole sequence is encoded in one pass
        self.bidirectional = config_name == "EmbeddingGemma2Config"

        if getattr(config, "multimodal", False):
            # Base multi-modal model (no LM head): encodes images / audio and merges them into the sequence
            if config_name == "Qwen3_5Config":
                from transformers.models.qwen3_5 import Qwen3_5Model as ModelClass
            elif config_name == "Gemma4Config":
                from transformers.models.gemma4 import Gemma4Model as ModelClass
            elif config_name == "EmbeddingGemma2Config":
                from transformers.models.embedding_gemma2 import EmbeddingGemma2Model as ModelClass
            else:
                raise ValueError(
                    f"multimodal decoder-kv requires Qwen3.5, Gemma 4 or EmbeddingGemma 2. Got: {config_name}"
                )
        elif config_name == "EmbeddingGemma2Config":
            from transformers.models.embedding_gemma2 import EmbeddingGemma2Model as ModelClass

            # text only: skip the vision / audio towers (their weights are ignored on load)
            config.encoder_config.vision_config = None
            config.encoder_config.audio_config = None
        elif config_name == "Qwen3_5TextConfig":
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
            raise ValueError(f"decoder-kv architecture requires Qwen3, Qwen3.5 or EmbeddingGemma 2. Got: {config_name}")

        if from_pretrained and self.bidirectional:
            # pass the config so a text-only one (no vision / audio config) skips the media towers
            self.decoder_model = ModelClass.from_pretrained(config.encoder_model_name, config=config.encoder_config)
        elif from_pretrained:
            self.decoder_model = ModelClass.from_pretrained(config.encoder_model_name)
        else:
            self.decoder_model = ModelClass(config.encoder_config)
            # multi-modal backbones build their sub-models in the config's dtype (e.g. bf16); match the scorer
            self.decoder_model.to(torch.get_default_dtype())

        if config.vocab_size is not None and hasattr(self.decoder_model, "resize_token_embeddings"):
            current_vocab = self.decoder_model.config.get_text_config().vocab_size
            if current_vocab != config.vocab_size:
                self.decoder_model.resize_token_embeddings(config.vocab_size)

        if getattr(config, "multimodal", False):
            from .multimodal import freeze_media_encoders, match_media_feature_dtype

            match_media_feature_dtype(self.decoder_model)
            if getattr(config, "freeze_media_encoders", True):
                freeze_media_encoders(self.decoder_model)

        from .scorers import DecoderKVScorer

        self.scorer = DecoderKVScorer(config)

        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.num_labels = -1

        self.sep_token_id = config.sep_token_index

    def get_loss(self, logits, labels, valid_labels=None):
        loss = None
        if labels is not None:
            # Sequence truncation may cut label tokens → align labels to logits
            num_labels = logits.shape[-1]
            # single-label targets are class indices (batch,); only per-label targets need aligning
            if labels.dim() > 1 and labels.shape[-1] != num_labels:
                labels = labels[:, :num_labels]
            loss = self._per_sample_loss(logits, labels, valid_labels).mean()
        return loss

    def _resolve_recurrence(self, use_recurrence=None, max_recurrent_steps=None, halt_threshold=None):
        """Pick (num_steps, halt_threshold) for this call.

        Training: depth sampled uniformly from [recurrent_min_steps, max], no halting.
        Inference: up to max steps, halting per example once probabilities stop changing.
        """
        if not self.scorer.recurrent:
            return 1, None
        if use_recurrence is None:
            use_recurrence = self.training or self.config.recurrent_inference
        if not use_recurrence:
            return 1, None

        if self.training:
            max_steps = max_recurrent_steps or self.config.recurrent_steps
            min_steps = min(self.config.recurrent_min_steps or 1, max_steps)
            return int(torch.randint(min_steps, max_steps + 1, (1,)).item()), None

        max_steps = max_recurrent_steps or self.config.recurrent_inference_max_steps or self.config.recurrent_steps
        if halt_threshold is None:
            halt_threshold = self.config.recurrent_halt_threshold
        return max_steps, halt_threshold

    def _per_sample_loss(self, logits, labels, valid_labels=None):
        """Unreduced loss per example, shape (batch,).

        valid_labels (batch, num_labels) marks the example's real labels. Slots past them only exist because
        the batch is padded to its longest label list, so they are left out; otherwise the loss would depend
        on which examples share a batch.
        """
        if self.config.problem_type == "multi_label_classification":
            from .loss_functions import focal_loss_with_logits

            all_losses = focal_loss_with_logits(
                logits,
                labels,
                self.config.focal_loss_alpha,
                self.config.focal_loss_gamma,
                "none",
            )
            valid = labels.ne(self.config.ignore_index)
            if valid_labels is not None:
                valid = valid & valid_labels
            return (all_losses * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        elif self.config.problem_type == "single_label_classification":
            if valid_labels is not None:
                logits = logits.masked_fill(~valid_labels, torch.finfo(logits.dtype).min)
            return nn.functional.cross_entropy(
                logits, labels.view(-1), ignore_index=self.config.ignore_index, reduction="none"
            )
        raise NotImplementedError(f"{self.config.problem_type} is not implemented.")

    def _per_sample_entropy(self, logits, valid_labels):
        """Mean uncertainty per example over its real labels, shape (batch,).

        Multi-label: binary entropy of each sigmoid probability (0 when p is 0 or 1, max at 0.5).
        Single-label: entropy of the softmax over the example's labels.
        """
        if self.config.problem_type == "single_label_classification":
            log_probs = torch.log_softmax(logits.masked_fill(~valid_labels, float("-inf")), dim=-1)
            return -(log_probs.exp() * log_probs.masked_fill(~valid_labels, 0.0)).sum(-1)
        probs = torch.sigmoid(logits)
        entropy = nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction="none")
        return (entropy * valid_labels).sum(-1) / valid_labels.sum(-1).clamp_min(1)

    def get_recurrent_loss(self, step_logits, labels, valid_labels=None):
        """Loss over all reasoning steps.

        loss = mean_t L_t                                                 (focal loss after every step)
             + coef * mean_{t>=3} relu(l_t - (1 - margin) * sg(l_{t-1}))  (deeper must beat shallower)
             + conf_coef * confidence term on recurrent steps t >= 2      (deeper must be more certain)

        The improvement term starts at step 3 vs step 2, so step 1 is only ever trained by its own
        loss. The stop-gradient on l_{t-1} means the hinge can only push the deeper step down; it
        never rewards making the earlier step worse.

        Confidence term (recurrent_confidence_coef > 0), with H_t the mean label entropy of step t:
            "relative": relu(H_t - (1 - conf_margin) * sg(H_{t-1}))  each recurrence less uncertain than the last
            "absolute": H_t                                          every recurrence as certain as possible

        Returns:
            loss: scalar
            step_losses: (num_steps,) detached mean loss per step, for monitoring
            step_entropies: (num_steps,) detached mean label entropy per step, for monitoring
        """
        num_labels = step_logits[-1].shape[-1]
        if labels.dim() > 1 and labels.shape[-1] != num_labels:
            labels = labels[:, :num_labels]
        if valid_labels is None:
            valid_labels = torch.ones_like(step_logits[-1], dtype=torch.bool)

        per_sample = torch.stack(
            [self._per_sample_loss(logits, labels, valid_labels) for logits in step_logits]
        )  # (T, B)
        step_losses = per_sample.mean(dim=1)
        loss = step_losses.mean()

        if len(step_logits) >= 3 and self.config.recurrent_improvement_coef > 0:
            target = (1.0 - self.config.recurrent_improvement_margin) * per_sample[1:-1].detach()
            improvement = torch.relu(per_sample[2:] - target).mean()
            loss = loss + self.config.recurrent_improvement_coef * improvement

        entropies = torch.stack([self._per_sample_entropy(logits, valid_labels) for logits in step_logits])  # (T, B)
        confidence_coef = getattr(self.config, "recurrent_confidence_coef", 0.0)
        if len(step_logits) >= 2 and confidence_coef > 0:
            mode = getattr(self.config, "recurrent_confidence_mode", "relative")
            if mode == "relative":
                margin = getattr(self.config, "recurrent_confidence_margin", 0.0)
                confidence = torch.relu(entropies[1:] - (1.0 - margin) * entropies[:-1].detach()).mean()
            elif mode == "absolute":
                confidence = entropies[1:].mean()
            else:
                raise ValueError(f"Unknown recurrent_confidence_mode: {mode}")
            loss = loss + confidence_coef * confidence

        return loss, step_losses.detach(), entropies.mean(dim=1).detach()

    def _extract_label_section(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_text_section: bool = False,
    ) -> tuple[torch.Tensor, ...]:
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
            with return_text_section, also everything before the label section:
            text_hidden:   (batch, max_text_len, hidden_size)
            text_mask:     (batch, max_text_len)
        """
        batch_size = hidden_states.shape[0]
        sep_id = self.sep_token_id
        label_id = self.config.class_token_index

        slices_h, slices_ids, text_slices = [], [], []

        for i in range(batch_size):
            real_len = int(attention_mask[i].sum().item())
            ids_i = input_ids[i, :real_len]

            label_pos = (ids_i == label_id).nonzero(as_tuple=False)
            sep_pos = (ids_i == sep_id).nonzero(as_tuple=False)

            if label_pos.numel() == 0 or sep_pos.numel() == 0:
                # Fallback: use everything up to real_len
                slices_h.append(hidden_states[i, :real_len])
                slices_ids.append(ids_i)
                text_slices.append(hidden_states[i, :real_len])
                continue

            first_label = label_pos[0].item()
            # last <<SEP>> strictly before first <<LABEL>>
            seps_before = sep_pos[sep_pos < first_label]

            if seps_before.numel() == 0:
                slices_h.append(hidden_states[i, :real_len])
                slices_ids.append(ids_i)
                text_slices.append(hidden_states[i, :real_len])
                continue

            # start right after that SEP → first token of "label1<<LABEL>>...<<SEP>>"
            start = int(seps_before[-1].item()) + 1
            slices_h.append(hidden_states[i, start:real_len])
            slices_ids.append(ids_i[start:real_len])
            # text section ends with that SEP, so it is never empty
            text_slices.append(hidden_states[i, :start])

        padded_hidden = pad_sequence(slices_h, batch_first=True)
        padded_ids = pad_sequence(slices_ids, batch_first=True)

        section_lengths = torch.tensor(
            [section.shape[0] for section in slices_h],
            device=attention_mask.device,
        )
        positions = torch.arange(padded_hidden.shape[1], device=attention_mask.device)
        label_mask = (positions.unsqueeze(0) < section_lengths.unsqueeze(1)).to(attention_mask.dtype)

        if not return_text_section:
            return padded_hidden, padded_ids, label_mask

        text_hidden = pad_sequence(text_slices, batch_first=True)
        text_lengths = torch.tensor([section.shape[0] for section in text_slices], device=attention_mask.device)
        text_positions = torch.arange(text_hidden.shape[1], device=attention_mask.device)
        text_mask = (text_positions.unsqueeze(0) < text_lengths.unsqueeze(1)).to(attention_mask.dtype)
        return padded_hidden, padded_ids, label_mask, text_hidden, text_mask

    def _require_kv_cache(self):
        if self.bidirectional:
            raise ValueError(
                f"{type(self.decoder_model).__name__} is a bidirectional encoder without a KV cache; "
                "use the classic (non-streaming) decoder-kv pipeline."
            )

    def update_decoder_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
    ):
        """Extend a text-only decoder cache without running the scorer."""
        self._require_kv_cache()
        return self.decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def classify_from_decoder_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
        use_recurrence: bool | None = None,
        max_recurrent_steps: int | None = None,
        halt_threshold: float | None = None,
        text_hidden_states: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
    ) -> GLiClassOutput:
        """Classify label tokens against a text cache without persisting labels.

        With config.recurrent_read_text, pass the decoder's last hidden states of the cached text
        (text_hidden_states / text_attention_mask); the KV cache alone does not contain them.
        """
        self._require_kv_cache()
        decoder_outputs = self.decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )

        label_hidden = decoder_outputs.last_hidden_state[:, -input_ids.shape[1] :, :]
        scorer_device = next(self.scorer.parameters()).device
        num_steps, halt_threshold = self._resolve_recurrence(use_recurrence, max_recurrent_steps, halt_threshold)
        label_ids = input_ids.to(scorer_device)
        label_mask = label_mask.to(scorer_device)
        step_logits, steps_taken, (text_repr, label_repr) = self.scorer.recurrent_forward(
            hidden_states=label_hidden.to(scorer_device),
            input_ids=label_ids,
            attention_mask=label_mask,
            num_steps=num_steps,
            halt_threshold=halt_threshold,
            text_hidden_states=None if text_hidden_states is None else text_hidden_states.to(scorer_device),
            text_attention_mask=None if text_attention_mask is None else text_attention_mask.to(scorer_device),
            return_representations=True,
        )
        logits = step_logits[-1]
        return GLiClassOutput(
            logits=logits,
            recurrent_num_steps=steps_taken,
            text_embeddings=text_repr if output_text_embeddings else None,
            class_embeddings=label_repr if output_class_embeddings else None,
            class_mask=self.scorer._valid_label_mask(label_ids, label_mask, logits.shape[-1]).long(),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        use_cache: bool = False,
        use_recurrence: bool | None = None,
        max_recurrent_steps: int | None = None,
        halt_threshold: float | None = None,
        output_recurrent_logits: bool = False,
        output_text_embeddings: bool | None = None,
        output_class_embeddings: bool | None = None,
        max_num_classes: int | None = None,
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
            use_recurrence: Enable/disable scorer recurrence for this call (default: config)
            max_recurrent_steps: Max reasoning steps for this call (training samples depth up to it)
            halt_threshold: Inference early-stop threshold on label probability change (0 disables)
            output_recurrent_logits: Return logits of every reasoning step
            output_text_embeddings / output_class_embeddings: Return the scorer's text / label
                representations of the final step
            max_num_classes: Unused, accepted for pipeline compatibility

        Returns:
            GLiClassOutput with logits (of the last step), loss, and optionally past_key_values
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.bidirectional:
            if past_key_values is not None or use_cache:
                self._require_kv_cache()
            # media positions come from the placeholder token ids; padding mm_token_type_ids are not accepted
            kwargs.pop("mm_token_type_ids", None)
            cache_kwargs = {}
        else:
            cache_kwargs = {"past_key_values": past_key_values, "use_cache": use_cache}

        decoder_outputs = self.decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **cache_kwargs,
            **kwargs,
        )

        hidden_states = decoder_outputs.last_hidden_state

        num_steps, halt_threshold = self._resolve_recurrence(use_recurrence, max_recurrent_steps, halt_threshold)

        text_hidden = text_mask = None
        if self.scorer.read_text and num_steps > 1:
            label_hidden, label_ids, label_mask, text_hidden, text_mask = self._extract_label_section(
                hidden_states, input_ids, attention_mask, return_text_section=True
            )
        else:
            label_hidden, label_ids, label_mask = self._extract_label_section(hidden_states, input_ids, attention_mask)

        step_logits, steps_taken, (text_repr, label_repr) = self.scorer.recurrent_forward(
            hidden_states=label_hidden,
            input_ids=label_ids,
            attention_mask=label_mask,
            num_steps=num_steps,
            halt_threshold=halt_threshold,
            return_all_steps=labels is not None or output_recurrent_logits,
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            return_representations=True,
        )
        logits = step_logits[-1]

        if labels is not None:
            self.num_labels = logits.shape[-1]

        recurrent_losses = recurrent_entropies = None
        valid_labels = self.scorer._valid_label_mask(label_ids, label_mask, logits.shape[-1])
        if len(step_logits) > 1 and labels is not None:
            loss, recurrent_losses, recurrent_entropies = self.get_recurrent_loss(step_logits, labels, valid_labels)
        else:
            loss = self.get_loss(logits, labels, valid_labels)

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
            past_key_values=decoder_outputs.past_key_values if use_cache else None,
            recurrent_logits=tuple(step_logits) if output_recurrent_logits else None,
            recurrent_losses=recurrent_losses,
            recurrent_entropies=recurrent_entropies,
            recurrent_num_steps=steps_taken if num_steps > 1 else None,
            text_embeddings=text_repr if output_text_embeddings else None,
            class_embeddings=label_repr if output_class_embeddings else None,
            class_mask=valid_labels.long(),
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
        self.calibrator = build_calibrator(config) if getattr(config, "use_calibrator", False) else None
        self.post_init()

    def add_calibrator(self, hidden_size=None, beta_max=None, dropout=None, use_bias=None):
        """Attach a fresh calibrator (beta == 1, bias == 0, i.e. identity) and record it in the config."""
        if hidden_size is not None:
            self.config.calibrator_hidden_size = hidden_size
        if beta_max is not None:
            self.config.calibrator_beta_max = beta_max
        if dropout is not None:
            self.config.calibrator_dropout = dropout
        if use_bias is not None:
            self.config.calibrator_use_bias = use_bias
        self.config.use_calibrator = True
        reference = next(self.model.parameters())
        self.calibrator = build_calibrator(self.config).to(device=reference.device)
        return self.calibrator

    def remove_calibrator(self):
        self.config.use_calibrator = False
        self.calibrator = None

    def _calibrate(self, outputs, keep_text_embeddings, keep_class_embeddings):
        logits = outputs.logits
        beta, bias = self.calibrator(outputs.text_embeddings, outputs.class_embeddings, logits, outputs.class_mask)
        outputs.uncalibrated_logits = logits
        outputs.inverse_temperatures = beta
        outputs.calibration_biases = bias
        outputs.logits = (beta * logits.float() + bias).to(logits.dtype)
        if not keep_text_embeddings:
            outputs.text_embeddings = None
        if not keep_class_embeddings:
            outputs.class_embeddings = None
        return outputs

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
        self.config.encoder_config.get_text_config().vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        if kwargs.get("adapter_ids") is None:
            kwargs.pop("adapter_ids", None)
        if self.calibrator is None:
            return self.model(*args, **kwargs)

        keep_text = bool(kwargs.get("output_text_embeddings"))
        keep_class = bool(kwargs.get("output_class_embeddings"))
        kwargs.update(output_text_embeddings=True, output_class_embeddings=True, return_dict=True)
        outputs = self.model(*args, **kwargs)
        return self._calibrate(outputs, keep_text, keep_class)

    def update_decoder_cache(self, **kwargs):
        """Extend a decoder-KV text cache."""
        if self.config.architecture_type != "decoder-kv":
            raise ValueError("Decoder cache updates require architecture_type='decoder-kv'.")
        return self.model.update_decoder_cache(**kwargs)

    def classify_from_decoder_cache(self, **kwargs) -> GLiClassOutput:
        """Classify labels against a decoder-KV text cache."""
        if self.config.architecture_type != "decoder-kv":
            raise ValueError("Cached classification requires architecture_type='decoder-kv'.")
        if self.calibrator is None:
            return self.model.classify_from_decoder_cache(**kwargs)

        keep_text = bool(kwargs.get("output_text_embeddings"))
        keep_class = bool(kwargs.get("output_class_embeddings"))
        kwargs.update(output_text_embeddings=True, output_class_embeddings=True)
        outputs = self.model.classify_from_decoder_cache(**kwargs)
        return self._calibrate(outputs, keep_text, keep_class)
