"""Multi-modal (image / audio) input helpers for the decoder-kv architecture.

Media enter a decoder-kv sequence as placeholder tokens in the context part:

    [prompt][examples]<image><image><audio>text<<SEP>>label1<<LABEL>>...<<SEP>>

The HF processor expands every placeholder into the item's soft tokens and returns the media tensors;
the multi-modal backbone (Qwen3_5Model, Gemma4Model, EmbeddingGemma2Model) encodes them and writes the features
into those positions. Placeholders are matched to media in order, so the i-th image placeholder is the i-th image.
"""

import os

import torch
from transformers.processing_utils import ProcessorMixin

# Processor outputs indexed by media item rather than by sample: concatenated across the batch.
MEDIA_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "image_position_ids",
    "pixel_values_videos",
    "video_grid_thw",
    "video_position_ids",
    "input_features",
    "input_features_mask",
)
# Padding values for trailing dims of media tensors that differ between samples ((-1, -1) = padded patch).
MEDIA_PAD_VALUES = {"image_position_ids": -1, "video_position_ids": -1}


def is_processor(obj) -> bool:
    return isinstance(obj, ProcessorMixin)


def get_tokenizer(tokenizer_or_processor):
    """Text tokenizer of a processor, or the tokenizer itself."""
    if is_processor(tokenizer_or_processor):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def media_placeholder(processor, kind: str) -> str:
    """The placeholder to write in the text for one image or audio item.

    Qwen expects <|vision_start|><|image_pad|><|vision_end|>; Gemma adds its begin/end markers itself.
    """
    if kind == "image":
        token = processor.image_token
        if getattr(processor, "vision_start_token", None) is not None:
            return f"{processor.vision_start_token}{token}{processor.vision_end_token}"
        return token
    if kind == "audio":
        token = getattr(processor, "audio_token", None)
        if token is None:
            raise ValueError(f"{type(processor).__name__} does not support audio inputs.")
        return token
    raise ValueError(f"Unknown media kind: {kind}")


def format_media_prefix(processor, num_images: int = 0, num_audio: int = 0) -> str:
    prefix = media_placeholder(processor, "image") * num_images if num_images else ""
    if num_audio:
        prefix += media_placeholder(processor, "audio") * num_audio
    return prefix


def load_images(images):
    """Load images given as paths, URLs or PIL images."""
    from transformers.image_utils import load_image

    return [load_image(image) for image in images or []]


def load_audio(audio, sampling_rate: int):
    """Load audio given as paths, URLs or 1D arrays (arrays must already be at `sampling_rate`)."""
    from transformers.audio_utils import load_audio as _load_audio

    audio = audio or []
    for item in audio:
        # transformers returns a missing local path unchanged, which only fails later inside feature extraction
        if isinstance(item, str) and not item.startswith(("http://", "https://")) and not os.path.isfile(item):
            raise FileNotFoundError(f"Audio file not found: {item}")
    return [_load_audio(item, sampling_rate=sampling_rate) for item in audio]


def process_multimodal(processor, text: str, images=None, audio=None) -> dict:
    """Run the processor on one sample; returns 1D input_ids/attention_mask/mm_token_type_ids plus media tensors.

    Media may be paths, URLs, PIL images or arrays. The text must already contain one placeholder per item.
    """
    images = load_images(images)
    kwargs = {}
    if images:
        kwargs["images"] = [images]
    if audio:
        sampling_rate = processor.feature_extractor.sampling_rate
        kwargs["audio"] = load_audio(audio, sampling_rate)
    outputs = processor(text=[text], return_tensors="pt", **kwargs)

    sample = {}
    for key, value in outputs.items():
        if key in MEDIA_KEYS:
            sample[key] = value
        elif isinstance(value, torch.Tensor):
            sample[key] = value[0]
    return sample


def process_multimodal_with_budget(processor, build, text: str, images, audio, max_length: int, reserved: int = 0):
    """Process build(text, media_placeholders) so that it fits max_length - reserved tokens.

    The processed sequence is never truncated (that would cut through expanded media placeholders);
    when it is too long, the text is shortened by the overflow and processed again.
    """
    images = load_images(images)
    if audio:
        audio = load_audio(audio, processor.feature_extractor.sampling_rate)
    media = format_media_prefix(processor, len(images), len(audio or []))
    sample = process_multimodal(processor, build(text, media), images, audio)
    overflow = len(sample["input_ids"]) + reserved - max_length
    if overflow > 0:
        tokenizer = get_tokenizer(processor)
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        text = tokenizer.decode(text_ids[: max(0, len(text_ids) - overflow)])
        sample = process_multimodal(processor, build(text, media), images, audio)
    return sample


def _pad_trailing(tensor: torch.Tensor, shape, value) -> torch.Tensor:
    pad = []
    for dim in reversed(range(1, tensor.dim())):
        pad.extend((0, shape[dim] - tensor.shape[dim]))
    if not any(pad):
        return tensor
    return torch.nn.functional.pad(tensor, pad, value=value)


def collate_media(samples: list[dict]) -> dict:
    """Concatenate media tensors of a batch along dim 0 (in batch order), padding trailing dims if needed."""
    batch = {}
    for key in MEDIA_KEYS:
        tensors = [sample[key] for sample in samples if sample.get(key) is not None]
        if not tensors:
            continue
        shape = [max(tensor.shape[dim] for tensor in tensors) for dim in range(tensors[0].dim())]
        value = MEDIA_PAD_VALUES.get(key, False if tensors[0].dtype == torch.bool else 0)
        batch[key] = torch.cat([_pad_trailing(tensor, shape, value) for tensor in tensors], dim=0)
    return batch


def freeze_media_encoders(model) -> None:
    """Freeze the vision / audio towers and their projectors of a multi-modal backbone."""
    for name in ("visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)


def match_media_feature_dtype(model) -> None:
    """Cast projected image / audio features to the text embedding dtype before they are merged.

    Under bf16 autocast the projectors return bf16 while the fp32 text embeddings stay fp32. Gemma 4 casts
    image features itself but not audio features, so masked_scatter fails on the mixed dtypes.
    """
    for name in ("embed_vision", "embed_audio"):
        module = getattr(model, name, None)
        if module is not None:
            module.register_forward_hook(
                lambda _module, _inputs, output: output.to(model.get_input_embeddings().weight.dtype)
            )
