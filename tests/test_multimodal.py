"""Tests for gliclass.multimodal helpers (no model downloads)."""

import pytest
import torch

from gliclass.data_processing import DataCollatorWithPadding, format_decoder_kv_sequence
from gliclass.multimodal import collate_media, format_media_prefix, media_placeholder
from gliclass.pipeline import BaseZeroShotClassificationPipeline


class QwenLikeProcessor:
    image_token = "<|image_pad|>"
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"


class GemmaLikeProcessor:
    image_token = "<|image|>"
    audio_token = "<|audio|>"


def test_media_placeholders_per_backbone():
    assert media_placeholder(QwenLikeProcessor(), "image") == "<|vision_start|><|image_pad|><|vision_end|>"
    assert media_placeholder(GemmaLikeProcessor(), "image") == "<|image|>"
    assert format_media_prefix(GemmaLikeProcessor(), num_images=2, num_audio=1) == "<|image|><|image|><|audio|>"
    # no audio placeholder is needed (or required) when there is no audio
    assert format_media_prefix(QwenLikeProcessor(), num_images=1) == "<|vision_start|><|image_pad|><|vision_end|>"
    with pytest.raises(ValueError):
        format_media_prefix(QwenLikeProcessor(), num_audio=1)


def test_media_goes_into_context_before_text():
    sequence = format_decoder_kv_sequence("text", ["a", "b"], prompt="P:", media="<|image|>")
    assert sequence == "P:<|image|>text<<SEP>>a<<LABEL>>b<<LABEL>><<SEP>>"


def test_collate_media_concatenates_items_in_batch_order():
    samples = [
        {"pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        {},
        {"pixel_values": torch.full((2, 3), 2.0), "image_grid_thw": torch.tensor([[1, 1, 2]])},
    ]
    batch = collate_media(samples)
    assert batch["pixel_values"].shape == (6, 3)
    assert batch["pixel_values"][4:].eq(2).all()
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 1, 2]]


def test_collate_media_pads_trailing_dims():
    samples = [
        {
            "input_features": torch.ones(1, 5, 2),
            "input_features_mask": torch.ones(1, 5, dtype=torch.bool),
            "image_position_ids": torch.zeros(1, 2, 2, dtype=torch.long),
        },
        {
            "input_features": torch.ones(2, 3, 2),
            "input_features_mask": torch.ones(2, 3, dtype=torch.bool),
            "image_position_ids": torch.zeros(1, 4, 2, dtype=torch.long),
        },
    ]
    batch = collate_media(samples)
    assert batch["input_features"].shape == (3, 5, 2)
    assert batch["input_features_mask"].sum(dim=1).tolist() == [5, 3, 3]
    assert batch["image_position_ids"].shape == (2, 4, 2)
    assert batch["image_position_ids"][0, 2:].eq(-1).all()


def test_collator_handles_samples_with_and_without_media():
    collator = DataCollatorWithPadding(device="cpu")
    batch = collator(
        [
            {"input_ids": torch.tensor([1, 2]), "mm_token_type_ids": torch.tensor([0, 0]), "labels": torch.tensor([1.0])},
            {
                "input_ids": torch.tensor([1, 9, 9]),
                "mm_token_type_ids": torch.tensor([0, 1, 1]),
                "labels": torch.tensor([0.0]),
                "pixel_values": torch.ones(8, 4),
                "image_grid_thw": torch.tensor([[1, 2, 4]]),
            },
        ]
    )
    assert batch["input_ids"].shape == (2, 3)
    assert batch["mm_token_type_ids"].tolist() == [[0, 0, 0], [0, 1, 1]]
    assert batch["pixel_values"].shape == (8, 4)
    assert batch["image_grid_thw"].tolist() == [[1, 2, 4]]


def test_normalize_media_accepts_per_text_and_flat_lists():
    normalize = BaseZeroShotClassificationPipeline._normalize_media
    assert normalize(["img1", "img2"], 1) == [["img1", "img2"]]
    assert normalize([["img1"], None, "img2"], 3) == [["img1"], [], ["img2"]]
    with pytest.raises(ValueError):
        normalize([["img1"]], 2)
