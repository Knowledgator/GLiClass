"""Tests for the decoder-KV architecture."""

from types import SimpleNamespace

import torch

from gliclass.model import GLiClassOutput, GLiClassDecoderKV


def make_decoder_kv_stub():
    return SimpleNamespace(
        sep_token_id=98,
        config=SimpleNamespace(class_token_index=99),
    )


def test_gliclass_output_exposes_past_key_values():
    cache = object()
    output = GLiClassOutput(logits=torch.zeros(1, 1), past_key_values=cache)
    assert output.past_key_values is cache


def test_extract_label_section_pads_uneven_sections():
    model = make_decoder_kv_stub()
    hidden_states = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    input_ids = torch.tensor(
        [
            [10, 98, 11, 98, 20, 99, 98, 0, 0, 99],
            [98, 12, 98, 21, 22, 99, 23, 99, 98, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )

    padded_hidden, padded_ids, label_mask = GLiClassDecoderKV._extract_label_section(
        model,
        hidden_states,
        input_ids,
        attention_mask,
    )

    expected_hidden = hidden_states.new_zeros(2, 6, 3)
    expected_hidden[0, :3] = hidden_states[0, 4:7]
    expected_hidden[1] = hidden_states[1, 3:9]
    expected_ids = input_ids.new_zeros(2, 6)
    expected_ids[0, :3] = input_ids[0, 4:7]
    expected_ids[1] = input_ids[1, 3:9]
    expected_mask = attention_mask.new_tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )

    torch.testing.assert_close(padded_hidden, expected_hidden)
    torch.testing.assert_close(padded_ids, expected_ids)
    torch.testing.assert_close(label_mask, expected_mask)


def test_extract_label_section_preserves_fallback_and_gradients():
    model = make_decoder_kv_stub()
    hidden_states = torch.randn(1, 5, 3, requires_grad=True)
    input_ids = torch.tensor([[1, 99, 2, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    padded_hidden, padded_ids, label_mask = GLiClassDecoderKV._extract_label_section(
        model,
        hidden_states,
        input_ids,
        attention_mask,
    )

    torch.testing.assert_close(padded_hidden, hidden_states[:, :3])
    torch.testing.assert_close(padded_ids, input_ids[:, :3])
    torch.testing.assert_close(label_mask, attention_mask[:, :3])

    padded_hidden.sum().backward()
    torch.testing.assert_close(hidden_states.grad[:, :3], torch.ones_like(hidden_states[:, :3]))
    torch.testing.assert_close(hidden_states.grad[:, 3:], torch.zeros_like(hidden_states[:, 3:]))
