"""CPU unit tests for the GLiClass serve logic around adapters (no Ray, no GPU, no PolyLoRA)."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from gliclass.model import GLiClassBiEncoder
from gliclass.serve.server import GLiClassServer


def _server(enable_polylora=True, mixes_adapters=True):
    server = GLiClassServer.__new__(GLiClassServer)
    server.config = Mock(enable_polylora=enable_polylora, polylora_base_adapter_id="__base__")
    server._polylora_model = None
    server._polylora_mixes_adapters = mixes_adapters
    server.ensure_adapter_loaded = lambda adapter_id: adapter_id or ("__base__" if enable_polylora else None)
    return server


def test_one_request_adapter_is_expanded_to_every_row():
    server = _server()

    assert server._resolve_adapter_ids("task-a", 3) == ["task-a", "task-a", "task-a"]
    assert server._resolve_adapter_ids(None, 2) == ["__base__", "__base__"]
    assert server._resolve_adapter_ids(["task-a", None], 2) == ["task-a", "__base__"]
    assert _server(enable_polylora=False)._resolve_adapter_ids(None, 2) is None


def test_polylora_rows_reach_the_model_as_cache_slots_not_strings():
    server = _server()
    slots = torch.tensor([1, 0, 2])
    server._polylora_model = Mock(resolve_adapters=Mock(return_value=slots))

    kwargs = server._adapter_run_kwargs(["a", "__base__", "b"])

    server._polylora_model.resolve_adapters.assert_called_once_with(["a", "__base__", "b"])
    assert kwargs == {"weight_indices": slots}
    assert server._adapter_run_kwargs(None) == {}


def test_mixed_adapter_batch_runs_one_adapter_at_a_time_when_encoder_cannot_mix():
    server = _server(mixes_adapters=False)
    calls = []

    def pipeline(texts, labels, threshold, batch_size, classification_type, examples, prompt, adapter_ids):
        calls.append((texts, labels, threshold, classification_type, adapter_ids))
        return [f"{text}:{adapter}" for text, adapter in zip(texts, adapter_ids, strict=True)]

    server.pipeline = Mock(side_effect=pipeline, pipe=SimpleNamespace(_examples_are_per_text=lambda examples: False))

    results = server._run_batch_internal(
        ["a", "b", "c", "d"],
        [["x"], ["y"], ["z"], ["w"]],
        threshold=[0.1, 0.2, 0.3, 0.4],
        multi_label=True,
        adapter_ids=["p", None, "p", "q"],
    )

    assert results == ["a:p", "b:__base__", "c:p", "d:q"]
    assert calls == [
        (["a", "c"], [["x"], ["z"]], [0.1, 0.3], "multi-label", ["p", "p"]),
        (["b"], [["y"]], [0.2], "multi-label", ["__base__"]),
        (["d"], [["w"]], [0.4], "multi-label", ["q"]),
    ]


def test_bi_encoder_passes_adapter_slots_to_its_text_encoder():
    calls = []

    def encoder(*args, **kwargs):
        calls.append(kwargs)
        return ("hidden",)

    model = SimpleNamespace(encoder_model=encoder, pool_outputs=lambda outputs: outputs)
    slots = torch.tensor([1, 2])
    ids = torch.ones(2, 1, 4, dtype=torch.long)

    GLiClassBiEncoder.encode_text(model, ids, ids, weight_indices=slots)
    GLiClassBiEncoder.encode_text(model, ids, ids)

    assert calls[0]["weight_indices"] is slots
    assert "weight_indices" not in calls[1] and "adapter_ids" not in calls[1]
