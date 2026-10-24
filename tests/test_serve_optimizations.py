"""GLiClass serve with torch.compile, FlashDeBERTa, PolyLoRA and FP8.

Each test starts a ``GLiClassServer`` (the model side of the Ray Serve deployment;
Ray itself is not needed) on ``knowledgator/gliclass-small-v1.0-init`` (DeBERTa-v3-small
encoder) and checks it against an fp32 eager Hugging Face reference run one text at a
time. Adapters are applied to the reference with PEFT, so PolyLoRA outputs are checked
against an independent LoRA implementation.

Covered configurations:

- eager DeBERTa + compilation
- eager DeBERTa + PolyLoRA
- FlashDeBERTa + compilation
- FlashDeBERTa + PolyLoRA + compilation + FP8
- the last one behind Ray Serve (HTTP, dynamic batching)

These tests need a CUDA GPU (FP8: compute capability 8.9+) and take a few minutes,
mostly in torch.compile.
"""

import copy

import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MODEL = "knowledgator/gliclass-small-v1.0-init"
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "dense"]
LABELS = ["technology", "sports", "finance", "politics", "health"]
TEXTS = [
    "The new smartphone has a faster processor and a much better camera.",
    "The striker scored twice as the home team won the championship final.",
    "The central bank raised interest rates again to slow down inflation.",
    "Doctors recommend regular exercise and a balanced diet to prevent heart disease.",
]
EXPECTED_TOP_LABELS = ["technology", "sports", "finance", "health"]
ADAPTERS = ["legal", "medical"]
ADAPTER_IDS = ["legal", "__base__", "medical", "legal"]  # a mixed batch, with a base-only row

# Logit error limits (max, mean), in units of the reference logit std. The reference runs
# in fp32; the server in bf16. The bf16 paths (compiled, FlashDeBERTa, PolyLoRA) measure
# about (0.02, 0.007). FP8 measures about (0.15, 0.05-0.07), eager and compiled alike, mostly
# from the FP8 linear layers: with only one logit per (text, label) pair the mean does not
# average out as it does over GLiNER's span logits. The test adapters move the logits by
# well over both limits.
BF16_TOL = (0.1, 0.015)
FP8_TOL = (0.4, 0.1)


def _fp8_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # The server sets USE_FLASHDEBERTA in os.environ; never let it leak into other tests.
    monkeypatch.delenv("USE_FLASHDEBERTA", raising=False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def model_dir():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL)
    except Exception as exc:  # offline and not cached
        pytest.skip(f"cannot download {MODEL}: {exc}")


@pytest.fixture(scope="module")
def adapter_cache_dir(model_dir, tmp_path_factory):
    """A PolyLoRA disk cache holding two PEFT adapters for the encoder."""
    peft = pytest.importorskip("peft")
    pytest.importorskip("polylora")
    from polylora.store import DiskAdapterCache

    encoder = _load_reference(model_dir).model.encoder_model
    root = tmp_path_factory.mktemp("adapters")
    cache = DiskAdapterCache(root / "cache")
    for seed, name in enumerate(ADAPTERS):
        torch.manual_seed(100 + seed)
        peft_model = peft.get_peft_model(
            copy.deepcopy(encoder),
            peft.LoraConfig(r=8, lora_alpha=16, target_modules=TARGET_MODULES, lora_dropout=0.0),
        )
        with torch.no_grad():
            for module_name, param in peft_model.named_parameters():
                if "lora_B" in module_name:  # PEFT initialises B to zero: make the adapter do something
                    param.normal_(0.0, 0.1)
        peft_model.save_pretrained(root / name, save_embedding_layers=False)
        cache.add_adapter(name, root / name)
    return str(root / "cache"), {name: str(root / name) for name in ADAPTERS}


def _load_reference(model_dir):
    """fp32 eager Hugging Face DeBERTa GLiClass (no FlashDeBERTa, no compile)."""
    from gliclass import GLiClassModel

    return GLiClassModel.from_pretrained(model_dir).to("cuda", torch.float32).eval()


def _make_server(model_dir, monkeypatch, **overrides):
    from gliclass.serve.config import GLiClassServeConfig
    from gliclass.serve.server import GLiClassServer

    config = {
        "model": model_dir,
        "device": "cuda",
        "dtype": "bfloat16",
        "max_model_len": 512,
        "max_batch_size": 4,
        "precompiled_batch_sizes": [1, 2, 4],
        "warmup_iterations": 1,
        "tokenizer_threads": 0,
        "enable_compilation": False,
        "polylora_adapter_weight_modules": TARGET_MODULES,
    }
    config.update(overrides)
    if not config.get("enable_flashdeberta"):
        monkeypatch.delenv("USE_FLASHDEBERTA", raising=False)
    return GLiClassServer(GLiClassServeConfig(**config))


def _pipe(model, model_dir):
    from transformers import AutoTokenizer

    from gliclass import ZeroShotClassificationPipeline

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return ZeroShotClassificationPipeline(model, tokenizer, device=model.device, progress_bar=False).pipe


@torch.inference_mode()
def _logits(pipe, texts, model_kwargs=None):
    """(texts, labels) logits, through the same pipeline inputs the server uses."""
    inputs = pipe.prepare_inputs(texts, LABELS, same_labels=True)
    output = pipe.model(**inputs, max_num_classes=len(LABELS), **(model_kwargs or {}))
    return output.logits.float()


def _server_logits(server, texts, adapter_ids=None):
    """Logits through the server's own adapter resolution."""
    resolved = server._resolve_adapter_ids(adapter_ids, len(texts))
    return _logits(server.pipeline.pipe, texts, server._adapter_run_kwargs(resolved))


@torch.inference_mode()
def _reference_rows(reference_pipe, texts, adapter_ids=None, adapter_paths=None):
    """Logits of every text on its own, with the row's adapter applied through PEFT."""
    inner = reference_pipe.model.model
    base = inner.encoder_model
    rows = []
    for index, text in enumerate(texts):
        adapter_id = None if adapter_ids is None else adapter_ids[index]
        if adapter_id not in (None, "__base__"):
            from peft import PeftModel

            inner.encoder_model = PeftModel.from_pretrained(copy.deepcopy(base), adapter_paths[adapter_id]).cuda()
        try:
            rows.append(_logits(reference_pipe, [text])[0])
        finally:
            inner.encoder_model = base
    return torch.stack(rows)


def _assert_close(actual, expected, tol, what):
    assert actual.shape == expected.shape, f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), f"{what}: non-finite logits"
    diffs = (actual - expected).abs() / expected.std()
    max_err, mean_err = diffs.max().item(), diffs.mean().item()
    print(f"{what}: error max {max_err:.4f}, mean {mean_err:.4f} x logit std")
    assert max_err < tol[0], f"{what}: max error {max_err:.4f} x logit std exceeds {tol[0]}"
    assert mean_err < tol[1], f"{what}: mean error {mean_err:.4f} x logit std exceeds {tol[1]}"


def _adapter_effect(with_adapters, base):
    effect = ((with_adapters - base).abs().max() / base.std()).item()
    print(f"adapter effect: max {effect:.4f} x logit std")
    return effect


def _encoder(server):
    return server.model.model.encoder_model


def _excluded_from_compile(fn) -> bool:
    return getattr(fn, "_torchdynamo_disable", False) and getattr(fn, "_torchdynamo_disable_recursive", True)


def _assert_compiled(server):
    """The whole model is compiled, encoder included (it is not excluded with torch.compiler.disable).

    ``PolyLoraModel.forward`` itself runs eagerly by design (``disable(recursive=False)``,
    so adapter ids never become guards); the base encoder it calls is compiled.
    """
    assert server.model._compiled_call_impl is not None, "model is not compiled"
    encoder = _encoder(server)
    for module in (encoder, getattr(encoder, "base_model", encoder)):
        assert not _excluded_from_compile(module.forward), f"{type(module).__name__} is excluded from compile"


def _assert_predict_works(server, adapter_id=None):
    results = server.predict(TEXTS, LABELS, threshold=0.0, adapter_id=adapter_id)
    assert len(results) == len(TEXTS)
    for result in results:
        assert {item["label"] for item in result} == set(LABELS)
    if adapter_id is None:  # the test adapters are random and may change predictions
        top = [max(result, key=lambda item: item["score"])["label"] for result in results]
        assert top == EXPECTED_TOP_LABELS
    return results


@pytest.fixture(scope="module")
def reference_pipe(model_dir):
    return _pipe(_load_reference(model_dir), model_dir)


def test_eager_model_with_compilation(model_dir, reference_pipe, monkeypatch):
    reference = _reference_rows(reference_pipe, TEXTS)
    server = _make_server(model_dir, monkeypatch, enable_compilation=True)

    assert type(_encoder(server)).__name__ == "DebertaV2Model"
    _assert_compiled(server)
    _assert_close(_server_logits(server, TEXTS), reference, BF16_TOL, "compiled eager model")
    # Unseen batch sizes and label counts reuse the warmed-up graphs.
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    server.predict(TEXTS[:3], LABELS[:4])
    server.predict(TEXTS + TEXTS[:1], LABELS[:2])
    server.predict(TEXTS[:2], LABELS[:1])
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs, "a request recompiled"
    _assert_predict_works(server)


def test_eager_model_with_polylora(model_dir, reference_pipe, adapter_cache_dir, monkeypatch):
    cache_dir, adapter_paths = adapter_cache_dir
    base_reference = _reference_rows(reference_pipe, TEXTS)
    server = _make_server(model_dir, monkeypatch, enable_polylora=True, polylora_disk_cache_dir=cache_dir)

    assert type(_encoder(server)).__name__ == "PolyLoraModel"
    assert type(_encoder(server).base_model).__name__ == "DebertaV2Model"
    _assert_close(_server_logits(server, TEXTS), base_reference, BF16_TOL, "PolyLoRA base rows")
    for adapter_id in ADAPTERS:
        adapter_ids = [adapter_id] * len(TEXTS)
        expected = _reference_rows(reference_pipe, TEXTS, adapter_ids, adapter_paths)
        _assert_close(_server_logits(server, TEXTS, adapter_ids), expected, BF16_TOL, f"PolyLoRA adapter {adapter_id}")
        # The adapter changes the output by much more than the tolerance.
        assert _adapter_effect(expected, base_reference) > 3 * BF16_TOL[0]
    # HF DeBERTa projects the shared relative-position embeddings through query/key_proj
    # once per batch, so one forward carries one adapter. The server splits a mixed batch
    # (as Ray's batcher builds) by adapter and keeps the order.
    assert not server._polylora_mixes_adapters
    mixed = server._run_batch_internal(TEXTS, LABELS, threshold=0.0, adapter_ids=ADAPTER_IDS)
    for text, adapter_id, result in zip(TEXTS, ADAPTER_IDS, mixed, strict=True):
        single = server._run_batch_internal([text], LABELS, threshold=0.0, adapter_ids=adapter_id)[0]
        assert [item["label"] for item in result] == [item["label"] for item in single]
        assert [item["score"] for item in result] == pytest.approx([item["score"] for item in single], abs=0.02)
    _assert_predict_works(server)
    _assert_predict_works(server, adapter_id="medical")
    assert server.adapter_cache_status("medical")["gpu_resident"]


def test_flash_model_with_compilation(model_dir, reference_pipe, monkeypatch):
    pytest.importorskip("flashdeberta")
    reference = _reference_rows(reference_pipe, TEXTS)
    server = _make_server(model_dir, monkeypatch, enable_flashdeberta=True, enable_compilation=True)

    assert type(_encoder(server)).__name__ == "FlashDebertaV2Model"
    _assert_compiled(server)
    _assert_close(_server_logits(server, TEXTS), reference, BF16_TOL, "compiled FlashDeBERTa")
    _assert_predict_works(server)


@pytest.mark.skipif(not _fp8_supported(), reason="FP8 needs compute capability 8.9+")
def test_flash_model_with_polylora_compilation_and_fp8(model_dir, reference_pipe, adapter_cache_dir, monkeypatch):
    pytest.importorskip("flashdeberta")
    pytest.importorskip("torchao")
    cache_dir, adapter_paths = adapter_cache_dir
    reference = _reference_rows(reference_pipe, TEXTS, ADAPTER_IDS, adapter_paths)
    # The adapters change the output by more than the FP8 limit, so a dropped adapter fails.
    assert _adapter_effect(reference, _reference_rows(reference_pipe, TEXTS)) > 2 * FP8_TOL[0]
    server = _make_server(
        model_dir,
        monkeypatch,
        enable_flashdeberta=True,
        enable_polylora=True,
        enable_compilation=True,
        quantization="fp8",
        polylora_disk_cache_dir=cache_dir,
    )

    from polylora.layers import MultiLoraLinear

    wrapped = _encoder(server)
    assert type(wrapped).__name__ == "PolyLoraModel"
    assert type(wrapped.base_model).__name__ == "FlashDebertaV2Model"
    assert server._polylora_mixes_adapters
    _assert_compiled(server)
    lora_layers = [m for m in wrapped.modules() if isinstance(m, MultiLoraLinear)]
    assert lora_layers
    for layer in lora_layers:  # FP8 base layers; adapters stay in the activation dtype
        assert "Float8" in type(layer.base_layer.weight).__name__, type(layer.base_layer.weight)
    attention = [m for m in wrapped.modules() if type(m).__name__ == "FlashDisentangledSelfAttention"]
    assert attention and all(m.attention_precision == "fp8_qk" for m in attention)

    actual = _server_logits(server, TEXTS, ADAPTER_IDS)
    _assert_close(actual, reference, FP8_TOL, "Flash + PolyLoRA + compile + FP8, mixed adapters")
    # New adapter mixes reuse the compiled graphs (no recompile) and stay correct.
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for mix in (["medical", "legal", "__base__", "medical"], ["__base__", "medical", "medical", "legal"]):
        expected = _reference_rows(reference_pipe, TEXTS, mix, adapter_paths)
        _assert_close(_server_logits(server, TEXTS, mix), expected, FP8_TOL, f"adapter mix {mix}")
        server._run_batch_internal(TEXTS, LABELS, adapter_ids=mix)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs, "a new adapter mix recompiled"
    _assert_predict_works(server)
    _assert_predict_works(server, adapter_id="legal")


@pytest.mark.skipif(not _fp8_supported(), reason="FP8 needs compute capability 8.9+")
def test_ray_serve_flash_polylora_compilation_fp8(model_dir, adapter_cache_dir):
    """The full stack behind Ray Serve: HTTP, dynamic batching of mixed-adapter requests."""
    pytest.importorskip("flashdeberta")
    pytest.importorskip("torchao")
    ray = pytest.importorskip("ray")
    pytest.importorskip("ray.serve")
    import socket
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    import requests

    from gliclass.serve import GLiClassClient, GLiClassServeConfig, shutdown, serve_gliclass

    cache_dir, _ = adapter_cache_dir
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    torch.cuda.empty_cache()
    # Replicas must import this checkout of gliclass.
    repo_root = str(Path(__file__).resolve().parents[1])
    ray.init(num_cpus=4, include_dashboard=False, runtime_env={"env_vars": {"PYTHONPATH": repo_root}})
    try:
        serve_gliclass(
            GLiClassServeConfig(
                model=model_dir,
                max_model_len=512,
                max_batch_size=8,
                precompiled_batch_sizes=[1, 2, 4, 8],
                warmup_iterations=1,
                enable_flashdeberta=True,
                enable_polylora=True,
                enable_compilation=True,
                quantization="fp8",
                polylora_adapter_weight_modules=TARGET_MODULES,
                polylora_disk_cache_dir=cache_dir,
                http_port=port,
            )
        )
        client = GLiClassClient(f"http://localhost:{port}/gliclass")

        def top(result):
            return max(result, key=lambda item: item["score"])["label"]

        base = client(TEXTS, LABELS, threshold=0.0)
        assert [top(result) for result in base] == EXPECTED_TOP_LABELS

        # Adapters reach the model: scores move.
        legal = client(TEXTS, LABELS, threshold=0.0, adapter_id="legal")
        base_scores = [item["score"] for result in base for item in result]
        legal_scores = [item["score"] for result in legal for item in result]
        assert max(abs(a - b) for a, b in zip(base_scores, legal_scores, strict=True)) > 0.01

        # Concurrent requests with different adapters are batched together by Ray and
        # must give the same results as one request at a time.
        def scores(job):
            result = client.classify(TEXTS[job[0]], LABELS, threshold=0.0, adapter_id=job[1])
            return {item["label"]: item["score"] for item in result}

        jobs = [(i % len(TEXTS), [None, "legal", "medical"][i % 3]) for i in range(48)]
        with ThreadPoolExecutor(16) as pool:
            concurrent = list(pool.map(scores, jobs))
        one_by_one = {job: scores(job) for job in set(jobs)}
        for job, result in zip(jobs, concurrent, strict=True):
            expected = one_by_one[job]
            assert max(expected, key=expected.get) == max(result, key=result.get), job
            assert result == pytest.approx(expected, abs=0.05), job

        status = client.adapter_cache_status("legal")
        assert status["enabled"] and status["cpu_resident"] and status["gpu_resident"]
        with pytest.raises(requests.HTTPError, match="404"):
            client(TEXTS[0], LABELS, adapter_id="unknown")
    finally:
        shutdown()
        ray.shutdown()
