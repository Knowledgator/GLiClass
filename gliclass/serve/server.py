"""Ray Serve deployment for GLiClass with dynamic batching."""

import os
import re
import json
import math
import asyncio
import logging
import functools
from typing import Any

import torch
from transformers import AutoTokenizer

from gliclass.model import GLiClassModel
from gliclass.pipeline import ZeroShotClassificationPipeline, flatten_hierarchical_labels

from .api import APIError, ProcessEngine, capabilities_from_config
from .stats import BatchStats
from .config import GLiClassServeConfig
from .memory import GLiClassMemoryEstimator
from .api.errors import is_error_payload

logger = logging.getLogger("ray.serve")

# Process rows run as multi-label rows with this threshold, so every label comes back with
# its independent score; each task applies its own selection afterwards.
PROCESS_ALL_SCORES = -1.0


class GLiClassServer:
    """GLiClass Ray Serve deployment with dynamic batching."""

    def __init__(self, config: GLiClassServeConfig):
        """Initialize GLiClass server deployment.

        Args:
            config: Server configuration with model and serving parameters
        """
        self.config = config
        self._polylora_model = None
        self._polylora_mixes_adapters = True
        self._adapter_id_re = re.compile(config.polylora_adapter_id_pattern)

        env_vars = config.to_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value

        if config.tokenizer_threads > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            torch.set_num_threads(config.tokenizer_threads)

        torch.set_float32_matmul_precision("high")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(config.dtype.lower(), torch.bfloat16)
        self.device = torch.device(config.device)

        self.memory_estimator = GLiClassMemoryEstimator(
            safety_factor=config.memory_overhead_factor,
            target_memory_fraction=config.target_memory_fraction,
            calibration_probe_batch_size=config.calibration_probe_batch_size,
        )

        if torch.cuda.is_available():
            self.memory_estimator.measure_cuda_context()

        logger.info("Loading model: %s", config.model)
        if config.enable_flashdeberta:
            logger.info("FlashDeBERTa enabled")

        self.model = GLiClassModel.from_pretrained(config.model)
        self.model.config.max_labels_alloc = config.max_labels_alloc
        # Before PolyLoRA: its GPU adapter cache is allocated on the base model's device and dtype.
        self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()
        if config.enable_polylora:
            self._initialize_polylora()

        if config.quantization:
            if not hasattr(self.model, "quantize"):
                raise ValueError("quantization is configured, but this GLiClass model does not support quantize()")
            logger.info("Applying quantization: %s", config.quantization)
            self.model.quantize(config.quantization)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_classes": config.max_labels if config.max_labels > 0 else 100,
            "max_length": config.max_model_len,
            "device": self.device,
            "progress_bar": False,
        }
        self.pipeline = ZeroShotClassificationPipeline(
            classification_type="multi-label",
            **pipeline_kwargs,
        )
        self.pipeline.pipe.adapter_model_kwargs = self._adapter_run_kwargs

        if torch.cuda.is_available():
            self.memory_estimator.measure_model_memory()

        if config.enable_compilation:
            logger.info("Compiling GLiClass model")
            # Warmup compiles a few graphs per precompiled batch size (see _precompile). Past
            # the limit torch silently runs uncompiled, so leave room for all of them.
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 2 * len(config.precompiled_batch_sizes)
            )
            self.model.compile()

        if config.precompile_on_startup or config.enable_compilation:
            self._precompile()

        if config.calibrate_on_startup and torch.cuda.is_available():
            self._calibrate_memory()

        self.batch_stats.reset()  # count request traffic only, not warmup and calibration
        logger.info("GLiClass server initialized successfully")

    @property
    def batch_stats(self) -> BatchStats:
        """Forward passes by batch size since start-up (or the last reset)."""
        stats = self.__dict__.get("_batch_stats")
        if stats is None:
            stats = self.__dict__["_batch_stats"] = BatchStats()
        return stats

    def _initialize_polylora(self) -> None:
        try:
            from polylora import PolyLoraModel, PolyLoraConfig
        except ImportError as exc:
            raise ImportError("enable_polylora=True requires the polylora package to be importable") from exc

        target_model = self._get_polylora_target_model()
        polylora_config = PolyLoraConfig(
            max_gpu_adapters=self.config.polylora_max_gpu_adapters,
            max_cpu_adapters=self.config.polylora_max_cpu_adapters,
            disk_cache_dir=self.config.polylora_disk_cache_dir,
            max_disk_adapters=self.config.polylora_max_disk_adapters,
            max_rank=self.config.polylora_max_rank,
            target_modules=self.config.polylora_adapter_weight_modules,
            base_adapter_id=self.config.polylora_base_adapter_id,
            use_triton_kernels=self.config.polylora_use_triton_kernels,
        )
        self._polylora_model = PolyLoraModel(target_model, polylora_config)
        self._set_polylora_target_model(self._polylora_model)
        # Stock HF DeBERTa projects its relative position embeddings once for the whole batch,
        # so one forward can carry only one adapter on those layers. FlashDeBERTa projects them
        # once per adapter (MultiLoraLinear.project_shared). Other encoders share no inputs.
        self._polylora_mixes_adapters = not any(
            type(module).__name__ == "DisentangledSelfAttention" for module in target_model.modules()
        )
        if not self._polylora_mixes_adapters:
            logger.info("PolyLoRA on HF DeBERTa: batches with mixed adapters run one adapter at a time")

        logger.info(
            "PolyLoRA enabled with %d GPU slots, max rank %d, disk cache %s",
            self.config.polylora_max_gpu_adapters,
            self.config.polylora_max_rank,
            self.config.polylora_disk_cache_dir or "disabled",
        )

    def _get_polylora_target_model(self):
        architecture = self.model.config.architecture_type
        if architecture in {"uni-encoder", "bi-encoder", "bi-encoder-fused"}:
            return self.model.model.encoder_model
        if architecture in {"encoder-decoder", "encoder-decoder-cls"}:
            return self.model.model.encoder_decoder_model
        raise NotImplementedError(f"PolyLoRA is not implemented for architecture {architecture!r}")

    def _set_polylora_target_model(self, wrapped_model) -> None:
        architecture = self.model.config.architecture_type
        if architecture in {"uni-encoder", "bi-encoder", "bi-encoder-fused"}:
            self.model.model.encoder_model = wrapped_model
        elif architecture in {"encoder-decoder", "encoder-decoder-cls"}:
            self.model.model.encoder_decoder_model = wrapped_model
        else:
            raise NotImplementedError(f"PolyLoRA is not implemented for architecture {architecture!r}")

    def _validate_adapter_id(self, adapter_id: str) -> None:
        if not isinstance(adapter_id, str) or not self._adapter_id_re.fullmatch(adapter_id):
            raise ValueError("adapter_id must match polylora_adapter_id_pattern")
        if adapter_id == self.config.polylora_base_adapter_id:
            raise ValueError(f"{adapter_id!r} is reserved for base-only inference")

    def adapter_cache_status(self, adapter_id: str | None = None) -> dict[str, Any]:
        if not self.config.enable_polylora or self._polylora_model is None:
            return {"enabled": False, "base_adapter_id": self.config.polylora_base_adapter_id}
        store = self._polylora_model.adapter_store
        disk_cache = getattr(store, "disk_cache", None)
        response: dict[str, Any] = {
            "enabled": True,
            "base_adapter_id": self.config.polylora_base_adapter_id,
            "loaded": sorted(store.adapters.keys()),
            "disk_cached": sorted(disk_cache.entries.keys()) if disk_cache is not None else [],
            "disk_cache_dir": str(disk_cache.cache_dir) if disk_cache is not None else None,
            "max_disk_adapters": disk_cache.max_adapters if disk_cache is not None else None,
            "gpu_slots": list(self._polylora_model.adapter_cache.slot_to_adapter),
        }
        if adapter_id is not None:
            if adapter_id == self.config.polylora_base_adapter_id:
                response["adapter_id"] = adapter_id
                response["cached"] = True
                response["cpu_resident"] = False
                response["gpu_resident"] = True
                return response
            self._validate_adapter_id(adapter_id)
            response["adapter_id"] = adapter_id
            response["cached"] = disk_cache is not None and adapter_id in disk_cache
            response["cpu_resident"] = adapter_id in store.adapters
            response["gpu_resident"] = adapter_id in self._polylora_model.adapter_cache.adapter_to_slot
        return response

    def ensure_adapter_loaded(self, adapter_id: str | None) -> str | None:
        if adapter_id is None:
            return self.config.polylora_base_adapter_id if self.config.enable_polylora else None
        if adapter_id == self.config.polylora_base_adapter_id:
            return adapter_id
        if not self.config.enable_polylora or self._polylora_model is None:
            raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")
        self._validate_adapter_id(adapter_id)
        if adapter_id in self._polylora_model.adapter_store:
            return adapter_id
        raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")

    def _resolve_adapter_ids(self, adapter_ids: str | list[str | None] | None, num_texts: int) -> list[str] | None:
        """One loaded adapter id per text, or None when adapters are off.

        With PolyLoRA every row gets an id (the base adapter id for none), so all batches
        reach the model with the same arguments.
        """
        if isinstance(adapter_ids, list):
            if len(adapter_ids) != num_texts:
                raise ValueError(f"Expected {num_texts} adapter ids, got {len(adapter_ids)}")
            resolved = [self.ensure_adapter_loaded(adapter_id) for adapter_id in adapter_ids]
        else:
            resolved = [self.ensure_adapter_loaded(adapter_ids)] * num_texts
        if all(adapter_id is None for adapter_id in resolved):
            return None
        return resolved

    def _adapter_run_kwargs(self, adapter_ids: list[str] | None) -> dict[str, Any]:
        """Model kwargs selecting one adapter per batch row.

        With PolyLoRA the ids are resolved to GPU cache slots here, right before the
        forward pass, and passed as the ``weight_indices`` tensor. Passing the id strings
        into a compiled model would make torch.compile guard on them and recompile for
        new adapter mixes.
        """
        if adapter_ids is None:
            return {}
        if self._polylora_model is None:
            return {"adapter_ids": adapter_ids}
        return {"weight_indices": self._polylora_model.resolve_adapters(adapter_ids)}

    def _precompile(self) -> None:
        logger.info("Precompiling model for batch sizes: %s", self.config.precompiled_batch_sizes)

        # torch.compile(dynamic=True) specialises sizes 0 and 1, so a batch of one text and a
        # request with one label each get their own graph; larger sizes share one graph, so
        # request batches of any size and label count reuse the warmed-up graphs.
        label_sets = [["person", "organization", "location"], ["person"]]

        for batch_size in self.config.precompiled_batch_sizes:
            dummy_texts = [f"Sample text number {i} for precompilation warmup." for i in range(batch_size)]

            for _ in range(self.config.warmup_iterations):
                for dummy_labels in label_sets:
                    self._run_batch_internal(
                        dummy_texts,
                        dummy_labels,
                        threshold=0.5,
                        multi_label=True,
                    )

            logger.info("  Batch size %d: compiled", batch_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("Precompilation complete.")

    def _calibrate_memory(self) -> None:
        logger.info("Calibrating memory table...")

        self.memory_estimator.calibrate(
            self._run_batch_internal,
            max_seq_len=self.config.max_model_len,
            min_seq_len=self.config.calibration_min_seq_len,
        )

        logger.info("Memory calibration complete.")

    def batch_size_fn(self, seq_len: int | None = None) -> int:
        """Largest precompiled batch size that fits at seq_len.

        Args:
            seq_len: Sequence length (text + label words). If None, uses max_model_len.

        Returns:
            Optimal batch size from precompiled sizes
        """
        if not self.config.use_memory_aware_batching or not torch.cuda.is_available():
            return self.config.precompiled_batch_sizes[-1]

        if not self.memory_estimator.per_sample_table:
            logger.warning("Memory-aware batching is enabled, but memory estimator is not calibrated")
            return self.config.precompiled_batch_sizes[-1]

        if seq_len is None:
            seq_len = self.config.max_model_len

        return self.memory_estimator.batch_size_fn(
            seq_len=seq_len,
            precompiled_sizes=self.config.precompiled_batch_sizes,
        )

    def observed_seq_len(
        self,
        texts: list[str],
        labels: list[str] | list[list[str]] | None = None,
    ) -> int:
        """Total input word count: longest text + all label words.

        Labels are concatenated into input, so they extend effective seq length
        for every sample in the batch.

        Args:
            texts: Input texts
            labels: Label list

        Returns:
            Estimated sequence length
        """
        max_text_words = max((len(t.split()) for t in texts if t.strip()), default=0)
        prompt_words = 0
        if labels:
            if isinstance(labels[0], list):
                prompt_words += max(sum(len(label.split()) for label in label_set) for label_set in labels)
            else:
                prompt_words += sum(len(label.split()) for label in labels)
        total = max_text_words + prompt_words
        return min(max(total, self.config.calibration_min_seq_len), self.config.max_model_len)

    def _filter_labels(self, labels: list[str]) -> list[str]:
        if self.config.max_labels > 0 and len(labels) > self.config.max_labels:
            logger.warning("Truncating labels from %d to %d", len(labels), self.config.max_labels)
            return labels[: self.config.max_labels]
        return labels

    @torch.inference_mode()
    def _run_batch_internal(
        self,
        texts: list[str],
        labels: list[str] | list[list[str]],
        threshold: float | list[float] = 0.5,
        multi_label: bool | list[bool] = True,
        examples: list[dict[str, Any]] | list[list[dict[str, Any]] | None] | None = None,
        prompt: str | list[str] | None = None,
        adapter_ids: str | list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Run batch inference using the shared zero-shot pipeline.

        Args:
            texts: List of input texts
            labels: Shared label list or one label list per text
            threshold: Shared threshold or one threshold per text
            multi_label: Shared mode or one mode flag per text
            examples: Shared examples or one example set per text
            prompt: Shared prompt or one prompt per text
            adapter_ids: Shared LoRA adapter id or one adapter id per text

        Returns:
            List of prediction dicts
        """
        adapter_ids = self._resolve_adapter_ids(adapter_ids, len(texts))
        if not self._polylora_mixes_adapters and adapter_ids is not None and len(set(adapter_ids)) > 1:
            return self._run_batch_per_adapter(texts, labels, threshold, multi_label, examples, prompt, adapter_ids)

        if isinstance(multi_label, list):
            classification_type = ["multi-label" if item else "single-label" for item in multi_label]
        else:
            classification_type = "multi-label" if multi_label else "single-label"

        self.batch_stats.record(len(texts))
        return self.pipeline(
            texts,
            labels,
            threshold=threshold,
            batch_size=max(len(texts), 1),
            classification_type=classification_type,
            examples=examples,
            prompt=prompt,
            adapter_ids=adapter_ids,
        )

    def _run_batch_per_adapter(self, texts, labels, threshold, multi_label, examples, prompt, adapter_ids):
        """Run a mixed-adapter batch as one sub-batch per adapter, in the original order.

        Per-text arguments (lists with one value per text) are split with the texts; shared
        ones (one label list, one threshold, shared examples) go to every sub-batch.
        """
        num_texts = len(texts)

        def is_per_text(value) -> bool:
            return isinstance(value, list) and len(value) == num_texts

        per_text = {
            "labels": bool(labels) and isinstance(labels[0], (list, dict)),
            "threshold": is_per_text(threshold),
            "multi_label": is_per_text(multi_label),
            "examples": self.pipeline.pipe._examples_are_per_text(examples),
            "prompt": is_per_text(prompt),
        }
        values = {"labels": labels, "threshold": threshold, "multi_label": multi_label}
        values.update(examples=examples, prompt=prompt)

        groups: dict[str, list[int]] = {}
        for index, adapter_id in enumerate(adapter_ids):
            groups.setdefault(adapter_id, []).append(index)

        results: list[Any] = [None] * num_texts
        for adapter_id, indices in groups.items():
            kwargs = {key: [value[i] for i in indices] if per_text[key] else value for key, value in values.items()}
            group_results = self._run_batch_internal([texts[i] for i in indices], adapter_ids=adapter_id, **kwargs)
            for position, index in enumerate(indices):
                results[index] = group_results[position]
        return results

    def predict(
        self,
        texts: str | list[str],
        labels: list[str],
        threshold: float | None = None,
        multi_label: bool = True,
        examples: list[dict[str, Any]] | None = None,
        prompt: str | list[str] | None = None,
        adapter_id: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        if isinstance(texts, str):
            texts = [texts]

        if threshold is None:
            threshold = self.config.default_threshold

        labels = self._filter_labels(labels)

        results = self._run_batch_internal(
            texts,
            labels,
            threshold=threshold,
            multi_label=multi_label,
            examples=examples,
            prompt=prompt,
            adapter_ids=adapter_id,
        )

        return results

    # -- process API -------------------------------------------------------------

    def _examples_supported(self) -> bool:
        """Whether the tokenizer knows the few-shot example token the checkpoint was trained with."""
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        try:
            return self.pipeline.pipe.example_token in tokenizer.get_vocab()
        except Exception:
            return False

    def _label_capacity(self) -> int | None:
        """Most labels the model can score at once, when it allocates a fixed number of label slots."""
        alloc = self.config.max_labels_alloc
        if isinstance(alloc, int) or (isinstance(alloc, str) and alloc.isdigit()):
            return int(alloc)
        if alloc == "fixed":
            return int(getattr(self.model.config, "max_num_classes", 0)) or None
        return None

    @property
    def process_engine(self) -> ProcessEngine:
        """Request planner and result assembler of ``/v1/process``."""
        engine = self.__dict__.get("_process_engine")
        if engine is None:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)

            @functools.lru_cache(maxsize=4096)
            def count_tokens(text: str) -> int:
                return len(tokenizer(text, add_special_tokens=False)["input_ids"])

            engine = self.__dict__["_process_engine"] = ProcessEngine(
                capabilities_from_config(
                    self.config,
                    examples_supported=self._examples_supported(),
                    max_num_classes=self._label_capacity(),
                ),
                max_tokens=self.config.max_model_len,
                count_tokens=count_tokens,
                resolve_adapter=self.ensure_adapter_loaded,
            )
        return engine

    def run_rows(self, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Score process rows in-process, in chunks of ``max_batch_size``: every label of every row."""
        outputs: list[list[dict[str, Any]]] = []
        chunk = max(1, self.config.max_batch_size)
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            examples = [row["examples"] for row in part]
            outputs.extend(
                self._run_batch_internal(
                    [row["text"] for row in part],
                    [row["labels"] for row in part],
                    threshold=PROCESS_ALL_SCORES,
                    multi_label=True,
                    examples=examples if any(examples) else None,
                    prompt=[row["prompt"] for row in part],
                    adapter_ids=[row["adapter_id"] for row in part],
                )
            )
        return outputs

    def process(self, body: Any) -> dict[str, Any]:
        """``POST /v1/process`` in-process. Raises ``APIError`` for a rejected request."""
        return self.process_engine.process(body, self.run_rows)

    def process_batch(self, body: Any) -> dict[str, Any]:
        """``POST /v1/process/batch`` in-process. Raises ``APIError`` for a bad envelope."""
        return self.process_engine.process_batch(body, self.run_rows)

    def capabilities(self, model_id: str | None = None) -> dict[str, Any]:
        info = self.process_engine.capabilities
        if model_id is not None and model_id not in info.model_ids():
            raise APIError("MODEL_NOT_FOUND", f"Model {model_id!r} is not served here", "/model")
        return info.to_dict()

    def models(self) -> dict[str, Any]:
        info = self.process_engine.capabilities
        return {"data": [{"id": info.model, "aliases": list(info.aliases)}]}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.process_engine.capabilities.model,
            "device": self.config.device,
            "cuda_available": torch.cuda.is_available(),
            "polylora": self.config.enable_polylora,
            "max_batch_size": self.batch_size_fn(),
            "task_types": list(self.process_engine.capabilities.task_types),
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_label_tree(node: Any) -> bool:
    if isinstance(node, dict):
        return all(isinstance(key, str) and _is_label_tree(value) for key, value in node.items())
    return isinstance(node, str) or _is_string_list(node)


def _check_labels(labels: Any, separator: str) -> list[str]:
    """Labels of one text as a flat list; hierarchical dicts become dotted labels."""
    if isinstance(labels, dict):
        if not _is_label_tree(labels):
            raise ValueError("hierarchical labels must nest dicts whose leaves are strings or lists of strings")
        labels = flatten_hierarchical_labels(labels, separator=separator)
    if not _is_string_list(labels):
        raise ValueError("'labels' must be a list of strings or a hierarchical dict of them")
    return labels


def _check_examples(examples: Any) -> None:
    if examples is None:
        return
    if not isinstance(examples, list) or not all(isinstance(example, dict) for example in examples):
        raise ValueError("'examples' must be a list of {'text', 'labels'} objects")
    for example in examples:
        labels = example.get("labels", example.get("true_labels", []))
        if not isinstance(example.get("text", ""), str) or not (isinstance(labels, str) or _is_string_list(labels)):
            raise ValueError("each example needs a string 'text' and a list of string 'labels'")


def check_row(
    text: Any,
    labels: Any,
    threshold: Any,
    multi_label: Any,
    examples: Any,
    prompt: Any,
    adapter_id: Any,
    separator: str = ".",
) -> list[str]:
    """Validate one text's request before it joins a batch; return its flat labels.

    Anything that would fail inside the batch must be rejected here: a failing batch fails
    every request batched with it.
    """
    if not isinstance(text, str):
        raise ValueError("'text' must be a string")
    if threshold is not None and not _is_number(threshold):
        raise ValueError("'threshold' must be a number")
    if not isinstance(multi_label, bool):
        raise ValueError("'multi_label' must be a boolean")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("'prompt' must be a string, or one string per text")
    if adapter_id is not None and not isinstance(adapter_id, str):
        raise ValueError("'adapter_id' must be a string")
    _check_examples(examples)
    return _check_labels(labels, separator)


def _per_text(value: Any, num_texts: int, is_item) -> bool:
    """Whether ``value`` holds one item per text (a list of items) rather than one shared item."""
    return isinstance(value, list) and bool(value) and all(is_item(item) for item in value) and len(value) == num_texts


async def _read_json_body(request, max_bytes: int) -> Any:
    """Parse a JSON request body, refusing bodies over ``max_bytes`` (``PAYLOAD_TOO_LARGE``)."""
    content_length = request.headers.get("content-length")
    if max_bytes > 0 and content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise APIError("PAYLOAD_TOO_LARGE", f"Request body exceeds {max_bytes} bytes")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if 0 < max_bytes < len(body):
            raise APIError("PAYLOAD_TOO_LARGE", f"Request body exceeds {max_bytes} bytes")
    try:
        return json.loads(bytes(body))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError("INVALID_REQUEST", f"Request body is not valid JSON: {exc}", "") from exc


def _build_deployment(config: GLiClassServeConfig):
    from ray import serve
    from starlette.responses import JSONResponse

    batch_wait_s = max(config.batch_wait_timeout_ms, 0.0) / 1000.0
    initial_max_batch_size = config.max_batch_size

    @serve.deployment(
        num_replicas=config.num_replicas,
        ray_actor_options={
            "num_gpus": config.num_gpus_per_replica,
            "num_cpus": config.num_cpus_per_replica,
        },
        max_ongoing_requests=config.max_ongoing_requests,
        max_queued_requests=config.queue_capacity,
    )
    class GLiClassDeployment:
        def __init__(self, serve_config: GLiClassServeConfig):
            self.server = GLiClassServer(serve_config)
            # Initialize dynamic batch sizing
            self._infer_batch.set_max_batch_size(self.server.batch_size_fn())
            logger.info(
                "Ray Serve batch size initialized to %d (precompiled: %s)",
                self.server.batch_size_fn(),
                serve_config.precompiled_batch_sizes,
            )

        @serve.batch(
            max_batch_size=initial_max_batch_size,
            batch_wait_timeout_s=batch_wait_s,
        )
        async def _infer_batch(
            self,
            texts: list[str],
            labels_list: list[list[str]],
            thresholds: list[float],
            multi_label_list: list[bool],
            examples_list: list[list[dict[str, Any]] | None],
            prompts_list: list[str | None],
            adapter_ids: list[str | None],
        ) -> list[list[dict[str, Any]]]:
            """Single forward pass over the Ray-accumulated batch.

            Before dispatch, re-sizes Ray's batcher via set_max_batch_size
            using batch_size_fn on the observed seq length — so the next
            accumulation picks the largest precompiled size that fits.

            Supports heterogeneous request parameters by passing per-text
            thresholds, classification types, labels, examples, and prompts
            through to the shared pipeline.
            """
            # Dynamically adjust batch size based on observed sequence length
            next_max_batch = self.server.batch_size_fn(
                seq_len=self.server.observed_seq_len(
                    texts,
                    labels=labels_list,
                )
            )
            self._infer_batch.set_max_batch_size(next_max_batch)

            # Process entire batch at once
            results = self.server._run_batch_internal(
                texts,
                labels_list,
                threshold=thresholds,
                multi_label=multi_label_list,
                examples=examples_list,
                prompt=prompts_list,
                adapter_ids=adapter_ids,
            )

            return results

        async def predict(
            self,
            text: str,
            labels: list[str],
            threshold: float | None = None,
            multi_label: bool = True,
            examples: list[dict[str, Any]] | None = None,
            prompt: str | None = None,
            adapter_id: str | None = None,
        ) -> list[dict[str, Any]]:
            """Single prediction endpoint - one text per request."""
            # Reject bad input and unknown adapters here: inside the batch, the error would
            # fail every request batched with this one.
            labels = check_row(
                text,
                labels,
                threshold,
                multi_label,
                examples,
                prompt,
                adapter_id,
                separator=getattr(self.server.pipeline, "label_separator", "."),
            )
            if threshold is None:
                threshold = self.server.config.default_threshold
            self.server.ensure_adapter_loaded(adapter_id)

            # Call batched method - Ray will accumulate these
            results = await self._infer_batch(
                text,
                labels,
                threshold,
                multi_label,
                examples,
                prompt,
                adapter_id,
            )
            return results

        def batch_stats(self, reset: bool = False) -> dict[str, Any]:
            """Forward passes by batch size; ``reset`` starts a new count."""
            stats = self.server.batch_stats.to_dict()
            if reset:
                self.server.batch_stats.reset()
            return stats

        # -- process API -----------------------------------------------------------

        async def _run_process_row(self, row: dict[str, Any]) -> list[dict[str, Any]]:
            """One process row through the dynamic batcher, shared with ``predict``: every label's score."""
            return await self._infer_batch(
                row["text"],
                row["labels"],
                PROCESS_ALL_SCORES,
                True,
                row["examples"],
                row["prompt"],
                row["adapter_id"],
            )

        async def process(self, body: Any) -> dict[str, Any]:
            """``/v1/process``: the response, or an ``{"error": ...}`` object."""
            engine = self.server.process_engine
            outcome = (await engine.process_many_async([body], self._run_process_row))[0]
            return outcome.to_dict() if isinstance(outcome, APIError) else outcome

        async def process_batch(self, body: Any) -> dict[str, Any]:
            """``/v1/process/batch``: the response, or an ``{"error": ...}`` object."""
            try:
                return await self.server.process_engine.process_batch_async(body, self._run_process_row)
            except APIError as exc:
                return exc.to_dict()

        def capabilities(self, model_id: str | None = None) -> dict[str, Any]:
            try:
                return self.server.capabilities(model_id)
            except APIError as exc:
                return exc.to_dict()

        def models(self) -> dict[str, Any]:
            return self.server.models()

        def health(self) -> dict[str, Any]:
            return self.server.health()

        async def _handle_v1(self, request, path: str):  # noqa: PLR0911
            """Routes of the process API (``{route_prefix}/v1/...``)."""

            def respond(payload: dict[str, Any]) -> JSONResponse:
                status = APIError.from_dict(payload).status if is_error_payload(payload) else 200
                return JSONResponse(payload, status_code=status)

            route = path[path.find("/v1/") :]
            method = request.method
            if route in ("/v1/process", "/v1/process/batch"):
                if method != "POST":
                    return respond(APIError("INVALID_REQUEST", f"Use POST for {route}").to_dict())
                try:
                    body = await _read_json_body(request, self.server.config.max_request_bytes)
                except APIError as exc:
                    return respond(exc.to_dict())
                if route == "/v1/process":
                    return respond(await self.process(body))
                return respond(await self.process_batch(body))
            capabilities_match = re.fullmatch(r"/v1/models/(.+)/capabilities", route)
            if route not in ("/v1/health", "/v1/ready", "/v1/models", "/v1/adapter-cache") and not capabilities_match:
                return JSONResponse(
                    {"error": {"code": "INVALID_REQUEST", "message": f"Unknown route {route}"}}, status_code=404
                )
            if method != "GET":
                return respond(APIError("INVALID_REQUEST", f"Use GET for {route}").to_dict())
            if route in ("/v1/health", "/v1/ready"):
                return respond(self.health())
            if route == "/v1/models":
                return respond(self.models())
            if route == "/v1/adapter-cache":
                try:
                    return respond(self.server.adapter_cache_status(request.query_params.get("adapter_id")))
                except ValueError as exc:
                    return respond(APIError("INVALID_REQUEST", str(exc), "/adapter_id").to_dict())
            return respond(self.capabilities(capabilities_match.group(1)))

        async def __call__(self, request) -> list[dict[str, Any]]:  # noqa: PLR0911
            """HTTP endpoint: classify ``text`` or every text in ``texts``; ``/v1/...`` is the process API."""
            path = request.url.path.rstrip("/")
            if "/v1/" in path:
                return await self._handle_v1(request, path)
            if path.endswith("/adapter-cache"):
                adapter_id = request.query_params.get("adapter_id")
                try:
                    return self.server.adapter_cache_status(adapter_id)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)

            try:
                payload = json.loads(await request.body())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return JSONResponse({"error": f"request body is not valid JSON: {exc}"}, status_code=400)
            if not isinstance(payload, dict):
                return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
            if "labels" not in payload:
                return JSONResponse({"error": "missing 'labels'"}, status_code=400)
            # {"text": "..."} returns that text's predictions; {"texts": [...]} returns one
            # result per text. Every text goes through the batcher as its own request.
            texts = payload.get("texts", payload.get("text", ""))
            single = isinstance(texts, str) and "texts" not in payload
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, list) or not texts:
                return JSONResponse({"error": "'texts' must be a non-empty list of strings"}, status_code=400)
            labels = payload["labels"]
            per_text_labels = isinstance(labels, list) and bool(labels) and isinstance(labels[0], (list, dict))
            prompt = payload.get("prompt")
            per_text_prompt = isinstance(prompt, list)
            examples = payload.get("examples")
            per_text_examples = _per_text(examples, len(texts), lambda item: item is None or isinstance(item, list))
            if (per_text_labels and len(labels) != len(texts)) or (per_text_prompt and len(prompt) != len(texts)):
                return JSONResponse({"error": "per-text labels/prompt must match the texts"}, status_code=400)
            try:
                results = await asyncio.gather(
                    *(
                        self.predict(
                            text=text,
                            labels=labels[index] if per_text_labels else labels,
                            threshold=payload.get("threshold"),
                            multi_label=payload.get("multi_label", True),
                            examples=examples[index] if per_text_examples else examples,
                            prompt=prompt[index] if per_text_prompt else prompt,
                            adapter_id=payload.get("adapter_id"),
                        )
                        for index, text in enumerate(texts)
                    )
                )
            except KeyError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return results[0] if single else results

    return GLiClassDeployment.bind(config)


def serve_gliclass(
    config: GLiClassServeConfig,
    blocking: bool = False,
    host: str = "127.0.0.1",
) -> Any:
    import ray
    from ray import serve

    if not ray.is_initialized():
        ray.init(address=config.ray_address, ignore_reinit_error=True)

    serve.start(
        detached=True,
        http_options={"host": host, "port": config.http_port},
    )
    # Deployments inherit the job's runtime environment; serve.run waits until they are ready.
    handle = serve.run(_build_deployment(config), name="gliclass", route_prefix=config.route_prefix)

    logger.info(
        "GLiClass server running at http://%s:%d%s",
        host,
        config.http_port,
        config.route_prefix,
    )

    if blocking:
        import time
        import signal

        shutdown_event = False

        def handle_signal(_signum, _frame):
            nonlocal shutdown_event
            shutdown_event = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not shutdown_event:
            time.sleep(1)

        serve.shutdown()

    return handle


def shutdown() -> None:
    from ray import serve

    serve.shutdown()


class GLiClassFactory:
    """Synchronous facade: config → deploy → predict → shutdown in one object.

    Pass list of texts to preserve dynamic batching - Ray Serve accumulates
    concurrent requests into single forward pass.

    Example:
        >>> from serve import GLiClassFactory
        >>> llm = GLiClassFactory(model="knowledgator/gliclass-edge-v3.0")
        >>> outputs = llm.predict(
        ...     ["Great product!", "Terrible service"],
        ...     labels=["positive", "negative", "neutral"],
        ... )
        >>> llm.shutdown()

        Or as context manager:
        >>> with GLiClassFactory(model="knowledgator/gliclass-edge-v3.0") as llm:
        ...     out = llm.predict("Great product!", ["positive", "negative"])
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        config: GLiClassServeConfig | None = None,
        **kwargs,
    ):
        """Pass either `config` or `model`/kwargs, not both."""
        if config is not None:
            if model is not None or kwargs:
                raise ValueError("Pass either `config` or `model`/kwargs, not both.")
        else:
            if model is None:
                raise ValueError("Must provide either `model` or `config`.")
            config = GLiClassServeConfig(model=model, **kwargs)

        self.config = config
        self._handle = serve_gliclass(config, blocking=False)
        self._closed = False

    @property
    def handle(self):
        """Underlying Ray Serve deployment handle for async/advanced use."""
        return self._handle

    def predict(
        self,
        texts: str | list[str],
        labels: list[str],
        threshold: float | None = None,
        multi_label: bool = True,
        examples: list[dict[str, Any]] | None = None,
        prompt: str | list[str] | None = None,
        adapter_id: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Blocking prediction. Returns dict for str input, list for list input."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        refs = [
            self._handle.predict.remote(
                t,
                labels,
                threshold,
                multi_label,
                examples,
                prompt,
                adapter_id,
            )
            for t in items
        ]
        results = [ref.result() for ref in refs]
        return results[0] if single else results

    async def predict_async(
        self,
        texts: str | list[str],
        labels: list[str],
        threshold: float | None = None,
        multi_label: bool = True,
        examples: list[dict[str, Any]] | None = None,
        prompt: str | list[str] | None = None,
        adapter_id: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Async prediction. Concurrent calls accumulate into one batch."""
        import asyncio

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        refs = [
            self._handle.predict.remote(
                t,
                labels,
                threshold,
                multi_label,
                examples,
                prompt,
                adapter_id,
            )
            for t in items
        ]
        results = list(await asyncio.gather(*refs))
        return results[0] if single else results

    # -- process API -------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.config.model_id or self.config.model

    def _process_body(self, input, tasks, defaults, options, model, body) -> dict[str, Any]:
        if body is not None:
            return body
        request: dict[str, Any] = {"model": model or self.model_id, "input": input, "tasks": tasks}
        if defaults is not None:
            request["defaults"] = defaults
        if options is not None:
            request["options"] = options
        return request

    def _batch_body(self, items, defaults, options, model) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model or self.model_id, "items": items}
        if defaults is not None:
            body["defaults"] = defaults
        if options is not None:
            body["options"] = options
        return body

    @staticmethod
    def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
        if is_error_payload(payload):
            raise APIError.from_dict(payload)
        return payload

    def process(
        self,
        input: Any = None,
        tasks: dict[str, Any] | None = None,
        *,
        defaults: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one ``/v1/process`` request; raises ``APIError`` when it is rejected.

        Pass ``input`` and ``tasks`` (``model`` defaults to the served model), or a
        complete request as ``body``. ``defaults`` holds settings shared by the tasks.
        """
        request = self._process_body(input, tasks, defaults, options, model, body)
        return self._unwrap(self._handle.process.remote(request).result())

    def process_batch(
        self,
        items: list[dict[str, Any]],
        *,
        defaults: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run a ``/v1/process/batch`` request; item errors are returned per item."""
        request = self._batch_body(items, defaults, options, model)
        return self._unwrap(self._handle.process_batch.remote(request).result())

    async def process_async(
        self,
        input: Any = None,
        tasks: dict[str, Any] | None = None,
        *,
        defaults: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async ``process``. Concurrent calls share model batches."""
        request = self._process_body(input, tasks, defaults, options, model, body)
        return self._unwrap(await self._handle.process.remote(request))

    async def process_batch_async(
        self,
        items: list[dict[str, Any]],
        *,
        defaults: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        request = self._batch_body(items, defaults, options, model)
        return self._unwrap(await self._handle.process_batch.remote(request))

    def capabilities(self, model_id: str | None = None) -> dict[str, Any]:
        return self._unwrap(self._handle.capabilities.remote(model_id).result())

    def shutdown(self) -> None:
        """Tear down Ray Serve deployment and Ray runtime.

        Idempotent. Shutting down Ray after Serve avoids leaving driver
        attached to detached Serve instance.
        """
        if self._closed:
            return
        import ray
        from ray import serve

        serve.shutdown()
        if ray.is_initialized():
            ray.shutdown()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
