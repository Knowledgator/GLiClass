"""Ray Serve deployment for GLiClass with dynamic batching."""

import os
import re
import logging
from typing import Any

import torch
from ray import serve
from ray.serve._private import api as serve_private_api
from ray.serve._private.build_app import build_app
from starlette.responses import JSONResponse
from transformers import AutoTokenizer

from gliclass.model import GLiClassModel
from gliclass.pipeline import ZeroShotClassificationPipeline

from .config import GLiClassServeConfig
from .memory import GLiClassMemoryEstimator

logger = logging.getLogger("ray.serve")


class GLiClassServer:
    """GLiClass Ray Serve deployment with dynamic batching."""

    def __init__(self, config: GLiClassServeConfig):
        """Initialize GLiClass server deployment.

        Args:
            config: Server configuration with model and serving parameters
        """
        self.config = config
        self._polylora_model = None
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

        self.model = GLiClassModel.from_pretrained(config.model)
        if config.enable_polylora:
            self._initialize_polylora()
        self.model.config.max_labels_alloc = config.max_labels_alloc
        self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()

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

        if torch.cuda.is_available():
            self.memory_estimator.measure_model_memory()

        if config.enable_compilation:
            self._precompile()

        if torch.cuda.is_available():
            self._calibrate_memory()

        logger.info("GLiClass server initialized successfully")

    def _initialize_polylora(self) -> None:
        try:
            from polylora import PolyLoraConfig, PolyLoraModel
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

    def _precompile(self) -> None:
        logger.info("Precompiling model for batch sizes: %s", self.config.precompiled_batch_sizes)

        if hasattr(self.model, "compile"):
            self.model.compile()

        dummy_labels = ["person", "organization", "location"]

        for batch_size in self.config.precompiled_batch_sizes:
            dummy_texts = [f"Sample text number {i} for precompilation warmup." for i in range(batch_size)]

            for _ in range(self.config.warmup_iterations):
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
        if not torch.cuda.is_available():
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
        if isinstance(multi_label, list):
            classification_type = ["multi-label" if item else "single-label" for item in multi_label]
        else:
            classification_type = "multi-label" if multi_label else "single-label"

        if isinstance(adapter_ids, list):
            adapter_ids = [self.ensure_adapter_loaded(adapter_id) for adapter_id in adapter_ids]
        elif adapter_ids is not None:
            adapter_ids = self.ensure_adapter_loaded(adapter_ids)

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


def _build_deployment(config: GLiClassServeConfig):
    batch_wait_s = max(config.batch_wait_timeout_ms, 0.0) / 1000.0
    initial_max_batch_size = config.max_batch_size

    @serve.deployment(
        num_replicas=config.num_replicas,
        ray_actor_options={
            "num_gpus": config.num_gpus_per_replica,
            "num_cpus": config.num_cpus_per_replica,
        },
        max_ongoing_requests=config.max_ongoing_requests,
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
            if threshold is None:
                threshold = self.server.config.default_threshold

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

        async def __call__(self, request) -> list[dict[str, Any]]:
            """HTTP endpoint - accepts single text per request."""
            path = request.url.path.rstrip("/")
            if path.endswith("/adapter-cache"):
                adapter_id = request.query_params.get("adapter_id")
                try:
                    return self.server.adapter_cache_status(adapter_id)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)

            payload = await request.json()
            text = payload.get("text") or payload.get("texts")
            if isinstance(text, list):
                # If list provided, take first element for compatibility
                text = text[0] if text else ""
            try:
                return await self.predict(
                    text=text,
                    labels=payload["labels"],
                    threshold=payload.get("threshold"),
                    multi_label=payload.get("multi_label", True),
                    examples=payload.get("examples"),
                    prompt=payload.get("prompt"),
                    adapter_id=payload.get("adapter_id"),
                )
            except KeyError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

    return GLiClassDeployment.bind(config)


def serve_gliclass(
    config: GLiClassServeConfig,
    blocking: bool = False,
    host: str = "127.0.0.1",
) -> Any:
    import ray

    if not ray.is_initialized():
        ray.init(address=config.ray_address, ignore_reinit_error=True)

    serve.start(
        detached=True,
        http_options={"host": host, "port": config.http_port},
    )
    serve_client = serve_private_api._get_global_client(_health_check_controller=True)

    app = _build_deployment(config)
    built_app = build_app(
        app,
        name="gliclass",
        route_prefix=config.route_prefix,
        default_runtime_env=ray.get_runtime_context().runtime_env,
    )
    handle = serve_client.deploy_applications([built_app])[0]
    serve_client.wait_for_proxies_serving()

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

    def shutdown(self) -> None:
        """Tear down Ray Serve deployment and Ray runtime.

        Idempotent. Shutting down Ray after Serve avoids leaving driver
        attached to detached Serve instance.
        """
        if self._closed:
            return
        import ray

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
