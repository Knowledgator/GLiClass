"""Ray Serve deployment for GLiClass with dynamic batching."""

import os
import logging
from typing import Any

import torch
from ray import serve
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
    ) -> list[list[dict[str, Any]]]:
        """Run batch inference using the shared zero-shot pipeline.

        Args:
            texts: List of input texts
            labels: Shared label list or one label list per text
            threshold: Shared threshold or one threshold per text
            multi_label: Shared mode or one mode flag per text
            examples: Shared examples or one example set per text
            prompt: Shared prompt or one prompt per text

        Returns:
            List of prediction dicts
        """
        if isinstance(multi_label, list):
            classification_type = ["multi-label" if item else "single-label" for item in multi_label]
        else:
            classification_type = "multi-label" if multi_label else "single-label"

        return self.pipeline(
            texts,
            labels,
            threshold=threshold,
            batch_size=max(len(texts), 1),
            classification_type=classification_type,
            examples=examples,
            prompt=prompt,
        )

    def predict(
        self,
        texts: str | list[str],
        labels: list[str],
        threshold: float | None = None,
        multi_label: bool = True,
        examples: list[dict[str, Any]] | None = None,
        prompt: str | list[str] | None = None,
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
            )
            return results

        async def __call__(self, request) -> list[dict[str, Any]]:
            """HTTP endpoint - accepts single text per request."""
            payload = await request.json()
            text = payload.get("text") or payload.get("texts")
            if isinstance(text, list):
                # If list provided, take first element for compatibility
                text = text[0] if text else ""
            return await self.predict(
                text=text,
                labels=payload["labels"],
                threshold=payload.get("threshold"),
                multi_label=payload.get("multi_label", True),
                examples=payload.get("examples"),
                prompt=payload.get("prompt"),
            )

    return GLiClassDeployment.bind(config)


def serve_gliclass(
    config: GLiClassServeConfig,
    blocking: bool = False,
) -> Any:
    import ray

    if not ray.is_initialized():
        ray.init(address=config.ray_address, ignore_reinit_error=True)

    serve.start(detached=True, http_options={"port": config.http_port})

    app = _build_deployment(config)
    handle = serve.run(app, name="gliclass", route_prefix=config.route_prefix)

    logger.info("GLiClass server running at http://localhost:%d%s", config.http_port, config.route_prefix)

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
