"""Client for GLiClass serving endpoint."""

from typing import Any

import requests


class GLiClassAPIError(RuntimeError):
    """The process API rejected a request with an error object (``code``, ``message``, ``path``)."""

    def __init__(self, code: str, message: str, *, status=None, path=None, details=None):
        super().__init__(f"{code}: {message}" + (f" (at {path})" if path else ""))
        self.code = code
        self.message = message
        self.status = status
        self.path = path
        self.details = details


class GLiClassClient:
    """Client for interacting with GLiClass Ray Serve deployment."""

    def __init__(self, url: str = "http://localhost:8000/gliclass", model: str | None = None, timeout: float = 30.0):
        """Initialize the client.

        Args:
            url: Base URL of the GLiClass server
            model: Model id for process requests; defaults to the first id ``GET /v1/models`` reports
            timeout: Per-request timeout in seconds
        """
        self.url = url.rstrip("/")
        self._model = model
        self.timeout = timeout

    def __call__(
        self,
        texts: str | list[str],
        labels: list[str] | list[list[str]],
        threshold: float = 0.5,
        multi_label: bool = True,
        examples: list[dict] | None = None,
        prompt: str | list[str] | None = None,
        adapter_id: str | None = None,
    ) -> list[list[dict]]:
        """Classify text(s) - same interface as pipeline.

        Args:
            texts: Single text or list of texts to classify
            labels: List of labels (same for all) or list of label lists (per text)
            threshold: Confidence threshold for predictions
            multi_label: Whether to enable multi-label classification
            examples: Optional list of example classifications
            prompt: Optional task description prompt (string or list)
            adapter_id: Optional LoRA adapter id

        Returns:
            List of results, one per text. Each result is a list of {"label": ..., "score": ...}
        """
        payload = {
            "texts": texts,
            "labels": labels,
            "threshold": threshold,
            "multi_label": multi_label,
        }
        if examples is not None:
            payload["examples"] = examples
        if prompt is not None:
            payload["prompt"] = prompt
        if adapter_id is not None:
            payload["adapter_id"] = adapter_id

        response = requests.post(self.url, json=payload, timeout=30)
        response.raise_for_status()

        return response.json()

    def classify(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
        multi_label: bool = True,
        examples: list[dict] | None = None,
        prompt: str | None = None,
        adapter_id: str | None = None,
    ) -> list[dict]:
        """Classify a single text (convenience method).

        Args:
            text: Input text to classify
            labels: List of possible labels
            threshold: Confidence threshold for predictions
            multi_label: Whether to enable multi-label classification
            examples: Optional list of example classifications
            prompt: Optional task description prompt
            adapter_id: Optional LoRA adapter id

        Returns:
            List of predictions: [{"label": ..., "score": ...}, ...]
        """
        results = self(text, labels, threshold, multi_label, examples, prompt, adapter_id)
        return results[0]

    def adapter_cache_status(self, adapter_id: str | None = None) -> dict:
        params = {"adapter_id": adapter_id} if adapter_id is not None else None
        response = requests.get(f"{self.url}/adapter-cache", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def is_adapter_cached(self, adapter_id: str) -> bool:
        return bool(self.adapter_cache_status(adapter_id).get("cached"))

    # -- process API (``/v1``) ---------------------------------------------------------

    def _request(self, method: str, suffix: str, payload: Any = None) -> Any:
        """Call ``{url}/v1{suffix}``; API error objects raise ``GLiClassAPIError``."""
        response = requests.request(method, f"{self.url}/v1{suffix}", json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            try:
                error = response.json().get("error")
            except ValueError:
                error = None
            if isinstance(error, dict):
                raise GLiClassAPIError(
                    error.get("code", "INTERNAL"),
                    error.get("message", ""),
                    status=response.status_code,
                    path=error.get("path"),
                    details=error.get("details"),
                )
            response.raise_for_status()
        return response.json()

    @property
    def model(self) -> str:
        """Model id sent with process requests: the one given, else the server's first model."""
        if self._model is None:
            self._model = self.models()["data"][0]["id"]
        return self._model

    def _body(self, defaults, options, model, **fields) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model or self.model, **fields}
        if defaults is not None:
            body["defaults"] = defaults
        if options is not None:
            body["options"] = options
        return body

    def process(
        self,
        input: Any,
        tasks: dict[str, dict[str, Any]],
        *,
        defaults: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/process``: one input, many named ``classify`` / ``boolean`` / ``score`` tasks.

        Example:
            >>> client.process(
            ...     "I was charged twice, please refund me.",
            ...     tasks={
            ...         "intent": {"type": "classify", "labels": ["refund", "technical issue", "question"]},
            ...         "refund": {"type": "boolean", "instruction": "asks for a refund"},
            ...         "urgency": {"type": "score", "labels": ["low", "high"], "order": ["low", "high"]},
            ...     },
            ...     defaults={"adapter_id": None},
            ... )["results"]["intent"]["value"]
        """
        return self._request("POST", "/process", self._body(defaults, options, model, input=input, tasks=tasks))

    def process_batch(
        self,
        items: list[dict[str, Any]],
        *,
        defaults: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """``POST /v1/process/batch``: items are ``{"id"?, "input", "tasks", "defaults"?, "options"?}``."""
        return self._request("POST", "/process/batch", self._body(defaults, options, model, items=items))

    async def process_async(self, input: Any, tasks: dict[str, dict[str, Any]], **kwargs) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.process, input, tasks, **kwargs)

    async def process_batch_async(self, items: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.process_batch, items, **kwargs)

    def judge(
        self,
        input: Any,
        instruction: str,
        *,
        threshold: float | None = None,
        adapter_id: str | None = None,
    ) -> bool | None:
        """One ``boolean`` task: is ``instruction`` (a statement, e.g. "asks for a refund") true of the input?"""
        task: dict[str, Any] = {"type": "boolean", "instruction": instruction, "adapter_id": adapter_id}
        if threshold is not None:
            task["threshold"] = threshold
        return self.process(input, {"decision": task})["results"]["decision"]["value"]

    def score(
        self,
        input: Any,
        *,
        labels: list[str] | None = None,
        scale: dict[str, float] | None = None,
        instruction: str | None = None,
        adapter_id: str | None = None,
    ) -> Any:
        """One ``score`` task: ordered ``labels`` (low to high) or a numeric ``scale`` ``{min, max, step?}``."""
        task: dict[str, Any] = {"type": "score", "instruction": instruction, "adapter_id": adapter_id}
        if scale is not None:
            task["scale"] = scale
        else:
            task.update(labels=labels, order=labels)
        return self.process(input, {"score": task})["results"]["score"]["value"]

    def models(self) -> dict[str, Any]:
        """``GET /v1/models``."""
        return self._request("GET", "/models")

    def capabilities(self, model: str | None = None) -> dict[str, Any]:
        """``GET /v1/models/{model}/capabilities``."""
        return self._request("GET", f"/models/{model or self.model}/capabilities")

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.url}/v1/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


if __name__ == "__main__":
    client = GLiClassClient()

    # Single text
    result = client.classify(
        text="This is a great product! I love it.",
        labels=["positive", "negative", "neutral"],
        threshold=0.3,
    )
    print("Single prediction:", result)

    # Batch
    results = client(
        texts=["Great product!", "Terrible experience"],
        labels=["positive", "negative", "neutral"],
        threshold=0.3,
    )
    print("Batch predictions:", results)
