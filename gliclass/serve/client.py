"""Client for GLiClass serving endpoint."""

import requests


class GLiClassClient:
    """Client for interacting with GLiClass Ray Serve deployment."""

    def __init__(self, url: str = "http://localhost:8000/gliclass"):
        """Initialize the client.

        Args:
            url: Base URL of the GLiClass server
        """
        self.url = url.rstrip("/")

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

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.url}/-/healthz", timeout=5)
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
