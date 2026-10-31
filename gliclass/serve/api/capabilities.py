"""What the served GLiClass checkpoint can run (NEW_API.md §13.2)."""

from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

from .labels import DEFAULT_LABEL_TEMPLATE

# Every task type of the unified API. GLiClass serves ``classify``, ``boolean``
# and ``score``; the rest are rejected with ``UNSUPPORTED_TASK_TYPE``.
ALL_TASK_TYPES = (
    "classify",
    "boolean",
    "score",
    "span",
    "relation",
    "json",
    "embedding",
    "rank",
    "box",
    "mask",
    "count",
    "time_span",
)
SERVED_TASK_TYPES = ("classify", "boolean", "score")
RESERVED_TASK_TYPES = ("text", "link")


@dataclass
class Capabilities:
    model: str
    aliases: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=lambda: list(SERVED_TASK_TYPES))
    label_template: str = DEFAULT_LABEL_TEMPLATE
    default_thresholds: dict[str, float] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)
    adapters_enabled: bool = False
    examples_supported: bool = False  # the checkpoint knows the few-shot example token
    max_scale_steps: int = 21

    def has_task(self, task_type: str) -> bool:
        return task_type in self.task_types

    def model_ids(self) -> list[str]:
        return [self.model, *self.aliases]

    def limit(self, name: str) -> int:
        """A limit, with ``0`` for "unlimited"."""
        value = self.limits.get(name) or 0
        return value if value > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "aliases": list(self.aliases),
            "modalities": ["text"],
            "task_types": list(self.task_types),
            "json_field_types": [],
            "json_text_formats": [],
            "label_template": self.label_template,
            "default_thresholds": dict(self.default_thresholds),
            # Non-positive limits mean "unlimited" and are reported as null.
            "limits": {key: (value if value and value > 0 else None) for key, value in self.limits.items()},
            "adapters": {"enabled": self.adapters_enabled},
            "extensions": {
                "examples": self.examples_supported,
                "score_scale": "step_labels",
                "max_scale_steps": self.max_scale_steps,
            },
        }


def capabilities_from_config(
    config: Any, *, examples_supported: bool = False, max_num_classes: int | None = None
) -> Capabilities:
    """Capabilities of a GLiClass deployment from its serve config.

    ``max_num_classes`` is the model's label capacity when it allocates a fixed
    number of label slots; it caps ``max_labels_per_task``.
    """
    model_id = getattr(config, "model_id", None) or config.model
    aliases = [alias for alias in [config.model, *(getattr(config, "model_aliases", None) or [])] if alias != model_id]
    max_labels = int(config.max_labels) if config.max_labels and config.max_labels > 0 else 0
    if max_num_classes:
        max_labels = min(max_labels, max_num_classes) if max_labels else max_num_classes
    return Capabilities(
        model=model_id,
        aliases=list(dict.fromkeys(aliases)),
        task_types=list(SERVED_TASK_TYPES),
        label_template=getattr(config, "label_template", DEFAULT_LABEL_TEMPLATE),
        default_thresholds={"classify": float(config.default_threshold)},
        limits={
            "max_input_tokens": int(config.max_model_len),
            "max_input_chars": int(getattr(config, "max_input_chars", 0)),
            "max_input_bytes": int(getattr(config, "max_request_bytes", 0)),
            "max_tasks": int(getattr(config, "max_tasks", 0)),
            "max_labels_per_task": max_labels,
            "max_batch_items": int(getattr(config, "max_batch_items", 0)),
        },
        adapters_enabled=bool(config.enable_polylora),
        examples_supported=examples_supported,
        max_scale_steps=int(getattr(config, "max_scale_steps", 21)),
    )
