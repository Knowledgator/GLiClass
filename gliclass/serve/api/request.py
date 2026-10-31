"""Validation of process requests into task specs (NEW_API.md §4, §8, §13.4).

GLiClass serves three task types, all scored by one zero-shot classification row:

* ``classify`` -- ``labels``; single-label by default, ``multi_label: true`` for many.
* ``boolean`` -- a yes/no decision. GLiClass scores one specific label: the ``true``
  label's title if given, otherwise the task's ``instruction`` (phrase it as a
  statement, e.g. "asks for a refund"). ``truth_score`` is that label's score.
* ``score`` -- ordinal (``labels`` + ``order``) or numeric (``scale``: ``{min, max,
  step?}``, run as one label per step; at most ``max_scale_steps`` steps).

Every task takes ``adapter_id``, ``instruction`` (shown to the model as the task
prompt), ``threshold`` and, when the checkpoint supports it, few-shot ``examples``
(``[{"text", "labels": [label ids]}]``).

Settings shared by many tasks can be given once in the request's ``defaults``:
``adapter_id``, ``threshold``, ``instruction``, ``multi_label`` and ``examples``.
A task's own value wins; an explicit ``null`` in a task skips the request default.
``defaults.threshold`` applies to multi-label ``classify`` only: single-label
decisions take a cutoff only when the task sets one, because a cutoff turns a
low-scoring top label into ``null``. ``options.adapter_id`` is an alias of
``defaults.adapter_id``.
"""

from __future__ import annotations

import math
from typing import Any
from datetime import date, datetime
from dataclasses import field, dataclass

from .errors import APIError, invalid, pointer
from .labels import Label, render_label, normalize_labels
from .render import RenderedInput, render_input
from .capabilities import ALL_TASK_TYPES, SERVED_TASK_TYPES, RESERVED_TASK_TYPES, Capabilities

OPTION_KEYS = frozenset(
    {
        "include_label_definitions",
        "include_usage",
        "include_evidence",
        "include_layout_text",
        "reference_date",
        "adapter_id",
    }
)
# ``include_evidence`` / ``include_layout_text`` / ``reference_date`` belong to task and
# input types GLiClass does not serve; they are validated and have no effect.
BOOLEAN_OPTIONS = ("include_label_definitions", "include_usage", "include_evidence", "include_layout_text")
SETTINGS = frozenset({"adapter_id", "threshold", "instruction", "examples"})
TASK_FIELDS = {
    "classify": SETTINGS | {"type", "labels", "multi_label"},
    "boolean": SETTINGS | {"type", "labels"},
    "score": SETTINGS | {"type", "labels", "order", "scale"},
}
DEFAULT_KEYS = SETTINGS | {"multi_label"}
REQUEST_KEYS = frozenset({"model", "input", "tasks", "options", "defaults"})
BATCH_KEYS = frozenset({"model", "items", "options", "defaults"})
BOOLEAN_IDS = ("true", "false")


@dataclass
class TaskSpec:
    name: str
    type: str
    path: str
    labels: list[Label]  # what the model scores, in prompt order
    definitions: list[Label] = field(default_factory=list)  # reported with include_label_definitions
    multi_label: bool = False
    threshold: float | None = None  # the cutoff applied; None when no cutoff applies
    instruction: str | None = None  # the task prompt shown to the model
    examples: list[dict[str, Any]] | None = None  # few-shot examples with model-facing label texts
    order: list[str] | None = None
    scale: dict[str, float] | None = None
    scale_values: list[float] | None = None
    adapter_id: str | None = None
    adapter_path: str | None = None  # where the adapter was set, for error paths

    @property
    def single_label(self) -> bool:
        return not (self.type == "classify" and self.multi_label)


@dataclass
class Options:
    include_label_definitions: bool = False
    include_usage: bool = False
    include_evidence: bool = False
    include_layout_text: bool = False
    reference_date: str | None = None
    adapter_id: str | None = None


@dataclass
class ParsedRequest:
    model: str
    rendered: RenderedInput
    tasks: list[TaskSpec]
    options: Options


def _threshold(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise invalid("A threshold must be a number", where)
    if not 0.0 <= float(value) <= 1.0:
        raise invalid("A threshold must be between 0 and 1", where)
    return float(value)


def _bool(value: Any, name: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid(f"'{name}' must be a boolean", where)
    return value


def _number(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise invalid("Expected a number", where)
    return float(value)


def _examples(value: Any, where: str) -> list[dict[str, Any]]:
    """Few-shot examples as ``[{"text": str, "labels": [label id, ...]}]``."""
    if not isinstance(value, list):
        raise invalid("'examples' must be a list of {'text', 'labels'} objects", where)
    examples = []
    for index, example in enumerate(value):
        at = f"{where}/{index}"
        if not isinstance(example, dict) or set(example) - {"text", "labels"} or "text" not in example:
            raise invalid("An example must be {'text', 'labels'}", at)
        if not isinstance(example["text"], str):
            raise invalid("An example's 'text' must be a string", f"{at}/text")
        labels = example.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise invalid("An example's 'labels' must be a list of label IDs", f"{at}/labels")
        examples.append({"text": example["text"], "labels": labels})
    return examples


def _check_setting(key: str, value: Any, where: str) -> Any:
    """Validate one setting value (``null`` means "use the deployment default")."""
    if value is None:
        return None
    if key == "threshold":
        return _threshold(value, where)
    if key == "multi_label":
        return _bool(value, key, where)
    if key in ("adapter_id", "instruction"):
        if not isinstance(value, str):
            raise invalid(f"'{key}' must be a string or null", where)
        if key == "instruction":
            return " ".join(value.split()) or None
        return value
    if key == "examples":
        return _examples(value, where)
    raise invalid(f"Unknown setting '{key}'", where)


def parse_options(raw: Any, path: str = "/options") -> Options:
    if raw is None:
        return Options()
    if not isinstance(raw, dict):
        raise invalid("'options' must be an object", path)
    for key in raw:
        if key not in OPTION_KEYS:
            raise invalid(f"Unknown option '{key}'", f"{path}/{key}")
    options = Options()
    for key in BOOLEAN_OPTIONS:
        if key in raw:
            setattr(options, key, _bool(raw[key], key, f"{path}/{key}"))
    if raw.get("reference_date") is not None:
        value = raw["reference_date"]
        if not isinstance(value, str):
            raise invalid("'reference_date' must be an ISO 8601 string", f"{path}/reference_date")
        try:
            (datetime if "T" in value else date).fromisoformat(value)
        except ValueError as exc:
            raise invalid(f"'reference_date' is not ISO 8601: {exc}", f"{path}/reference_date") from exc
        options.reference_date = value
    if raw.get("adapter_id") is not None:
        if not isinstance(raw["adapter_id"], str):
            raise invalid("'adapter_id' must be a string", f"{path}/adapter_id")
        options.adapter_id = raw["adapter_id"]
    return options


def parse_defaults(raw: Any, options: Options, path: str = "/defaults") -> dict[str, tuple[Any, str]]:
    """Request-level settings as ``{key: (value, path)}``."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise invalid("'defaults' must be an object", path)
    defaults: dict[str, tuple[Any, str]] = {}
    for key, raw_value in raw.items():
        if key not in DEFAULT_KEYS:
            raise invalid(f"Unknown default '{key}'", f"{path}/{key}")
        defaults[key] = (_check_setting(key, raw_value, f"{path}/{key}"), f"{path}/{key}")
    if options.adapter_id is not None:
        if "adapter_id" in defaults and defaults["adapter_id"][0] != options.adapter_id:
            raise invalid("'options.adapter_id' and 'defaults.adapter_id' disagree", "/options/adapter_id")
        defaults.setdefault("adapter_id", (options.adapter_id, "/options/adapter_id"))
    return defaults


def merge_options(batch: Any, item: Any) -> Any:
    """Item-level options (or defaults) override batch-level ones."""
    if batch is None:
        return item
    if item is None:
        return batch
    if not isinstance(batch, dict) or not isinstance(item, dict):
        return item
    return {**batch, **item}


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.10g}"


class RequestParser:
    """Validate requests against the contract and the model's capabilities."""

    def __init__(self, capabilities: Capabilities) -> None:
        self.capabilities = capabilities

    def check_model(self, model: Any, path: str = "/model") -> str:
        if not isinstance(model, str) or not model:
            raise invalid("'model' must be a non-empty string", path)
        if model not in self.capabilities.model_ids():
            raise APIError(
                "MODEL_NOT_FOUND",
                f"Model {model!r} is not served here",
                path,
                {"available": self.capabilities.model_ids()},
            )
        return self.capabilities.model

    def render(self, value: Any) -> RenderedInput:
        rendered = render_input(value)
        limit = self.capabilities.limit("max_input_chars")
        if limit and len(rendered.text) > limit:
            raise APIError(
                "INPUT_TOO_LONG",
                f"Rendered input has {len(rendered.text)} characters; the limit is {limit}",
                "/input",
            )
        return rendered

    def parse(self, body: Any, *, check_model: bool = True) -> ParsedRequest:
        if not isinstance(body, dict):
            raise invalid("The request body must be a JSON object", "")
        for key in body:
            if key not in REQUEST_KEYS:
                raise invalid(f"Unknown request field '{key}'", pointer(key))
        model = self.check_model(body.get("model")) if check_model else self.capabilities.model
        if "input" not in body:
            raise invalid("'input' is required", "/input")
        options = parse_options(body.get("options"))
        defaults = parse_defaults(body.get("defaults"), options)
        tasks_raw = body.get("tasks")
        if not isinstance(tasks_raw, dict) or not tasks_raw:
            raise invalid("'tasks' must be a non-empty object", "/tasks")
        max_tasks = self.capabilities.limit("max_tasks")
        if max_tasks and len(tasks_raw) > max_tasks:
            raise invalid(f"At most {max_tasks} tasks are allowed per request", "/tasks")
        tasks = [self.parse_task(name, raw, pointer("tasks", name), defaults) for name, raw in tasks_raw.items()]
        rendered = self.render(body["input"])
        return ParsedRequest(model=model, rendered=rendered, tasks=tasks, options=options)

    # -- tasks ----------------------------------------------------------------

    def parse_task(
        self, name: str, raw: Any, path: str, defaults: dict[str, tuple[Any, str]] | None = None
    ) -> TaskSpec:
        if not name:
            raise invalid("Task names must be non-empty", path)
        if not isinstance(raw, dict):
            raise invalid("A task must be an object", path)
        task_type = raw.get("type")
        if task_type is None:
            raise invalid("Every task needs a 'type'", f"{path}/type")
        if task_type in RESERVED_TASK_TYPES or (task_type in ALL_TASK_TYPES and task_type not in SERVED_TASK_TYPES):
            raise APIError(
                "UNSUPPORTED_TASK_TYPE",
                f"GLiClass does not serve '{task_type}' tasks",
                f"{path}/type",
                {"model": self.capabilities.model, "supported": list(self.capabilities.task_types)},
            )
        if task_type not in ALL_TASK_TYPES:
            raise invalid(
                f"Unknown task type {task_type!r}; expected one of {', '.join(ALL_TASK_TYPES)}", f"{path}/type"
            )
        for key in raw:
            if key not in TASK_FIELDS[task_type]:
                raise invalid(f"Field '{key}' is not valid for '{task_type}' tasks", f"{path}/{key}")
        defaults = defaults or {}

        def own(key: str) -> tuple[Any, str] | None:
            """The task's own value and path, or None when the task does not set the key."""
            if key not in raw:
                return None
            return _check_setting(key, raw[key], f"{path}/{key}"), f"{path}/{key}"

        def setting(key: str) -> tuple[Any, str | None]:
            """The task's value, else the request default; ``(None, None)`` when neither is set."""
            return own(key) or defaults.get(key, (None, None))

        task = TaskSpec(name=name, type=task_type, path=path, labels=[])
        task.adapter_id, adapter_path = setting("adapter_id")
        task.adapter_path = adapter_path or f"{path}/adapter_id"
        task.instruction = setting("instruction")[0]
        getattr(self, f"parse_{task_type}")(task, raw, path, own, setting, defaults)
        self.check_label_limit(task, path)
        examples, examples_path = setting("examples")
        if examples:
            if not self.capabilities.examples_supported:
                raise APIError(
                    "UNSUPPORTED_FIELD",
                    "This checkpoint was not trained with few-shot examples (its tokenizer has no example token)",
                    examples_path,
                )
            task.examples = self.model_examples(task, examples, examples_path)
        return task

    def labels(self, spec: Any, where: str) -> list[Label] | None:
        return normalize_labels(spec, where, template=self.capabilities.label_template)

    def check_label_limit(self, task: TaskSpec, path: str) -> None:
        limit = self.capabilities.limit("max_labels_per_task")
        if limit and len(task.labels) > limit:
            where = f"{path}/scale" if task.scale is not None else f"{path}/labels"
            raise invalid(f"At most {limit} labels are allowed per task; this task has {len(task.labels)}", where)

    def parse_classify(self, task: TaskSpec, raw: dict[str, Any], path: str, own, setting, defaults) -> None:
        task.multi_label = bool(setting("multi_label")[0])
        task.labels = self.labels(raw.get("labels"), f"{path}/labels") or []
        if not task.labels:
            raise invalid("'classify' tasks need at least one label", f"{path}/labels")
        task.definitions = task.labels
        if task.multi_label:
            threshold = setting("threshold")[0]
            task.threshold = (
                threshold if threshold is not None else float(self.capabilities.default_thresholds["classify"])
            )
        else:
            task.threshold = (own("threshold") or (None, None))[0]

    def parse_boolean(self, task: TaskSpec, raw: dict[str, Any], path: str, own, setting, defaults) -> None:
        given = self.labels(raw.get("labels"), f"{path}/labels") or []
        for label in given:
            if label.id not in BOOLEAN_IDS:
                raise invalid("'boolean' labels may only define 'true' and 'false'", f"{path}/labels/{label.id}")
        task.definitions = given
        true_label = next((label for label in given if label.id == "true"), None)
        own_instruction = (own("instruction") or (None, None))[0]
        if true_label is not None and (true_label.title is not None or true_label.description is not None):
            text = true_label.text
        elif own_instruction is not None:
            # The task's instruction is the statement to check; only a request-wide
            # instruction is left as the prompt.
            text = render_label("true", own_instruction, None, "{title}")
            task.instruction = defaults.get("instruction", (None, None))[0]
        else:
            raise invalid(
                "A 'boolean' task needs an 'instruction' (the statement to check, e.g. 'asks for a refund') "
                "or a 'true' label with a title",
                f"{path}/instruction",
            )
        task.labels = [Label(id="true", text=text)]
        task.threshold = (own("threshold") or (None, None))[0]

    def parse_score(self, task: TaskSpec, raw: dict[str, Any], path: str, own, setting, defaults) -> None:
        has_scale = raw.get("scale") is not None
        has_labels = raw.get("labels") is not None or raw.get("order") is not None
        if has_scale == has_labels:
            raise invalid("A 'score' task needs exactly one of 'scale' or 'labels' + 'order'", path)
        if has_scale:
            if raw.get("threshold") is not None:
                raise invalid("'threshold' is not valid with 'scale'", f"{path}/threshold")
            self.parse_scale(task, raw["scale"], f"{path}/scale")
            return
        task.labels = self.labels(raw.get("labels"), f"{path}/labels") or []
        if len(task.labels) < 2:
            raise invalid("An ordinal 'score' needs at least two labels", f"{path}/labels")
        order = raw.get("order")
        ids = [label.id for label in task.labels]
        if (
            not isinstance(order, list)
            or not all(isinstance(item, str) for item in order)
            or sorted(order) != sorted(ids)
            or len(set(order)) != len(order)
        ):
            raise invalid("'order' must list every label ID exactly once", f"{path}/order")
        by_id = {label.id: label for label in task.labels}
        task.order = list(order)
        task.labels = [by_id[label_id] for label_id in order]  # the model sees the labels in order
        task.definitions = task.labels
        task.threshold = (own("threshold") or (None, None))[0]

    def parse_scale(self, task: TaskSpec, scale: Any, where: str) -> None:
        if not isinstance(scale, dict) or set(scale) - {"min", "max", "step"} or not {"min", "max"} <= set(scale):
            raise invalid("'scale' must be {min, max, step?}", where)
        low = _number(scale["min"], f"{where}/min")
        high = _number(scale["max"], f"{where}/max")
        if low >= high:
            raise invalid("'scale.min' must be below 'scale.max'", where)
        step = scale.get("step")
        if step is None:
            if not (low.is_integer() and high.is_integer()):
                raise invalid("'scale.step' is required when 'min' or 'max' is not an integer", f"{where}/step")
            step = 1.0
        step = _number(step, f"{where}/step")
        if step <= 0:
            raise invalid("'scale.step' must be positive", f"{where}/step")
        count = math.floor((high - low) / step + 1e-9) + 1
        limit = self.capabilities.max_scale_steps
        if count > limit:
            raise invalid(f"The scale has {count} steps; at most {limit} are supported", where)
        values = [low + index * step for index in range(count)]
        task.scale = {key: scale[key] for key in ("min", "max", "step") if key in scale}
        task.scale_values = values
        task.labels = [Label(id=_number_text(value), text=_number_text(value)) for value in values]

    @staticmethod
    def model_examples(task: TaskSpec, examples: list[dict[str, Any]], where: str) -> list[dict[str, Any]]:
        """Examples with label IDs replaced by the label text the model sees."""
        if task.type == "boolean":
            texts = {"true": task.labels[0].text, "false": None}
        else:
            texts = {label.id: label.text for label in task.labels}
        result = []
        for index, example in enumerate(examples):
            labels = []
            for label_id in example["labels"]:
                if label_id not in texts:
                    raise invalid(
                        f"Example label {label_id!r} is not a label of task '{task.name}'",
                        f"{where}/{index}/labels",
                    )
                if texts[label_id] is not None:
                    labels.append(texts[label_id])
            result.append({"text": example["text"], "labels": labels})
        return result
