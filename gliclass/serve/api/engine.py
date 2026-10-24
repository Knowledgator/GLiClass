"""Plan model rows for process requests, run them, and assemble results.

Every task becomes one zero-shot classification row: the rendered input, the
task's labels in prompt order, its instruction as the task prompt, its
few-shot examples and its adapter. A row returns an independent (sigmoid)
score for every label, and each task then applies its own selection rule:

* single-label (``classify``, ordinal ``score``, ``boolean``): the top label,
  or ``null`` below an explicit threshold;
* multi-label ``classify``: every label at or above the threshold.

Since the selection happens after the model runs, tasks that show the model
the same thing -- same labels, instruction, examples and adapter -- share one
row even when their mode or threshold differ. Tasks with different labels get
their own rows, because labels attend to each other in the prompt; all rows of
a request, and of every request batched with it, still share the dynamic
batcher's forward passes.
"""

from __future__ import annotations

import json
import time
import asyncio
import logging
from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

from .errors import APIError, invalid
from .request import BATCH_KEYS, TaskSpec, ParsedRequest, RequestParser, merge_options

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Awaitable

    from .capabilities import Capabilities

logger = logging.getLogger(__name__)

Row = dict[str, Any]
RowOutput = list[dict[str, Any]]  # [{"label": text, "score": float}, ...] for every label of the row


@dataclass
class RowRef:
    task: TaskSpec
    row: Row
    tokens: int
    slot: int = -1


@dataclass
class Plan:
    parsed: ParsedRequest
    refs: list[RowRef] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    outputs: list[RowOutput] | None = None
    error: APIError | None = None


class ProcessEngine:
    """Validate, plan, run and assemble process requests for GLiClass.

    The engine does not run the model. ``process*`` take a row runner:

    * sync: ``run_rows(rows) -> outputs``, one output per row;
    * async: ``run_row(row) -> awaitable output``, called once per row so a
      dynamic batcher can merge rows from many requests.

    A row is ``{"text", "labels", "prompt", "examples", "adapter_id"}``; its output
    lists every label with its independent score.
    """

    def __init__(
        self,
        capabilities: Capabilities,
        *,
        max_tokens: int,
        count_tokens: Callable[[str], int] | None = None,
        resolve_adapter: Callable[[str | None], Any] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.parser = RequestParser(capabilities)
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens or (lambda text: len(text.split()))
        self.resolve_adapter = resolve_adapter or (lambda adapter_id: adapter_id)

    # -- public: sync -----------------------------------------------------------

    def process(self, body: Any, run_rows: Callable[[list[Row]], list[RowOutput]]) -> dict[str, Any]:
        """One request; raises ``APIError`` when it is rejected or fails."""
        outcome = self.process_many([body], run_rows)[0]
        if isinstance(outcome, APIError):
            raise outcome
        return outcome

    def process_many(
        self,
        bodies: Sequence[Any],
        run_rows: Callable[[list[Row]], list[RowOutput]],
        *,
        check_model: bool = True,
    ) -> list[dict[str, Any] | APIError]:
        """Independent requests run together; each succeeds or fails alone."""
        outcomes, plans = self._prepare(bodies, check_model=check_model)
        rows = self._schedule(plans)
        if rows:
            try:
                results: list[Any] = list(run_rows(rows))
                if len(results) != len(rows):
                    raise RuntimeError(f"Row runner returned {len(results)} outputs for {len(rows)} rows")
            except Exception as exc:
                results = [exc] * len(rows)
            self._distribute(plans, results)
        return self._finish(outcomes, plans)

    def process_batch(self, body: Any, run_rows: Callable[[list[Row]], list[RowOutput]]) -> dict[str, Any]:
        """``/v1/process/batch``: many items, per-item errors, order kept."""
        items, model = self.parse_batch(body)
        outcomes = self.process_many([item for _, item in items], run_rows, check_model=False)
        return self.batch_response(items, model, outcomes)

    # -- public: async ----------------------------------------------------------

    async def process_async(self, body: Any, run_row: Callable[[Row], Awaitable[RowOutput]]) -> dict[str, Any]:
        outcome = (await self.process_many_async([body], run_row))[0]
        if isinstance(outcome, APIError):
            raise outcome
        return outcome

    async def process_many_async(
        self,
        bodies: Sequence[Any],
        run_row: Callable[[Row], Awaitable[RowOutput]],
        *,
        check_model: bool = True,
    ) -> list[dict[str, Any] | APIError]:
        outcomes, plans = self._prepare(bodies, check_model=check_model)
        rows = self._schedule(plans)
        if rows:
            results = await asyncio.gather(*(run_row(row) for row in rows), return_exceptions=True)
            self._distribute(plans, list(results))
        return self._finish(outcomes, plans)

    async def process_batch_async(self, body: Any, run_row: Callable[[Row], Awaitable[RowOutput]]) -> dict[str, Any]:
        items, model = self.parse_batch(body)
        outcomes = await self.process_many_async([item for _, item in items], run_row, check_model=False)
        return self.batch_response(items, model, outcomes)

    # -- batch envelope -----------------------------------------------------------

    def parse_batch(self, body: Any) -> tuple[list[tuple[str | None, Any]], str]:
        if not isinstance(body, dict):
            raise invalid("The request body must be a JSON object", "")
        for key in body:
            if key not in BATCH_KEYS:
                raise invalid(f"Unknown batch field '{key}'", f"/{key}")
        model = self.parser.check_model(body.get("model"))
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise invalid("'items' must be a non-empty list", "/items")
        limit = self.capabilities.limit("max_batch_items")
        if limit and len(raw_items) > limit:
            raise invalid(f"At most {limit} items are allowed per batch", "/items")
        batch_shared = {}
        for key in ("options", "defaults"):
            value = body.get(key)
            if value is not None and not isinstance(value, dict):
                raise invalid(f"'{key}' must be an object", f"/{key}")
            batch_shared[key] = value
        items: list[tuple[str | None, Any]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                items.append((None, item))  # fails on its own as an invalid request
                continue
            item_id = item.get("id")
            if item_id is not None:
                if not isinstance(item_id, str):
                    raise invalid("Item 'id' must be a string", f"/items/{index}/id")
                if item_id in seen_ids:
                    raise invalid(f"Duplicate item ID {item_id!r}", f"/items/{index}/id")
                seen_ids.add(item_id)
            item_body = {key: value for key, value in item.items() if key != "id"}
            for key, batch_value in batch_shared.items():  # item values override batch values
                merged = merge_options(batch_value, item.get(key))
                if merged is None:
                    item_body.pop(key, None)
                else:
                    item_body[key] = merged
            items.append((item_id, item_body))
        return items, model

    @staticmethod
    def batch_response(
        items: list[tuple[str | None, Any]],
        model: str,
        outcomes: Sequence[dict[str, Any] | APIError],
    ) -> dict[str, Any]:
        response_items = []
        for index, ((item_id, _), outcome) in enumerate(zip(items, outcomes, strict=True)):
            entry: dict[str, Any] = {}
            if item_id is not None:
                entry["id"] = item_id
            if isinstance(outcome, APIError):
                entry.update(outcome.with_prefix(f"/items/{index}").to_dict())
            else:
                entry["results"] = outcome["results"]
                if "usage" in outcome:
                    entry["usage"] = outcome["usage"]
            response_items.append(entry)
        return {"model": model, "items": response_items}

    # -- planning -------------------------------------------------------------------

    def _prepare(
        self, bodies: Sequence[Any], *, check_model: bool
    ) -> tuple[list[dict[str, Any] | APIError | None], list[tuple[int, Plan]]]:
        outcomes: list[dict[str, Any] | APIError | None] = [None] * len(bodies)
        plans: list[tuple[int, Plan]] = []
        for index, body in enumerate(bodies):
            try:
                plans.append((index, self.plan(self.parser.parse(body, check_model=check_model))))
            except APIError as exc:
                outcomes[index] = exc
        return outcomes, plans

    def row_tokens(self, row: Row) -> int:
        """Tokens of the full model input: text, prompt, labels (plus a marker each) and examples."""
        parts = [row["text"], row["prompt"] or ""]
        parts += [f"{example['text']} {' '.join(example['labels'])}" for example in row["examples"] or []]
        return sum(self.count_tokens(part) for part in parts if part) + sum(
            self.count_tokens(label) + 1 for label in row["labels"]
        )

    def plan(self, parsed: ParsedRequest) -> Plan:
        # Check every distinct adapter before anything runs; errors point to where it was set.
        checked: set[str | None] = set()
        for task in parsed.tasks:
            if task.adapter_id in checked:
                continue
            try:
                self.resolve_adapter(task.adapter_id)
            except KeyError as exc:
                raise APIError("UNKNOWN_ADAPTER", str(exc).strip("'\""), task.adapter_path) from exc
            except ValueError as exc:
                raise invalid(str(exc), task.adapter_path) from exc
            checked.add(task.adapter_id)

        plan = Plan(parsed=parsed)
        for task in parsed.tasks:
            row = {
                "text": parsed.rendered.text,
                "labels": [label.text for label in task.labels],
                "prompt": task.instruction,
                "examples": task.examples,
                "adapter_id": task.adapter_id,
            }
            tokens = self.row_tokens(row)
            if tokens > self.max_tokens:
                # Classification reads the whole input; truncating it would silently drop text.
                raise APIError(
                    "INPUT_TOO_LONG",
                    f"Task '{task.name}' needs about {tokens} tokens (input, prompt, labels, examples); "
                    f"the model window is {self.max_tokens}",
                    "/input",
                    {"task": task.name, "tokens": tokens, "max_input_tokens": self.max_tokens},
                )
            plan.refs.append(RowRef(task=task, row=row, tokens=tokens))
        return plan

    @staticmethod
    def _row_key(row: Row) -> tuple:
        examples = json.dumps(row["examples"], sort_keys=True) if row["examples"] else None
        return (row["text"], tuple(row["labels"]), row["prompt"], examples, row["adapter_id"])

    def _schedule(self, plans: list[tuple[int, Plan]]) -> list[Row]:
        """Unique rows of all plans; every ``RowRef`` gets the slot of its row."""
        slots: dict[tuple, int] = {}
        rows: list[Row] = []
        for _, plan in plans:
            for ref in plan.refs:
                key = self._row_key(ref.row)
                slot = slots.get(key)
                if slot is None:
                    slot = slots[key] = len(rows)
                    rows.append(ref.row)
                ref.slot = slot
        return rows

    @staticmethod
    def _distribute(plans: list[tuple[int, Plan]], results: list[Any]) -> None:
        for _, plan in plans:
            outputs = []
            for ref in plan.refs:
                result = results[ref.slot]
                if isinstance(result, BaseException):
                    plan.error = _execution_error(result)
                    break
                outputs.append(result)
            else:
                plan.outputs = outputs

    def _finish(
        self, outcomes: list[dict[str, Any] | APIError | None], plans: list[tuple[int, Plan]]
    ) -> list[dict[str, Any] | APIError]:
        for index, plan in plans:
            if plan.error is not None:
                outcomes[index] = plan.error
                continue
            try:
                outcomes[index] = self.assemble(plan)
            except APIError as exc:
                outcomes[index] = exc
            except Exception as exc:
                logger.exception("Failed to assemble process results")
                outcomes[index] = APIError("INTERNAL", f"Result assembly failed: {exc}")
        return outcomes  # type: ignore[return-value]

    # -- assembly ---------------------------------------------------------------------

    def assemble(self, plan: Plan) -> dict[str, Any]:
        parsed = plan.parsed
        results: dict[str, Any] = {}
        for ref, output in zip(plan.refs, plan.outputs or [], strict=True):
            task = ref.task
            scores = self.label_scores(task, output)
            if task.type == "boolean":
                result = self.boolean_result(task, scores)
            elif task.scale is not None:
                result = self.scale_result(task, scores)
            else:
                result = self.choice_result(task, scores)
            result["adapter_id"] = task.adapter_id
            if parsed.options.include_label_definitions and task.definitions:
                result["label_definitions"] = {label.id: label.definition() for label in task.definitions}
            results[task.name] = result
        response: dict[str, Any] = {"model": parsed.model, "results": results}
        if parsed.options.include_usage:
            response["usage"] = {
                "rows": len(plan.refs),
                "tokens": sum(ref.tokens for ref in plan.refs),
                "latency_ms": round((time.perf_counter() - plan.started) * 1000.0, 1),
            }
        return response

    @staticmethod
    def label_scores(task: TaskSpec, output: RowOutput) -> dict[str, float]:
        """Score per label ID, in the task's label order."""
        by_text = {item.get("label"): float(item.get("score", 0.0)) for item in output or []}
        return {label.id: by_text.get(label.text, 0.0) for label in task.labels}

    @staticmethod
    def top(scores: dict[str, float]) -> str | None:
        return max(scores, key=scores.__getitem__) if scores else None  # ties keep label order

    def choice_result(self, task: TaskSpec, scores: dict[str, float]) -> dict[str, Any]:
        """``classify`` (single or multi-label) and ordinal ``score``."""
        if task.single_label:
            best = self.top(scores)
            value: Any = best
            if task.threshold is not None and best is not None and scores[best] < task.threshold:
                value = None
        else:
            value = [label_id for label_id, score in scores.items() if score >= task.threshold]
        return {"type": task.type, "value": value, "scores": scores, "threshold": task.threshold}

    @staticmethod
    def boolean_result(task: TaskSpec, scores: dict[str, float]) -> dict[str, Any]:
        truth = scores.get("true", 0.0)
        value: bool | None = truth >= 0.5
        top_score = truth if value else 1.0 - truth
        if task.threshold is not None and top_score < task.threshold:
            value = None
        return {
            "type": "boolean",
            "value": value,
            "truth_score": truth,
            # GLiClass scores the "true" statement only; "false" is its complement.
            "scores": {"true": truth, "false": 1.0 - truth},
            "threshold": task.threshold,
        }

    def scale_result(self, task: TaskSpec, scores: dict[str, float]) -> dict[str, Any]:
        best = self.top(scores)
        value: float | None = None
        if best is not None and task.scale_values:
            value = task.scale_values[[label.id for label in task.labels].index(best)]
            if float(value).is_integer():
                value = int(value)
        return {"type": "score", "value": value, "scale": dict(task.scale or {})}


def _execution_error(exc: BaseException) -> APIError:
    if isinstance(exc, APIError):
        return exc
    logger.error("Process execution failed: %s: %s", type(exc).__name__, exc)
    return APIError("INTERNAL", f"Model execution failed: {type(exc).__name__}: {exc}")
