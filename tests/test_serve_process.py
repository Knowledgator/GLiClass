"""Process API (``/v1/process``) for GLiClass serving: validation, planning, selection rules, assembly.

The engine is tested with a fake row runner that scores a label high when its
text appears in the input, so no model, GPU or Ray is needed.
"""

import asyncio
from unittest.mock import Mock

import pytest

from gliclass.serve.api import APIError, ProcessEngine
from gliclass.serve.server import PROCESS_ALL_SCORES, GLiClassServer
from gliclass.serve.api.capabilities import Capabilities

MODEL = "test/gliclass"


class FakeModel:
    """Scores 0.9 for labels whose text occurs in the input, 0.2 otherwise (0.6 for 'maybe ...')."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def score(text, label):
        if label.lower() in text.lower():
            return 0.9
        return 0.6 if label.startswith("maybe") else 0.2

    def run_row(self, row):
        return [{"label": label, "score": self.score(row["text"], label)} for label in row["labels"]]

    def run_rows(self, rows):
        self.calls.append([dict(row) for row in rows])
        return [self.run_row(row) for row in rows]


def make_engine(*, examples=False, max_tokens=512, resolve_adapter=None, **limits):
    capabilities = Capabilities(
        model=MODEL,
        aliases=["alias/gliclass"],
        default_thresholds={"classify": 0.5},
        limits={"max_tasks": 8, "max_labels_per_task": 6, "max_batch_items": 4, "max_input_chars": 10_000, **limits},
        examples_supported=examples,
        max_scale_steps=11,
    )
    return ProcessEngine(capabilities, max_tokens=max_tokens, resolve_adapter=resolve_adapter)


def process(engine, input, tasks, **extra):
    model = FakeModel()
    response = engine.process({"model": MODEL, "input": input, "tasks": tasks, **extra}, model.run_rows)
    return response, model


def error_of(engine, body):
    with pytest.raises(APIError) as info:
        engine.process(body, FakeModel().run_rows)
    return info.value


def known_adapters(*known):
    def resolve(adapter_id):
        if adapter_id is not None and adapter_id not in known:
            raise KeyError(f"Unknown LoRA adapter id: {adapter_id}")

    return resolve


TEXT = "I was charged twice, please send a refund. The app also crashes."


# -- classify ------------------------------------------------------------------------


def test_single_label_classify_returns_the_top_label_and_every_score():
    labels = {"refund": {"title": "refund", "metadata": {"team": "billing"}}, "crash": "app crash", "sales": None}
    response, _ = process(make_engine(), TEXT, {"intent": {"type": "classify", "labels": labels}})

    assert response["results"]["intent"] == {
        "type": "classify",
        "value": "refund",
        "scores": {"refund": 0.9, "crash": 0.9, "sales": 0.2},
        "threshold": None,
        "adapter_id": None,
    }


def test_single_label_classify_with_a_threshold_returns_null_below_it():
    tasks = {"t": {"type": "classify", "labels": ["sales", "login"], "threshold": 0.5}}
    response, _ = process(make_engine(), TEXT, tasks)

    assert (response["results"]["t"]["value"], response["results"]["t"]["threshold"]) == (None, 0.5)


def test_multi_label_classify_uses_the_deployment_default_threshold():
    tasks = {"t": {"type": "classify", "multi_label": True, "labels": ["refund", "app", "maybe sales", "login"]}}
    response, _ = process(make_engine(), TEXT, tasks)

    assert response["results"]["t"]["value"] == ["refund", "app", "maybe sales"]
    assert response["results"]["t"]["threshold"] == 0.5


def test_tasks_with_the_same_labels_share_one_row_whatever_their_mode():
    labels = ["refund", "app", "maybe sales"]
    tasks = {
        "single": {"type": "classify", "labels": labels},
        "multi": {"type": "classify", "labels": labels, "multi_label": True, "threshold": 0.7},
        "other": {"type": "classify", "labels": ["login"]},
    }
    response, model = process(make_engine(), TEXT, tasks)

    assert len(model.calls) == 1 and len(model.calls[0]) == 2
    assert response["results"]["single"]["value"] == "refund"
    assert response["results"]["multi"]["value"] == ["refund", "app"]


# -- boolean -------------------------------------------------------------------------


def test_boolean_scores_the_instruction_as_one_statement():
    tasks = {
        "refund": {"type": "boolean", "instruction": "send a refund"},
        "login": {"type": "boolean", "instruction": "cannot log in"},
    }
    response, model = process(make_engine(), TEXT, tasks)

    assert [row["labels"] for row in model.calls[0]] == [["send a refund"], ["cannot log in"]]
    assert all(row["prompt"] is None for row in model.calls[0])
    refund, login = response["results"]["refund"], response["results"]["login"]
    assert refund == {
        "type": "boolean",
        "value": True,
        "truth_score": 0.9,
        "scores": {"true": 0.9, "false": pytest.approx(0.1)},
        "threshold": None,
        "adapter_id": None,
    }
    assert (login["value"], login["truth_score"]) == (False, 0.2)


def test_boolean_with_a_true_title_uses_it_as_the_statement_and_the_instruction_as_prompt():
    tasks = {
        "t": {"type": "boolean", "instruction": "Read the ticket.", "labels": {"true": {"title": "send a refund"}}}
    }
    response, model = process(make_engine(), TEXT, tasks, options={"include_label_definitions": True})

    assert (model.calls[0][0]["labels"], model.calls[0][0]["prompt"]) == (["send a refund"], "Read the ticket.")
    assert response["results"]["t"]["label_definitions"] == {"true": {"title": "send a refund"}}


def test_boolean_threshold_turns_unsure_decisions_into_null():
    tasks = {"t": {"type": "boolean", "instruction": "maybe sales", "threshold": 0.8}}
    response, _ = process(make_engine(), TEXT, tasks)

    assert (response["results"]["t"]["value"], response["results"]["t"]["truth_score"]) == (None, 0.6)


def test_boolean_keeps_a_request_wide_instruction_as_its_prompt():
    tasks = {"t": {"type": "boolean", "instruction": "send a refund"}}
    _, model = process(make_engine(), TEXT, tasks, defaults={"instruction": "Classify this ticket."})

    assert (model.calls[0][0]["labels"], model.calls[0][0]["prompt"]) == (["send a refund"], "Classify this ticket.")


# -- score ---------------------------------------------------------------------------


def test_ordinal_score_shows_labels_in_order_and_returns_the_top_one():
    tasks = {
        "t": {
            "type": "score",
            "labels": {"high": "high", "low": "low", "app": "app"},
            "order": ["low", "app", "high"],
        }
    }
    response, model = process(make_engine(), TEXT, tasks)

    assert model.calls[0][0]["labels"] == ["low", "app", "high"]
    assert response["results"]["t"]["value"] == "app"
    assert list(response["results"]["t"]["scores"]) == ["low", "app", "high"]


def test_scale_score_runs_one_label_per_step():
    response, model = process(make_engine(), "rating: 4", {"t": {"type": "score", "scale": {"min": 1, "max": 5}}})

    assert model.calls[0][0]["labels"] == ["1", "2", "3", "4", "5"]
    assert response["results"]["t"] == {
        "type": "score",
        "value": 4,
        "scale": {"min": 1, "max": 5},
        "adapter_id": None,
    }


def test_scale_with_fractional_steps():
    rows = []

    def run_rows(batch):
        rows.extend(batch)
        return [[{"label": label, "score": 0.9 if label == "0.5" else 0.1} for label in row["labels"]] for row in batch]

    body = {
        "model": MODEL,
        "input": "about half",
        "tasks": {"t": {"type": "score", "scale": {"min": 0, "max": 1, "step": 0.25}}},
    }
    response = make_engine().process(body, run_rows)

    assert rows[0]["labels"] == ["0", "0.25", "0.5", "0.75", "1"]
    assert response["results"]["t"]["value"] == 0.5


# -- validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "code", "path"),
    [
        ({"type": "span", "labels": ["a"]}, "UNSUPPORTED_TASK_TYPE", "/tasks/t/type"),
        ({"type": "text"}, "UNSUPPORTED_TASK_TYPE", "/tasks/t/type"),
        ({"type": "nope"}, "INVALID_REQUEST", "/tasks/t/type"),
        ({"type": "classify"}, "INVALID_REQUEST", "/tasks/t/labels"),
        ({"type": "classify", "labels": ["a"], "order": ["a"]}, "INVALID_REQUEST", "/tasks/t/order"),
        ({"type": "classify", "labels": list("abcdefg")}, "INVALID_REQUEST", "/tasks/t/labels"),
        ({"type": "classify", "labels": ["a"], "threshold": 2}, "INVALID_REQUEST", "/tasks/t/threshold"),
        ({"type": "classify", "labels": ["a"], "multi_label": "yes"}, "INVALID_REQUEST", "/tasks/t/multi_label"),
        ({"type": "classify", "labels": ["a"], "instruction": 3}, "INVALID_REQUEST", "/tasks/t/instruction"),
        ({"type": "boolean"}, "INVALID_REQUEST", "/tasks/t/instruction"),
        ({"type": "boolean", "instruction": "x", "labels": {"maybe": "m"}}, "INVALID_REQUEST", "/tasks/t/labels/maybe"),
        ({"type": "boolean", "instruction": "x", "multi_label": True}, "INVALID_REQUEST", "/tasks/t/multi_label"),
        ({"type": "score", "labels": ["a", "b"]}, "INVALID_REQUEST", "/tasks/t/order"),
        ({"type": "score", "labels": ["a", "b"], "order": ["a"]}, "INVALID_REQUEST", "/tasks/t/order"),
        ({"type": "score", "labels": ["a"], "order": ["a"]}, "INVALID_REQUEST", "/tasks/t/labels"),
        ({"type": "score", "scale": {"min": 1, "max": 5}, "labels": ["a"]}, "INVALID_REQUEST", "/tasks/t"),
        ({"type": "score", "scale": {"min": 0, "max": 100}}, "INVALID_REQUEST", "/tasks/t/scale"),
        ({"type": "score", "scale": {"min": 0.5, "max": 2}}, "INVALID_REQUEST", "/tasks/t/scale/step"),
        ({"type": "score", "scale": {"min": 5, "max": 1}}, "INVALID_REQUEST", "/tasks/t/scale"),
        ({"type": "score", "scale": {"min": 1, "max": 5}, "threshold": 0.5}, "INVALID_REQUEST", "/tasks/t/threshold"),
        (
            {"type": "classify", "labels": ["a"], "examples": [{"text": "x", "labels": ["a"]}]},
            "UNSUPPORTED_FIELD",
            "/tasks/t/examples",
        ),
    ],
)
def test_invalid_tasks_are_rejected_with_a_path(task, code, path):
    error = error_of(make_engine(), {"model": MODEL, "input": "x", "tasks": {"t": task}})

    assert (error.code, error.path) == (code, path)


def test_input_longer_than_the_model_window_is_rejected():
    error = error_of(
        make_engine(max_tokens=10),
        {"model": MODEL, "input": "word " * 20, "tasks": {"t": {"type": "classify", "labels": ["a"]}}},
    )

    assert (error.code, error.status, error.path, error.details["task"]) == ("INPUT_TOO_LONG", 413, "/input", "t")


def test_structured_input_is_rendered_as_one_text():
    data = {"premise": "Alice works at Acme.", "hypothesis": "Alice has a job."}
    _, model = process(make_engine(), data, {"t": {"type": "classify", "labels": ["entailment", "neutral"]}})

    assert model.calls[0][0]["text"] == "# premise\nAlice works at Acme.\n\n# hypothesis\nAlice has a job."


# -- examples --------------------------------------------------------------------------


def test_examples_reach_the_model_with_label_texts():
    tasks = {
        "c": {
            "type": "classify",
            "labels": {"fr": {"title": "feature request"}, "bug": {"title": "bug report"}},
            "examples": [{"text": "Add PDF export.", "labels": ["fr"]}, {"text": "Save fails.", "labels": "bug"}],
        },
        "b": {"type": "boolean", "instruction": "reports a bug", "examples": [{"text": "x", "labels": ["false"]}]},
    }
    _, model = process(make_engine(examples=True), TEXT, tasks)

    assert model.calls[0][0]["examples"] == [
        {"text": "Add PDF export.", "labels": ["feature request"]},
        {"text": "Save fails.", "labels": ["bug report"]},
    ]
    assert model.calls[0][1]["examples"] == [{"text": "x", "labels": []}]


def test_example_labels_must_belong_to_the_task():
    body = {
        "model": MODEL,
        "input": "x",
        "tasks": {"t": {"type": "classify", "labels": ["a"], "examples": [{"text": "x", "labels": ["b"]}]}},
    }
    error = error_of(make_engine(examples=True), body)

    assert (error.code, error.path) == ("INVALID_REQUEST", "/tasks/t/examples/0/labels")


# -- defaults and adapters ---------------------------------------------------------------


def test_defaults_apply_to_tasks_that_do_not_set_their_own():
    engine = make_engine(resolve_adapter=known_adapters("news", "legal"))
    tasks = {
        "multi": {"type": "classify", "multi_label": True, "labels": ["refund", "maybe sales"]},
        "multi_own": {"type": "classify", "multi_label": True, "labels": ["refund", "maybe x"], "threshold": 0.5},
        "single": {"type": "classify", "labels": ["refund", "login"]},
        "legal": {"type": "boolean", "instruction": "send a refund", "adapter_id": "legal"},
        "base": {"type": "classify", "labels": ["refund"], "adapter_id": None, "instruction": None},
    }
    defaults = {"threshold": 0.7, "adapter_id": "news", "instruction": "Read the ticket."}
    response, model = process(engine, TEXT, tasks, defaults=defaults)

    rows = model.calls[0]
    assert [(row["adapter_id"], row["prompt"]) for row in rows] == [
        ("news", "Read the ticket."),
        ("news", "Read the ticket."),
        ("news", "Read the ticket."),
        ("legal", "Read the ticket."),
        (None, None),
    ]
    results = response["results"]
    assert (results["multi"]["value"], results["multi"]["threshold"]) == (["refund"], 0.7)
    assert (results["multi_own"]["value"], results["multi_own"]["threshold"]) == (["refund", "maybe x"], 0.5)
    assert results["single"]["threshold"] is None  # defaults.threshold does not apply to single-label tasks
    assert [results[name]["adapter_id"] for name in tasks] == ["news", "news", "news", "legal", None]


def test_default_multi_label_applies_to_classify_tasks_only():
    tasks = {
        "c": {"type": "classify", "labels": ["refund", "app"]},
        "b": {"type": "boolean", "instruction": "send a refund"},
    }
    response, _ = process(make_engine(), TEXT, tasks, defaults={"multi_label": True})

    assert response["results"]["c"]["value"] == ["refund", "app"]
    assert response["results"]["b"]["value"] is True


@pytest.mark.parametrize(
    ("body_extra", "task_extra", "path"),
    [
        ({}, {"adapter_id": "missing"}, "/tasks/t/adapter_id"),
        ({"defaults": {"adapter_id": "missing"}}, {}, "/defaults/adapter_id"),
        ({"options": {"adapter_id": "missing"}}, {}, "/options/adapter_id"),
    ],
)
def test_unknown_adapter_error_points_to_where_it_was_set(body_extra, task_extra, path):
    body = {
        "model": MODEL,
        "input": "x",
        "tasks": {"t": {"type": "classify", "labels": ["a"], **task_extra}},
        **body_extra,
    }
    error = error_of(make_engine(resolve_adapter=known_adapters()), body)

    assert (error.code, error.path) == ("UNKNOWN_ADAPTER", path)


@pytest.mark.parametrize(
    ("defaults", "path"),
    [
        ({"labels": ["x"]}, "/defaults/labels"),
        ({"threshold": "high"}, "/defaults/threshold"),
        ({"multi_label": 1}, "/defaults/multi_label"),
        ({"adapter_id": 5}, "/defaults/adapter_id"),
        ({"examples": "x"}, "/defaults/examples"),
        ("x", "/defaults"),
    ],
)
def test_bad_defaults_are_rejected_with_a_path(defaults, path):
    body = {"model": MODEL, "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"]}}, "defaults": defaults}

    assert error_of(make_engine(), body).path == path


# -- batch and async ------------------------------------------------------------------------


def test_batch_merges_defaults_dedupes_rows_and_reports_item_errors():
    engine = make_engine(resolve_adapter=known_adapters("news"))
    task = {"t": {"type": "classify", "multi_label": True, "labels": ["refund", "maybe x"]}}
    body = {
        "model": MODEL,
        "defaults": {"threshold": 0.7},
        "items": [
            {"id": "a", "input": TEXT, "tasks": task},
            {"id": "b", "input": TEXT, "tasks": task, "defaults": {"threshold": 0.5}},
            {"id": "c", "input": TEXT, "tasks": {"t": {"type": "json", "labels": ["x"]}}},
            {"id": "d", "input": TEXT, "tasks": task, "defaults": {"adapter_id": "news"}},
        ],
    }
    model = FakeModel()
    response = engine.process_batch(body, model.run_rows)

    assert len(model.calls[0]) == 2  # a and b share a row; d runs with its adapter
    items = response["items"]
    assert items[0]["results"]["t"]["value"] == ["refund"]
    assert items[1]["results"]["t"]["value"] == ["refund", "maybe x"]
    assert (items[2]["error"]["code"], items[2]["error"]["path"]) == ("UNSUPPORTED_TASK_TYPE", "/items/2/tasks/t/type")
    assert items[3]["results"]["t"]["adapter_id"] == "news"


def test_async_path_runs_rows_concurrently():
    model = FakeModel()
    seen = []

    async def run_row(row):
        seen.append(row["labels"])
        await asyncio.sleep(0)
        return model.run_row(row)

    body = {
        "model": MODEL,
        "input": TEXT,
        "tasks": {"a": {"type": "classify", "labels": ["refund", "x"]}, "b": {"type": "boolean", "instruction": "app"}},
        "options": {"include_usage": True},
    }
    response = asyncio.run(make_engine().process_async(body, run_row))

    assert seen == [["refund", "x"], ["app"]]
    assert response["results"]["a"]["value"] == "refund" and response["results"]["b"]["value"] is True
    assert response["usage"]["rows"] == 2


def test_runner_failure_is_an_internal_error():
    def broken(rows):
        raise RuntimeError("boom")

    with pytest.raises(APIError) as info:
        make_engine().process(
            {"model": MODEL, "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"]}}}, broken
        )

    assert info.value.code == "INTERNAL"


# -- server glue -------------------------------------------------------------------------------


def test_server_runs_rows_as_all_score_multi_label_rows_in_chunks():
    server = GLiClassServer.__new__(GLiClassServer)
    server.config = Mock(max_batch_size=2)
    calls = []

    def run_batch_internal(texts, labels, threshold, multi_label, examples, prompt, adapter_ids):
        calls.append((texts, labels, threshold, multi_label, examples, prompt, adapter_ids))
        return [[{"label": label, "score": 0.5} for label in row_labels] for row_labels in labels]

    server._run_batch_internal = run_batch_internal
    rows = [
        {"text": f"t{i}", "labels": [f"l{i}"], "prompt": "p" if i == 1 else None, "examples": None, "adapter_id": None}
        for i in range(3)
    ]
    outputs = server.run_rows(rows)

    assert [call[0] for call in calls] == [["t0", "t1"], ["t2"]]
    assert all(call[2] == PROCESS_ALL_SCORES and call[3] is True for call in calls)
    assert calls[0][5] == [None, "p"] and calls[0][4] is None
    assert outputs == [
        [{"label": "l0", "score": 0.5}],
        [{"label": "l1", "score": 0.5}],
        [{"label": "l2", "score": 0.5}],
    ]
