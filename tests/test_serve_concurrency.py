"""Concurrent traffic on GLiClass serving (``POST /gliclass``) behind Ray Serve.

Threads send a mix of requests at the same time: single texts and text lists
(longer than the batch size), shared and per-text labels, single- and
multi-label, thresholds, prompts, few-shot examples, hierarchical labels, and
invalid requests. Every response must match the same request sent alone,
errors must stay with the request that caused them, and the GPU must run the
rows in batches no larger than ``max_batch_size``.

Needs Ray, a CUDA GPU and ``knowledgator/gliclass-small-v1.0-init``; takes about a minute.
"""

import json
import random
import socket
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MODEL = "knowledgator/gliclass-small-v1.0-init"
MAX_BATCH = 8
LABELS = ["technology", "sports", "finance", "politics", "health"]
TEXTS = [
    "The new smartphone has a faster processor and a much better camera.",
    "The striker scored twice as the home team won the championship final.",
    "The central bank raised interest rates again to slow down inflation.",
    "Doctors recommend regular exercise and a balanced diet to prevent heart disease.",
    "Parliament passed the new election law after a long debate.",
    "The startup raised fifty million dollars to build faster chips.",
]
EXAMPLES = [
    {"text": "The court ruled on the merger of the two banks.", "labels": ["finance", "politics"]},
    {"text": "The team signed a new goalkeeper.", "labels": ["sports"]},
]
# Rows batched with other rows are padded differently, which moves scores slightly.
SCORE_TOL = 0.02

_SERVED = {}


@pytest.fixture(scope="module")
def server_url():
    ray = pytest.importorskip("ray")
    pytest.importorskip("ray.serve")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL)
    except Exception as exc:
        pytest.skip(f"cannot download {MODEL}: {exc}")

    from gliclass.serve import GLiClassServeConfig, shutdown, serve_gliclass

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Replicas must import this checkout of gliclass.
    repo_root = str(Path(__file__).resolve().parents[1])
    ray.init(num_cpus=4, include_dashboard=False, runtime_env={"env_vars": {"PYTHONPATH": repo_root}})
    try:
        _SERVED["handle"] = serve_gliclass(
            GLiClassServeConfig(
                model=MODEL,
                dtype="float32",
                max_model_len=512,
                max_batch_size=MAX_BATCH,
                precompiled_batch_sizes=[1, 2, 4, MAX_BATCH],
                batch_wait_timeout_ms=10.0,
                http_port=port,
            )
        )
        yield f"http://127.0.0.1:{port}/gliclass"
    finally:
        shutdown()
        ray.shutdown()


def _post(url, body=None, *, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"error": body.decode(errors="replace")[:200]}


def _send(url, item):
    if isinstance(item, bytes):
        return _post(url, raw=item)
    return _post(url, item)


def _valid_workload():
    work = []
    for index, text in enumerate(TEXTS):
        threshold = 0.2 + 0.1 * (index % 3)
        work += [
            {"text": text, "labels": LABELS, "threshold": threshold},
            {"text": text, "labels": LABELS, "multi_label": False},
            {"text": text, "labels": LABELS[: 2 + index % 4], "threshold": 0.3},
            {"text": text, "labels": LABELS, "prompt": "Classify the topic of this news sentence:"},
            {"text": text, "labels": LABELS, "examples": EXAMPLES, "threshold": 0.3},
            {"text": text, "labels": {"topic": ["technology", "sports"], "domain": ["finance", "health"]}},
        ]
    work += [
        {"texts": TEXTS * 2, "labels": LABELS, "threshold": 0.3},  # 12 texts: more than one batch
        {"texts": TEXTS, "labels": [LABELS[: 2 + i % 4] for i in range(len(TEXTS))], "multi_label": False},
        {"texts": TEXTS[:3], "labels": LABELS, "prompt": ["Topic:", "Subject:", "Category:"]},
        {"texts": TEXTS[:2], "labels": [{"topic": ["technology", "sports"]}, ["finance", "health"]]},
        {"texts": TEXTS[:2], "labels": LABELS, "examples": [EXAMPLES, None]},
        {"text": TEXTS[0], "labels": []},  # nothing to score: no predictions
    ]
    return work


def _invalid_workload():
    return [
        ({"text": TEXTS[0]}, 400),  # no labels
        ({"text": TEXTS[0], "labels": LABELS, "adapter_id": "missing"}, 404),
        ({"texts": TEXTS[:2], "labels": [LABELS]}, 400),  # per-text labels do not match the texts
        ({"texts": TEXTS[:2], "labels": LABELS, "prompt": ["only one"]}, 400),
        ({"text": TEXTS[0], "labels": {"topic": [1]}}, 400),
        ({"text": TEXTS[0], "labels": LABELS, "multi_label": "yes"}, 400),
        ({"text": TEXTS[0], "labels": LABELS, "examples": [{"text": "x", "labels": [1]}]}, 400),
        ({"text": TEXTS[0], "labels": LABELS, "examples": "x"}, 400),
        ({"texts": [TEXTS[0], 7], "labels": LABELS}, 400),
        (b"[1, 2]", 400),
        ({"text": 12345, "labels": LABELS}, 400),
        ({"text": TEXTS[0], "labels": LABELS, "threshold": "high"}, 400),
        ({"text": TEXTS[0], "labels": [1, 2]}, 400),
        (b"{not json", 400),
    ]


def _predictions_equal(got, ref, threshold, single_label):
    """Same predicted labels, scores within tolerance; labels may differ only near the cutoff."""
    got = {p["label"]: p["score"] for p in got}
    ref = {p["label"]: p["score"] for p in ref}
    for label in got.keys() | ref.keys():
        if label in got and label in ref:
            if abs(got[label] - ref[label]) > SCORE_TOL:
                return False
        elif single_label:
            return False  # the top label changed
        elif abs(got.get(label, ref.get(label)) - threshold) > SCORE_TOL:
            return False
    return True


def _responses_equal(item, got, ref):
    (got_status, got_body), (ref_status, ref_body) = got, ref
    if got_status != ref_status:
        return False
    if ref_status != 200:
        return True
    threshold = item.get("threshold", 0.5)
    single_label = item.get("multi_label") is False
    if "texts" in item:
        return len(got_body) == len(ref_body) and all(
            _predictions_equal(g, r, threshold, single_label) for g, r in zip(got_body, ref_body, strict=True)
        )
    return _predictions_equal(got_body, ref_body, threshold, single_label)


def test_requests_alone_give_sensible_results(server_url):
    status, result = _send(server_url, {"text": TEXTS[1], "labels": LABELS, "multi_label": False})
    assert status == 200
    assert result[0]["label"] == "sports"
    status, results = _send(server_url, {"texts": TEXTS[:4], "labels": LABELS, "threshold": 0.0})
    assert status == 200 and len(results) == 4
    assert [max(r, key=lambda p: p["score"])["label"] for r in results] == ["technology", "sports", "finance", "health"]


def test_invalid_requests_are_rejected_on_their_own(server_url):
    for item, status in _invalid_workload():
        got_status, body = _send(server_url, item)
        assert got_status == status, (item, got_status, body)


def test_a_bad_request_does_not_fail_requests_batched_with_it(server_url):
    good = {"text": TEXTS[0], "labels": LABELS}
    for item, status in _invalid_workload():
        with ThreadPoolExecutor(8) as pool:
            bad = pool.submit(_send, server_url, item)
            neighbours = [pool.submit(_send, server_url, good) for _ in range(7)]
        assert bad.result()[0] == status, item
        assert [future.result()[0] for future in neighbours] == [200] * 7, item


def test_hierarchical_labels_are_flattened_to_dotted_labels(server_url):
    status, result = _send(
        server_url, {"text": TEXTS[1], "labels": {"topic": ["sports", "finance"]}, "multi_label": False}
    )
    assert (status, result[0]["label"]) == (200, "topic.sports")


def test_concurrent_mixed_traffic_matches_requests_sent_alone(server_url):
    valid = _valid_workload()
    invalid = _invalid_workload()
    alone = [_send(server_url, item) for item in valid]
    assert all(status == 200 for status, _ in alone), [r for r in alone if r[0] != 200][:3]

    jobs = [("valid", index) for index in range(len(valid)) for _ in range(3)]
    jobs += [("invalid", index) for index in range(len(invalid)) for _ in range(3)]
    random.Random(0).shuffle(jobs)

    def run(job):
        kind, index = job
        return _send(server_url, valid[index] if kind == "valid" else invalid[index][0])

    with ThreadPoolExecutor(48) as pool:
        results = list(pool.map(run, jobs))

    mismatches = []
    for (kind, index), got in zip(jobs, results, strict=True):
        if kind == "valid" and not _responses_equal(valid[index], got, alone[index]):
            mismatches.append((json.dumps(valid[index])[:100], got[0], str(got[1])[:200]))
        if kind == "invalid" and got[0] != invalid[index][1]:
            mismatches.append((str(invalid[index][0])[:100], got[0], str(got[1])[:200]))
    assert not mismatches, mismatches[:5]


def test_large_requests_run_in_gpu_batches_of_bounded_size(server_url):
    """Requests with more texts than the batch size are split; texts of all requests share capped batches."""
    handle = _SERVED["handle"]
    work = [{"texts": TEXTS * 3, "labels": LABELS[: 2 + i % 4], "threshold": 0.3} for i in range(6)]  # 18 texts each
    work += [{"text": text, "labels": LABELS} for text in TEXTS]
    alone = [_send(server_url, item) for item in work]

    handle.batch_stats.remote(reset=True).result()
    with ThreadPoolExecutor(len(work)) as pool:
        concurrent = list(pool.map(lambda item: _send(server_url, item), work))
    stats = handle.batch_stats.remote().result()

    for item, got, ref in zip(work, concurrent, alone, strict=True):
        assert _responses_equal(item, got, ref), item
    assert stats["rows"] == 6 * 18 + len(TEXTS)
    assert stats["max_rows"] <= MAX_BATCH
    unmerged_passes = 6 * -(-18 // MAX_BATCH) + len(TEXTS)
    assert stats["forward_passes"] < unmerged_passes  # texts of different requests merged


def test_concurrency_batches_requests(server_url):
    """Many concurrent requests finish faster than the same requests sent one at a time."""
    import time

    work = [{"text": text, "labels": LABELS, "threshold": 0.3} for text in TEXTS * 10]

    start = time.perf_counter()
    for item in work:
        _send(server_url, item)
    sequential = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(48) as pool:
        statuses = [status for status, _ in pool.map(lambda item: _send(server_url, item), work)]
    concurrent = time.perf_counter() - start

    assert statuses == [200] * len(work)
    assert concurrent < sequential, (concurrent, sequential)


# -- process API (/v1/process) ----------------------------------------------------------------


def _v1(url, suffix):
    return url + "/v1" + suffix


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _process_tasks(index):
    return {
        "topic": {"type": "classify", "labels": LABELS},
        "topics": {"type": "classify", "labels": LABELS, "multi_label": True, "threshold": 0.3 + 0.1 * (index % 3)},
        "is_sport": {"type": "boolean", "instruction": "is about sports"},
        "money": {"type": "boolean", "labels": {"true": {"title": "mentions money"}}, "threshold": 0.6},
        "tone": {"type": "score", "labels": ["calm", "neutral", "excited"], "order": ["calm", "neutral", "excited"]},
        "rating": {"type": "score", "scale": {"min": 1, "max": 5}},
    }


def _process_workload():
    work = []
    for index, text in enumerate(TEXTS):
        work.append(("process", {"model": MODEL, "input": text, "tasks": _process_tasks(index)}))
        work.append(
            (
                "process",
                {
                    "model": MODEL,
                    "input": {"headline": text, "section": "news"},
                    "defaults": {"instruction": "Classify the news item:", "threshold": 0.4},
                    "tasks": {
                        "topics": {"type": "classify", "labels": LABELS, "multi_label": True},
                        "own": {"type": "classify", "labels": LABELS, "multi_label": True, "threshold": 0.6},
                        "plain": {"type": "classify", "labels": LABELS, "instruction": None},
                    },
                },
            )
        )
    multi = {"t": {"type": "classify", "labels": LABELS, "multi_label": True}}
    batch = {
        "model": MODEL,
        "defaults": {"threshold": 0.3},
        "items": [
            {"id": "a", "input": TEXTS[0], "tasks": multi},
            {"id": "b", "input": TEXTS[1], "tasks": multi, "defaults": {"threshold": 0.6}},
            {"id": "c", "input": TEXTS[2], "tasks": {"t": {"type": "span", "labels": ["x"]}}},
        ],
    }
    work.append(("batch", batch))
    invalid = [
        {"model": MODEL, "input": "x", "tasks": {"t": {"type": "span", "labels": ["a"]}}},
        {"model": MODEL, "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"], "adapter_id": "missing"}}},
        {"model": MODEL, "input": "x", "tasks": {"t": {"type": "boolean"}}},
        {"model": MODEL, "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"]}}, "defaults": {"x": 1}},
        {"model": "nope", "input": "x", "tasks": {"t": {"type": "classify", "labels": ["a"]}}},
    ]
    work += [("process", body) for body in invalid]
    work.append(("process-raw", b"{not json"))
    return work


def _send_process(url, item):
    kind, body = item
    if kind == "process":
        return _post(_v1(url, "/process"), body)
    if kind == "batch":
        return _post(_v1(url, "/process/batch"), body)
    return _post(_v1(url, "/process"), raw=body)


def _close(a, b):
    return abs(a - b) <= SCORE_TOL


def _task_results_equal(got, ref):
    """Same values with scores within tolerance; values may differ only on near-ties or near the cutoff."""
    if got["type"] != ref["type"] or got.get("threshold") != ref.get("threshold"):
        return False
    if "scores" in ref:
        if got["scores"].keys() != ref["scores"].keys():
            return False
        if not all(_close(got["scores"][k], ref["scores"][k]) for k in ref["scores"]):
            return False
    if got["value"] == ref["value"]:
        return True
    scores = ref.get("scores", {})
    ranked = sorted(scores.values(), reverse=True)
    near_tie = len(ranked) > 1 and _close(ranked[0], ranked[1])
    threshold = ref.get("threshold")
    near_cutoff = threshold is not None and any(_close(score, threshold) for score in scores.values())
    near_half = ref["type"] == "boolean" and _close(ref["truth_score"], 0.5)
    return near_tie or near_cutoff or near_half or (ref["type"] == "score" and "scale" in ref)


def _process_responses_equal(item, got, ref):
    (got_status, got_body), (ref_status, ref_body) = got, ref
    if got_status != ref_status:
        return False
    if ref_status != 200:
        return got_body == ref_body
    if item[0] == "batch":
        return all(
            g.get("id") == r.get("id")
            and (
                g == r
                if "error" in r
                else all(_task_results_equal(g["results"][n], r["results"][n]) for n in r["results"])
            )
            for g, r in zip(got_body["items"], ref_body["items"], strict=True)
        )
    return got_body["results"].keys() == ref_body["results"].keys() and all(
        _task_results_equal(got_body["results"][name], ref_body["results"][name]) for name in ref_body["results"]
    )


def test_process_results_are_shaped_by_task_type(server_url):
    status, body = _post(_v1(server_url, "/process"), {"model": MODEL, "input": TEXTS[1], "tasks": _process_tasks(0)})
    assert status == 200
    results = body["results"]
    assert results["topic"]["value"] in LABELS and results["topic"]["threshold"] is None
    assert set(results["topic"]["scores"]) == set(LABELS)
    assert isinstance(results["topics"]["value"], list)
    assert isinstance(results["is_sport"]["value"], bool) and 0 <= results["is_sport"]["truth_score"] <= 1
    assert results["tone"]["value"] in ("calm", "neutral", "excited")
    assert results["rating"]["value"] in (1, 2, 3, 4, 5) and results["rating"]["scale"] == {"min": 1, "max": 5}


def test_process_and_legacy_traffic_together_match_requests_sent_alone(server_url):
    process_work = _process_workload()
    legacy_work = _valid_workload()
    process_alone = [_send_process(server_url, item) for item in process_work]
    legacy_alone = [_send(server_url, item) for item in legacy_work]

    jobs = [("process", i) for i in range(len(process_work)) for _ in range(3)]
    jobs += [("legacy", i) for i in range(len(legacy_work)) for _ in range(2)]
    random.Random(1).shuffle(jobs)

    def run(job):
        kind, index = job
        if kind == "process":
            return _send_process(server_url, process_work[index])
        return _send(server_url, legacy_work[index])

    with ThreadPoolExecutor(48) as pool:
        results = list(pool.map(run, jobs))

    mismatches = []
    for (kind, index), got in zip(jobs, results, strict=True):
        if kind == "process" and not _process_responses_equal(process_work[index], got, process_alone[index]):
            mismatches.append(("process", json.dumps(process_work[index][1])[:100], got[0], str(got[1])[:300]))
        if kind == "legacy" and not _responses_equal(legacy_work[index], got, legacy_alone[index]):
            mismatches.append(("legacy", json.dumps(legacy_work[index])[:100], got[0], str(got[1])[:200]))
    assert not mismatches, mismatches[:5]
    statuses = {status for status, _ in process_alone}
    assert {200, 404, 422} <= statuses


def test_process_and_legacy_agree_on_the_same_question(server_url):
    """Multi-label scores are the same sigmoid scores on both endpoints; single-label picks the same top label."""
    for text in TEXTS:
        legacy_multi = _send(server_url, {"text": text, "labels": LABELS, "threshold": 0.0, "prompt": "Topic:"})[1]
        legacy_single = _send(server_url, {"text": text, "labels": LABELS, "multi_label": False})[1]
        tasks = {
            "multi": {
                "type": "classify",
                "labels": LABELS,
                "multi_label": True,
                "threshold": 0.0,
                "instruction": "Topic:",
            },
            "single": {"type": "classify", "labels": LABELS},
        }
        results = _post(_v1(server_url, "/process"), {"model": MODEL, "input": text, "tasks": tasks})[1]["results"]
        legacy_scores = {p["label"]: p["score"] for p in legacy_multi}
        assert all(_close(results["multi"]["scores"][label], legacy_scores[label]) for label in LABELS)
        assert results["single"]["value"] == legacy_single[0]["label"]


def test_many_task_process_requests_run_in_gpu_batches_of_bounded_size(server_url):
    handle = _SERVED["handle"]
    work = [
        {
            "model": MODEL,
            "input": text,
            "tasks": {
                f"t{i}": {"type": "classify", "labels": LABELS[: 1 + i % 5], "instruction": f"Task {i}:"}
                for i in range(20)
            },
        }
        for text in TEXTS
    ]
    alone = [_post(_v1(server_url, "/process"), body) for body in work]

    handle.batch_stats.remote(reset=True).result()
    with ThreadPoolExecutor(len(work)) as pool:
        concurrent = list(pool.map(lambda body: _post(_v1(server_url, "/process"), body), work))
    stats = handle.batch_stats.remote().result()

    for body, got, ref in zip(work, concurrent, alone, strict=True):
        assert _process_responses_equal(("process", body), got, ref)
    assert stats["rows"] == len(TEXTS) * 20
    assert stats["max_rows"] <= MAX_BATCH
    assert stats["forward_passes"] < len(TEXTS) * -(-20 // MAX_BATCH)  # rows of different requests merged


def test_v1_info_routes_and_http_errors(server_url):
    status, models = _get(_v1(server_url, "/models"))
    assert (status, models["data"][0]["id"]) == (200, MODEL)
    status, capabilities = _get(_v1(server_url, f"/models/{MODEL}/capabilities"))
    assert status == 200 and capabilities["task_types"] == ["classify", "boolean", "score"]
    assert capabilities["extensions"]["examples"] in (True, False)
    assert _get(_v1(server_url, "/health"))[1]["status"] == "ok"
    assert _get(_v1(server_url, "/ready"))[0] == 200
    assert _get(_v1(server_url, "/models/nope/capabilities"))[1]["error"]["code"] == "MODEL_NOT_FOUND"
    assert _get(_v1(server_url, "/nope"))[0] == 404
    assert _get(_v1(server_url, "/process"))[0] == 422  # GET on a POST route
    status, body = _post(_v1(server_url, "/process"), raw=b"x" * 10_000_001)
    assert (status, body["error"]["code"]) == (413, "PAYLOAD_TOO_LARGE")
