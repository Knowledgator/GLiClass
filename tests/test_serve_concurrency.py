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
