"""Characterization (R0): the structured log events the API emits (JSON on stdout).

Pinned per event: the set of keys (never their order), the constant values, and the privacy
rules (no input values, no API key, no client address, no query string).

* ``http_request`` (logger ``trinet_api.http``): one per request, exact key set;
* ``prediction`` (logger ``trinet_api.predictions``): required keys, ``input_sha256`` is the
  SHA-256 of the raw request body (JSON) or of the uploaded file (image), never the input;
* ``unhandled_error`` and ``rate_limited`` (logger ``trinet_api.http``);
* log lines written from the worker thread of an inference still carry the ``request_id``.

The ``logger`` field is pinned ONLY for the two literal names ``trinet_api.http`` and
``trinet_api.predictions``. The module ``__name__`` loggers (inference service, loading, app,
runtimes) change when the packages move: that is accepted, so only the ``trinet_api.`` prefix
is asserted for them.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import make_dummy_artifacts as dummy
import numpy as np
import pytest
import structlog
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from PIL import Image
from tests.conftest import API_KEY
from tests.fakes import FakeRuntime

# Update these imports if packages move (refactor R1/R2): the only place this file names them.
from trinet_api.domain.entities import ModelBundle, ModelRegistry

AUTH = {"X-API-Key": API_KEY}
PREDICT = "/v1/models/{}/predictions"
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z")
HEX32 = re.compile(r"[0-9a-f]{32}")

HTTP_REQUEST_KEYS = {
    "event",
    "level",
    "logger",
    "timestamp",
    "request_id",
    "method",
    "route",
    "model",
    "status",
    "duration_ms",
}
PREDICTION_REQUIRED_KEYS = {
    "event",
    "level",
    "logger",
    "timestamp",
    "request_id",
    "model",
    "version",
    "runtime",
    "batch_size",
    "predicted",
    "latency_ms",
    "input_sha256",
}
RATE_LIMITED_KEYS = {"event", "level", "logger", "timestamp", "request_id", "bucket", "limit"}
UNHANDLED_ERROR_KEYS = {"event", "level", "logger", "timestamp", "request_id", "exception"}


@pytest.fixture(autouse=True)
def _no_leaked_request_id() -> Iterator[None]:
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


# --------------------------------------------------------------------------- helpers


CLIENT_NOISE = ("httpx", "httpcore")  # the TestClient logs every request URL at INFO


def server_text(out: str) -> str:
    """The JSON lines of everything but the HTTP client (its logger prints raw URLs)."""
    lines = [line for line in out.splitlines() if line.startswith("{")]
    return "\n".join(
        line for line in lines if not json.loads(line).get("logger", "").startswith(CLIENT_NOISE)
    )


def records(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def events(out: str, name: str) -> list[dict[str, Any]]:
    return [r for r in records(out) if r.get("event") == name]


def golden_record(root: Path, model: str) -> dict[str, Any]:
    golden = json.loads((root / model / dummy.DUMMY_VERSION / "golden.json").read_text())
    record: dict[str, Any] = golden["cases"][0]["record"]
    return record


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (20, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def post_json(client: TestClient, model: str, body: bytes, **headers: str) -> Any:
    return client.post(
        PREDICT.format(model),
        headers={**AUTH, "Content-Type": "application/json", **headers},
        content=body,
    )


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


class ThreadSpyRuntime(FakeRuntime):
    """A runtime that remembers which thread ran ``predict`` (and fails when told to)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thread_ids: list[int] = []

    def predict(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        self.thread_ids.append(threading.get_ident())
        return super().predict(batch)


def registry_with(real: ModelRegistry, runtime: FakeRuntime) -> ModelRegistry:
    bundle = real.get("credit-default-mlp")
    return ModelRegistry({"credit-default-mlp": ModelBundle(bundle.manifest, runtime)})


# --------------------------------------------------------------------------- http_request


@pytest.mark.parametrize(
    ("method", "path", "headers", "status", "route", "model"),
    [
        ("GET", "/health/live", {}, 200, "/health/live", "unknown"),
        ("GET", "/health/ready", {}, 200, "/health/ready", "unknown"),
        ("GET", "/openapi.json", {}, 200, "/openapi.json", "unknown"),
        ("GET", "/v1/models", AUTH, 200, "/v1/models", "unknown"),
        ("GET", "/v1/models", {}, 401, "/v1/models", "unknown"),
        (
            "GET",
            "/v1/models/credit-default-mlp",
            AUTH,
            200,
            "/v1/models/{name}",
            "credit-default-mlp",
        ),
        ("GET", "/v1/models/nope", AUTH, 404, "/v1/models/{name}", "unknown"),
        ("GET", "/v1/models/Bad_Name", AUTH, 422, "/v1/models/{name}", "unknown"),
        ("GET", "/nowhere", {}, 404, "unmatched", "unknown"),
        ("DELETE", "/v1/models", AUTH, 405, "/v1/models", "unknown"),
        (
            "POST",
            "/v1/models/volatility-mlp/predictions",
            AUTH,
            415,
            "/v1/models/{name}/predictions",
            "volatility-mlp",
        ),
    ],
)
def test_http_request_has_exactly_these_keys_and_values(
    make_client: Callable[..., TestClient],
    capsys: pytest.CaptureFixture[str],
    method: str,
    path: str,
    headers: dict[str, str],
    status: int,
    route: str,
    model: str,
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()  # drop the start-up line
    response = client.request(method, path, headers=headers)
    assert response.status_code == status
    (event,) = events(server_text(capsys.readouterr().out), "http_request")
    assert set(event) == HTTP_REQUEST_KEYS
    assert event["event"] == "http_request"
    assert event["level"] == "info"
    assert event["logger"] == "trinet_api.http"
    assert (event["method"], event["route"], event["model"], event["status"]) == (
        method,
        route,
        model,
        status,
    )
    assert isinstance(event["duration_ms"], float)
    assert event["duration_ms"] >= 0
    assert round(event["duration_ms"], 2) == event["duration_ms"]
    assert TIMESTAMP.fullmatch(event["timestamp"])
    assert event["request_id"] == response.headers["x-request-id"]


def test_the_request_id_is_taken_from_a_valid_header_and_never_leaks_between_requests(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()
    client.get("/health/live", headers={"X-Request-ID": "char-request-0001"})
    client.get("/health/live", headers={"X-Request-ID": "char-request-0002"})
    client.get("/health/live", headers={"X-Request-ID": "short"})  # invalid: replaced
    client.get("/health/live", headers={"X-Request-ID": "has spaces in it!"})  # invalid: replaced
    client.get("/health/live")
    ids = [e["request_id"] for e in events(server_text(capsys.readouterr().out), "http_request")]
    assert ids[:2] == ["char-request-0001", "char-request-0002"]
    assert all(HEX32.fullmatch(i) for i in ids[2:])
    assert len(set(ids)) == 5


def test_the_access_log_never_contains_the_query_string_the_path_or_the_client_address(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()
    client.get("/v1/models?token=sekrit-query-value", headers=AUTH)
    client.get("/v1/models/attacker-supplied-name", headers=AUTH)
    out = server_text(capsys.readouterr().out)
    assert "sekrit-query-value" not in out
    assert "attacker-supplied-name" not in out  # not even the raw path of a 404
    assert "testclient" not in out  # the peer address of the TestClient


# --------------------------------------------------------------------------- prediction


def check_prediction_event(event: dict[str, Any], *, model: str, batch: int, digest: str) -> None:
    assert PREDICTION_REQUIRED_KEYS <= set(event)
    assert event["event"] == "prediction"
    assert event["level"] == "info"
    assert event["logger"] == "trinet_api.predictions"
    assert event["model"] == model
    assert event["version"] == dummy.DUMMY_VERSION
    assert event["runtime"] == "onnx"
    assert event["batch_size"] == batch
    assert isinstance(event["latency_ms"], float)
    assert event["latency_ms"] >= 0
    assert round(event["latency_ms"], 3) == event["latency_ms"]
    assert re.fullmatch(r"[0-9a-f]{64}", event["input_sha256"])
    assert event["input_sha256"] == digest
    assert TIMESTAMP.fullmatch(event["timestamp"])
    assert isinstance(event["predicted"], list)


def test_a_tabular_prediction_is_logged_with_the_hash_of_the_raw_body(
    make_client: Callable[..., TestClient],
    artifacts_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()
    record = golden_record(artifacts_root, "credit-default-mlp")
    body = json.dumps({"instances": [record, record]}).encode()
    response = post_json(
        client, "credit-default-mlp", body, **{"X-Request-ID": "char-predict-0001"}
    )
    assert response.status_code == 200
    out = server_text(capsys.readouterr().out)
    assert [r["event"] for r in records(out)] == ["prediction", "http_request"]  # in this order
    (event,) = events(out, "prediction")
    check_prediction_event(
        event,
        model="credit-default-mlp",
        batch=2,
        digest=hashlib.sha256(body).hexdigest(),
    )
    assert event["request_id"] == "char-predict-0001"
    assert len(event["predicted"]) == 2
    assert all(isinstance(label, str) for label in event["predicted"])
    # never the input: neither the values nor the field names
    for name, value in record.items():
        if len(str(value)) >= 6:
            assert str(value) not in out, (name, value)
        if "_" in name:
            assert name not in out, name


def test_a_regression_prediction_logs_no_predicted_classes(
    make_client: Callable[..., TestClient],
    artifacts_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()
    body = json.dumps({"instances": [golden_record(artifacts_root, "volatility-mlp")]}).encode()
    assert post_json(client, "volatility-mlp", body).status_code == 200
    (event,) = events(server_text(capsys.readouterr().out), "prediction")
    check_prediction_event(
        event, model="volatility-mlp", batch=1, digest=hashlib.sha256(body).hexdigest()
    )
    assert event["predicted"] == []


def test_an_image_prediction_hashes_the_uploaded_file_not_the_multipart_envelope(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()
    png = png_bytes()
    response = client.post(
        PREDICT.format("candlestick-cnn"),
        headers=AUTH,
        files={"file": ("../../etc/passwd", png, "application/octet-stream")},
    )
    assert response.status_code == 200
    out = server_text(capsys.readouterr().out)
    (event,) = events(out, "prediction")
    check_prediction_event(
        event, model="candlestick-cnn", batch=1, digest=hashlib.sha256(png).hexdigest()
    )
    assert len(event["predicted"]) == 1
    assert "passwd" not in out  # the client-supplied file name is not logged either


def test_an_invalid_input_is_never_echoed_into_the_log(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="DEBUG")
    capsys.readouterr()
    body = json.dumps({"instances": [{"limit_bal": "SENTINEL-INPUT-VALUE-7731"}]}).encode()
    assert post_json(client, "credit-default-mlp", body).status_code == 422
    out = server_text(capsys.readouterr().out)
    assert "SENTINEL-INPUT-VALUE-7731" not in out
    assert events(out, "prediction") == []  # nothing was predicted
    (access,) = events(out, "http_request")
    assert access["status"] == 422


# --------------------------------------------------------------------------- the API key


def test_the_api_key_never_reaches_the_log_under_any_outcome(
    make_client: Callable[..., TestClient],
    artifacts_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrong = "wrong-key-0123456789abcdef"
    client = make_client(log_level="DEBUG", rate_limit="2/minute")
    record = golden_record(artifacts_root, "credit-default-mlp")
    body = json.dumps({"instances": [record]}).encode()
    client.get("/v1/models", headers=AUTH)
    post_json(client, "credit-default-mlp", body)
    client.get("/v1/models", headers=AUTH)  # third hit of the key bucket: 429
    for _ in range(3):
        client.get("/v1/models", headers={"X-API-Key": wrong})  # 401, 401, 429
    out = server_text(capsys.readouterr().out)
    assert [e["bucket"] for e in events(out, "rate_limited")] == ["key", "ip"]
    digest = hashlib.sha256(API_KEY.encode()).hexdigest()
    for secret in (API_KEY, wrong, API_KEY[:12], digest, digest[:16]):
        assert secret not in out, secret


# --------------------------------------------------------------------------- rate_limited


def test_rate_limited_for_the_valid_key_names_the_bucket_kind_and_the_limit(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO", rate_limit="1/minute")
    capsys.readouterr()
    assert client.get("/v1/models", headers=AUTH).status_code == 200
    limited = client.get("/v1/models", headers=AUTH)
    assert limited.status_code == 429
    out = server_text(capsys.readouterr().out)
    (event,) = events(out, "rate_limited")
    assert set(event) == RATE_LIMITED_KEYS
    assert event["level"] == "warning"
    assert event["logger"] == "trinet_api.http"
    assert event["bucket"] == "key"
    assert event["limit"] == "1 per 1 minute"
    assert event["request_id"] == limited.headers["x-request-id"]
    assert TIMESTAMP.fullmatch(event["timestamp"])
    # the 429 is still access-logged, after the warning
    names = [r["event"] for r in records(out)]
    assert names == ["http_request", "rate_limited", "http_request"]
    assert [r["status"] for r in events(out, "http_request")] == [200, 429]


def test_rate_limited_for_anyone_else_is_an_ip_bucket_without_the_address(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO", rate_limit="1/minute")
    capsys.readouterr()
    assert (
        client.get("/v1/models", headers={"X-API-Key": "guess-0123456789abcdef"}).status_code == 401
    )
    assert client.get("/v1/models").status_code == 429  # no key at all: same per-address bucket
    (event,) = events(server_text(capsys.readouterr().out), "rate_limited")
    assert set(event) == RATE_LIMITED_KEYS
    assert event["bucket"] == "ip"
    assert event["limit"] == "1 per 1 minute"
    assert "testclient" not in json.dumps(event)


def test_probes_are_not_rate_limited_and_log_no_warning(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO", rate_limit="1/minute")
    capsys.readouterr()
    for _ in range(5):
        assert client.get("/health/live").status_code == 200
    assert events(server_text(capsys.readouterr().out), "rate_limited") == []


# --------------------------------------------------------------------------- unhandled_error


def test_unhandled_error_is_logged_with_the_traceback_before_the_access_line(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client(log_level="INFO")
    capsys.readouterr()

    def boom() -> None:
        raise RuntimeError("visible-only-in-the-log")

    client.app.add_api_route("/boom", boom)  # type: ignore[attr-defined]
    response = client.get("/boom", headers=AUTH | {"X-Request-ID": "char-boom-0001"})
    assert response.status_code == 500
    assert "visible-only-in-the-log" not in response.text
    out = server_text(capsys.readouterr().out)
    assert [r["event"] for r in records(out)] == ["unhandled_error", "http_request"]
    (event,) = events(out, "unhandled_error")
    assert set(event) == UNHANDLED_ERROR_KEYS
    assert event["level"] == "error"
    assert event["logger"] == "trinet_api.http"
    assert event["request_id"] == "char-boom-0001"
    assert "Traceback (most recent call last)" in event["exception"]
    assert "RuntimeError: visible-only-in-the-log" in event["exception"]
    (access,) = events(out, "http_request")
    assert (access["route"], access["status"], access["model"]) == ("/boom", 500, "unknown")
    assert access["request_id"] == "char-boom-0001"


# --------------------------------------------------------------------------- worker thread


def test_a_log_line_from_the_inference_worker_thread_carries_the_request_id(
    make_client: Callable[..., TestClient],
    registry: ModelRegistry,
    artifacts_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = ThreadSpyRuntime((25,), 2, error=RuntimeError("kernel panic"))
    client = make_client(registry_override=registry_with(registry, runtime), log_level="INFO")
    loop_threads: list[int] = []

    async def where() -> dict[str, int]:
        loop_threads.append(threading.get_ident())
        return {}

    client.app.add_api_route("/where", where)  # type: ignore[attr-defined]
    client.get("/where")
    capsys.readouterr()

    body = json.dumps({"instances": [golden_record(artifacts_root, "credit-default-mlp")]}).encode()
    response = post_json(client, "credit-default-mlp", body, **{"X-Request-ID": "char-worker-0001"})
    assert response.status_code == 500
    assert "kernel panic" not in response.text

    assert len(runtime.thread_ids) == 1
    assert runtime.thread_ids[0] != loop_threads[0]  # really ran in a worker thread
    out = server_text(capsys.readouterr().out)
    (failure,) = [r for r in records(out) if r["event"].startswith("runtime ")]
    assert failure["event"] == "runtime fake failed for model credit-default-mlp"
    assert failure["level"] == "error"
    assert failure["logger"].startswith("trinet_api.")  # exact module name may change: accepted
    assert failure["request_id"] == "char-worker-0001"
    assert "RuntimeError: kernel panic" in failure["exception"]
    assert events(out, "prediction") == []  # a failed inference is not a prediction
    (access,) = events(out, "http_request")
    assert (access["status"], access["request_id"]) == (500, "char-worker-0001")


def test_concurrent_worker_thread_logs_keep_their_own_request_id(
    make_client: Callable[..., TestClient],
    registry: ModelRegistry,
    artifacts_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = threading.Event()
    runtime = ThreadSpyRuntime((25,), 2, error=RuntimeError("kernel panic"), gate=gate)
    client = make_client(registry_override=registry_with(registry, runtime), log_level="INFO")
    capsys.readouterr()
    body = json.dumps({"instances": [golden_record(artifacts_root, "credit-default-mlp")]}).encode()
    ids = ["char-concurrent-0001", "char-concurrent-0002"]
    answered: dict[str, str] = {}

    def call(request_id: str) -> None:
        response = post_json(client, "credit-default-mlp", body, **{"X-Request-ID": request_id})
        answered[request_id] = response.headers["x-request-id"]

    threads = [threading.Thread(target=call, args=(i,)) for i in ids]
    try:
        for thread in threads:
            thread.start()
        wait_for(lambda: len(runtime.calls) == 2)  # both are inside the runtime at once
    finally:
        gate.set()
    for thread in threads:
        thread.join(timeout=10)
    assert answered == {i: i for i in ids}
    out = server_text(capsys.readouterr().out)
    failures = [r for r in records(out) if r["event"].startswith("runtime ")]
    assert sorted(f["request_id"] for f in failures) == ids
    assert len(set(runtime.thread_ids)) == 2  # two different worker threads
    assert sorted(a["request_id"] for a in events(out, "http_request")) == ids
