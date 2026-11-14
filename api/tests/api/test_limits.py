"""Rate limiting, concurrency (backpressure) and request-size limits."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.api.conftest import multipart
from tests.fakes import FakeRuntime

from trinet_api.domain.entities import ModelRegistry

PREDICT = "/v1/models/{}/predictions"


# --------------------------------------------------------------------------- 429


def test_the_key_bucket_is_exhausted_and_answers_429_with_retry_after(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="3/minute")
    assert [client.get("/v1/models", headers=auth).status_code for _ in range(3)] == [200] * 3
    response = client.get("/v1/models", headers=auth)
    assert response.status_code == 429
    assert response.headers["content-type"] == "application/problem+json"
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert response.json()["code"] == "rate_limited"


def test_a_flood_of_bad_keys_cannot_starve_the_valid_key(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="3/minute")
    statuses = [
        client.get("/v1/models", headers={"X-API-Key": f"guess-{i:016d}"}).status_code
        for i in range(6)
    ]
    assert statuses == [401, 401, 401, 429, 429, 429]  # attackers share one per-IP bucket
    assert client.get("/v1/models", headers=auth).status_code == 200  # the real key has its own


def test_rate_limited_responses_still_carry_the_security_headers(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="1/minute")
    client.get("/v1/models", headers=auth)
    response = client.get("/v1/models", headers=auth)
    assert response.status_code == 429
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]


def test_probes_are_exempt_from_the_rate_limit(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="1/minute")
    assert {client.get("/health/live").status_code for _ in range(10)} == {200}
    assert {client.get("/health/ready").status_code for _ in range(10)} == {200}
    assert client.get("/v1/models", headers=auth).status_code == 200  # nothing was consumed


def test_the_limit_applies_to_predictions_too(
    make_client: Callable[..., TestClient], auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    client = make_client(rate_limit="2/minute")
    body = {"instances": [golden_records["credit-default-mlp"]]}
    codes = [
        client.post(PREDICT.format("credit-default-mlp"), headers=auth, json=body).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


# --------------------------------------------------------------------------- 503


@pytest.fixture
def gated(
    fake_registry: Callable[..., ModelRegistry],
) -> Iterator[tuple[ModelRegistry, FakeRuntime, threading.Event]]:
    gate = threading.Event()
    runtime = FakeRuntime((25,), 2, gate=gate)
    yield fake_registry({"credit-default-mlp": runtime}), runtime, gate
    gate.set()  # never leave a worker thread blocked


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.005)


def test_a_saturated_service_sheds_load_with_503_and_retry_after(
    make_client: Callable[..., TestClient],
    gated: tuple[ModelRegistry, FakeRuntime, threading.Event],
    auth: dict[str, str],
    golden_records: dict[str, Any],
) -> None:
    registry, runtime, gate = gated
    client = make_client(registry_override=registry, max_concurrency=1, queue_timeout_s=0.1)
    body = {"instances": [golden_records["credit-default-mlp"]]}
    first: list[int] = []
    worker = threading.Thread(
        target=lambda: first.append(
            client.post(PREDICT.format("credit-default-mlp"), headers=auth, json=body).status_code
        )
    )
    worker.start()
    wait_for(lambda: len(runtime.calls) == 1)  # the only slot is now busy

    shed = client.post(PREDICT.format("credit-default-mlp"), headers=auth, json=body)
    assert shed.status_code == 503
    assert shed.json()["code"] == "overloaded"
    assert shed.headers["content-type"] == "application/problem+json"
    assert int(shed.headers["retry-after"]) >= 1
    assert client.get("/health/live").status_code == 200  # the process itself is fine
    assert client.get("/v1/models", headers=auth).status_code == 200  # only inference is shed

    gate.set()
    worker.join(timeout=5)
    assert first == [200]
    assert (
        client.post(PREDICT.format("credit-default-mlp"), headers=auth, json=body).status_code
        == 200
    )


def test_requests_wait_in_the_queue_when_a_slot_frees_up_in_time(
    make_client: Callable[..., TestClient],
    gated: tuple[ModelRegistry, FakeRuntime, threading.Event],
    auth: dict[str, str],
    golden_records: dict[str, Any],
) -> None:
    registry, runtime, gate = gated
    client = make_client(registry_override=registry, max_concurrency=1, queue_timeout_s=5.0)
    body = {"instances": [golden_records["credit-default-mlp"]]}
    results: list[int] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                client.post(
                    PREDICT.format("credit-default-mlp"), headers=auth, json=body
                ).status_code
            )
        )
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    wait_for(lambda: len(runtime.calls) >= 1)
    gate.set()
    for thread in threads:
        thread.join(timeout=10)
    assert results == [200, 200, 200]


# --------------------------------------------------------------------------- 413


def test_oversized_json_is_413_before_it_is_parsed(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(max_json_bytes=2048)
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "application/json"},
        content=b'{"instances": [' + b"1," * 5000 + b"1]}",
    )
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_lying_content_length_cannot_smuggle_a_big_body(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(max_json_bytes=2048)

    def chunks() -> Iterator[bytes]:  # chunked upload: no Content-Length to check up front
        for _ in range(20):
            yield b" " * 1000

    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "application/json"},
        content=chunks(),
    )
    assert response.status_code == 413


def test_oversized_multipart_is_413(
    make_client: Callable[..., TestClient], auth: dict[str, str], png_bytes: bytes
) -> None:
    client = make_client(max_upload_bytes=1024)
    response = client.post(
        PREDICT.format("candlestick-cnn"), headers=auth, **multipart(png_bytes + b"\x00" * 4000)
    )
    assert response.status_code == 413


def test_bodies_at_the_limit_pass(
    make_client: Callable[..., TestClient], auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    body = json.dumps({"instances": [golden_records["credit-default-mlp"]]}).encode()
    client = make_client(max_json_bytes=max(len(body), 1024))
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "application/json"},
        content=body,
    )
    assert response.status_code == 200


def test_get_requests_are_not_subject_to_the_body_limit(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(max_json_bytes=1024)
    assert client.get("/v1/models", headers=auth).status_code == 200
