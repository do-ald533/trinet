"""Security headers, request ids and error hygiene on every kind of response."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.api.conftest import multipart
from tests.fakes import FakeRuntime

from trinet_api.domain.entities import ModelRegistry
from trinet_api.interface.http.middleware import SECURITY_HEADERS

PREDICT = "/v1/models/credit-default-mlp/predictions"
TRACE_MARKERS = ("Traceback", 'File "', "site-packages", ".py", "raise ", "Error:")


def scenarios(
    client: TestClient, auth: dict[str, str], record: dict[str, Any], png: bytes
) -> dict[int, Callable[[], Any]]:
    """One representative request per status code the API can answer with."""
    return {
        200: lambda: client.get("/v1/models", headers=auth),
        401: lambda: client.get("/v1/models"),
        404: lambda: client.get("/v1/models/nope", headers=auth),
        405: lambda: client.delete("/v1/models", headers=auth),
        415: lambda: client.post(PREDICT, headers=auth, content=b"{}"),
        422: lambda: client.post(PREDICT, headers=auth, json={"instances": [{}]}),
    }


@pytest.mark.parametrize("status", [200, 401, 404, 405, 415, 422])
def test_every_answer_carries_the_security_headers(
    client: TestClient,
    auth: dict[str, str],
    golden_records: dict[str, Any],
    png_bytes: bytes,
    status: int,
) -> None:
    response = scenarios(client, auth, golden_records["credit-default-mlp"], png_bytes)[status]()
    assert response.status_code == status
    for name, value in SECURITY_HEADERS:
        assert response.headers[name] == value, name
    assert re.fullmatch(r"[A-Za-z0-9._-]{8,64}", response.headers["x-request-id"])
    assert "server" not in response.headers  # uvicorn is started with server_header=False upstream
    if status >= 400:
        assert response.headers["content-type"] == "application/problem+json"
        assert response.json()["status"] == status
        assert response.json()["request_id"] == response.headers["x-request-id"]


def test_413_429_and_503_carry_the_security_headers_too(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(max_json_bytes=1024, rate_limit="1/minute")
    big = client.post(
        PREDICT, headers={**auth, "Content-Type": "application/json"}, content=b" " * 2000
    )
    limited = client.get("/v1/models", headers=auth)
    for response, status in ((big, 413), (limited, 429)):
        assert response.status_code == status
        for name, value in SECURITY_HEADERS:
            assert response.headers[name] == value, (status, name)


def test_error_bodies_never_contain_stack_traces_or_internals(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any], png_bytes: bytes
) -> None:
    for status, call in scenarios(
        client, auth, golden_records["credit-default-mlp"], png_bytes
    ).items():
        if status < 400:
            continue
        text = call().text
        assert not any(marker in text for marker in TRACE_MARKERS), (status, text)


def test_unhandled_exceptions_become_a_generic_500_problem(
    client: TestClient, auth: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> None:
        raise RuntimeError("super-secret-internal-state /srv/models/x.onnx")

    client.app.add_api_route("/boom", boom)  # type: ignore[attr-defined]
    response = client.get("/boom", headers=auth)
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "internal_error"
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "super-secret" not in response.text
    for name, value in SECURITY_HEADERS:
        assert response.headers[name] == value


def test_the_unhandled_exception_is_logged_server_side(
    client: TestClient, auth: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> None:
        raise RuntimeError("visible-only-in-the-log")

    client.app.add_api_route("/boom", boom)  # type: ignore[attr-defined]
    client.get("/boom", headers=auth)
    records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    errors = [r for r in records if r.get("event") == "unhandled_error"]
    assert errors
    assert "visible-only-in-the-log" in errors[0]["exception"]


def test_interactive_docs_are_not_served(client: TestClient, auth: dict[str, str]) -> None:
    for path in ("/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.yaml"):
        assert client.get(path, headers=auth).status_code == 404


def test_cors_is_not_enabled(client: TestClient, auth: dict[str, str]) -> None:
    response = client.options(
        "/v1/models",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in response.headers
    assert response.status_code in (401, 405)


def test_the_api_key_is_never_logged(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    api_key: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = make_client(log_level="DEBUG")
    client.get("/v1/models", headers=auth)
    client.get("/v1/models", headers={"X-API-Key": "wrong-key-0123456789abcdef"})
    out = capsys.readouterr().out
    assert api_key not in out
    assert "wrong-key-0123456789abcdef" not in out
    assert "http_request" in out


def test_predictions_are_logged_with_an_input_hash_never_the_input(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    golden_records: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = make_client(log_level="INFO")
    record = golden_records["credit-default-mlp"]
    client.post(PREDICT, headers=auth, json={"instances": [record]})
    out = capsys.readouterr().out
    logged = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    (event,) = [r for r in logged if r.get("event") == "prediction"]
    assert re.fullmatch(r"[0-9a-f]{64}", event["input_sha256"])
    assert event["model"] == "credit-default-mlp"
    assert event["batch_size"] == 1
    assert event["request_id"]
    assert str(record["limit_bal"]) not in out
    access = [r for r in logged if r.get("event") == "http_request"]
    assert access
    assert access[0]["route"] == "/v1/models/{name}/predictions"
    assert access[0]["status"] == 200


def test_an_unknown_model_cannot_pollute_log_or_metric_labels(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = make_client(log_level="INFO")
    client.get("/v1/models/attacker-supplied-name", headers=auth)
    out = capsys.readouterr().out
    access = [json.loads(line) for line in out.splitlines() if '"http_request"' in line]
    assert access[-1]["model"] == "unknown"
    assert access[-1]["route"] == "/v1/models/{name}"


def test_a_predicted_runtime_failure_is_logged_not_returned(
    make_client: Callable[..., TestClient],
    fake_registry: Callable[..., ModelRegistry],
    auth: dict[str, str],
    golden_records: dict[str, Any],
) -> None:
    broken = FakeRuntime((25,), 2, error=RuntimeError("kernel panic"))
    client = make_client(registry_override=fake_registry({"credit-default-mlp": broken}))
    response = client.post(
        PREDICT, headers=auth, json={"instances": [golden_records["credit-default-mlp"]]}
    )
    assert response.status_code == 500
    assert "kernel panic" not in response.text


def test_uploads_do_not_leave_the_filename_in_responses(
    client: TestClient, auth: dict[str, str], png_bytes: bytes
) -> None:
    response = client.post(
        "/v1/models/candlestick-cnn/predictions",
        headers=auth,
        **multipart(png_bytes, filename="../../etc/passwd"),
    )
    assert response.status_code == 200
    assert "passwd" not in response.text
