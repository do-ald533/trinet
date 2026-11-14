from __future__ import annotations

import socket
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.conftest import make_settings

from trinet_api.domain.entities import ModelRegistry
from trinet_api.interface.http.app import create_app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        assert response.status == 200
        body: bytes = response.read()
        return body.decode()


# --------------------------------------------------------------------------- health


def test_live_is_always_ok(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_lists_every_model(client: TestClient) -> None:
    body = client.get("/health/ready").json()
    assert body == {
        "status": "ok",
        "models": {"credit-default-mlp": True, "volatility-mlp": True, "candlestick-cnn": True},
    }


def test_ready_is_degraded_but_200_while_at_least_one_model_serves(
    make_client: Callable[..., TestClient], fake_registry: Callable[..., ModelRegistry]
) -> None:
    response = make_client(registry_override=fake_registry()).get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["models"]["broken-model"] is False
    assert body["models"]["credit-default-mlp"] is True


def test_ready_is_503_when_nothing_can_be_served(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(registry_override=ModelRegistry({}))
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "not_ready"
    assert client.get("/health/live").status_code == 200  # alive, just not ready


def test_a_missing_artifacts_directory_starts_alive_but_not_ready(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "missing"))
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get(
            "/v1/models", headers={"X-API-Key": "test-api-key-0123456789abcdef"}
        ).json() == {"models": []}


# --------------------------------------------------------------------------- metrics


def test_the_metrics_server_exposes_red_and_prediction_metrics(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    golden_records: dict[str, Any],
) -> None:
    port = free_port()
    client = make_client(metrics_enabled=True, metrics_host="127.0.0.1", metrics_port=port)
    client.post(
        "/v1/models/credit-default-mlp/predictions",
        headers=auth,
        json={"instances": [golden_records["credit-default-mlp"]] * 2},
    )
    client.post("/v1/models/credit-default-mlp/predictions", headers=auth, json={"instances": [{}]})
    client.get("/v1/models/not-a-model-xyz", headers=auth)
    client.get("/v1/models")  # 401
    text = scrape(port)
    route = 'route="/v1/models/{name}/predictions"'
    counter = 'trinet_http_requests_total{method="POST",model="credit-default-mlp",'
    assert f'{counter}{route},status="200"}} 1.0' in text
    assert f'{counter}{route},status="422"}} 1.0' in text
    assert 'model="unknown"' in text
    assert "not-a-model-xyz" not in text  # unbounded labels would be a cardinality bomb
    assert 'status="401"' in text
    assert "trinet_http_request_duration_seconds_bucket" in text
    assert (
        'trinet_inference_duration_seconds_count{model="credit-default-mlp",runtime="onnx"} 1.0'
        in text
    )
    predicted = [
        line for line in text.splitlines() if line.startswith("trinet_predicted_class_total{")
    ]
    assert sum(float(line.rsplit(" ", 1)[1]) for line in predicted) == 2.0
    info = 'trinet_model_info{model="candlestick-cnn",runtime="onnx",status="dummy",'
    assert f'{info}version="0.0.0-dummy"}} 1.0' in text
    assert "trinet_inflight_inferences 0.0" in text


def test_metrics_are_not_served_on_the_public_port(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert client.get("/metrics", headers=auth).status_code == 404


def test_the_metrics_server_stops_with_the_app(make_client: Callable[..., TestClient]) -> None:
    port = free_port()
    client = make_client(metrics_enabled=True, metrics_host="127.0.0.1", metrics_port=port)
    assert "trinet_http_inflight_requests" in scrape(port)
    client.__exit__(None, None, None)
    with pytest.raises(OSError):
        scrape(port)


def test_a_busy_metrics_port_does_not_stop_the_api(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        client = make_client(
            metrics_enabled=True, metrics_host="127.0.0.1", metrics_port=blocker.getsockname()[1]
        )
        assert client.get("/v1/models", headers=auth).status_code == 200
