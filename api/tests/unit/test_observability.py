from __future__ import annotations

import json
import logging

import pytest
import structlog
from prometheus_client import generate_latest

from trinet_api.infrastructure.observability import UNKNOWN, Metrics, configure_logging


@pytest.fixture
def metrics() -> Metrics:
    instance = Metrics()
    instance.register_models(
        [
            ("alpha", "1.0.0", "candidate", "onnx", True),
            ("beta", "1.0.0", "unavailable", "none", False),
        ]
    )
    return instance


def scrape(metrics: Metrics) -> str:
    return generate_latest(metrics.registry).decode()


def json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


# --------------------------------------------------------------------------- logging


def test_structlog_events_are_json_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    structlog.contextvars.bind_contextvars(request_id="abc12345")
    try:
        structlog.get_logger("trinet_api.test").info("hello", answer=42)
    finally:
        structlog.contextvars.clear_contextvars()
    (record,) = json_lines(capsys.readouterr().out)
    assert record["event"] == "hello"
    assert record["answer"] == 42
    assert record["level"] == "info"
    assert record["logger"] == "trinet_api.test"
    assert record["request_id"] == "abc12345"
    assert str(record["timestamp"]).endswith("Z")


def test_third_party_stdlib_loggers_are_json_too(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logging.getLogger("some.library").warning("careful %s", "now")
    (record,) = json_lines(capsys.readouterr().out)
    assert record["event"] == "careful now"
    assert record["level"] == "warning"


def test_the_level_filters_and_exceptions_are_rendered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("warning")
    log = structlog.get_logger("trinet_api.test")
    log.info("dropped")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed")
    (record,) = json_lines(capsys.readouterr().out)
    assert record["event"] == "failed"
    assert "RuntimeError: boom" in str(record["exception"])


def test_reconfiguring_does_not_duplicate_handlers(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    structlog.get_logger("trinet_api.test").info("once")
    assert len(json_lines(capsys.readouterr().out)) == 1


def test_the_uvicorn_access_log_is_silenced() -> None:
    configure_logging("INFO")
    assert logging.getLogger("uvicorn.access").disabled


# --------------------------------------------------------------------------- metrics


def test_unknown_models_and_methods_collapse_to_fixed_labels(metrics: Metrics) -> None:
    for attacker_value in ("evil-1", "evil-2", "../../etc/passwd"):
        metrics.observe_http(
            method="BREW", route="unmatched", model=attacker_value, status=404, seconds=0.001
        )
    text = scrape(metrics)
    assert 'method="OTHER"' in text
    assert f'model="{UNKNOWN}"' in text
    assert "evil" not in text
    assert "passwd" not in text
    assert metrics.model_label("alpha") == "alpha"
    assert metrics.model_label(None) == UNKNOWN


def test_red_metrics_per_model(metrics: Metrics) -> None:
    metrics.observe_http(
        method="POST",
        route="/v1/models/{name}/predictions",
        model="alpha",
        status=200,
        seconds=0.02,
    )
    metrics.observe_http(
        method="POST",
        route="/v1/models/{name}/predictions",
        model="alpha",
        status=422,
        seconds=0.01,
    )
    text = scrape(metrics)
    assert (
        'trinet_http_requests_total{method="POST",model="alpha",'
        'route="/v1/models/{name}/predictions",status="200"} 1.0' in text
    )
    assert 'status="422"} 1.0' in text
    assert "trinet_http_request_duration_seconds_bucket" in text


def test_prediction_metrics_count_each_predicted_class(metrics: Metrics) -> None:
    metrics.observe_prediction(
        model="alpha",
        runtime="onnx",
        inference_seconds=0.004,
        labels=["default", "default", "no_default"],
    )
    text = scrape(metrics)
    assert 'trinet_predicted_class_total{label="default",model="alpha"} 2.0' in text
    assert 'trinet_predicted_class_total{label="no_default",model="alpha"} 1.0' in text
    assert 'trinet_inference_duration_seconds_count{model="alpha",runtime="onnx"} 1.0' in text


def test_model_info_gauge_reports_serving_state(metrics: Metrics) -> None:
    text = scrape(metrics)
    assert (
        'trinet_model_info{model="alpha",runtime="onnx",status="candidate",version="1.0.0"} 1.0'
        in text
    )
    assert (
        'trinet_model_info{model="beta",runtime="none",status="unavailable",version="1.0.0"} 0.0'
        in text
    )


def test_each_instance_has_a_private_registry() -> None:
    first, second = Metrics(), Metrics()
    first.inflight_requests.inc()
    assert "trinet_http_inflight_requests 1.0" in scrape(first)
    assert "trinet_http_inflight_requests 0.0" in scrape(second)
