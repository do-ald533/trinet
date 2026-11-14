"""Structured logging (structlog -> JSON on stdout) and Prometheus metrics.

Cardinality is bounded by construction: label values come from fixed sets (HTTP methods,
route templates, loaded model names, manifest class labels, status codes), never from
request data.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from http import HTTPMethod
from typing import TextIO

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.process_collector import ProcessCollector

UNKNOWN = "unknown"
_METHODS = frozenset(m.value for m in HTTPMethod)
_LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class _StdoutHandler(logging.StreamHandler[TextIO]):
    """Writes to whatever ``sys.stdout`` is *now* (survives stream swaps such as test capture)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self) -> TextIO:
        return sys.stdout

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        return None


def configure_logging(level: str = "INFO") -> None:
    """JSON logs for both structlog loggers and the stdlib loggers of third-party code."""
    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    handler = _StdoutHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).disabled = True  # one access log line comes from our middleware


class Metrics:
    """RED metrics per model plus predicted-class counters, on a private registry."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        ProcessCollector(registry=self.registry)
        self.requests = Counter(
            "trinet_http_requests_total",
            "HTTP requests by method, route template, model and status code.",
            ("method", "route", "model", "status"),
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "trinet_http_request_duration_seconds",
            "HTTP request latency by method, route template and model.",
            ("method", "route", "model"),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inference_seconds = Histogram(
            "trinet_inference_duration_seconds",
            "Time spent inside the model runtime per prediction request.",
            ("model", "runtime"),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predicted_class = Counter(
            "trinet_predicted_class_total",
            "Predictions per model and predicted class (labels come from the manifest).",
            ("model", "label"),
            registry=self.registry,
        )
        self.inflight_requests = Gauge(
            "trinet_http_inflight_requests",
            "HTTP requests currently being served.",
            registry=self.registry,
        )
        self.inflight_inferences = Gauge(
            "trinet_inflight_inferences",
            "Inferences currently holding a concurrency slot.",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "trinet_model_info",
            "Loaded models (value 1 = serving, 0 = unavailable).",
            ("model", "version", "status", "runtime"),
            registry=self.registry,
        )
        self._models: frozenset[str] = frozenset()

    def register_models(self, models: Iterable[tuple[str, str, str, str, bool]]) -> None:
        """Declare ``(name, version, status, runtime, serving)`` per model; fixes the label set."""
        names = set()
        for name, version, status, runtime, serving in models:
            names.add(name)
            self.model_info.labels(name, version, status, runtime).set(1 if serving else 0)
        self._models = frozenset(names)

    def model_label(self, name: str | None) -> str:
        return name if name in self._models else UNKNOWN

    def observe_http(
        self, *, method: str, route: str, model: str | None, status: int, seconds: float
    ) -> None:
        method_label = method if method in _METHODS else "OTHER"
        model_label = self.model_label(model)
        self.requests.labels(method_label, route, model_label, str(status)).inc()
        self.request_seconds.labels(method_label, route, model_label).observe(seconds)

    def observe_prediction(
        self, *, model: str, runtime: str, inference_seconds: float, labels: Iterable[str]
    ) -> None:
        self.inference_seconds.labels(model, runtime).observe(inference_seconds)
        for label in labels:
            self.predicted_class.labels(model, label).inc()
