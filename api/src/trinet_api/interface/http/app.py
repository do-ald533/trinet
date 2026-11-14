"""Composition root: wires settings, model loading, service, middlewares and routes.

``create_app`` is a factory (``uvicorn trinet_api.interface.http.app:create_app --factory``).
Models are loaded in the lifespan, so the process is alive (``/health/live``) while loading
and ``/health/ready`` flips once every model has been verified and warmed up.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from PIL import Image
from prometheus_client import start_http_server
from slowapi import Limiter
from slowapi.middleware import SlowAPIASGIMiddleware
from starlette.concurrency import run_in_threadpool

from trinet_api.application.service import ConcurrencyLimiter, PredictionService
from trinet_api.domain.entities import ModelBundle, ModelRegistry
from trinet_api.infrastructure.loading import RuntimeOptions, load_registry
from trinet_api.infrastructure.observability import Metrics, configure_logging
from trinet_api.interface.http.middleware import BodyLimitMiddleware, RequestContextMiddleware
from trinet_api.interface.http.problems import PROBLEM_MEDIA_TYPE, register_problem_handlers
from trinet_api.interface.http.routers import (
    JSON,
    health_live,
    health_ready,
    health_router,
    models_router,
)
from trinet_api.interface.http.security import API_KEY_HEADER, key_matches
from trinet_api.settings import Settings

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Inference API for the TriNet models (Dry Bean, California Housing, Intel scenes). "
    "Behaviour is driven by each model's `manifest.json`; errors follow RFC 9457."
)


def _package_version() -> str:
    try:
        return version("trinet-api")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "0.0.0"


def rate_limit_key(request: Request) -> str:
    """Bucket per valid API key; everything else shares a per-IP bucket (bounded memory)."""
    expected: bytes = request.app.state.api_key_bytes
    if key_matches(request.headers.get(API_KEY_HEADER), expected):
        return "key:" + hashlib.sha256(expected).hexdigest()[:16]
    return f"ip:{request.client.host if request.client else 'unknown'}"


def _model_rows(registry: ModelRegistry) -> list[tuple[str, str, str, str, bool]]:
    rows = []
    for entry in registry:
        if isinstance(entry, ModelBundle):
            rows.append(
                (entry.name, entry.version, entry.manifest.status.value, entry.runtime.kind, True)
            )
        else:
            rows.append((entry.name, entry.version or "", "unavailable", "none", False))
    return rows


def _build_openapi(app: FastAPI, settings: Settings) -> dict[str, Any]:
    if app.openapi_schema is not None:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            for code, response in operation.get("responses", {}).items():
                content = response.get("content", {})
                if code.isdigit() and int(code) >= 400 and JSON in content:
                    content[PROBLEM_MEDIA_TYPE] = content.pop(JSON)
    predictions = schema["paths"]["/v1/models/{name}/predictions"]["post"]
    predictions["requestBody"]["content"][JSON]["schema"]["properties"]["instances"]["maxItems"] = (
        settings.max_batch
    )
    app.openapi_schema = schema
    return schema


def create_app(
    settings: Settings | None = None, *, registry: ModelRegistry | None = None
) -> FastAPI:
    """Build the application. ``registry`` skips artifact loading (tests, embedding)."""
    settings = settings or Settings()
    configure_logging(settings.log_level)
    api_key = settings.resolve_api_key()  # fail closed at start-up
    metrics = Metrics()
    limiter = Limiter(
        key_func=rate_limit_key,
        application_limits=[settings.rate_limit],
        strategy="moving-window",
        storage_uri="memory://",
        headers_enabled=False,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
        loaded = (
            registry
            if registry is not None
            else await run_in_threadpool(
                load_registry,
                settings.artifacts_dir,
                RuntimeOptions(intra_op_threads=settings.ort_intra_op_threads),
            )
        )
        concurrency = ConcurrencyLimiter(settings.max_concurrency, settings.queue_timeout_s)
        app.state.service = PredictionService(
            loaded,
            concurrency,
            max_batch=settings.max_batch,
            max_image_pixels=settings.max_image_pixels,
        )
        metrics.register_models(_model_rows(loaded))
        metrics.inflight_inferences.set_function(lambda: float(concurrency.inflight))
        server = None
        if settings.metrics_enabled:
            try:
                server, _thread = start_http_server(
                    settings.metrics_port, addr=settings.metrics_host, registry=metrics.registry
                )
            except OSError:
                logger.exception(
                    "could not start the metrics server on port %s", settings.metrics_port
                )
        logger.info("startup complete: %d model(s), %d serving", len(loaded), len(loaded.available))
        try:
            yield
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()

    app = FastAPI(
        title="TriNet inference API",
        version=_package_version(),
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url=None,  # the strict CSP would break Swagger UI; /openapi.json stays available
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.api_key_bytes = api_key.get_secret_value().encode("utf-8")
    app.state.metrics = metrics
    app.state.limiter = limiter

    # Innermost first: the last middleware added is the outermost.
    app.add_middleware(
        BodyLimitMiddleware,
        max_json_bytes=settings.max_json_bytes,
        max_upload_bytes=settings.max_upload_bytes,
    )
    app.add_middleware(SlowAPIASGIMiddleware)
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    register_problem_handlers(app)

    app.include_router(health_router)
    app.include_router(models_router)
    limiter.exempt(health_live)  # type: ignore[no-untyped-call]
    limiter.exempt(health_ready)  # type: ignore[no-untyped-call]
    app.openapi = lambda: _build_openapi(app, settings)  # type: ignore[method-assign]
    return app
