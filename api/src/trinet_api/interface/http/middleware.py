"""Pure-ASGI middlewares: request context (id, security headers, access log, metrics) and
body-size limiting (enforced *before* any JSON or multipart parsing).

Order, outermost first: ``RequestContextMiddleware`` -> rate limiter -> ``BodyLimitMiddleware``
-> routing. The request-context layer also turns any unhandled exception into a generic
problem+json 500 so the response still carries the request id and security headers.
"""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from trinet_api.infrastructure.observability import Metrics
from trinet_api.interface.http.problems import problem_response

log = structlog.get_logger("trinet_api.http")

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Strict-Transport-Security", "max-age=63072000; includeSubDomains"),
    ("Cache-Control", "no-store"),
)
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _route_and_model(scope: Scope) -> tuple[str, str | None]:
    route = scope.get("route")
    template = getattr(route, "path", None)
    params = scope.get("path_params") or {}
    if not isinstance(template, str):
        template = "/openapi.json" if scope.get("path") == "/openapi.json" else "unmatched"
    return template, params.get("name")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                for name, value in SECURITY_HEADERS:
                    headers.setdefault(name, value)
            await send(message)

        self.metrics.inflight_requests.inc()
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("unhandled_error")
            if response_started:
                raise  # too late for a clean answer: let the server drop the connection
            await problem_response(
                500,
                "internal_error",
                detail="An unexpected error occurred.",
                request_id=request_id,
                scope_path=scope.get("path"),
            )(scope, receive, send_wrapper)
        finally:
            self.metrics.inflight_requests.dec()
            seconds = time.perf_counter() - started
            route, model = _route_and_model(scope)
            self.metrics.observe_http(
                method=scope["method"], route=route, model=model, status=status, seconds=seconds
            )
            log.info(
                "http_request",
                method=scope["method"],
                route=route,
                model=self.metrics.model_label(model),
                status=status,
                duration_ms=round(seconds * 1000.0, 2),
            )
            structlog.contextvars.unbind_contextvars("request_id")


class _BodyTooLargeError(Exception):
    pass


class BodyLimitMiddleware:
    """413 as soon as the declared or streamed body exceeds the limit for its content type."""

    def __init__(self, app: ASGIApp, *, max_json_bytes: int, max_upload_bytes: int) -> None:
        self.app = app
        self.max_json_bytes = max_json_bytes
        self.max_upload_bytes = max_upload_bytes

    def _limit(self, headers: Headers) -> int:
        media = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        return self.max_upload_bytes if media == "multipart/form-data" else self.max_json_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        limit = self._limit(headers)
        declared = headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await self._reject(scope, receive, send, limit)
            return
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if response_started:
                raise
            await self._reject(scope, receive, send, limit)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        request_id = scope.get("state", {}).get("request_id")
        response = problem_response(
            413,
            "payload_too_large",
            detail=f"The request body exceeds {limit} bytes.",
            request_id=request_id,
            scope_path=scope.get("path"),
        )
        await response(scope, receive, send)
