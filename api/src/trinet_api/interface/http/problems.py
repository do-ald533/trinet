"""RFC 9457 Problem Details (``application/problem+json``) for every error response.

No stack trace, exception message of unexpected errors or request value is ever returned;
those go to the server log only.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from trinet_api.domain.errors import (
    InferenceFailed,
    InvalidInput,
    Issue,
    ModelNotAvailable,
    ModelNotFound,
    NotReady,
    Overloaded,
    PayloadTooLarge,
    TriNetError,
    UnsupportedInput,
)
from trinet_api.interface.http.security import Unauthorized

PROBLEM_MEDIA_TYPE = "application/problem+json"
log = structlog.get_logger(__name__)


class ProblemIssue(BaseModel):
    loc: list[str | int] = Field(description="Path to the offending value, e.g. instances.0.Area")
    code: str
    message: str


class Problem(BaseModel):
    """RFC 9457 problem object plus ``code``, ``request_id`` and (for 422) ``errors``."""

    type: str = Field(default="about:blank", description="URI identifying the problem type.")
    title: str
    status: int
    detail: str | None = None
    instance: str | None = Field(default=None, description="Request path.")
    code: str | None = Field(default=None, description="Stable machine-readable error code.")
    request_id: str | None = None
    errors: list[ProblemIssue] | None = None


_TITLES: dict[int, str] = {
    400: "Bad request",
    401: "Authentication required",
    404: "Not found",
    405: "Method not allowed",
    413: "Payload too large",
    415: "Unsupported media type",
    422: "Validation failed",
    429: "Too many requests",
    500: "Internal server error",
    503: "Service unavailable",
}

_STATUS_BY_ERROR: dict[type[TriNetError], int] = {
    ModelNotFound: 404,
    ModelNotAvailable: 404,
    InvalidInput: 422,
    UnsupportedInput: 415,
    PayloadTooLarge: 413,
    Overloaded: 503,
    InferenceFailed: 500,
    NotReady: 503,
}


def problem_response(
    status: int,
    code: str,
    *,
    detail: str | None = None,
    request: Request | None = None,
    scope_path: str | None = None,
    request_id: str | None = None,
    errors: list[Issue] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    if request is not None:
        scope_path = scope_path or request.url.path
        request_id = request_id or getattr(request.state, "request_id", None)
    body = Problem(
        type=f"urn:trinet:problem:{code}",
        title=_TITLES.get(status, "Error"),
        status=status,
        detail=detail,
        instance=scope_path,
        code=code,
        request_id=request_id,
        errors=[ProblemIssue(loc=list(i.loc), code=i.code, message=i.message) for i in errors]
        if errors
        else None,
    )
    return JSONResponse(
        body.model_dump(exclude_none=True),
        status_code=status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """``responses=`` entries so the OpenAPI document declares every error we can return."""
    documented: dict[int | str, dict[str, Any]] = {}
    for status in statuses:
        entry: dict[str, Any] = {"model": Problem, "description": _TITLES[status]}
        if status == 401:
            entry["headers"] = {
                "WWW-Authenticate": {"schema": {"type": "string"}, "description": "ApiKey"}
            }
        if status in (429, 503):
            entry["headers"] = {
                "Retry-After": {"schema": {"type": "integer"}, "description": "Seconds"}
            }
        documented[status] = entry
    return documented


# --------------------------------------------------------------------------- handlers


def _domain_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, TriNetError)  # noqa: S101 - registered for TriNetError only
    status = next((s for t, s in _STATUS_BY_ERROR.items() if isinstance(exc, t)), 500)
    headers: dict[str, str] | None = None
    errors: list[Issue] | None = None
    detail: str | None = str(exc)
    if isinstance(exc, Overloaded):
        headers = {"Retry-After": str(exc.retry_after_s)}
    elif isinstance(exc, InvalidInput):
        errors = list(exc.issues)
        detail = f"{len(exc.issues)} validation error(s); see 'errors'"
    elif isinstance(exc, InferenceFailed):
        detail = "The model could not produce a valid prediction."
    return problem_response(
        status, exc.code, detail=detail, request=request, errors=errors, headers=headers
    )


def _unauthorized(request: Request, _exc: Exception) -> Response:
    return problem_response(
        401,
        "unauthorized",
        detail="A valid X-API-Key header is required.",
        request=request,
        headers={"WWW-Authenticate": "ApiKey"},
    )


def _rate_limited(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RateLimitExceeded)  # noqa: S101 - registered for it only
    window = exc.limit.limit.get_expiry() if exc.limit is not None else 60
    return problem_response(
        429,
        "rate_limited",
        detail="Rate limit exceeded.",
        request=request,
        headers={"Retry-After": str(window)},
    )


def _http_exception(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered for it only
    code = {404: "not_found", 405: "method_not_allowed"}.get(
        exc.status_code, f"http_{exc.status_code}"
    )
    safe_detail = {404: "No such resource.", 405: "Method not allowed."}.get(exc.status_code)
    headers = {k: v for k, v in (exc.headers or {}).items() if k.lower() == "allow"}
    return problem_response(
        exc.status_code,
        code,
        detail=safe_detail
        or ("The request could not be processed." if exc.status_code < 500 else None),
        request=request,
        headers=headers or None,
    )


def _validation_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered for it only
    issues = [
        Issue(
            loc=tuple(part for part in err["loc"] if part != "body"),
            code=str(err["type"]),
            message=str(err["msg"]),
        )
        for err in exc.errors()
    ]
    return problem_response(
        422,
        "validation_error",
        detail=f"{len(issues)} validation error(s); see 'errors'",
        request=request,
        errors=issues,
    )


def register_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(TriNetError, _domain_error)
    app.add_exception_handler(Unauthorized, _unauthorized)
    app.add_exception_handler(RateLimitExceeded, _rate_limited)
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
