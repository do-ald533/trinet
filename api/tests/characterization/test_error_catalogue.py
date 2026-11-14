"""Every problem+json the API can answer with, triggered over real HTTP and pinned exactly.

Characterization of the CURRENT behaviour (R0 of docs/refactor-plan.md), quirks included: it
must stay green, untouched, through every move of the refactor. Each row asserts the status,
the ``application/problem+json`` content type, the complete body (type URN, title, detail,
instance, code, request id, ``errors[]`` with ``loc``/``code``/``message``) and its key order,
the seven security headers, ``X-Request-ID`` and the three status-specific headers
(``WWW-Authenticate: ApiKey`` on 401, ``Allow`` on 405, ``Retry-After`` on 429/503: present
exactly where expected and absent everywhere else).

Codes covered: unauthorized, model_not_found, model_not_available, invalid_input,
validation_error, unsupported_media_type, payload_too_large, rate_limited, overloaded,
not_ready, inference_failed, internal_error, not_found, method_not_allowed, http_<n>.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx2 import Response
from tests.characterization._http_helpers import (
    CANDLESTICK,
    CREDIT,
    JSON_CT,
    PREDICT_CANDLESTICK,
    PREDICT_CREDIT,
    assert_conditional_headers,
    problem_body,
)
from tests.fakes import FakeRuntime

PATTERN_MESSAGE = "String should match pattern '^[a-z0-9][a-z0-9-]{0,62}$'"
TABULAR_NEEDS_JSON = f"model '{CREDIT}' expects Content-Type application/json"
IMAGE_NEEDS_MULTIPART = f"model '{CANDLESTICK}' expects Content-Type multipart/form-data"
ONLY_PNG = "unsupported image format; allowed: PNG"
ONE_ERROR = "1 validation error(s); see 'errors'"


@dataclass(frozen=True)
class Ctx:
    """What a trigger needs; built once per test from the shared fixtures."""

    make_client: Callable[..., TestClient]
    make_empty_artifacts_client: Callable[..., TestClient]
    fake_registry: Callable[..., Any]
    auth: dict[str, str]
    record: dict[str, Any]
    png: bytes

    def upload(self, data: bytes, *, field: str = "file") -> dict[str, Any]:
        return {"files": {field: ("scene.png", data, "application/octet-stream")}}


@dataclass(frozen=True)
class Case:
    status: int
    code: str
    title: str
    detail: str | None
    instance: str
    trigger: Callable[[Ctx], Response]
    errors: list[dict[str, Any]] | None = None
    www_authenticate: str | None = None
    allow: str | None = None
    retry_after: bool = False


@pytest.fixture
def ctx(
    make_client: Callable[..., TestClient],
    make_empty_artifacts_client: Callable[..., TestClient],
    fake_registry: Callable[..., Any],
    auth: dict[str, str],
    golden_records: dict[str, Any],
    png_bytes: bytes,
) -> Ctx:
    return Ctx(
        make_client, make_empty_artifacts_client, fake_registry, auth, golden_records[CREDIT], png_bytes
    )


def issue(loc: list[str | int], code: str, message: str) -> dict[str, Any]:
    return {"loc": loc, "code": code, "message": message}


# --------------------------------------------------------------------------- triggers


def request(
    method: str,
    path: str,
    *,
    authed: bool = True,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> Callable[[Ctx], Response]:
    def trigger(c: Ctx) -> Response:
        merged = {**(c.auth if authed else {}), **(headers or {})}
        return c.make_client().request(method, path, headers=merged, **kwargs)

    return trigger


def get(
    path: str, *, authed: bool = True, headers: dict[str, str] | None = None
) -> Callable[[Ctx], Response]:
    return request("GET", path, authed=authed, headers=headers)


def post_json(path: str, payload: Callable[[Ctx], Any], **overrides: Any) -> Callable[[Ctx], Response]:
    def trigger(c: Ctx) -> Response:
        return c.make_client(**overrides).post(path, headers=c.auth, json=payload(c))

    return trigger


def post_raw(
    path: str, content: bytes, content_type: str | None = "application/json", **overrides: Any
) -> Callable[[Ctx], Response]:
    def trigger(c: Ctx) -> Response:
        headers = dict(c.auth)
        if content_type is not None:
            headers["Content-Type"] = content_type
        return c.make_client(**overrides).post(path, headers=headers, content=content)

    return trigger


def post_upload(
    path: str, data: Callable[[Ctx], bytes], *, field: str = "file", **overrides: Any
) -> Callable[[Ctx], Response]:
    def trigger(c: Ctx) -> Response:
        client = c.make_client(**overrides)
        return client.post(path, headers=c.auth, **c.upload(data(c), field=field))

    return trigger


def broken_registry_get(path: str, method: str = "GET") -> Callable[[Ctx], Response]:
    def trigger(c: Ctx) -> Response:
        client = c.make_client(registry_override=c.fake_registry())
        if method == "GET":
            return client.get(path, headers=c.auth)
        return client.post(path, headers={**c.auth, **JSON_CT}, content=b"{}")

    return trigger


def raise_http(
    status: int, *, detail: str | None = None, headers: dict[str, str] | None = None
) -> Callable[[Ctx], Response]:
    """A route (added to the running app) that raises ``HTTPException`` with these arguments."""

    def trigger(c: Ctx) -> Response:
        client = c.make_client()

        def synthetic() -> None:
            raise HTTPException(status, detail=detail, headers=headers)

        client.app.add_api_route("/synthetic", synthetic)  # type: ignore[attr-defined]
        return client.get("/synthetic")

    return trigger


def unhandled_exception(c: Ctx) -> Response:
    client = c.make_client()

    def boom() -> None:
        raise RuntimeError("super-secret-internal-state /srv/models/x.onnx")

    client.app.add_api_route("/boom", boom)  # type: ignore[attr-defined]
    return client.get("/boom", headers=c.auth)


def rate_limited(c: Ctx) -> Response:
    client = c.make_client(rate_limit="1/minute")
    assert client.get("/v1/models", headers=c.auth).status_code == 200
    return client.get("/v1/models", headers=c.auth)


def overloaded(c: Ctx) -> Response:
    gate = threading.Event()
    runtime = FakeRuntime((25,), 2, gate=gate)
    registry = c.fake_registry({CREDIT: runtime})
    client = c.make_client(registry_override=registry, max_concurrency=1, queue_timeout_s=0.1)
    body = {"instances": [c.record]}
    worker = threading.Thread(
        target=lambda: client.post(PREDICT_CREDIT, headers=c.auth, json=body), daemon=True
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not runtime.calls:  # the only inference slot is now taken
            assert time.monotonic() < deadline, "the first request never reached the runtime"
            time.sleep(0.005)
        return client.post(PREDICT_CREDIT, headers=c.auth, json=body)
    finally:
        gate.set()  # never leave a worker thread blocked
        worker.join(timeout=5)


def failing_runtime(error: Exception | None = None, *, wrong_rows: bool = False) -> Callable[[Ctx], Response]:
    def trigger(c: Ctx) -> Response:
        def two_rows_too_many(batch: Any) -> Any:
            return np.zeros((batch.shape[0] + 2, 2), dtype=np.float32)

        runtime = FakeRuntime((25,), 2, error=error, fn=two_rows_too_many if wrong_rows else None)
        client = c.make_client(registry_override=c.fake_registry({CREDIT: runtime}))
        return client.post(PREDICT_CREDIT, headers=c.auth, json={"instances": [c.record]})

    return trigger


def not_ready(c: Ctx) -> Response:
    return c.make_empty_artifacts_client().get("/health/ready")


# --------------------------------------------------------------------------- the table

UNAUTHORIZED = {
    "status": 401,
    "code": "unauthorized",
    "title": "Authentication required",
    "detail": "A valid X-API-Key header is required.",
    "www_authenticate": "ApiKey",
}
NOT_FOUND = {"status": 404, "title": "Not found"}
INVALID_INPUT = {"status": 422, "code": "invalid_input", "title": "Validation failed"}
UNSUPPORTED = {"status": 415, "code": "unsupported_media_type", "title": "Unsupported media type"}
TOO_LARGE = {"status": 413, "code": "payload_too_large", "title": "Payload too large"}
METHOD = {
    "status": 405,
    "code": "method_not_allowed",
    "title": "Method not allowed",
    "detail": "Method not allowed.",
}
HTTP_400 = {
    "status": 400,
    "code": "http_400",
    "title": "Bad request",
    "detail": "The request could not be processed.",
}
MODEL_PATH = "/v1/models/{}"
BIG = b" " * 2000

CASES: dict[str, Case] = {
    # ----- unauthorized
    "unauthorized-missing-key": Case(
        **UNAUTHORIZED, instance="/v1/models", trigger=get("/v1/models", authed=False)
    ),
    "unauthorized-wrong-key": Case(
        **UNAUTHORIZED,
        instance="/v1/models",
        trigger=get("/v1/models", authed=False, headers={"X-API-Key": "wrong-key-0123456789abcdef"}),
    ),
    "unauthorized-empty-key": Case(
        **UNAUTHORIZED,
        instance="/v1/models",
        trigger=get("/v1/models", authed=False, headers={"X-API-Key": ""}),
    ),
    "unauthorized-on-a-model-detail": Case(
        **UNAUTHORIZED, instance="/v1/models/nope", trigger=get("/v1/models/nope", authed=False)
    ),
    # ----- model_not_found / model_not_available
    "model_not_found-get": Case(
        **NOT_FOUND,
        code="model_not_found",
        detail="model 'nope' does not exist",
        instance="/v1/models/nope",
        trigger=get("/v1/models/nope"),
    ),
    "model_not_found-predict": Case(
        **NOT_FOUND,
        code="model_not_found",
        detail="model 'nope' does not exist",
        instance="/v1/models/nope/predictions",
        trigger=post_raw("/v1/models/nope/predictions", b"{}"),
    ),
    "model_not_available-get": Case(
        **NOT_FOUND,
        code="model_not_available",
        detail="model 'broken-model' is not available (integrity_check_failed)",
        instance="/v1/models/broken-model",
        trigger=broken_registry_get("/v1/models/broken-model"),
    ),
    "model_not_available-predict": Case(
        **NOT_FOUND,
        code="model_not_available",
        detail="model 'broken-model' is not available (integrity_check_failed)",
        instance="/v1/models/broken-model/predictions",
        trigger=broken_registry_get("/v1/models/broken-model/predictions", "POST"),
    ),
    # ----- invalid_input (the domain's own validation; ``loc`` has no "body" prefix)
    "invalid_input-not-json": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue([], "invalid_json", "the body is not valid JSON")],
        trigger=post_raw(PREDICT_CREDIT, b"{nope"),
    ),
    "invalid_input-nan-constant": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue([], "invalid_json", "the body is not valid JSON")],
        trigger=post_raw(PREDICT_CREDIT, b'{"instances": [{"age": NaN}]}'),
    ),
    "invalid_input-missing-instances": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue(["instances"], "missing", "Field required")],
        trigger=post_json(PREDICT_CREDIT, lambda c: {}),
    ),
    "invalid_input-extra-key": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue(["x"], "extra_forbidden", "Extra inputs are not permitted")],
        trigger=post_json(PREDICT_CREDIT, lambda c: {"instances": [], "x": 1}),
    ),
    "invalid_input-body-is-an-array": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[
            issue([], "model_type", "Input should be a valid dictionary or instance of PredictionRequest")
        ],
        trigger=post_json(PREDICT_CREDIT, lambda c: [1]),
    ),
    "invalid_input-instance-is-not-an-object": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue(["instances", 0], "dict_type", "Input should be a valid dictionary")],
        trigger=post_json(PREDICT_CREDIT, lambda c: {"instances": [1]}),
    ),
    "invalid_input-empty-batch": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue(["instances"], "empty_batch", "at least one instance required")],
        trigger=post_json(PREDICT_CREDIT, lambda c: {"instances": []}),
    ),
    "invalid_input-batch-too-large": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CREDIT,
        errors=[issue(["instances"], "batch_too_large", "at most 1 instances per request")],
        trigger=post_json(PREDICT_CREDIT, lambda c: {"instances": [c.record, c.record]}, max_batch=1),
    ),
    "invalid_input-bad-record": Case(
        **INVALID_INPUT,
        detail="4 validation error(s); see 'errors'",
        instance=PREDICT_CREDIT,
        errors=[
            issue(["instances", 0, "zzz"], "unknown_field", "unknown field"),
            issue(
                ["instances", 0, "education"],
                "not_in_vocabulary",
                "must be one of: 'graduate', 'university', 'high_school', 'other'",
            ),
            issue(["instances", 0, "age"], "below_minimum", "must be >= 18"),
            issue(["instances", 0, "pay_1"], "type_error", "must be a finite number"),
        ],
        trigger=post_json(
            PREDICT_CREDIT,
            lambda c: {
                "instances": [
                    {**c.record, "zzz": 1, "age": 5, "pay_1": "late", "education": "nope"}
                ]
            },
        ),
    ),
    "invalid_input-image-field-missing": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CANDLESTICK,
        errors=[issue(["file"], "required", "multipart field 'file' with the image is required")],
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: c.png, field="image"),
    ),
    "invalid_input-truncated-image": Case(
        **INVALID_INPUT,
        detail=ONE_ERROR,
        instance=PREDICT_CANDLESTICK,
        errors=[issue(["file"], "invalid_image", "the file is not a valid, complete image")],
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: c.png[:60]),
    ),
    # ----- validation_error (FastAPI's own validation: the ModelName path pattern)
    "validation_error-path-pattern-get": Case(
        **{"status": 422, "title": "Validation failed"},
        code="validation_error",
        detail=ONE_ERROR,
        instance="/v1/models/NOPE",
        errors=[issue(["path", "name"], "string_pattern_mismatch", PATTERN_MESSAGE)],
        trigger=get("/v1/models/NOPE"),
    ),
    "validation_error-path-pattern-predict": Case(
        **{"status": 422, "title": "Validation failed"},
        code="validation_error",
        detail=ONE_ERROR,
        instance="/v1/models/NOPE/predictions",
        errors=[issue(["path", "name"], "string_pattern_mismatch", PATTERN_MESSAGE)],
        trigger=post_raw("/v1/models/NOPE/predictions", b"{}"),
    ),
    "validation_error-path-too-long": Case(
        **{"status": 422, "title": "Validation failed"},
        code="validation_error",
        detail=ONE_ERROR,
        instance="/v1/models/" + "a" * 64,
        errors=[issue(["path", "name"], "string_pattern_mismatch", PATTERN_MESSAGE)],
        trigger=get("/v1/models/" + "a" * 64),
    ),
    # ----- unsupported_media_type
    "unsupported_media_type-tabular-needs-json": Case(
        **UNSUPPORTED,
        detail=TABULAR_NEEDS_JSON,
        instance=PREDICT_CREDIT,
        trigger=post_raw(PREDICT_CREDIT, b"{}", "text/plain"),
    ),
    "unsupported_media_type-no-content-type-at-all": Case(
        **UNSUPPORTED,
        detail=TABULAR_NEEDS_JSON,
        instance=PREDICT_CREDIT,
        trigger=post_raw(PREDICT_CREDIT, b"", None),
    ),
    "unsupported_media_type-image-needs-multipart": Case(
        **UNSUPPORTED,
        detail=IMAGE_NEEDS_MULTIPART,
        instance=PREDICT_CANDLESTICK,
        trigger=post_json(PREDICT_CANDLESTICK, lambda c: {"instances": []}),
    ),
    "unsupported_media_type-empty-file": Case(
        **UNSUPPORTED,
        detail=ONLY_PNG,
        instance=PREDICT_CANDLESTICK,
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: b""),
    ),
    "unsupported_media_type-not-an-image": Case(
        **UNSUPPORTED,
        detail=ONLY_PNG,
        instance=PREDICT_CANDLESTICK,
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: b"not an image"),
    ),
    "unsupported_media_type-jpeg-when-only-png-is-allowed": Case(
        **UNSUPPORTED,
        detail=ONLY_PNG,
        instance=PREDICT_CANDLESTICK,
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: b"\xff\xd8\xff\xe0" + b"0" * 64),
    ),
    # ----- payload_too_large (middleware, and the domain error of the image size)
    "payload_too_large-json-body": Case(
        **TOO_LARGE,
        detail="The request body exceeds 1024 bytes.",
        instance=PREDICT_CREDIT,
        trigger=post_raw(PREDICT_CREDIT, BIG, max_json_bytes=1024),
    ),
    "payload_too_large-multipart-body": Case(
        **TOO_LARGE,
        detail="The request body exceeds 1024 bytes.",
        instance=PREDICT_CANDLESTICK,
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: b"\x89PNG" + b"0" * 3000, max_upload_bytes=1024),
    ),
    "payload_too_large-image-above-the-manifest-limit": Case(
        **TOO_LARGE,
        detail="image larger than 1048576 bytes",
        instance=PREDICT_CANDLESTICK,
        trigger=post_upload(PREDICT_CANDLESTICK, lambda c: b"\x89PNG\r\n\x1a\n" + b"0" * 1_100_000),
    ),
    # ----- rate_limited / overloaded / not_ready
    "rate_limited": Case(
        status=429,
        code="rate_limited",
        title="Too many requests",
        detail="Rate limit exceeded.",
        instance="/v1/models",
        retry_after=True,
        trigger=rate_limited,
    ),
    "overloaded": Case(
        status=503,
        code="overloaded",
        title="Service unavailable",
        detail="the service is busy, retry later",
        instance=PREDICT_CREDIT,
        retry_after=True,
        trigger=overloaded,
    ),
    # Today the readiness 503 carries NO Retry-After although the OpenAPI document declares one
    # (R7 of the plan adds it: flip ``retry_after`` here, deliberately, in that commit).
    "not_ready": Case(
        status=503,
        code="not_ready",
        title="Service unavailable",
        detail="no model is loaded and serving",
        instance="/health/ready",
        trigger=not_ready,
    ),
    # ----- inference_failed / internal_error
    "inference_failed-runtime-raises": Case(
        status=500,
        code="inference_failed",
        title="Internal server error",
        detail="The model could not produce a valid prediction.",
        instance=PREDICT_CREDIT,
        trigger=failing_runtime(RuntimeError("kernel panic")),
    ),
    "inference_failed-wrong-output-shape": Case(
        status=500,
        code="inference_failed",
        title="Internal server error",
        detail="The model could not produce a valid prediction.",
        instance=PREDICT_CREDIT,
        trigger=failing_runtime(wrong_rows=True),
    ),
    "internal_error-unhandled-exception": Case(
        status=500,
        code="internal_error",
        title="Internal server error",
        detail="An unexpected error occurred.",
        instance="/boom",
        trigger=unhandled_exception,
    ),
    # ----- not_found / method_not_allowed (Starlette's own, mapped by the HTTPException handler)
    "not_found-unknown-path": Case(
        **NOT_FOUND,
        code="not_found",
        detail="No such resource.",
        instance="/nope",
        trigger=get("/nope"),
    ),
    "not_found-unknown-path-needs-no-key": Case(
        **NOT_FOUND,
        code="not_found",
        detail="No such resource.",
        instance="/v1/nope",
        trigger=get("/v1/nope", authed=False),
    ),
    "not_found-docs-are-not-served": Case(
        **NOT_FOUND,
        code="not_found",
        detail="No such resource.",
        instance="/docs",
        trigger=get("/docs"),
    ),
    "not_found-metrics-is-not-on-the-public-port": Case(
        **NOT_FOUND,
        code="not_found",
        detail="No such resource.",
        instance="/metrics",
        trigger=get("/metrics"),
    ),
    "method_not_allowed-delete-models": Case(
        **METHOD,
        instance="/v1/models",
        allow="GET",
        trigger=request("DELETE", "/v1/models"),
    ),
    "method_not_allowed-post-without-key": Case(
        **METHOD,
        instance="/v1/models",
        allow="GET",
        trigger=request("POST", "/v1/models", authed=False),
    ),
    "method_not_allowed-post-health": Case(
        **METHOD,
        instance="/health/live",
        allow="GET",
        trigger=request("POST", "/health/live", authed=False),
    ),
    "method_not_allowed-get-predictions": Case(
        **METHOD,
        instance=PREDICT_CREDIT,
        allow="POST",
        trigger=request("GET", PREDICT_CREDIT),
    ),
    "method_not_allowed-options": Case(
        **METHOD,
        instance="/v1/models",
        allow="GET",
        trigger=request("OPTIONS", "/v1/models"),
    ),
    # ----- http_<n> (any other HTTPException: only ``Allow`` survives, details are replaced)
    "http_400-multipart-without-boundary": Case(
        **HTTP_400,
        instance=PREDICT_CANDLESTICK,
        trigger=post_raw(PREDICT_CANDLESTICK, b"abc", "multipart/form-data"),
    ),
    "http_400-malformed-multipart-body": Case(
        **HTTP_400,
        instance=PREDICT_CANDLESTICK,
        trigger=post_raw(PREDICT_CANDLESTICK, b"junk", "multipart/form-data; boundary=xx"),
    ),
    "http_400-two-files": Case(
        **HTTP_400,
        instance=PREDICT_CANDLESTICK,
        trigger=lambda c: c.make_client().post(
            PREDICT_CANDLESTICK,
            headers=c.auth,
            files=[("file", ("a.png", c.png, "image/png")), ("file", ("b.png", c.png, "image/png"))],
        ),
    ),
    "http_418-unknown-4xx-gets-the-generic-title": Case(
        status=418,
        code="http_418",
        title="Error",
        detail="The request could not be processed.",
        instance="/synthetic",
        trigger=raise_http(418, detail="i am a teapot"),
    ),
    "http_502-5xx-has-no-detail": Case(
        status=502,
        code="http_502",
        title="Error",
        detail=None,
        instance="/synthetic",
        trigger=raise_http(502, detail="upstream said something private"),
    ),
    "http_409-title-is-generic-and-detail-replaced": Case(
        status=409,
        code="http_409",
        title="Error",
        detail="The request could not be processed.",
        instance="/synthetic",
        trigger=raise_http(409, detail="conflict with /srv/secret"),
    ),
    "http_429-raised-by-hand-loses-its-retry-after": Case(
        status=429,
        code="http_429",
        title="Too many requests",
        detail="The request could not be processed.",
        instance="/synthetic",
        trigger=raise_http(429, headers={"Retry-After": "5", "X-Leak": "1"}),
    ),
    "http_404-raised-by-hand-gets-the-fixed-detail": Case(
        **NOT_FOUND,
        code="not_found",
        detail="No such resource.",
        instance="/synthetic",
        trigger=raise_http(404, detail="the real reason"),
    ),
    "http_405-raised-by-hand-keeps-only-allow": Case(
        **METHOD,
        instance="/synthetic",
        allow="GET, POST",
        trigger=raise_http(405, headers={"Allow": "GET, POST", "X-Leak": "1"}),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_problem_is_exactly_what_the_catalogue_says(name: str, ctx: Ctx) -> None:
    case = CASES[name]
    response = case.trigger(ctx)
    problem_body(
        response,
        status=case.status,
        code=case.code,
        title=case.title,
        detail=case.detail,
        instance=case.instance,
        errors=case.errors,
    )
    assert_conditional_headers(
        response,
        www_authenticate=case.www_authenticate,
        allow=case.allow,
        retry_after=case.retry_after,
    )
    assert "x-leak" not in response.headers
    assert "server" not in response.headers


def test_rate_limited_retry_after_is_within_the_window(ctx: Ctx) -> None:
    response = rate_limited(ctx)
    assert 1 <= int(response.headers["retry-after"]) <= 60


def test_overloaded_retry_after_is_the_queue_timeout_rounded_up(ctx: Ctx) -> None:
    assert overloaded(ctx).headers["retry-after"] == "1"  # ceil(0.1 s)


def test_every_code_of_the_catalogue_is_covered() -> None:
    assert {case.code for case in CASES.values()} == {
        "unauthorized",
        "model_not_found",
        "model_not_available",
        "invalid_input",
        "validation_error",
        "unsupported_media_type",
        "payload_too_large",
        "rate_limited",
        "overloaded",
        "not_ready",
        "inference_failed",
        "internal_error",
        "not_found",
        "method_not_allowed",
        "http_400",
        "http_409",
        "http_418",
        "http_429",
        "http_502",
    }


def test_a_record_with_every_field_missing_reports_each_one(ctx: Ctx) -> None:
    response = ctx.make_client().post(PREDICT_CREDIT, headers=ctx.auth, json={"instances": [{}]})
    body = problem_body(
        response,
        status=422,
        code="invalid_input",
        title="Validation failed",
        detail="21 validation error(s); see 'errors'",
        instance=PREDICT_CREDIT,
        errors=response.json()["errors"],
    )
    assert len(body["errors"]) == 21
    assert {(tuple(e["loc"][:2]), e["code"], e["message"]) for e in body["errors"]} == {
        (("instances", 0), "required", "field required")
    }


def test_head_is_not_served_and_answers_405_without_a_body(ctx: Ctx) -> None:
    client = ctx.make_client()
    for path in ("/health/live", "/v1/models"):
        response = client.head(path, headers=ctx.auth)
        assert response.status_code == 405
        assert response.headers["allow"] == "GET"
        assert response.headers["content-type"] == "application/problem+json"
        assert response.content == b""


def test_the_documented_503_of_the_readiness_probe_has_no_retry_after_today(ctx: Ctx) -> None:
    """The OpenAPI document promises ``Retry-After`` on this 503; the handler does not send it."""
    client = ctx.make_empty_artifacts_client()
    documented = client.get("/openapi.json").json()["paths"]["/health/ready"]["get"]["responses"]
    assert "Retry-After" in documented["503"]["headers"]
    assert "retry-after" not in client.get("/health/ready").headers


def test_without_artifacts_the_app_is_alive_but_not_ready(ctx: Ctx) -> None:
    client = ctx.make_empty_artifacts_client()
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 503
    listing = client.get("/v1/models", headers=ctx.auth)
    assert (listing.status_code, listing.json()) == (200, {"models": []})
    assert client.get(f"/v1/models/{CREDIT}", headers=ctx.auth).json()["code"] == "model_not_found"
