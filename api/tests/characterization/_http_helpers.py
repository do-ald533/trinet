"""Helpers shared by the HTTP characterization tests (owned by R0-A).

Deliberately free of ``trinet_api`` imports and with every expected value written out as a
literal (not imported from the middleware module): a refactor that moves or rewrites the
production constants must not be able to move the expectation with it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from httpx2 import Response

PROBLEM_JSON = "application/problem+json"
URN_PREFIX = "urn:trinet:problem:"

#: ``SECURITY_HEADERS`` of the middleware, spelled out (lower-case names: httpx lower-cases).
SECURITY_HEADERS: dict[str, str] = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
    "cross-origin-resource-policy": "same-origin",
    "strict-transport-security": "max-age=63072000; includeSubDomains",
    "cache-control": "no-store",
}

#: What a client-supplied ``X-Request-ID`` must match to be echoed back.
ECHOED_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,64}")
#: What the server generates otherwise (``uuid.uuid4().hex``).
GENERATED_REQUEST_ID = re.compile(r"[0-9a-f]{32}")

#: Canonical key order of the problem+json bodies (they are serialised from a model).
PROBLEM_KEY_ORDER = ("type", "title", "status", "detail", "instance", "code", "request_id", "errors")

CREDIT = "credit-default-mlp"
VOLATILITY = "volatility-mlp"
CANDLESTICK = "candlestick-cnn"
PREDICT_CREDIT = f"/v1/models/{CREDIT}/predictions"
PREDICT_CANDLESTICK = f"/v1/models/{CANDLESTICK}/predictions"
JSON_CT = {"Content-Type": "application/json"}


def assert_envelope(response: Response) -> str:
    """Security headers and a request id on the response; returns the request id."""
    for name, value in SECURITY_HEADERS.items():
        assert response.headers.get(name) == value, (name, response.headers.get(name))
    request_id = response.headers["x-request-id"]
    assert ECHOED_REQUEST_ID.fullmatch(request_id), request_id
    return request_id


def assert_conditional_headers(
    response: Response, *, www_authenticate: str | None, allow: str | None, retry_after: bool
) -> None:
    """The three headers that exist only on some statuses are present exactly when expected."""
    assert response.headers.get("www-authenticate") == www_authenticate
    assert response.headers.get("allow") == allow
    value = response.headers.get("retry-after")
    if retry_after:
        assert value is not None
        assert value.isdigit(), value
        assert int(value) >= 1
    else:
        assert value is None, value


def problem_body(
    response: Response,
    *,
    status: int,
    code: str,
    title: str,
    detail: str | None,
    instance: str,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assert the whole problem+json envelope and return the parsed body."""
    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_JSON
    request_id = assert_envelope(response)
    body: dict[str, Any] = response.json()
    expected: dict[str, Any] = {
        "type": f"{URN_PREFIX}{code}",
        "title": title,
        "status": status,
        "instance": instance,
        "code": code,
        "request_id": request_id,
    }
    if detail is not None:
        expected["detail"] = detail
    if errors is not None:
        expected["errors"] = errors
    assert body == expected
    assert list(body) == [k for k in PROBLEM_KEY_ORDER if k in expected]
    return body


def snapshot_update_requested() -> bool:
    return os.environ.get("UPDATE_SNAPSHOTS") == "1"


def dump_json(document: Any) -> str:
    """The on-disk format of every snapshot of this directory (stable, diff-friendly)."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
