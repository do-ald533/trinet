"""API-key authentication (``X-API-Key``, constant-time comparison)."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    scheme_name="ApiKeyAuth",
    description="Shared secret configured through TRINET_API_KEY_FILE / TRINET_API_KEY.",
    auto_error=False,  # we raise our own problem+json 401
)


class Unauthorized(Exception):
    """Missing or wrong API key (HTTP 401)."""


def key_matches(supplied: str | None, expected: bytes) -> bool:
    if supplied is None:
        return False
    return secrets.compare_digest(supplied.encode("utf-8"), expected)


async def require_api_key(
    request: Request, supplied: Annotated[str | None, Security(api_key_header)]
) -> None:
    if not key_matches(supplied, request.app.state.api_key_bytes):
        raise Unauthorized
