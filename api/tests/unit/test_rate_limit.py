"""The rate limiter as a plain ASGI component (no FastAPI involved)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from limits import parse_many
from starlette.types import Message, Receive, Scope, Send

from trinet_api.interface.http.middleware import RateLimitMiddleware

KEY = b"a-valid-api-key-0123456789"


async def ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def scope(
    path: str = "/v1/models",
    *,
    key: bytes | None = None,
    client: tuple[str, int] | None = ("10.0.0.1", 5000),
    kind: str = "http",
) -> Scope:
    headers = [] if key is None else [(b"x-api-key", key)]
    return {"type": kind, "path": path, "headers": headers, "client": client, "method": "GET"}


async def hit(app: RateLimitMiddleware, sc: Scope) -> tuple[int, dict[str, str]]:
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(sc, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}


def call(app: RateLimitMiddleware, sc: Scope) -> tuple[int, dict[str, str]]:
    return asyncio.run(hit(app, sc))


def make(rate: str, **kwargs: Any) -> RateLimitMiddleware:
    return RateLimitMiddleware(ok_app, limits=parse_many(rate), api_key=KEY, **kwargs)


def test_requests_over_the_limit_get_429_with_a_sane_retry_after() -> None:
    app = make("2/minute")
    assert [(call(app, scope(key=KEY)))[0] for _ in range(2)] == [200, 200]
    status, headers = call(app, scope(key=KEY))
    assert status == 429
    assert 1 <= int(headers["retry-after"]) <= 60
    assert headers["content-type"] == "application/problem+json"


def test_every_limit_of_a_compound_rule_is_enforced() -> None:
    app = make("2/second; 3/minute")
    codes = [(call(app, scope(key=KEY)))[0] for _ in range(4)]
    assert codes == [200, 200, 429, 429]


def test_valid_key_and_clients_have_separate_buckets() -> None:
    app = make("1/minute")
    assert (call(app, scope(key=KEY)))[0] == 200
    assert (call(app, scope(key=KEY)))[0] == 429
    assert (call(app, scope(key=b"guess", client=("10.0.0.1", 1))))[0] == 200
    assert (call(app, scope(key=b"other-guess", client=("10.0.0.1", 2))))[0] == 429
    assert (call(app, scope(client=("10.0.0.2", 1))))[0] == 200  # another address


def test_requests_without_a_peer_address_share_one_bucket() -> None:
    app = make("1/minute")
    assert (call(app, scope(client=None)))[0] == 200
    assert (call(app, scope(client=None)))[0] == 429


def test_exempt_paths_are_never_counted() -> None:
    app = make("1/minute", exempt_paths=["/health/live"])
    assert {(call(app, scope("/health/live")))[0] for _ in range(5)} == {200}
    assert (call(app, scope()))[0] == 200


def test_non_http_scopes_pass_straight_through() -> None:
    app = make("1/minute")
    seen: list[str] = []

    async def lifespan_app(sc: Scope, receive: Receive, send: Send) -> None:
        seen.append(sc["type"])

    app.app = lifespan_app
    for _ in range(3):
        asyncio.run(app(scope(kind="lifespan"), None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan"] * 3


def test_the_key_never_appears_in_the_bucket_name() -> None:
    app = make("5/minute")
    assert KEY.decode() not in app.bucket(scope(key=KEY))
    assert app.bucket(scope(key=KEY)).startswith("key:")
    assert app.bucket(scope()).startswith("ip:")


@pytest.mark.parametrize("rate", ["", "x", "10/decade"])
def test_invalid_rules_are_rejected_by_the_parser(rate: str) -> None:
    with pytest.raises(ValueError):
        parse_many(rate)
