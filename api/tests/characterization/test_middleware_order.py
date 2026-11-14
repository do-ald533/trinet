"""Black-box facts about the ORDER in which the API decides, pinned from the outside.

Outermost first, today: request context (id, security headers, 500 net) -> rate limit -> body
limit -> routing (405/404, then the router dependencies: API key) -> path validation -> model
lookup -> media type -> body parse. Everything here is observed over HTTP only, so it must hold
whatever the package layout (R0 of docs/refactor-plan.md, section 13).

Several facts are surprising on purpose and are pinned as they are: a body over the limit is a
413 even without an API key, a flood of unknown paths is rate limited like any other traffic,
the trailing-slash redirect is answered before authentication, and the probe exemption is an
exact path match.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from tests.characterization._http_helpers import (
    ECHOED_REQUEST_ID,
    GENERATED_REQUEST_ID,
    JSON_CT,
    PREDICT_CANDLESTICK,
    PREDICT_CREDIT,
    SECURITY_HEADERS,
    assert_envelope,
)

LIMIT = 1024
BIG = b" " * 2000  # over LIMIT, and not valid JSON: parsing it would be a 422


def chunked(parts: int = 20, size: int = 1000) -> Iterator[bytes]:
    """A body with no Content-Length (chunked), larger than LIMIT."""
    for _ in range(parts):
        yield b" " * size


def codes(responses: list[Response]) -> list[int]:
    return [r.status_code for r in responses]


# --------------------------------------------------------------------------- 429 first


def test_a_flood_of_bad_keys_is_401_until_the_bucket_is_empty_then_429(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(rate_limit="3/minute")
    responses = [
        client.get("/v1/models", headers={"X-API-Key": f"guess-{i:016d}"}) for i in range(6)
    ]
    assert codes(responses) == [401, 401, 401, 429, 429, 429]
    assert [r.json()["code"] for r in responses] == ["unauthorized"] * 3 + ["rate_limited"] * 3


def test_rate_limiting_precedes_routing_so_404_and_405_are_counted_too(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="2/minute")
    unknown = [client.get("/nope", headers=auth) for _ in range(4)]
    assert codes(unknown) == [404, 404, 429, 429]
    client = make_client(rate_limit="2/minute")
    wrong_verb = [client.delete("/v1/models", headers=auth) for _ in range(3)]
    assert codes(wrong_verb) == [405, 405, 429]


def test_rate_limiting_precedes_the_body_limit(make_client: Callable[..., TestClient]) -> None:
    client = make_client(rate_limit="1/minute", max_json_bytes=LIMIT)
    first = client.post(PREDICT_CREDIT, headers=JSON_CT, content=BIG)
    second = client.post(PREDICT_CREDIT, headers=JSON_CT, content=BIG)
    assert codes([first, second]) == [413, 429]  # the 413 itself consumed the only hit


def test_the_valid_key_has_its_own_bucket_apart_from_the_attackers(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="2/minute")
    assert codes([client.get("/v1/models", headers={"X-API-Key": "x" * 20}) for _ in range(3)]) == [
        401,
        401,
        429,
    ]
    assert codes([client.get("/v1/models", headers=auth) for _ in range(3)]) == [200, 200, 429]


# --------------------------------------------------------------------------- probes exempt


def test_health_probes_are_exempt_from_the_rate_limit(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(rate_limit="1/minute")
    assert {client.get("/health/live").status_code for _ in range(8)} == {200}
    assert {client.get("/health/ready").status_code for _ in range(8)} == {200}
    assert client.get("/v1/models", headers=auth).status_code == 200  # nothing was consumed
    assert client.get("/v1/models", headers=auth).status_code == 429


def test_the_probe_exemption_is_by_exact_path_not_by_method(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(rate_limit="1/minute")
    assert {client.post("/health/live").status_code for _ in range(5)} == {405}
    assert {client.delete("/health/ready").status_code for _ in range(5)} == {405}


def test_the_trailing_slash_variant_of_a_probe_is_not_exempt(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(rate_limit="1/minute")
    first = client.get("/health/live/", follow_redirects=False)
    second = client.get("/health/live/", follow_redirects=False)
    assert codes([first, second]) == [307, 429]


def test_the_openapi_document_is_rate_limited_like_everything_else(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(rate_limit="2/minute")
    assert codes([client.get("/openapi.json") for _ in range(3)]) == [200, 200, 429]


# --------------------------------------------------------------------------- 413 first


def over_limit_requests() -> list[Any]:
    """(method, path, headers, authed): every one of them carries a 2000-byte body."""
    return [
        pytest.param("POST", PREDICT_CREDIT, JSON_CT, True, id="valid-request"),
        pytest.param("POST", PREDICT_CREDIT, JSON_CT, False, id="no-api-key-before-401"),
        pytest.param(
            "POST", "/v1/models/nope/predictions", JSON_CT, True, id="unknown-model-before-404"
        ),
        pytest.param(
            "POST", "/v1/models/NOPE/predictions", JSON_CT, True, id="bad-path-before-422"
        ),
        pytest.param(
            "POST",
            PREDICT_CREDIT,
            {"Content-Type": "text/plain"},
            True,
            id="wrong-media-type-before-415",
        ),
        pytest.param("POST", PREDICT_CANDLESTICK, JSON_CT, True, id="image-model-before-415"),
        pytest.param("POST", "/nope", JSON_CT, True, id="unknown-path-before-404"),
        pytest.param("DELETE", "/v1/models", JSON_CT, True, id="wrong-verb-before-405"),
        pytest.param("PUT", "/health/live", JSON_CT, False, id="probe-path-with-a-body"),
    ]


@pytest.mark.parametrize(("method", "path", "headers", "authed"), over_limit_requests())
def test_an_oversized_body_is_413_before_any_other_check(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    method: str,
    path: str,
    headers: dict[str, str],
    authed: bool,
) -> None:
    client = make_client(max_json_bytes=LIMIT)
    sent = {**headers, **(auth if authed else {})}
    response = client.request(method, path, headers=sent, content=BIG)
    assert response.status_code == 413
    body = response.json()
    assert body["code"] == "payload_too_large"
    assert body["detail"] == f"The request body exceeds {LIMIT} bytes."
    assert body["instance"] == path
    assert_envelope(response)


def test_the_body_limit_boundary_is_inclusive(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(max_json_bytes=LIMIT)
    at_limit = client.post(PREDICT_CREDIT, headers={**auth, **JSON_CT}, content=b" " * LIMIT)
    over = client.post(PREDICT_CREDIT, headers={**auth, **JSON_CT}, content=b" " * (LIMIT + 1))
    assert (at_limit.status_code, at_limit.json()["code"]) == (422, "invalid_input")
    assert (over.status_code, over.json()["code"]) == (413, "payload_too_large")


def test_the_limit_depends_on_the_content_type_json_versus_multipart(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(max_json_bytes=LIMIT, max_upload_bytes=64 * 1024)
    as_json = client.post(PREDICT_CANDLESTICK, headers={**auth, **JSON_CT}, content=BIG)
    as_form = client.post(
        PREDICT_CANDLESTICK,
        headers=auth,
        files={"file": ("scene.png", b"\x89PNG\r\n\x1a\n" + b"0" * 4000, "image/png")},
    )
    assert as_json.status_code == 413
    assert (as_form.status_code, as_form.json()["code"]) == (422, "invalid_input")  # past the limit
    strict = make_client(max_upload_bytes=LIMIT)
    too_big = strict.post(
        PREDICT_CANDLESTICK, headers=auth, files={"file": ("a.png", b"\x89PNG" + b"0" * 3000, "x")}
    )
    assert too_big.status_code == 413


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_only_methods_that_carry_a_body_are_size_limited(
    make_client: Callable[..., TestClient], auth: dict[str, str], method: str
) -> None:
    client = make_client(max_json_bytes=LIMIT)
    response = client.request(method, "/v1/models", headers=auth, content=BIG)
    assert response.status_code == (200 if method == "GET" else 405)


def test_a_streamed_body_is_only_checked_when_the_handler_reads_it(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """No Content-Length (chunked): the 413 appears at the first read, after every other check."""
    client = make_client(max_json_bytes=LIMIT)
    json_headers = {**auth, **JSON_CT}
    no_key = client.post(PREDICT_CREDIT, headers=JSON_CT, content=chunked())
    unknown = client.post("/v1/models/nope/predictions", headers=json_headers, content=chunked())
    bad_path = client.post("/v1/models/NOPE/predictions", headers=json_headers, content=chunked())
    plain = client.post(
        PREDICT_CREDIT, headers={**auth, "Content-Type": "text/plain"}, content=chunked()
    )
    read = client.post(PREDICT_CREDIT, headers=json_headers, content=chunked())
    assert [r.json()["code"] for r in (no_key, bad_path, unknown, plain, read)] == [
        "unauthorized",
        "validation_error",
        "model_not_found",
        "unsupported_media_type",
        "payload_too_large",
    ]
    assert [r.status_code for r in (no_key, bad_path, unknown, plain, read)] == [
        401,
        422,
        404,
        415,
        413,
    ]
    assert read.json()["detail"] == f"The request body exceeds {LIMIT} bytes."


# --------------------------------------------------------------------------- the ladder


@dataclass(frozen=True)
class Step:
    method: str
    path: str
    authed: bool = True
    content_type: str | None = "application/json"
    body: bytes = b"{nope"
    broken_registry: bool = False


LADDER: dict[str, tuple[Step, int, str]] = {
    # 1. unknown route / wrong verb are decided by routing, before any dependency runs
    "unknown-route-needs-no-key": (Step("GET", "/v1/nope", authed=False), 404, "not_found"),
    "wrong-verb-needs-no-key": (Step("POST", "/v1/models", authed=False), 405, "method_not_allowed"),
    # 2. the API key beats everything that follows
    "key-before-path-validation": (
        Step("POST", "/v1/models/NOPE/predictions", authed=False, content_type="text/plain"),
        401,
        "unauthorized",
    ),
    "key-before-model-lookup": (
        Step("POST", "/v1/models/nope/predictions", authed=False),
        401,
        "unauthorized",
    ),
    "key-before-media-type": (
        Step("POST", PREDICT_CREDIT, authed=False, content_type="text/plain"),
        401,
        "unauthorized",
    ),
    "key-before-body-parse": (Step("POST", PREDICT_CREDIT, authed=False), 401, "unauthorized"),
    "key-before-model-detail-lookup": (
        Step("GET", "/v1/models/nope", authed=False, content_type=None),
        401,
        "unauthorized",
    ),
    "key-before-path-validation-on-get": (
        Step("GET", "/v1/models/NOPE", authed=False, content_type=None),
        401,
        "unauthorized",
    ),
    # 3. then path validation, then the model lookup, then the media type, then the body
    "path-validation-before-model-lookup": (
        Step("POST", "/v1/models/NOPE/predictions", content_type="text/plain"),
        422,
        "validation_error",
    ),
    "model-lookup-before-media-type": (
        Step("POST", "/v1/models/nope/predictions", content_type="text/plain"),
        404,
        "model_not_found",
    ),
    "unavailable-model-before-media-type": (
        Step("POST", "/v1/models/broken-model/predictions", content_type="text/plain", broken_registry=True),
        404,
        "model_not_available",
    ),
    "media-type-before-body-parse": (
        Step("POST", PREDICT_CREDIT, content_type="text/plain"),
        415,
        "unsupported_media_type",
    ),
    "image-media-type-before-multipart-parse": (
        Step("POST", PREDICT_CANDLESTICK),
        415,
        "unsupported_media_type",
    ),
    "tabular-body-parse": (Step("POST", PREDICT_CREDIT), 422, "invalid_input"),
    "multipart-parse": (
        Step("POST", PREDICT_CANDLESTICK, content_type="multipart/form-data"),
        400,
        "http_400",
    ),
}


@pytest.mark.parametrize("name", list(LADDER))
def test_the_checks_run_in_the_documented_order(
    name: str,
    make_client: Callable[..., TestClient],
    fake_registry: Callable[..., Any],
    auth: dict[str, str],
) -> None:
    step, status, code = LADDER[name]
    headers = dict(auth) if step.authed else {}
    if step.content_type is not None:
        headers["Content-Type"] = step.content_type
    client = (
        make_client(registry_override=fake_registry()) if step.broken_registry else make_client()
    )
    content = step.body if step.method == "POST" else None
    response = client.request(step.method, step.path, headers=headers, content=content)
    assert (response.status_code, response.json()["code"]) == (status, code)


def test_the_trailing_slash_redirect_is_answered_before_authentication(
    client: TestClient,
) -> None:
    for path in ("/v1/models/", "/v1/models/credit-default-mlp/", "/health/live/"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 307, path
        assert response.headers["location"] == f"http://testserver{path.rstrip('/')}", path
        assert response.content == b""
        assert_envelope(response)
    followed = client.get("/v1/models/")  # followed by the client: now the key is needed
    assert followed.status_code == 401


# --------------------------------------------------------------------------- envelope


@pytest.fixture
def scenario_clients(
    make_client: Callable[..., TestClient],
    make_empty_artifacts_client: Callable[..., TestClient],
    auth: dict[str, str],
) -> dict[int, Callable[[], Response]]:
    """One request per status the API answers with (including the 307 and the 500 net)."""

    def limited() -> Response:
        client = make_client(rate_limit="1/minute")
        client.get("/v1/models", headers=auth)
        return client.get("/v1/models", headers=auth)

    def boom() -> Response:
        client = make_client()

        def raises() -> None:
            raise RuntimeError("internal detail that must never leak")

        client.app.add_api_route("/boom", raises)  # type: ignore[attr-defined]
        return client.get("/boom", headers=auth)

    plain = make_client()
    return {
        200: lambda: plain.get("/v1/models", headers=auth),
        307: lambda: plain.get("/v1/models/", follow_redirects=False),
        401: lambda: plain.get("/v1/models"),
        404: lambda: plain.get("/nope"),
        405: lambda: plain.delete("/v1/models", headers=auth),
        413: lambda: make_client(max_json_bytes=LIMIT).post(PREDICT_CREDIT, content=BIG),
        415: lambda: plain.post(PREDICT_CREDIT, headers=auth, content=b"{}"),
        422: lambda: plain.get("/v1/models/NOPE", headers=auth),
        429: limited,
        500: boom,
        503: lambda: make_empty_artifacts_client().get("/health/ready"),
    }


@pytest.mark.parametrize("status", [200, 307, 401, 404, 405, 413, 415, 422, 429, 500, 503])
def test_security_headers_and_a_request_id_are_on_every_status(
    scenario_clients: dict[int, Callable[[], Response]], status: int
) -> None:
    response = scenario_clients[status]()
    assert response.status_code == status
    request_id = assert_envelope(response)
    if status >= 400:
        assert response.json()["request_id"] == request_id
    assert set(SECURITY_HEADERS) <= set(response.headers)


# --------------------------------------------------------------------------- request ids


@pytest.mark.parametrize(
    "supplied",
    [
        "abcdefgh",  # 8 characters: the shortest accepted
        "a" * 64,  # 64 characters: the longest accepted
        "req-0001",
        "Req_ID.1.2-3",
        "00000000",
        "A" * 8,
    ],
)
def test_a_well_formed_request_id_is_echoed(client: TestClient, supplied: str) -> None:
    assert ECHOED_REQUEST_ID.fullmatch(supplied)
    response = client.get("/health/live", headers={"X-Request-ID": supplied})
    assert response.headers["x-request-id"] == supplied


@pytest.mark.parametrize(
    "supplied",
    [
        pytest.param(b"abcdefg", id="7-chars"),
        pytest.param(b"a" * 65, id="65-chars"),
        pytest.param(b"a" * 500, id="500-chars"),
        pytest.param(b"", id="empty"),
        pytest.param(b"abcdefgh ijkl", id="space"),
        pytest.param(b"abcdefgh/ijkl", id="slash"),
        pytest.param(b"abcdefgh:ijkl", id="colon"),
        pytest.param(b"abcdefgh,ijkl", id="comma"),
        pytest.param(b"abcdefgh+ijkl", id="plus"),
        pytest.param(b"abcdefgh=", id="equals"),
        pytest.param(b"abcdefgh\xe9", id="latin1-byte"),
        pytest.param("abcdefghé".encode(), id="utf8-char"),
        pytest.param(b"abcdefgh\t", id="tab"),
        pytest.param(b"<script>alert(1)</script>", id="markup"),
    ],
)
def test_a_malformed_request_id_is_replaced_by_a_generated_one(
    client: TestClient, supplied: bytes
) -> None:
    response = client.get("/health/live", headers=[(b"X-Request-ID", supplied)])
    echoed = response.headers["x-request-id"]
    assert echoed != supplied.decode("latin-1")
    assert GENERATED_REQUEST_ID.fullmatch(echoed)


def test_a_generated_request_id_is_a_fresh_uuid4_hex_each_time(client: TestClient) -> None:
    ids = [client.get("/health/live").headers["x-request-id"] for _ in range(5)]
    assert len(set(ids)) == 5
    for value in ids:
        assert GENERATED_REQUEST_ID.fullmatch(value)
        assert value[12] == "4"  # uuid version nibble
        assert value[16] in "89ab"  # uuid variant


def test_with_two_request_id_headers_the_first_one_wins(client: TestClient) -> None:
    response = client.get(
        "/health/live", headers=[("X-Request-ID", "first-id-0001"), ("X-Request-ID", "second-id-0002")]
    )
    assert response.headers["x-request-id"] == "first-id-0001"


def test_the_header_name_is_case_insensitive(client: TestClient) -> None:
    response = client.get("/health/live", headers={"x-request-id": "lower-case-0001"})
    assert response.headers["x-request-id"] == "lower-case-0001"


def test_the_supplied_id_ends_up_in_error_bodies_and_headers(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    supplied = {"X-Request-ID": "trace-me-0001"}
    plain = make_client()
    unknown = plain.get("/nope", headers={**auth, **supplied})
    over = make_client(max_json_bytes=LIMIT).post(
        PREDICT_CREDIT, headers={**supplied, **JSON_CT}, content=BIG
    )
    limited_client = make_client(rate_limit="1/minute")
    limited_client.get("/v1/models", headers=auth)
    limited = limited_client.get("/v1/models", headers={**auth, **supplied})
    for response, status in ((unknown, 404), (over, 413), (limited, 429)):
        assert response.status_code == status
        assert response.headers["x-request-id"] == "trace-me-0001"
        assert response.json()["request_id"] == "trace-me-0001"


def test_the_prediction_response_carries_the_same_request_id_as_the_header(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    response = client.post(
        PREDICT_CREDIT,
        headers={**auth, "X-Request-ID": "predict-0001"},
        json={"instances": [golden_records["credit-default-mlp"]]},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "predict-0001"
    assert response.json()["request_id"] == "predict-0001"
