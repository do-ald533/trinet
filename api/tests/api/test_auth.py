from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from tests.conftest import API_KEY

PROTECTED = [
    ("GET", "/v1/models"),
    ("GET", "/v1/models/credit-default-mlp"),
    ("POST", "/v1/models/credit-default-mlp/predictions"),
    ("GET", "/v1/models/no-such-model"),  # auth is checked before the model lookup
    ("POST", "/v1/models/no-such-model/predictions"),
]
BAD_KEYS = [
    None,
    "",
    " ",
    "wrong-key-0123456789abcdef",
    API_KEY[:-1],
    API_KEY + "x",
    API_KEY.upper(),
    " " + API_KEY + " ",
    "é" * 20,
]


def call(client: TestClient, method: str, path: str, key: str | None) -> tuple[int, dict[str, str]]:
    headers = {} if key is None else {"X-API-Key": key}
    try:
        response = client.request(method, path, headers=headers, json={"instances": []})
    except UnicodeEncodeError:  # non-latin-1 header values cannot even be sent
        return 401, {}
    return response.status_code, dict(response.headers)


@pytest.mark.parametrize(("method", "path"), PROTECTED)
@pytest.mark.parametrize("key", BAD_KEYS)
def test_missing_or_wrong_key_is_401(
    client: TestClient, method: str, path: str, key: str | None
) -> None:
    status, headers = call(client, method, path, key)
    assert status == 401
    if headers:
        assert headers["www-authenticate"] == "ApiKey"
        assert headers["content-type"] == "application/problem+json"


def test_the_401_body_is_a_problem_document_and_leaks_nothing(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"X-API-Key": "wrong-key-0123456789abcdef"})
    body = response.json()
    assert body["status"] == 401
    assert body["code"] == "unauthorized"
    assert body["instance"] == "/v1/models"
    assert "wrong-key" not in response.text
    assert API_KEY not in response.text


def test_the_key_is_only_accepted_in_the_header(client: TestClient) -> None:
    assert client.get("/v1/models", params={"api_key": API_KEY}).status_code == 401
    assert client.get("/v1/models", params={"X-API-Key": API_KEY}).status_code == 401
    assert (
        client.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 401
    )


def test_the_right_key_is_accepted(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get("/v1/models", headers=auth).status_code == 200


def test_health_and_openapi_need_no_key(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_an_app_without_a_key_refuses_to_start(
    settings_factory: Callable[..., object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trinet_api.interface.http.app import create_app
    from trinet_api.settings import ConfigurationError

    settings = settings_factory(api_key=None)
    with pytest.raises(ConfigurationError):
        create_app(settings)  # type: ignore[arg-type]
