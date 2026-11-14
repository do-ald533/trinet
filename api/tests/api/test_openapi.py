"""The published OpenAPI document: it must describe every answer the API can give."""

from __future__ import annotations

import difflib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

SNAPSHOT = Path(__file__).with_name("openapi.snapshot.json")
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


@pytest.fixture(scope="module")
def spec(registry: Any, artifacts_root: Path) -> dict[str, Any]:
    from tests.conftest import make_settings

    from trinet_api.interface.http.app import create_app

    app = create_app(make_settings(artifacts_root), registry=registry)
    with TestClient(app) as client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        document: dict[str, Any] = response.json()
        return document


def operations(spec: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (path, method, op)
        for path, item in spec["paths"].items()
        for method, op in item.items()
        if method in HTTP_METHODS
    ]


def test_the_document_matches_the_committed_snapshot(spec: dict[str, Any]) -> None:
    """Unintended API changes show up as a diff; refresh with ``UPDATE_SNAPSHOTS=1 pytest``."""
    current = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.write_text(current)
    expected = SNAPSHOT.read_text()
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(True), current.splitlines(True), "snapshot", "current", n=2
        )
    )
    assert current == expected, f"OpenAPI drifted (UPDATE_SNAPSHOTS=1 to accept):\n{diff[:3000]}"


def test_expected_operations_exist(spec: dict[str, Any]) -> None:
    assert {(m.upper(), p) for p, m, _ in operations(spec)} == {
        ("GET", "/health/live"),
        ("GET", "/health/ready"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models/{name}"),
        ("POST", "/v1/models/{name}/predictions"),
    }
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "TriNet inference API"


def test_operation_ids_are_unique_and_snake_case(spec: dict[str, Any]) -> None:
    ids = [op["operationId"] for _, _, op in operations(spec)]
    assert len(set(ids)) == len(ids)
    assert all(i.islower() and " " not in i for i in ids)


def test_v1_operations_require_the_api_key_and_probes_do_not(spec: dict[str, Any]) -> None:
    schemes = spec["components"]["securitySchemes"]
    (name,) = schemes
    assert schemes[name]["type"] == "apiKey"
    assert schemes[name]["in"] == "header"
    assert schemes[name]["name"] == "X-API-Key"
    for path, _, op in operations(spec):
        if path.startswith("/v1"):
            assert op["security"] == [{name: []}], path
            assert {"401", "429"} <= set(op["responses"]), path
        else:
            assert "security" not in op, path


def test_every_error_response_is_problem_json(spec: dict[str, Any]) -> None:
    for path, method, op in operations(spec):
        for status, response in op["responses"].items():
            if (int(status) >= 400 and status != "422") or (
                status == "422" and "content" in response
            ):
                content = response["content"]
                assert list(content) == ["application/problem+json"], (path, method, status)
                ref = content["application/problem+json"]["schema"]["$ref"]
                assert ref.endswith("/Problem"), (path, method, status)


def test_status_codes_are_documented_where_the_api_can_return_them(spec: dict[str, Any]) -> None:
    predict = spec["paths"]["/v1/models/{name}/predictions"]["post"]["responses"]
    assert {"200", "401", "404", "413", "415", "422", "429", "503"} <= set(predict)
    ready = spec["paths"]["/health/ready"]["get"]["responses"]
    assert {"200", "503"} <= set(ready)
    assert "Retry-After" in predict["429"]["headers"]
    assert "Retry-After" in predict["503"]["headers"]
    assert "WWW-Authenticate" in predict["401"]["headers"]
    assert "X-Model-Version" in predict["200"]["headers"]


def test_the_prediction_request_body_documents_both_encodings(spec: dict[str, Any]) -> None:
    body = spec["paths"]["/v1/models/{name}/predictions"]["post"]["requestBody"]
    assert body["required"] is True
    assert set(body["content"]) == {"application/json", "multipart/form-data"}
    instances = body["content"]["application/json"]["schema"]["properties"]["instances"]
    assert instances["minItems"] == 1
    assert instances["maxItems"] == 256  # TRINET_MAX_BATCH, patched into the served document
    assert body["content"]["multipart/form-data"]["schema"]["required"] == ["file"]


def test_max_items_follows_the_configured_batch_limit(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(max_batch=7)
    operation = client.get("/openapi.json").json()["paths"]["/v1/models/{name}/predictions"]["post"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert schema["properties"]["instances"]["maxItems"] == 7


def test_model_name_parameter_is_constrained(spec: dict[str, Any]) -> None:
    (parameter,) = spec["paths"]["/v1/models/{name}"]["get"]["parameters"]
    assert parameter["in"] == "path"
    assert parameter["schema"]["pattern"] == r"^[a-z0-9][a-z0-9-]{0,62}$"


def test_the_document_is_cached_and_stable(client: TestClient) -> None:
    first = client.get("/openapi.json")
    assert first.headers["content-type"] == "application/json"
    assert client.get("/openapi.json").json() == first.json()
