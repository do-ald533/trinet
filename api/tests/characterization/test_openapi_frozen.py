"""The published OpenAPI document, frozen in two complementary ways (R0 of docs/refactor-plan.md).

``api/tests/api/test_openapi.py`` already snapshots the document, but it compares
``json.dumps(..., sort_keys=True)``, so it cannot see a change in key ORDER, and it must be
refreshed whenever a route is added. This module adds:

1. ``openapi.rawhash.json``: the SHA-256 (and length) of the RAW bytes of ``GET /openapi.json``.
   It catches anything that changes the served bytes: key order, whitespace, escaping, a
   different serializer. It is meant to be IDENTICAL through the pure-move phases (R1-R4), and
   it is EXPECTED to change deliberately when routes are added (R6). Refresh it, in that
   commit only, with ``UPDATE_SNAPSHOTS=1``.
2. ``openapi.frozen_operations.json``: a FROZEN SUBSET. For exactly the five operations that
   exist today (``health_live``, ``health_ready``, ``list_models``, ``get_model``,
   ``create_prediction``) it records method, path, tags, parameters, requestBody, responses
   (every status with its description, content types, schemas and headers) and security,
   plus the full transitive closure of the component schemas and security schemes they
   reference. The test extracts the same subset from the live document and compares it. New
   routes, new components and new tags elsewhere in the document do NOT touch it, so after the
   additive phases the raw hash may be refreshed while this subset must stay byte-identical:
   if it has to change, an existing operation's contract changed and that needs a decision.

Both files are produced from the app with default settings (``max_batch`` 256) and the package
version of ``trinet-api`` (``info.version``): a version bump or a FastAPI/Pydantic upgrade
legitimately changes the raw hash.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from tests.characterization._http_helpers import (
    dump_json,
    read_json,
    snapshot_update_requested,
)

HERE = Path(__file__).parent
RAW_HASH = HERE / "openapi.rawhash.json"
FROZEN = HERE / "openapi.frozen_operations.json"

FROZEN_OPERATION_IDS = (
    "health_live",
    "health_ready",
    "list_models",
    "get_model",
    "create_prediction",
)
HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
SCHEMA_REF = "#/components/schemas/"
OPERATION_KEYS = ("tags", "parameters", "requestBody", "responses", "security")


@pytest.fixture
def served(client: TestClient) -> Response:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    return response


@pytest.fixture
def document(served: Response) -> dict[str, Any]:
    parsed: dict[str, Any] = served.json()
    return parsed


# --------------------------------------------------------------------------- extraction


def _refs(node: Any) -> Iterator[str]:
    """Every ``$ref`` string below ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _refs(item)


def schema_closure(document: dict[str, Any], roots: list[Any]) -> dict[str, Any]:
    """The component schemas reachable from ``roots`` (transitively, through ``$ref``)."""
    schemas: dict[str, Any] = document.get("components", {}).get("schemas", {})
    wanted: list[str] = []
    pending = [ref for root in roots for ref in _refs(root)]
    while pending:
        ref = pending.pop()
        assert ref.startswith(SCHEMA_REF), f"unexpected reference kind: {ref}"
        name = ref.removeprefix(SCHEMA_REF)
        if name in wanted:
            continue
        assert name in schemas, f"dangling reference {ref}"
        wanted.append(name)
        pending.extend(_refs(schemas[name]))
    return {name: schemas[name] for name in sorted(wanted)}


def frozen_subset(document: dict[str, Any]) -> dict[str, Any]:
    """The frozen view of ``document`` (see the module docstring)."""
    found: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for path, item in document["paths"].items():
        for method in HTTP_METHODS:
            operation = item.get(method)
            if operation is not None and operation.get("operationId") in FROZEN_OPERATION_IDS:
                found[operation["operationId"]] = (method, path, operation)
    assert sorted(found) == sorted(FROZEN_OPERATION_IDS), "an operationId is gone or was renamed"

    operations: dict[str, Any] = {}
    for operation_id in FROZEN_OPERATION_IDS:
        method, path, operation = found[operation_id]
        operations[operation_id] = {
            "method": method,
            "path": path,
            **{key: operation.get(key) for key in OPERATION_KEYS},
        }
    schemes: dict[str, Any] = document.get("components", {}).get("securitySchemes", {})
    used = sorted(
        {name for op in operations.values() for req in op["security"] or [] for name in req}
    )
    return {
        "operations": operations,
        "components": {
            "schemas": schema_closure(document, list(operations.values())),
            "securitySchemes": {name: schemes[name] for name in used},
        },
    }


# --------------------------------------------------------------------------- tests


def test_the_raw_bytes_of_the_document_are_unchanged(served: Response) -> None:
    raw: bytes = served.content
    current = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if snapshot_update_requested():
        RAW_HASH.write_text(dump_json(current), encoding="utf-8")
    assert current == read_json(RAW_HASH), (
        "the bytes served at /openapi.json changed (key order, whitespace, content...). "
        "If routes were added on purpose: UPDATE_SNAPSHOTS=1 refreshes this hash, and "
        "test_the_frozen_operations_are_unchanged must still pass without a refresh."
    )


def test_the_frozen_operations_are_unchanged(document: dict[str, Any]) -> None:
    current = frozen_subset(document)
    if snapshot_update_requested():
        FROZEN.write_text(dump_json(current), encoding="utf-8")
    expected = read_json(FROZEN)
    diff = "".join(
        difflib.unified_diff(
            dump_json(expected).splitlines(True),
            dump_json(current).splitlines(True),
            "frozen",
            "current",
            n=2,
        )
    )
    assert current == expected, f"a frozen operation changed:\n{diff[:4000]}"


def test_the_subset_is_self_contained_and_covers_the_five_operations(
    document: dict[str, Any],
) -> None:
    subset = frozen_subset(document)
    assert [(o["method"], o["path"]) for o in subset["operations"].values()] == [
        ("get", "/health/live"),
        ("get", "/health/ready"),
        ("get", "/v1/models"),
        ("get", "/v1/models/{name}"),
        ("post", "/v1/models/{name}/predictions"),
    ]
    schemas = subset["components"]["schemas"]
    assert {r.removeprefix(SCHEMA_REF) for r in _refs(subset)} <= set(schemas)
    assert set(subset["components"]["securitySchemes"]) == {"ApiKeyAuth"}


def test_the_document_is_served_as_compact_json_with_a_stable_top_level_order(
    served: Response,
) -> None:
    raw: bytes = served.content
    assert raw.startswith(b'{"openapi":"3.1.0","info":')
    assert b"\n" not in raw
    parsed = json.loads(raw)
    assert list(parsed) == ["openapi", "info", "paths", "components"]
    assert list(parsed["paths"]) == [
        "/health/live",
        "/health/ready",
        "/v1/models",
        "/v1/models/{name}",
        "/v1/models/{name}/predictions",
    ]
    assert list(parsed["info"]) == ["title", "description", "version"]
