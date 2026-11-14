"""Property-based conformance of the running app to its own OpenAPI document (Schemathesis).

The app is exercised in-process (ASGI, no sockets) with the API key set, the dummy artifacts
loaded and a generous rate limit, ~50 generated cases per operation (``HYPOTHESIS_PROFILE=dev``
raises it). Everything the generated traffic provokes must be a documented status, with a
documented content type, headers and body schema, and never a 5xx.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import schemathesis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from schemathesis.checks import CHECKS
from schemathesis.specs.openapi import checks as openapi_checks
from tests.conftest import API_KEY, make_settings

from trinet_api.domain.entities import ModelRegistry
from trinet_api.interface.http.app import create_app

# Generated "valid" bodies satisfy the generic JSON schema but not the per-model manifest
# (field names/ranges live in /v1/models/{name}, not in the OpenAPI document), so a 422 on
# schema-valid data is correct behaviour here.
EXCLUDED = [openapi_checks.positive_data_acceptance]
EXAMPLES = 50


@pytest.fixture(scope="module")
def running_app(registry: ModelRegistry, artifacts_root: Path) -> Iterator[FastAPI]:
    app = create_app(
        make_settings(artifacts_root, rate_limit="100000000/minute", queue_timeout_s=2.0),
        registry=registry,
    )
    with TestClient(app):  # runs the lifespan so app.state.service exists for the ASGI client
        yield app


@pytest.fixture(scope="module")
def api_schema(running_app: FastAPI) -> Any:
    return schemathesis.openapi.from_asgi("/openapi.json", running_app)


schema = schemathesis.pytest.from_fixture("api_schema")


@schema.parametrize()
@settings(
    max_examples=EXAMPLES,
    deadline=None,
    suppress_health_check=list(HealthCheck),
    derandomize=True,
)
def test_the_api_conforms_to_its_openapi_document(case: Any) -> None:
    case.call_and_validate(headers={"X-API-Key": API_KEY}, excluded_checks=EXCLUDED)


def test_the_checks_in_use_are_the_expected_ones() -> None:
    """Guards the exclusion list: renaming a check upstream must not silently disable it."""
    names = {check.__name__ for check in CHECKS.get_all()}
    assert "not_a_server_error" in names
    assert {c.__name__ for c in EXCLUDED} == {"positive_data_acceptance"}
