"""Shared fixtures. Unit and API tests build their own artifacts in temporary directories."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import make_dummy_artifacts as dummy
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck
from hypothesis import settings as hypothesis_settings

from trinet_api.domain.entities import ModelRegistry
from trinet_api.infrastructure.loading import load_registry
from trinet_api.interface.http.app import create_app
from trinet_api.settings import Settings

hypothesis_settings.register_profile(
    "ci", max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
hypothesis_settings.register_profile("dev", max_examples=200, deadline=None)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

API_KEY = "test-api-key-0123456789abcdef"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def api_key() -> str:
    return API_KEY


@pytest.fixture(scope="session")
def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """All three dummy artifacts, generated once per session (and per xdist worker)."""
    root = tmp_path_factory.mktemp("artifacts")
    dummy.build_all(root)
    return root


@pytest.fixture
def artifacts_copy(artifacts_root: Path, tmp_path: Path) -> Path:
    """A private, mutable copy of the artifacts (for corruption tests)."""
    target = tmp_path / "artifacts"
    shutil.copytree(artifacts_root, target)
    return target


@pytest.fixture(scope="session")
def registry(artifacts_root: Path) -> ModelRegistry:
    return load_registry(artifacts_root)


def make_settings(artifacts_dir: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "artifacts_dir": artifacts_dir,
        "api_key": API_KEY,
        "api_key_file": None,
        "metrics_enabled": False,
        "rate_limit": "100000/minute",
        "queue_timeout_s": 0.2,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings_factory(artifacts_root: Path) -> Callable[..., Settings]:
    def factory(**overrides: Any) -> Settings:
        return make_settings(artifacts_root, **overrides)

    return factory


@pytest.fixture
def make_client(
    registry: ModelRegistry, settings_factory: Callable[..., Settings]
) -> Iterator[Callable[..., TestClient]]:
    """Factory of started ``TestClient`` s (lifespan run, torn down after the test)."""
    stack = ExitStack()

    def factory(*, registry_override: ModelRegistry | None = None, **overrides: Any) -> TestClient:
        app = create_app(settings_factory(**overrides), registry=registry_override or registry)
        return stack.enter_context(TestClient(app, raise_server_exceptions=False))

    yield factory
    stack.close()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
