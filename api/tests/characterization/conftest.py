"""Fixtures of the characterization tests (owned by R0-A; the other files only consume them).

``fake_registry``, ``golden_records`` and ``png_bytes`` live in ``tests/api/conftest.py``, which
pytest does not apply to this directory, so they are re-exported here. The one extra fixture
runs the real start-up path (lifespan, loader) against an artifacts directory that does not
exist, which is the only way to get a service that is alive but never ready.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.api.conftest import fake_registry, golden_records, png_bytes
from tests.conftest import make_settings

from trinet_api.interface.http.app import create_app

__all__ = ["fake_registry", "golden_records", "make_empty_artifacts_client", "png_bytes"]


@pytest.fixture
def make_empty_artifacts_client(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    """Factory of started clients whose artifacts directory is missing (no registry override)."""
    stack = ExitStack()

    def factory(**overrides: Any) -> TestClient:
        app = create_app(make_settings(tmp_path / "no-artifacts-here", **overrides))
        return stack.enter_context(TestClient(app, raise_server_exceptions=False))

    yield factory
    stack.close()
