"""Fixtures for the HTTP-level tests (TestClient against the real app factory)."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import make_dummy_artifacts as dummy
import pytest
from PIL import Image
from tests.fakes import FakeRuntime

from trinet_api.domain.entities import ModelBundle, ModelRegistry, UnavailableModel
from trinet_api.domain.ports import ModelRuntime


@pytest.fixture(scope="session")
def golden_records(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    """One valid raw record per tabular model (taken from the golden files)."""
    out = {}
    for name in ("credit-default-mlp", "volatility-mlp"):
        golden = json.loads(
            (artifacts_root / name / dummy.DUMMY_VERSION / "golden.json").read_text()
        )
        out[name] = golden["cases"][0]["record"]
    return out


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    """A small, valid PNG (the API resizes it to the model's input size)."""
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (20, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart(data: bytes, *, field: str = "file", filename: str = "scene.png") -> dict[str, Any]:
    return {"files": {field: (filename, data, "application/octet-stream")}}


@pytest.fixture
def fake_registry(registry: ModelRegistry) -> Callable[..., ModelRegistry]:
    """Registry with the real manifests but test-double runtimes (and one broken model)."""

    def build(
        runtimes: dict[str, ModelRuntime] | None = None, *, broken: bool = True
    ) -> ModelRegistry:
        shapes = {
            "credit-default-mlp": ((25,), 2),
            "volatility-mlp": ((25,), 1),
            "candlestick-cnn": ((64, 64, 3), 3),
        }
        entries: dict[str, Any] = {}
        for name, (trailing, width) in shapes.items():
            bundle = registry.get(name)
            runtime = (runtimes or {}).get(name) or FakeRuntime(trailing, width)
            entries[name] = ModelBundle(bundle.manifest, runtime)
        if broken:
            entries["broken-model"] = UnavailableModel(
                "broken-model", "1.0.0", "integrity_check_failed"
            )
        return ModelRegistry(entries)

    return build
