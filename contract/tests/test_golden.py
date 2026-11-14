from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trinet_contract import ImageGolden, TabularGolden, load_golden


def test_tabular_golden_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "kind": "tabular",
                "model": "housing-mlp",
                "version": "1.0.0",
                "cases": [
                    {
                        "record": {"a": 1.0, "b": None, "c": "INLAND"},
                        "features": [1.0],
                        "output": [2.0],
                        "label": 2.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    golden = load_golden(path)
    assert isinstance(golden, TabularGolden)
    assert golden.cases[0].record["b"] is None
    assert golden.tolerance.features_atol == pytest.approx(1e-6)


def test_image_golden_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    payload = {
        "kind": "image",
        "model": "intel-cnn",
        "version": "1.0.0",
        "cases": [{"file": "../etc/passwd", "output": [0.5, 0.5]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_golden(path)
    payload["cases"] = [{"file": "img_000.png", "output": [0.5, 0.5], "label": "sea"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert isinstance(load_golden(path), ImageGolden)
