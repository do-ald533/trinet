from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from trinet_contract import Manifest, Status, TabularInput, load_manifest, sha256_file
from trinet_contract.schema import write_schemas

SHA = "0" * 64


def tabular_manifest() -> dict[str, Any]:
    return {
        "name": "housing-mlp",
        "version": "1.0.0",
        "status": "candidate",
        "task": "regression",
        "created_at": "2026-11-10T12:00:00Z",
        "input": {
            "kind": "tabular",
            "fields": [
                {"kind": "numeric", "name": "total_rooms"},
                {"kind": "numeric", "name": "households", "minimum": 1},
                {"kind": "numeric", "name": "total_bedrooms", "impute": 435.0},
                {
                    "kind": "categorical",
                    "name": "ocean_proximity",
                    "vocabulary": ["<1H OCEAN", "INLAND", "NEAR BAY"],
                },
            ],
            "features": [
                {"kind": "passthrough", "source": "total_bedrooms"},
                {"kind": "one_hot", "source": "ocean_proximity"},
                {
                    "kind": "ratio",
                    "name": "rooms_per_household",
                    "numerator": "total_rooms",
                    "denominator": "households",
                },
            ],
        },
        "output": {"kind": "regression", "target": "median_house_value", "unit": "USD"},
        "inputs": [{"name": "inputs", "shape": [None, 5]}],
        "outputs": [{"name": "output_0", "shape": [None, 1]}],
        "runtimes": [
            {"kind": "onnx", "file": "model.onnx", "sha256": SHA, "opset": 17},
            {
                "kind": "numpy_mlp",
                "file": "weights.npz",
                "sha256": SHA,
                "layers": [
                    {"op": "normalize", "key": "norm"},
                    {"op": "dense", "key": "d0", "activation": "relu"},
                    {"op": "dense", "key": "out", "activation": "linear"},
                    {"op": "denormalize", "key": "target"},
                ],
            },
        ],
        "metrics": {"test": {"rmse": 48000.0}},
        "quality_gate": {
            "metric": "rmse",
            "threshold": 60000.0,
            "value": 50000.0,
            "higher_is_better": False,
        },
        "training": {
            "dataset": "California Housing Prices",
            "data_sha256": SHA,
            "seed": 42,
            "python": "3.13",
            "tensorflow": "2.21.0",
            "keras": "3.15.1",
        },
    }


def image_manifest() -> dict[str, Any]:
    return {
        "name": "intel-cnn",
        "version": "1.0.0",
        "status": "dummy",
        "task": "image_classification",
        "created_at": "2026-11-10T12:00:00Z",
        "input": {"kind": "image", "height": 128, "width": 128},
        "output": {"kind": "classification", "labels": ["a", "b", "c"]},
        "inputs": [{"name": "inputs", "shape": [None, 128, 128, 3]}],
        "outputs": [{"name": "output_0", "shape": [None, 3]}],
        "runtimes": [{"kind": "onnx", "file": "model.onnx", "sha256": SHA, "opset": 17}],
        "quality_gate": {"metric": "accuracy", "threshold": 0.8, "value": 0.1},
        "training": tabular_manifest()["training"],
    }


def test_valid_tabular_manifest_expands_features() -> None:
    manifest = Manifest.model_validate(tabular_manifest())
    assert isinstance(manifest.input, TabularInput)
    assert manifest.input.feature_names() == [
        "total_bedrooms",
        "ocean_proximity=<1H OCEAN",
        "ocean_proximity=INLAND",
        "ocean_proximity=NEAR BAY",
        "rooms_per_household",
    ]
    assert manifest.quality_gate.passed


def test_valid_image_manifest() -> None:
    assert Manifest.model_validate(image_manifest()).status is Status.DUMMY


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(task="classification"), "requires tabular input"),
        (lambda m: m["inputs"][0].update(shape=[None, 4]), "feature width"),
        (lambda m: m["outputs"][0].update(shape=[None, 2]), "output tensor width"),
        (lambda m: m["runtimes"][0].update(file="../model.onnx"), "String should match"),
        (lambda m: m["runtimes"][1].update(file="model.onnx"), "distinct files"),
        (lambda m: m["input"]["features"][0].update(source="nope"), "unknown field"),
        (lambda m: m["input"]["features"][1].update(source="total_rooms"), "CategoricalField"),
        (lambda m: m.update(extra_key=1), "Extra inputs are not permitted"),
        (
            lambda m: m.update(status="release", quality_gate={**m["quality_gate"], "value": 7e4}),
            "quality gate",
        ),
    ],
)
def test_invalid_tabular_manifests(mutate: Any, message: str) -> None:
    data = copy.deepcopy(tabular_manifest())
    mutate(data)
    with pytest.raises(ValidationError, match=message):
        Manifest.model_validate(data)


def test_image_manifest_rejects_numpy_runtime() -> None:
    data = image_manifest()
    data["runtimes"].append(tabular_manifest()["runtimes"][1])
    with pytest.raises(ValidationError, match="only supports tabular"):
        Manifest.model_validate(data)


def test_load_manifest_and_sha(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tabular_manifest()), encoding="utf-8")
    assert load_manifest(path).name == "housing-mlp"
    assert len(sha256_file(path)) == 64


def test_committed_schemas_are_up_to_date(tmp_path: Path) -> None:
    committed = Path(__file__).resolve().parents[1] / "schema"
    for generated in write_schemas(tmp_path):
        assert (committed / generated.name).read_text(encoding="utf-8") == generated.read_text(
            encoding="utf-8"
        ), f"{generated.name} drifted: run `uv run python -m trinet_contract.schema`"
