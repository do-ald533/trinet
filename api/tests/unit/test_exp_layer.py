"""``ExpLayer`` (``{"op": "exp"}``): manifest round-trip, NumPy fallback and ONNX parity.

The volatility model is trained on ``log(target)`` and its exported graph ends in
``denormalize -> exp``, so every serving path has to support that tail.
"""

from __future__ import annotations

import json
from pathlib import Path

import make_dummy_artifacts as dummy
import numpy as np
import onnx
import pytest
from pydantic import ValidationError

from trinet_api.infrastructure.loading import RuntimeOptions, _build_runtime
from trinet_api.infrastructure.runtimes.numpy_mlp import NumpyMlpRuntime
from trinet_contract import (
    DenormalizeLayer,
    DenseLayer,
    ExpLayer,
    Manifest,
    NormalizeLayer,
    NumpyMlpRuntimeSpec,
    TensorSpec,
)

IN = TensorSpec(name="x", shape=(None, 3))
OUT = TensorSpec(name="y", shape=(None, 1))
TAIL = (
    NormalizeLayer(key="n"),
    DenseLayer(key="o", activation="linear"),
    DenormalizeLayer(key="t"),
    ExpLayer(),
)


def arrays() -> dummy.Arrays:
    return {
        "n.mean": np.asarray([1, 2, 3], np.float32),
        "n.variance": np.asarray([1, 4, 9], np.float32),
        "o.kernel": np.asarray([[0.3], [-0.2], [0.1]], np.float32),
        "o.bias": np.asarray([0.05], np.float32),
        "t.mean": np.asarray([np.log(0.25)], np.float32),
        "t.variance": np.asarray([0.25], np.float32),
    }


# --------------------------------------------------------------------------- manifest


def test_the_layer_is_just_an_op_tag() -> None:
    assert ExpLayer().model_dump() == {"op": "exp"}
    assert json.loads(ExpLayer().model_dump_json()) == {"op": "exp"}


def test_a_numpy_mlp_spec_accepts_exp_in_json_and_keeps_it_through_a_roundtrip() -> None:
    payload = {
        "kind": "numpy_mlp",
        "file": "weights.npz",
        "sha256": "0" * 64,
        "layers": [
            {"op": "normalize", "key": "n"},
            {"op": "denormalize", "key": "t"},
            {"op": "exp"},
        ],
    }
    spec = NumpyMlpRuntimeSpec.model_validate(payload)
    assert isinstance(spec.layers[-1], ExpLayer)
    assert NumpyMlpRuntimeSpec.model_validate_json(spec.model_dump_json()) == spec
    assert json.loads(spec.model_dump_json())["layers"][-1] == {"op": "exp"}


@pytest.mark.parametrize("extra", [{"key": "x"}, {"base": 2}, {"unknown": True}])
def test_exp_takes_no_parameters(extra: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExpLayer.model_validate({"op": "exp", **extra})


def test_the_volatility_manifest_ends_in_denormalize_then_exp(artifacts_root: Path) -> None:
    path = artifacts_root / "volatility-mlp" / dummy.DUMMY_VERSION / "manifest.json"
    manifest = Manifest.model_validate_json(path.read_text())
    (spec,) = (r for r in manifest.runtimes if isinstance(r, NumpyMlpRuntimeSpec))
    assert [layer.op for layer in spec.layers[-2:]] == ["denormalize", "exp"]
    assert Manifest.model_validate_json(manifest.model_dump_json()) == manifest


# --------------------------------------------------------------------------- numpy_mlp


def test_numpy_runtime_matches_the_reference_and_is_strictly_positive() -> None:
    weights = arrays()
    x = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    got = NumpyMlpRuntime(dummy.npz_bytes(weights), TAIL, IN, OUT).predict(x)
    np.testing.assert_allclose(got, dummy.mlp_reference(x, TAIL, weights), rtol=1e-5)
    assert got.dtype == np.float32
    assert got.shape == (32, 1)
    assert np.all(got > 0.0)


def test_exp_needs_no_weights_of_its_own() -> None:
    only_exp = NumpyMlpRuntime(
        dummy.npz_bytes({}), (ExpLayer(),), IN, TensorSpec(name="y", shape=(None, 3))
    )
    x = np.asarray([[0.0, 1.0, -1.0]], np.float32)
    np.testing.assert_allclose(only_exp.predict(x), np.exp(x), rtol=1e-6)


def test_exp_keeps_the_feature_width_so_a_following_dense_layer_still_chains() -> None:
    layers = (ExpLayer(), DenseLayer(key="o", activation="linear"))
    weights = {k: v for k, v in arrays().items() if k.startswith("o.")}
    runtime = NumpyMlpRuntime(dummy.npz_bytes(weights), layers, IN, OUT)
    x = np.asarray([[0.0, 0.0, 0.0]], np.float32)
    expected = np.ones((1, 3), np.float32) @ weights["o.kernel"] + weights["o.bias"]
    np.testing.assert_allclose(runtime.predict(x), expected, rtol=1e-6)


def test_exp_overflow_is_refused_by_the_runtime_instead_of_returned_as_inf() -> None:
    runtime = NumpyMlpRuntime(
        dummy.npz_bytes({}), (ExpLayer(),), TensorSpec(name="x", shape=(None, 1)), OUT
    )
    with np.errstate(over="ignore"), pytest.raises(RuntimeError, match="non-finite"):
        runtime.predict(np.asarray([[200.0]], np.float32))  # exp(200) overflows float32


# --------------------------------------------------------------------------- onnx parity


def test_the_dummy_onnx_graph_contains_an_exp_node(artifacts_root: Path) -> None:
    graph = onnx.load(artifacts_root / "volatility-mlp" / dummy.DUMMY_VERSION / "model.onnx").graph
    assert [node.op_type for node in graph.node].count("Exp") == 1


def test_onnx_and_numpy_runtimes_agree_on_realistic_records(artifacts_root: Path) -> None:
    directory = artifacts_root / "volatility-mlp" / dummy.DUMMY_VERSION
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
    runtimes = {
        spec.kind: _build_runtime(spec, manifest, directory, RuntimeOptions())
        for spec in manifest.runtimes
    }
    rng = np.random.default_rng(1)
    batch = np.asarray(
        [dummy.volatility_features(dummy.volatility_record(rng)) for _ in range(48)], np.float32
    )
    onnx_out, numpy_out = runtimes["onnx"].predict(batch), runtimes["numpy_mlp"].predict(batch)
    assert np.all(onnx_out > 0.0)
    np.testing.assert_allclose(numpy_out, onnx_out, rtol=1e-4, atol=1e-5)
