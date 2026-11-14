"""One behavioural contract, run against every ``ModelRuntime`` implementation.

Real adapters are built from the dummy artifacts exactly like the loader does (every rung the
manifests declare), plus a tiny real ``.tflite`` when LiteRT is installed and the test double.
Anything the service relies on has to hold for all of them.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from numpy.typing import NDArray
from tests.fakes import FakeRuntime

from trinet_api.application.preprocessing import to_vector
from trinet_api.domain.ports import ModelRuntime
from trinet_api.infrastructure.loading import RuntimeOptions, _build_runtime
from trinet_api.infrastructure.runtimes import static_dims
from trinet_contract import Manifest, TabularInput

type Sampler = Callable[[np.random.Generator, int], NDArray[np.float32]]


@dataclass(frozen=True)
class Case:
    name: str
    runtime: ModelRuntime
    trailing: tuple[int, ...]
    width: int
    probabilities: bool
    sample: Sampler


def _uniform(trailing: tuple[int, ...], low: float, high: float) -> Sampler:
    def sample(rng: np.random.Generator, rows: int) -> NDArray[np.float32]:
        return rng.uniform(low, high, size=(rows, *trailing)).astype(np.float32)

    return sample


def _records(
    make_record: Callable[[np.random.Generator], dummy.Record], spec: TabularInput
) -> Sampler:
    """Realistic raw records, vectorised like the API does: the volatility head ends in exp."""

    def sample(rng: np.random.Generator, rows: int) -> NDArray[np.float32]:
        return np.stack([to_vector(make_record(rng), spec) for _ in range(rows)])

    return sample


_RECORDS = {"credit-default-mlp": dummy.credit_record, "volatility-mlp": dummy.volatility_record}


def _sampler(model: str, manifest: Manifest, trailing: tuple[int, ...]) -> Sampler:
    if model in _RECORDS:
        assert isinstance(manifest.input, TabularInput)
        return _records(_RECORDS[model], manifest.input)
    return _uniform(trailing, 0.0, 255.0)  # pixel values


def _artifact_cases(root: Path) -> list[Case]:
    cases = []
    for model in dummy.MODELS:
        directory = root / model / dummy.DUMMY_VERSION
        manifest = Manifest.model_validate(json.loads((directory / "manifest.json").read_text()))
        trailing, width = static_dims(manifest.inputs[0], manifest.outputs[0])
        classification = manifest.task != "regression"
        for spec in manifest.runtimes:
            runtime = _build_runtime(spec, manifest, directory, RuntimeOptions())
            cases.append(
                Case(
                    f"{model}/{spec.kind}",
                    runtime,
                    trailing,
                    width,
                    classification,
                    _sampler(model, manifest, trailing),
                )
            )
    return cases


def _litert_case() -> Case:
    pytest.importorskip("ai_edge_litert")
    from tests.litert_model import softmax_classifier

    from trinet_api.infrastructure.runtimes.litert import LiteRtRuntime
    from trinet_contract import TensorSpec

    rng = np.random.default_rng(3)
    blob = softmax_classifier(
        rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=4).astype(np.float32)
    )
    runtime = LiteRtRuntime(
        blob, TensorSpec(name="x", shape=(None, 5)), TensorSpec(name="y", shape=(None, 4))
    )
    return Case("tiny/litert", runtime, (5,), 4, True, _uniform((5,), -3.0, 3.0))


def _fake_cases() -> list[Case]:
    return [
        Case("fake/classifier", FakeRuntime((6,), 3), (6,), 3, True, _uniform((6,), -3.0, 3.0)),
        Case("fake/regressor", FakeRuntime((6,), 1), (6,), 1, False, _uniform((6,), -3.0, 3.0)),
    ]


CASE_IDS = [
    *(f"{m}/{k}" for m, kinds in (("credit-default-mlp", ("onnx", "numpy_mlp")),
                                  ("volatility-mlp", ("onnx", "numpy_mlp")),
                                  ("candlestick-cnn", ("onnx",))) for k in kinds),
    "tiny/litert",
    "fake/classifier",
    "fake/regressor",
]  # fmt: skip


@pytest.fixture(scope="module")
def all_cases(artifacts_root: Path) -> dict[str, Case]:
    cases = {case.name: case for case in [*_artifact_cases(artifacts_root), *_fake_cases()]}
    with contextlib.suppress(pytest.skip.Exception):  # LiteRT not installed
        cases["tiny/litert"] = _litert_case()
    return cases


@pytest.fixture(params=CASE_IDS)
def case(request: pytest.FixtureRequest, all_cases: dict[str, Case]) -> Case:
    if request.param not in all_cases:
        pytest.skip(f"{request.param} is not available in this environment")
    return all_cases[request.param]


def batch_of(case: Case, rows: int, seed: int = 0) -> NDArray[np.float32]:
    return case.sample(np.random.default_rng(seed), rows)


# --------------------------------------------------------------------------- output contract


def test_satisfies_the_port(case: Case) -> None:
    assert isinstance(case.runtime, ModelRuntime)
    assert isinstance(case.runtime.kind, str)
    assert case.runtime.kind


@pytest.mark.parametrize("rows", [1, 2, 7, 33])
def test_output_is_finite_float32_of_the_declared_shape(case: Case, rows: int) -> None:
    out = case.runtime.predict(batch_of(case, rows))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (rows, case.width)
    assert np.all(np.isfinite(out))


def test_classifiers_emit_probability_rows(case: Case) -> None:
    if not case.probabilities:
        pytest.skip("regression runtime")
    out = case.runtime.predict(batch_of(case, 16))
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_output_does_not_alias_runtime_state(case: Case) -> None:
    batch = batch_of(case, 3)
    first = case.runtime.predict(batch)
    expected = first.copy()
    first[:] = 123.0
    np.testing.assert_array_equal(case.runtime.predict(batch), expected)


# --------------------------------------------------------------------------- semantics


def test_rows_are_independent_of_their_batch_mates(case: Case) -> None:
    batch = batch_of(case, 6, seed=1)
    together = case.runtime.predict(batch)
    for i in range(len(batch)):
        alone = case.runtime.predict(batch[i : i + 1])
        np.testing.assert_allclose(alone[0], together[i], rtol=1e-4, atol=1e-5)


def test_row_order_is_preserved(case: Case) -> None:
    batch = batch_of(case, 5, seed=2)
    forward = case.runtime.predict(batch)
    backward = case.runtime.predict(batch[::-1].copy())
    np.testing.assert_allclose(backward[::-1], forward, rtol=1e-4, atol=1e-5)


def test_prediction_is_deterministic(case: Case) -> None:
    batch = batch_of(case, 4, seed=3)
    np.testing.assert_array_equal(case.runtime.predict(batch), case.runtime.predict(batch))


def test_input_is_not_mutated(case: Case) -> None:
    batch = batch_of(case, 4, seed=4)
    snapshot = batch.copy()
    case.runtime.predict(batch)
    np.testing.assert_array_equal(batch, snapshot)


def test_non_contiguous_input_gives_the_same_answer(case: Case) -> None:
    wide = np.repeat(batch_of(case, 4, seed=5), 2, axis=0)
    view = wide[::2]  # strided view of the same rows
    assert not view.flags.c_contiguous
    np.testing.assert_allclose(
        case.runtime.predict(view),
        case.runtime.predict(np.ascontiguousarray(view)),
        rtol=1e-5,
        atol=1e-6,
    )


# --------------------------------------------------------------------------- rejected inputs


def test_wrong_dtype_is_a_value_error(case: Case) -> None:
    with pytest.raises(ValueError, match="float32"):
        case.runtime.predict(batch_of(case, 2).astype(np.float64))


def test_wrong_width_is_a_value_error(case: Case) -> None:
    bad = np.zeros((2, *case.trailing[:-1], case.trailing[-1] + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        case.runtime.predict(bad)


def test_wrong_rank_is_a_value_error(case: Case) -> None:
    with pytest.raises(ValueError, match="shape"):
        case.runtime.predict(np.zeros(case.trailing, dtype=np.float32))


def test_empty_batch_is_a_value_error(case: Case) -> None:
    with pytest.raises(ValueError, match="row"):
        case.runtime.predict(np.zeros((0, *case.trailing), dtype=np.float32))


# --------------------------------------------------------------------------- concurrency


def test_concurrent_predictions_match_serial_ones(case: Case) -> None:
    batches = [batch_of(case, 1 + i % 5, seed=100 + i) for i in range(24)]
    expected = [case.runtime.predict(b) for b in batches]
    results: list[NDArray[np.float32] | None] = [None] * len(batches)
    errors: list[BaseException] = []

    def work(indices: range) -> None:
        try:
            for i in indices:
                results[i] = case.runtime.predict(batches[i])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(range(t, len(batches), 6),)) for t in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    for got, want in zip(results, expected, strict=True):
        assert got is not None
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("model", ["credit-default-mlp", "volatility-mlp"])
def test_onnx_and_numpy_mlp_agree(all_cases: dict[str, Case], model: str) -> None:
    onnx_case, numpy_case = all_cases[f"{model}/onnx"], all_cases[f"{model}/numpy_mlp"]
    batch = batch_of(onnx_case, 64, seed=9)
    np.testing.assert_allclose(
        numpy_case.runtime.predict(batch),
        onnx_case.runtime.predict(batch),
        rtol=1e-4,
        atol=1e-4,
    )
