"""Training-serving skew: replay every golden case of the committed artifacts through the API.

The golden file is written next to each artifact by the notebook (or, for the dummies, by
``tools/make_dummy_artifacts.py``) and records, per case, the raw request, the engineered
features and the model output as the *training* stack produced them. The tests here prove that
this service, which has its own preprocessing and its own runtimes, reproduces all of it:

* ``to_vector`` reproduces the golden features,
* every runtime rung the manifest declares (ONNX, NumPy MLP, LiteRT) reproduces the outputs,
* top-1 decisions agree wherever the golden margin is clear,
* the HTTP surface returns the golden outputs,
* re-scoring the labelled cases through the API reproduces the manifest's quality gate.

Tolerances come from the golden file (``features_atol``, ``output_atol``, ``top1_margin``) and
are applied as ``atol`` plus a small ``rtol``, because float32 features reach 1e5 and an absolute
tolerance alone would be tighter for them than for the outputs.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import ArrayLike

from trinet_api.application.preprocessing import decode_image, to_vector
from trinet_api.domain.entities import ModelBundle, ModelRegistry
from trinet_api.domain.ports import ModelRuntime
from trinet_api.infrastructure.loading import (
    ArtifactIntegrityError,
    RuntimeOptions,
    _build_runtime,
    latest_versions,
    load_registry,
)
from trinet_api.infrastructure.runtimes import RuntimeLoadError
from trinet_contract import (
    ClassificationOutput,
    ImageGolden,
    ImageInput,
    Manifest,
    RegressionOutput,
    Status,
    TabularGolden,
    TabularInput,
    load_golden,
)

ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"
LATEST = latest_versions(ARTIFACTS)
MODELS = sorted(LATEST)
PREDICT = "/v1/models/{}/predictions"
RTOL = 1e-5  # on top of the golden file's absolute tolerances


@dataclass(frozen=True)
class Golden:
    manifest: Manifest
    directory: Path
    golden: TabularGolden | ImageGolden


@functools.cache
def golden_for(model: str) -> Golden:
    directory = LATEST[model]
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
    assert manifest.golden is not None, f"{model} ships no golden file"
    return Golden(manifest, directory, load_golden(directory / manifest.golden.file))


def rungs(models: Sequence[str]) -> list[Any]:
    params = []
    for model in models:
        for spec in golden_for(model).manifest.runtimes:
            params.append(pytest.param(model, spec.kind, id=f"{model}/{spec.kind}"))
    return params


def runtime_for(model: str, kind: str) -> ModelRuntime:
    """One specific rung of the manifest's ladder (the loader only builds the first usable)."""
    data = golden_for(model)
    spec = next(s for s in data.manifest.runtimes if s.kind == kind)
    try:
        return _build_runtime(spec, data.manifest, data.directory, RuntimeOptions())
    except (RuntimeLoadError, ArtifactIntegrityError) as exc:
        pytest.skip(f"{model}/{kind} cannot be built in this environment: {exc}")


def tabular(model: str) -> tuple[Manifest, TabularInput, TabularGolden]:
    data = golden_for(model)
    if not (
        isinstance(data.manifest.input, TabularInput) and isinstance(data.golden, TabularGolden)
    ):
        pytest.skip(f"{model} is not a tabular model")
    return data.manifest, data.manifest.input, data.golden


def images(model: str) -> tuple[Manifest, ImageInput, ImageGolden]:
    data = golden_for(model)
    if not (isinstance(data.manifest.input, ImageInput) and isinstance(data.golden, ImageGolden)):
        pytest.skip(f"{model} is not an image model")
    return data.manifest, data.manifest.input, data.golden


def png_bytes(model: str, golden: ImageGolden, file: str) -> bytes:
    return (golden_for(model).directory / golden.directory / file).read_bytes()


def close(actual: ArrayLike, expected: ArrayLike, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        atol=atol,
        rtol=RTOL,
    )


def clear_top1(expected: Sequence[float], margin: float) -> bool:
    top = sorted(expected, reverse=True)
    return len(top) < 2 or top[0] - top[1] >= margin


TABULAR = [m for m in MODELS if isinstance(golden_for(m).manifest.input, TabularInput)]
IMAGE = [m for m in MODELS if isinstance(golden_for(m).manifest.input, ImageInput)]
CLASSIFIERS = [
    m for m in TABULAR if isinstance(golden_for(m).manifest.output, ClassificationOutput)
]

pytestmark = pytest.mark.skipif(not MODELS, reason=f"no artifacts under {ARTIFACTS}")


@pytest.fixture(scope="module")
def repo_registry() -> ModelRegistry:
    return load_registry(ARTIFACTS)


@pytest.fixture
def served(make_client: Callable[..., TestClient], repo_registry: ModelRegistry) -> TestClient:
    return make_client(registry_override=repo_registry)


# --------------------------------------------------------------------------- the artifact


@pytest.mark.parametrize("model", MODELS)
def test_the_golden_file_belongs_to_its_manifest(model: str) -> None:
    data = golden_for(model)
    assert (data.golden.model, data.golden.version) == (data.manifest.name, data.manifest.version)
    assert data.manifest.name == model
    assert data.manifest.quality_gate.passed, "a model that fails its own gate must not ship"


@pytest.mark.parametrize("model", MODELS)
def test_every_model_loads_in_the_registry(model: str, repo_registry: ModelRegistry) -> None:
    entry = repo_registry.get(model)
    assert isinstance(entry, ModelBundle), f"{model} did not load: {entry}"
    assert entry.version == golden_for(model).manifest.version


# --------------------------------------------------------------------------- tabular skew


@pytest.mark.parametrize("model", TABULAR)
def test_preprocessing_reproduces_the_golden_features(model: str) -> None:
    _, spec, golden = tabular(model)
    for case in golden.cases:
        close(to_vector(case.record, spec), case.features, golden.tolerance.features_atol)


@pytest.mark.parametrize(("model", "kind"), rungs(TABULAR))
def test_every_runtime_rung_reproduces_the_golden_outputs(model: str, kind: str) -> None:
    manifest, _, golden = tabular(model)
    runtime = runtime_for(model, kind)
    features = np.asarray([c.features for c in golden.cases], dtype=np.float32)
    got = runtime.predict(features)
    for row, case in zip(got, golden.cases, strict=True):
        close(row, case.output, golden.tolerance.output_atol)
    assert got.shape[1] == len(golden.cases[0].output) == manifest.outputs[0].shape[-1]


@pytest.mark.parametrize(("model", "kind"), rungs(CLASSIFIERS))
def test_top1_agrees_wherever_the_golden_margin_is_clear(model: str, kind: str) -> None:
    manifest, _, golden = tabular(model)
    if not isinstance(manifest.output, ClassificationOutput):
        pytest.skip("regression model")
    runtime = runtime_for(model, kind)
    features = np.asarray([c.features for c in golden.cases], dtype=np.float32)
    got = runtime.predict(features)
    checked = 0
    for row, case in zip(got, golden.cases, strict=True):
        if clear_top1(case.output, golden.tolerance.top1_margin):
            assert int(np.argmax(row)) == int(np.argmax(case.output))
            checked += 1
    assert checked > 0, "no golden case has a clear top-1 margin"


@pytest.mark.parametrize("model", TABULAR)
def test_http_replay_returns_the_golden_outputs(
    model: str, served: TestClient, auth: dict[str, str]
) -> None:
    manifest, _, golden = tabular(model)
    response = served.post(
        PREDICT.format(model), headers=auth, json={"instances": [c.record for c in golden.cases]}
    )
    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert len(predictions) == len(golden.cases)
    for prediction, case in zip(predictions, golden.cases, strict=True):
        if isinstance(manifest.output, ClassificationOutput):
            probabilities = [prediction["probabilities"][label] for label in manifest.output.labels]
            close(probabilities, case.output, golden.tolerance.output_atol)
        else:
            assert isinstance(manifest.output, RegressionOutput)
            assert prediction["unit"] == manifest.output.unit
            close([prediction["value"]], case.output, golden.tolerance.output_atol)


# --------------------------------------------------------------------------- quality gate


def accuracy(predicted: list[Any], truth: list[Any]) -> float:
    return float(np.mean([p == t for p, t in zip(predicted, truth, strict=True)]))


def rmse(predicted: list[Any], truth: list[Any]) -> float:
    diff = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(math.sqrt(float(np.mean(diff**2))))


def mae(predicted: list[Any], truth: list[Any]) -> float:
    diff = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


METRICS: dict[str, Callable[[list[Any], list[Any]], float]] = {
    "accuracy": accuracy,
    "rmse": rmse,
    "mae": mae,
}


@pytest.mark.parametrize("model", TABULAR)
def test_rescoring_the_labelled_cases_through_the_api_reproduces_the_manifest_gate(
    model: str, served: TestClient, auth: dict[str, str]
) -> None:
    manifest, _, golden = tabular(model)
    gate = manifest.quality_gate
    if gate.metric not in METRICS:
        pytest.skip(f"no re-scoring implementation for metric {gate.metric!r}")
    labelled = [c for c in golden.cases if c.label is not None]
    if not labelled:
        pytest.skip("the golden file has no labelled cases")
    response = served.post(
        PREDICT.format(model), headers=auth, json={"instances": [c.record for c in labelled]}
    )
    assert response.status_code == 200, response.text
    key = "label" if isinstance(manifest.output, ClassificationOutput) else "value"
    predicted = [p[key] for p in response.json()["predictions"]]
    rescored = METRICS[gate.metric](predicted, [c.label for c in labelled])
    assert math.isfinite(rescored)
    assert manifest.metrics[gate.split][gate.metric] == pytest.approx(gate.value)
    if manifest.status is Status.DUMMY:
        # The dummy generator computes the gate on exactly the labelled golden cases.
        assert rescored == pytest.approx(gate.value, rel=1e-4, abs=1e-6)
    # A real model's gate comes from a whole split, so the golden subset only has to be sane:
    # it may not be catastrophically worse than the threshold the model was allowed to ship at.
    elif gate.higher_is_better:
        assert rescored >= gate.threshold - 0.25
    else:
        assert rescored <= gate.threshold * 1.5 + 1e-9


# --------------------------------------------------------------------------- image skew


@pytest.mark.parametrize(("model", "kind"), rungs(IMAGE))
def test_image_pipeline_reproduces_the_golden_outputs(model: str, kind: str) -> None:
    manifest, spec, golden = images(model)
    runtime = runtime_for(model, kind)
    for case in golden.cases:
        array = decode_image(png_bytes(model, golden, case.file), spec)
        out = runtime.predict(array[np.newaxis])[0]
        close(out, case.output, golden.tolerance.output_atol)
        if clear_top1(case.output, golden.tolerance.top1_margin):
            assert int(np.argmax(out)) == int(np.argmax(case.output))
    assert isinstance(manifest.output, ClassificationOutput)


@pytest.mark.parametrize("model", IMAGE)
def test_image_http_replay_returns_the_golden_scores(
    model: str, served: TestClient, auth: dict[str, str]
) -> None:
    manifest, _, golden = images(model)
    assert isinstance(manifest.output, ClassificationOutput)
    labels = manifest.output.labels
    for case in golden.cases:
        data = png_bytes(model, golden, case.file)
        response = served.post(
            PREDICT.format(model),
            headers=auth,
            files={"file": (case.file, data, "application/octet-stream")},
        )
        assert response.status_code == 200, response.text
        (prediction,) = response.json()["predictions"]
        for entry in prediction["top_k"]:
            expected = case.output[labels.index(entry["label"])]
            assert entry["probability"] == pytest.approx(
                expected, abs=golden.tolerance.output_atol, rel=RTOL
            )
        if clear_top1(case.output, golden.tolerance.top1_margin):
            assert prediction["label"] == labels[int(np.argmax(case.output))]
