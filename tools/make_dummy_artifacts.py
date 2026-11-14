"""Generate DUMMY artifacts for the three TriNet models (real schemas, fake weights).

They let the API, the image and the CI run end to end before the notebook exports the real
models. Everything is deterministic (seeded RNGs, fixed timestamps, reproducible ``.npz`` and
ONNX bytes), so re-running the script does not churn the committed files.

Usage::

    uv run python tools/make_dummy_artifacts.py            # writes artifacts/*/0.0.0-dummy/
    uv run python tools/make_dummy_artifacts.py --out /tmp/x --version 0.0.1-dummy

The functions are also imported by the API test-suite to build artifacts in temporary
directories (``build_all``, ``build_drybean``, ``build_housing``, ``build_intel``).
The ``onnx`` package is a dev-only dependency (never imported by ``api/src``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from trinet_contract import (
    CategoricalField,
    ClassificationOutput,
    DenormalizeLayer,
    DenseLayer,
    GoldenRef,
    GoldenTolerance,
    ImageGolden,
    ImageGoldenCase,
    ImageInput,
    Manifest,
    NormalizeLayer,
    NumericField,
    NumpyMlpRuntimeSpec,
    OneHotFeature,
    OnnxRuntimeSpec,
    PassthroughFeature,
    QualityGate,
    RatioFeature,
    RegressionOutput,
    ServingSkew,
    Status,
    TabularGolden,
    TabularGoldenCase,
    TabularInput,
    Task,
    TensorSpec,
    TrainingInfo,
)
from trinet_contract.manifest import MlpLayer

DUMMY_VERSION = "0.0.0-dummy"
OPSET = 17
IR_VERSION = 8  # the IR version that goes with opset 17
CREATED_AT = datetime(2026, 11, 10, tzinfo=UTC)
MODELS = ("drybean-mlp", "housing-mlp", "intel-cnn")
EPS = np.float32(1e-7)  # Keras Normalization epsilon

type F32 = NDArray[np.float32]
type Arrays = dict[str, F32]

# --------------------------------------------------------------------------- bytes helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def npz_bytes(arrays: Arrays) -> bytes:
    """Uncompressed ``.npz`` with fixed timestamps (``np.savez`` embeds the current time)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, arrays[name], allow_pickle=False)
    return buffer.getvalue()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- MLP reference


def _std(variance: F32) -> F32:
    return np.maximum(np.sqrt(variance), EPS).astype(np.float32)


def mlp_reference(x: F32, layers: Sequence[MlpLayer], arrays: Arrays) -> F32:
    """Independent NumPy implementation of the manifest layers (float32)."""
    out = x.astype(np.float32)
    for layer in layers:
        if isinstance(layer, NormalizeLayer):
            out = (out - arrays[f"{layer.key}.mean"]) / _std(arrays[f"{layer.key}.variance"])
        elif isinstance(layer, DenormalizeLayer):
            out = out * _std(arrays[f"{layer.key}.variance"]) + arrays[f"{layer.key}.mean"]
        else:
            out = out @ arrays[f"{layer.key}.kernel"] + arrays[f"{layer.key}.bias"]
            if layer.activation == "relu":
                out = np.maximum(out, np.float32(0))
            elif layer.activation == "softmax":
                e = np.exp(out - out.max(axis=-1, keepdims=True))
                out = e / e.sum(axis=-1, keepdims=True)
        out = out.astype(np.float32)
    return out


def _dense(rng: np.random.Generator, key: str, n_in: int, n_out: int, gain: float) -> Arrays:
    kernel = rng.normal(0.0, gain * np.sqrt(2.0 / n_in), size=(n_in, n_out))
    bias = rng.normal(0.0, 0.05, size=(n_out,))
    return {f"{key}.kernel": kernel.astype(np.float32), f"{key}.bias": bias.astype(np.float32)}


def _stats(key: str, mean: Sequence[float], variance: Sequence[float]) -> Arrays:
    return {
        f"{key}.mean": np.asarray(mean, dtype=np.float32),
        f"{key}.variance": np.asarray(variance, dtype=np.float32),
    }


def mlp_onnx(layers: Sequence[MlpLayer], arrays: Arrays, n_in: int, n_out: int) -> bytes:
    """ONNX graph (opset 17) computing exactly what ``mlp_reference`` computes."""
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    current = "inputs"

    def const(name: str, value: F32) -> str:
        inits.append(numpy_helper.from_array(value, name=name))
        return name

    for i, layer in enumerate(layers):
        out = f"t{i}"
        if isinstance(layer, NormalizeLayer | DenormalizeLayer):
            mean = const(f"{layer.key}.mean", arrays[f"{layer.key}.mean"])
            std = const(f"{layer.key}.std", _std(arrays[f"{layer.key}.variance"]))
            if isinstance(layer, NormalizeLayer):
                nodes.append(helper.make_node("Sub", [current, mean], [f"{out}a"]))
                nodes.append(helper.make_node("Div", [f"{out}a", std], [out]))
            else:
                nodes.append(helper.make_node("Mul", [current, std], [f"{out}a"]))
                nodes.append(helper.make_node("Add", [f"{out}a", mean], [out]))
        else:
            kernel = const(f"{layer.key}.kernel", arrays[f"{layer.key}.kernel"])
            bias = const(f"{layer.key}.bias", arrays[f"{layer.key}.bias"])
            nodes.append(helper.make_node("MatMul", [current, kernel], [f"{out}a"]))
            nodes.append(helper.make_node("Add", [f"{out}a", bias], [f"{out}b"]))
            if layer.activation == "relu":
                nodes.append(helper.make_node("Relu", [f"{out}b"], [out]))
            elif layer.activation == "softmax":
                nodes.append(helper.make_node("Softmax", [f"{out}b"], [out], axis=-1))
            else:
                nodes.append(helper.make_node("Identity", [f"{out}b"], [out]))
        current = out
    nodes.append(helper.make_node("Identity", [current], ["output_0"]))
    return _serialize(nodes, inits, [("inputs", ["batch", n_in])], ("output_0", ["batch", n_out]))


def _serialize(
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    inputs: list[tuple[str, list[Any]]],
    output: tuple[str, list[Any]],
) -> bytes:
    graph = helper.make_graph(
        nodes,
        "trinet_dummy",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
        [helper.make_tensor_value_info(output[0], TensorProto.FLOAT, output[1])],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="trinet-dummy",
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model, full_check=True)
    return bytes(model.SerializeToString(deterministic=True))


def ort_run(model: bytes, batch: F32) -> F32:
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {session.get_inputs()[0].name: batch})
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- shared writer


def _training_info(name: str) -> TrainingInfo:
    return TrainingInfo(
        dataset=f"synthetic (dummy artifact for {name})",
        data_sha256=sha256_bytes(f"dummy:{name}".encode()),
        seed=0,
        python="3.13",
        tensorflow="none (dummy)",
        keras="none (dummy)",
        hardware="",
    )


def _tensors(n_in: tuple[int | None, ...], n_out: int) -> tuple[TensorSpec, TensorSpec]:
    return (
        TensorSpec(name="inputs", shape=n_in),
        TensorSpec(name="output_0", shape=(None, n_out)),
    )


def _finish(
    out_dir: Path,
    manifest: Manifest,
    results: dict[str, Any],
) -> Path:
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    (out_dir / "results.json").write_bytes(_json_bytes(results))
    return out_dir


def _results(manifest: Manifest) -> dict[str, Any]:
    return {
        "model": manifest.name,
        "version": manifest.version,
        "status": manifest.status.value,
        "note": "DUMMY artifact generated by tools/make_dummy_artifacts.py (fake weights).",
        "metrics": manifest.metrics,
        "quality_gate": json.loads(manifest.quality_gate.model_dump_json()),
        "serving_skew": json.loads(manifest.serving_skew.model_dump_json())
        if manifest.serving_skew
        else None,
    }


# --------------------------------------------------------------------------- tabular models

type Record = dict[str, float | str | None]
type Featurizer = Callable[[Record], list[float]]


def _round(x: float) -> float:
    return float(np.float32(x))


def _tabular_artifact(
    *,
    out_dir: Path,
    name: str,
    version: str,
    status: Status,
    description: str,
    task: Task,
    spec: TabularInput,
    output: ClassificationOutput | RegressionOutput,
    layers: tuple[MlpLayer, ...],
    arrays: Arrays,
    records: list[Record],
    featurize: Featurizer,
    labels_for: Callable[[int, F32], str | float],
    tolerance: GoldenTolerance,
    quality: Callable[[list[F32], list[str | float]], tuple[dict[str, float], QualityGate]],
) -> Path:
    n_in, n_out = spec.width, 1 if isinstance(output, RegressionOutput) else len(output.labels)
    onnx_bytes = mlp_onnx(layers, arrays, n_in, n_out)
    npz = npz_bytes(arrays)

    features = np.asarray([featurize(r) for r in records], dtype=np.float32)
    outputs = mlp_reference(features, layers, arrays)
    labels = [labels_for(i, row) for i, row in enumerate(outputs)]
    onnx_out = ort_run(onnx_bytes, features)
    max_diff = float(np.max(np.abs(onnx_out - outputs)))
    # metrics/gate are computed on the labelled half only, so the golden suite can re-score them
    metrics, gate = quality(list(outputs)[::2], labels[::2])

    golden = TabularGolden(
        model=name,
        version=version,
        tolerance=tolerance,
        cases=tuple(
            TabularGoldenCase(
                record=record,
                features=tuple(_round(v) for v in features[i]),
                output=tuple(_round(v) for v in outputs[i]),
                label=labels[i] if i % 2 == 0 else None,  # half of the cases are labelled
            )
            for i, record in enumerate(records)
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    golden_sha = _write(out_dir / "golden.json", (golden.model_dump_json(indent=2) + "\n").encode())
    onnx_sha = _write(out_dir / "model.onnx", onnx_bytes)
    npz_sha = _write(out_dir / "weights.npz", npz)

    in_spec, out_spec = _tensors((None, n_in), n_out)
    manifest = Manifest(
        name=name,
        version=version,
        status=status,
        task=task,
        description=description,
        created_at=CREATED_AT,
        input=spec,
        output=output,
        inputs=(in_spec,),
        outputs=(out_spec,),
        runtimes=(
            OnnxRuntimeSpec(file="model.onnx", sha256=onnx_sha, opset=OPSET),
            NumpyMlpRuntimeSpec(file="weights.npz", sha256=npz_sha, layers=layers),
        ),
        metrics={"validation": metrics, "test": metrics},
        quality_gate=gate,
        serving_skew=ServingSkew(
            samples=len(records),
            max_abs_diff=max_diff,
            top1_agreement=1.0 if task is Task.CLASSIFICATION else None,
        ),
        training=_training_info(name),
        golden=GoldenRef(file="golden.json", sha256=golden_sha),
    )
    return _finish(out_dir, manifest, _results(manifest))


def _accuracy_quality(
    n_labels: int, names: Sequence[str]
) -> Callable[[list[F32], list[str | float]], tuple[dict[str, float], QualityGate]]:
    def quality(
        outputs: list[F32], labels: list[str | float]
    ) -> tuple[dict[str, float], QualityGate]:
        hits = [names[int(np.argmax(o))] == lab for o, lab in zip(outputs, labels, strict=True)]
        accuracy = float(np.mean(hits))
        assert n_labels == len(names)  # noqa: S101 - generator sanity check
        gate = QualityGate(metric="accuracy", split="validation", threshold=0.5, value=accuracy)
        return {"accuracy": accuracy}, gate

    return quality


def _rmse_quality(
    outputs: list[F32], labels: list[str | float]
) -> tuple[dict[str, float], QualityGate]:
    pred = np.asarray([o[0] for o in outputs], dtype=np.float64)
    true = np.asarray(labels, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    metrics = {"rmse": rmse, "mae": mae, "mae_over_mean": mae / float(np.mean(true))}
    gate = QualityGate(
        metric="rmse", split="validation", threshold=rmse * 1.25, value=rmse, higher_is_better=False
    )
    return metrics, gate


# ---- Dry Bean

DRYBEAN_LABELS = ("SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON")
# name: (mean, std, minimum, maximum) - roughly the UCI statistics, only used to look plausible.
DRYBEAN_FIELDS: dict[str, tuple[float, float, float, float | None]] = {
    "Area": (53048.0, 29324.0, 0.0, None),
    "Perimeter": (855.0, 214.0, 0.0, None),
    "MajorAxisLength": (320.0, 86.0, 0.0, None),
    "MinorAxisLength": (202.0, 45.0, 0.0, None),
    "AspectRation": (1.58, 0.25, 0.0, None),
    "Eccentricity": (0.75, 0.09, 0.0, 1.0),
    "ConvexArea": (53768.0, 29775.0, 0.0, None),
    "EquivDiameter": (253.0, 59.0, 0.0, None),
    "Extent": (0.75, 0.05, 0.0, 1.0),
    "Solidity": (0.987, 0.005, 0.0, 1.0),
    "roundness": (0.873, 0.059, 0.0, 1.0),
    "Compactness": (0.80, 0.062, 0.0, 1.0),
    "ShapeFactor1": (0.0066, 0.0011, 0.0, 1.0),
    "ShapeFactor2": (0.0017, 0.0006, 0.0, 1.0),
    "ShapeFactor3": (0.64, 0.099, 0.0, 1.0),
    "ShapeFactor4": (0.995, 0.0043, 0.0, 1.0),
}


def drybean_spec() -> TabularInput:
    fields = tuple(
        NumericField(name=n, minimum=lo, maximum=hi) for n, (_, _, lo, hi) in DRYBEAN_FIELDS.items()
    )
    return TabularInput(
        fields=fields, features=tuple(PassthroughFeature(source=f.name) for f in fields)
    )


def build_drybean(
    root: Path, version: str = DUMMY_VERSION, status: Status = Status.DUMMY, seed: int = 11
) -> Path:
    rng = np.random.default_rng(seed)
    means = [m for m, _, _, _ in DRYBEAN_FIELDS.values()]
    variances = [s * s for _, s, _, _ in DRYBEAN_FIELDS.values()]
    layers: tuple[MlpLayer, ...] = (
        NormalizeLayer(key="norm"),
        DenseLayer(key="d0", activation="relu"),
        DenseLayer(key="d1", activation="relu"),
        DenseLayer(key="out", activation="softmax"),
    )
    arrays: Arrays = {
        **_stats("norm", means, variances),
        **_dense(rng, "d0", 16, 16, 1.0),
        **_dense(rng, "d1", 16, 12, 1.0),
        **_dense(rng, "out", 12, len(DRYBEAN_LABELS), 1.4),
    }

    def sample() -> Record:
        record: Record = {}
        for name, (mean, std, lo, hi) in DRYBEAN_FIELDS.items():
            value = rng.normal(mean, std)
            value = max(value, lo + 1e-6)
            record[name] = float(min(value, hi) if hi is not None else value)
        return record

    records = [sample() for _ in range(24)]

    def labels_for(i: int, row: F32) -> str:
        # ~2/3 of the labels agree with the model, so the metric is not trivially 1.0
        pick = int(np.argmax(row))
        return DRYBEAN_LABELS[pick if i % 3 else (pick + 1) % len(DRYBEAN_LABELS)]

    return _tabular_artifact(
        out_dir=root / "drybean-mlp" / version,
        name="drybean-mlp",
        version=version,
        status=status,
        description="DUMMY Dry Bean classifier (16 numeric features, 7 classes); random weights.",
        task=Task.CLASSIFICATION,
        spec=drybean_spec(),
        output=ClassificationOutput(labels=DRYBEAN_LABELS),
        layers=layers,
        arrays=arrays,
        records=records,
        featurize=lambda r: [float(r[n]) for n in DRYBEAN_FIELDS],  # type: ignore[arg-type]
        labels_for=labels_for,
        tolerance=GoldenTolerance(),
        quality=_accuracy_quality(len(DRYBEAN_LABELS), DRYBEAN_LABELS),
    )


# ---- California Housing

OCEAN = ("<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN")
BEDROOMS_IMPUTE = 435.0
TARGET_MEAN, TARGET_STD = 206856.0, 115396.0
# name: (mean, std, minimum, maximum)
HOUSING_FIELDS: dict[str, tuple[float, float, float, float]] = {
    "longitude": (-119.6, 2.0, -125.0, -114.0),
    "latitude": (35.6, 2.1, 32.0, 42.0),
    "housing_median_age": (28.6, 12.6, 0.0, 100.0),
    "total_rooms": (2636.0, 2182.0, 1.0, 100000.0),
    "total_bedrooms": (538.0, 421.0, 1.0, 20000.0),
    "population": (1425.0, 1132.0, 1.0, 100000.0),
    "households": (500.0, 382.0, 1.0, 20000.0),  # >= 1: it is a ratio denominator
    "median_income": (3.87, 1.9, 0.0, 20.0),
}


def housing_spec() -> TabularInput:
    fields: list[NumericField | CategoricalField] = [
        NumericField(
            name=n,
            minimum=lo,
            maximum=hi,
            impute=BEDROOMS_IMPUTE if n == "total_bedrooms" else None,
        )
        for n, (_, _, lo, hi) in HOUSING_FIELDS.items()
    ]
    fields.append(CategoricalField(name="ocean_proximity", vocabulary=OCEAN))
    features = (
        *(PassthroughFeature(source=n) for n in HOUSING_FIELDS),
        OneHotFeature(source="ocean_proximity"),
        RatioFeature(name="rooms_per_household", numerator="total_rooms", denominator="households"),
        RatioFeature(
            name="population_per_household", numerator="population", denominator="households"
        ),
    )
    return TabularInput(fields=tuple(fields), features=features)


def housing_features(record: Record) -> list[float]:
    """Reference feature engineering, written independently of the API's ``to_vector``."""
    values = {n: record[n] for n in HOUSING_FIELDS}
    if values["total_bedrooms"] is None:
        values["total_bedrooms"] = BEDROOMS_IMPUTE
    cols = [float(values[n]) for n in HOUSING_FIELDS]  # type: ignore[arg-type]
    cols.extend(1.0 if record["ocean_proximity"] == v else 0.0 for v in OCEAN)
    cols.append(float(values["total_rooms"]) / float(values["households"]))  # type: ignore[arg-type]
    cols.append(float(values["population"]) / float(values["households"]))  # type: ignore[arg-type]
    return cols


def _housing_record(rng: np.random.Generator, null_bedrooms: bool) -> Record:
    record: Record = {}
    for name, (mean, std, lo, hi) in HOUSING_FIELDS.items():
        record[name] = float(np.clip(rng.normal(mean, std), lo + 0.5, hi))
    record["total_bedrooms"] = None if null_bedrooms else record["total_bedrooms"]
    record["ocean_proximity"] = OCEAN[int(rng.integers(len(OCEAN)))]
    return record


def build_housing(
    root: Path, version: str = DUMMY_VERSION, status: Status = Status.DUMMY, seed: int = 12
) -> Path:
    rng = np.random.default_rng(seed)
    fit = np.asarray(
        [housing_features(_housing_record(rng, null_bedrooms=False)) for _ in range(2000)]
    )
    layers: tuple[MlpLayer, ...] = (
        NormalizeLayer(key="norm"),
        DenseLayer(key="d0", activation="relu"),
        DenseLayer(key="d1", activation="relu"),
        DenseLayer(key="out", activation="linear"),
        DenormalizeLayer(key="target"),
    )
    width = fit.shape[1]
    arrays: Arrays = {
        **_stats("norm", fit.mean(axis=0), fit.var(axis=0)),
        **_dense(rng, "d0", width, 16, 1.0),
        **_dense(rng, "d1", 16, 8, 1.0),
        **_dense(rng, "out", 8, 1, 0.5),
        **_stats("target", [TARGET_MEAN], [TARGET_STD**2]),
    }
    records = [_housing_record(rng, null_bedrooms=(i % 4 == 1)) for i in range(24)]

    def labels_for(i: int, row: F32) -> float:
        # a deterministic "ground truth" around the prediction (so RMSE is non-zero)
        return float(row[0] * (1.0 + 0.15 * np.sin(i * 1.7)))

    return _tabular_artifact(
        out_dir=root / "housing-mlp" / version,
        name="housing-mlp",
        version=version,
        status=status,
        description="DUMMY California Housing regressor (median house value, USD); random weights.",
        task=Task.REGRESSION,
        spec=housing_spec(),
        output=RegressionOutput(target="median_house_value", unit="USD"),
        layers=layers,
        arrays=arrays,
        records=records,
        featurize=housing_features,
        labels_for=labels_for,
        # outputs are ~1e5 USD: float32 resolution is ~0.01, so allow a cent-level tolerance
        tolerance=GoldenTolerance(output_atol=0.5),
        quality=_rmse_quality,
    )


# --------------------------------------------------------------------------- image model

INTEL_LABELS = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMAGE_SIZE = 150
POOL = 30  # 150 / 30 = 5x5 grid
# Mean RGB of each synthetic class, loosely evocative (a dummy "scene" is a colour field).
_BASE_COLOURS = {
    "buildings": (150, 135, 120),
    "forest": (40, 100, 45),
    "glacier": (205, 225, 240),
    "mountain": (125, 105, 90),
    "sea": (30, 90, 160),
    "street": (80, 80, 85),
}


def synthetic_image(label: str, rng: np.random.Generator) -> NDArray[np.uint8]:
    """A 150x150 RGB image: class colour, vertical gradient and a little noise."""
    base = np.asarray(_BASE_COLOURS[label], dtype=np.float64)
    gradient = np.linspace(-25.0, 25.0, IMAGE_SIZE)[:, None, None]
    # coarse blocks (cheap to compress, keeps the committed PNGs small) + a touch of fine noise
    blocks = rng.normal(0.0, 8.0, size=(IMAGE_SIZE // 10, IMAGE_SIZE // 10, 3))
    noise = np.kron(blocks, np.ones((10, 10, 1))) + rng.normal(
        0.0, 1.0, (IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    pixels = base[None, None, :] + gradient * (1.0 if label != "sea" else -1.0) + noise
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def pooled_features(images: F32) -> F32:
    """Reference for Mul(1/255) -> NCHW -> AveragePool(30) -> Flatten, on NHWC input."""
    x = images / np.float32(255.0)
    n = x.shape[0]
    grid = IMAGE_SIZE // POOL
    pooled = x.reshape(n, grid, POOL, grid, POOL, 3).mean(axis=(2, 4))  # (n, 5, 5, 3)
    return np.ascontiguousarray(pooled.transpose(0, 3, 1, 2).reshape(n, -1)).astype(np.float32)


def cnn_reference(images: F32, kernel: F32, bias: F32) -> F32:
    logits = pooled_features(images) @ kernel + bias
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def cnn_onnx(kernel: F32, bias: F32) -> bytes:
    n_features = kernel.shape[0]
    inits = [
        numpy_helper.from_array(np.asarray(1 / 255, dtype=np.float32), name="rescale"),
        numpy_helper.from_array(kernel, name="fc.kernel"),
        numpy_helper.from_array(bias, name="fc.bias"),
    ]
    nodes = [
        helper.make_node("Mul", ["inputs", "rescale"], ["scaled"]),
        helper.make_node("Transpose", ["scaled"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "AveragePool", ["nchw"], ["pooled"], kernel_shape=[POOL, POOL], strides=[POOL, POOL]
        ),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "fc.kernel"], ["logits0"]),
        helper.make_node("Add", ["logits0", "fc.bias"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["output_0"], axis=-1),
    ]
    assert kernel.shape == (n_features, len(INTEL_LABELS))  # noqa: S101 - generator sanity check
    return _serialize(
        nodes, inits, [("inputs", ["batch", IMAGE_SIZE, IMAGE_SIZE, 3])], ("output_0", ["batch", 6])
    )


def build_intel(
    root: Path, version: str = DUMMY_VERSION, status: Status = Status.DUMMY, seed: int = 13
) -> Path:
    rng = np.random.default_rng(seed)
    # "Train" a nearest-centroid classifier on synthetic images: logits = -k * ||x - c||^2.
    centroids = np.stack(
        [
            pooled_features(
                np.stack([synthetic_image(label, rng) for _ in range(8)]).astype(np.float32)
            ).mean(axis=0)
            for label in INTEL_LABELS
        ]
    )
    k = np.float32(1.5)
    kernel = (2 * k * centroids.T).astype(np.float32)  # (75, 6)
    bias = (-k * (centroids**2).sum(axis=1)).astype(np.float32)

    out_dir = root / "intel-cnn" / version
    golden_dir = out_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    files, labels = [], []
    for i in range(12):
        label = INTEL_LABELS[i % len(INTEL_LABELS)]
        name = f"g{i:02d}_{label}.png"
        Image.fromarray(synthetic_image(label, rng)).save(golden_dir / name, format="PNG")
        files.append(name)
        labels.append(label)
    images = np.stack(
        [np.asarray(Image.open(golden_dir / f).convert("RGB")) for f in files]
    ).astype(np.float32)
    outputs = cnn_reference(images, kernel, bias)
    onnx_bytes = cnn_onnx(kernel, bias)
    max_diff = float(np.max(np.abs(ort_run(onnx_bytes, images) - outputs)))
    accuracy = float(
        np.mean(
            [INTEL_LABELS[int(np.argmax(o))] == y for o, y in zip(outputs, labels, strict=True)]
        )
    )

    golden = ImageGolden(
        model="intel-cnn",
        version=version,
        cases=tuple(
            ImageGoldenCase(file=f, output=tuple(_round(v) for v in o), label=y)
            for f, o, y in zip(files, outputs, labels, strict=True)
        ),
    )
    golden_sha = _write(out_dir / "golden.json", (golden.model_dump_json(indent=2) + "\n").encode())
    onnx_sha = _write(out_dir / "model.onnx", onnx_bytes)
    in_spec, out_spec = _tensors((None, IMAGE_SIZE, IMAGE_SIZE, 3), len(INTEL_LABELS))
    manifest = Manifest(
        name="intel-cnn",
        version=version,
        status=status,
        task=Task.IMAGE_CLASSIFICATION,
        description="DUMMY Intel scenes classifier (150x150 RGB, 6 classes): pooling + linear.",
        created_at=CREATED_AT,
        input=ImageInput(height=IMAGE_SIZE, width=IMAGE_SIZE),
        output=ClassificationOutput(labels=INTEL_LABELS),
        inputs=(in_spec,),
        outputs=(out_spec,),
        runtimes=(OnnxRuntimeSpec(file="model.onnx", sha256=onnx_sha, opset=OPSET),),
        metrics={"validation": {"accuracy": accuracy}, "test": {"accuracy": accuracy}},
        quality_gate=QualityGate(
            metric="accuracy", split="validation", threshold=0.5, value=accuracy
        ),
        serving_skew=ServingSkew(
            samples=len(files), max_abs_diff=max_diff, top1_agreement=1.0, accuracy_delta_pp=0.0
        ),
        training=_training_info("intel-cnn"),
        golden=GoldenRef(file="golden.json", sha256=golden_sha),
    )
    return _finish(out_dir, manifest, _results(manifest))


# --------------------------------------------------------------------------- CLI

BUILDERS: dict[str, Callable[..., Path]] = {
    "drybean-mlp": build_drybean,
    "housing-mlp": build_housing,
    "intel-cnn": build_intel,
}


def build_all(
    root: Path,
    version: str = DUMMY_VERSION,
    status: Status = Status.DUMMY,
    models: Sequence[str] = MODELS,
) -> dict[str, Path]:
    """Write the requested dummy artifacts under ``root/<name>/<version>/``."""
    return {name: BUILDERS[name](root, version, status) for name in models}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    default_out = Path(__file__).resolve().parents[1] / "artifacts"
    parser.add_argument("--out", type=Path, default=default_out, help="artifacts root directory")
    parser.add_argument("--version", default=DUMMY_VERSION)
    parser.add_argument("--only", choices=MODELS, action="append", help="restrict to a model")
    args = parser.parse_args(argv)
    for name, path in build_all(args.out, args.version, models=args.only or MODELS).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
