"""Generate DUMMY artifacts for the three TriNet models (real schemas, fake weights).

Models: ``credit-default-mlp`` (binary), ``volatility-mlp`` (regression, ends in ``exp``) and
``candlestick-cnn`` (64x64 PNG charts). Educational project: nothing here is financial advice.

They let the API, the image and the CI run end to end before the notebook exports the real
models. Everything is deterministic (seeded RNGs, fixed timestamps, reproducible ``.npz`` and
ONNX bytes), so re-running the script does not churn the committed files.

Usage::

    uv run python tools/make_dummy_artifacts.py            # writes artifacts/*/0.0.0-dummy/
    uv run python tools/make_dummy_artifacts.py --out /tmp/x --version 0.0.1-dummy

The functions are also imported by the API test-suite to build artifacts in temporary
directories (``build_all``, ``build_credit``, ``build_volatility``, ``build_candlestick``).
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
    ExpLayer,
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
MODELS = ("credit-default-mlp", "volatility-mlp", "candlestick-cnn")
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
    out: NDArray[Any] = x.astype(np.float32)
    for layer in layers:
        match layer:
            case NormalizeLayer(key=key):
                out = (out - arrays[f"{key}.mean"]) / _std(arrays[f"{key}.variance"])
            case DenormalizeLayer(key=key):
                out = out * _std(arrays[f"{key}.variance"]) + arrays[f"{key}.mean"]
            case ExpLayer():
                out = np.exp(out)
            case DenseLayer(key=key, activation=activation):
                out = out @ arrays[f"{key}.kernel"] + arrays[f"{key}.bias"]
                if activation == "relu":
                    out = np.maximum(out, np.float32(0))
                elif activation == "softmax":
                    e = np.exp(out - out.max(axis=-1, keepdims=True))
                    out = e / e.sum(axis=-1, keepdims=True)
        out = out.astype(np.float32)
    return np.asarray(out, dtype=np.float32)


def _dense(rng: np.random.Generator, key: str, n_in: int, n_out: int, gain: float) -> Arrays:
    kernel = rng.normal(0.0, gain * np.sqrt(2.0 / n_in), size=(n_in, n_out))
    bias = rng.normal(0.0, 0.05, size=(n_out,))
    return {f"{key}.kernel": kernel.astype(np.float32), f"{key}.bias": bias.astype(np.float32)}


def _stats(
    key: str, mean: Sequence[float] | NDArray[Any], variance: Sequence[float] | NDArray[Any]
) -> Arrays:
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
        match layer:
            case NormalizeLayer(key=key) | DenormalizeLayer(key=key):
                mean = const(f"{key}.mean", arrays[f"{key}.mean"])
                std = const(f"{key}.std", _std(arrays[f"{key}.variance"]))
                if isinstance(layer, NormalizeLayer):
                    nodes.append(helper.make_node("Sub", [current, mean], [f"{out}a"]))
                    nodes.append(helper.make_node("Div", [f"{out}a", std], [out]))
                else:
                    nodes.append(helper.make_node("Mul", [current, std], [f"{out}a"]))
                    nodes.append(helper.make_node("Add", [f"{out}a", mean], [out]))
            case ExpLayer():
                nodes.append(helper.make_node("Exp", [current], [out]))
            case DenseLayer(key=key, activation=activation):
                kernel = const(f"{key}.kernel", arrays[f"{key}.kernel"])
                bias = const(f"{key}.bias", arrays[f"{key}.bias"])
                nodes.append(helper.make_node("MatMul", [current, kernel], [f"{out}a"]))
                nodes.append(helper.make_node("Add", [f"{out}a", bias], [f"{out}b"]))
                if activation == "relu":
                    nodes.append(helper.make_node("Relu", [f"{out}b"], [out]))
                elif activation == "softmax":
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


# ---- Default of credit card clients (binary classification)

CREDIT_LABELS = ("no_default", "default")
EDUCATION = ("graduate", "university", "high_school", "other")
CREDIT_MONTHS = range(1, 7)
# Numeric raw fields, in request/feature order: name -> (mean, std, minimum, maximum).
# Statistics only make the dummy look plausible (roughly the UCI #350 values).
CREDIT_NUMERIC: dict[str, tuple[float, float, float, float]] = {
    "limit_bal": (167484.0, 129747.0, 1.0, 10_000_000.0),  # >= 1: it is a ratio denominator
    "age": (35.5, 9.2, 18.0, 120.0),
    **{f"pay_{m}": (0.0, 1.1, -2.0, 8.0) for m in CREDIT_MONTHS},
    **{f"bill_amt{m}": (50000.0, 70000.0, -10_000_000.0, 10_000_000.0) for m in CREDIT_MONTHS},
    **{f"pay_amt{m}": (5500.0, 14000.0, 0.0, 10_000_000.0) for m in CREDIT_MONTHS},
}
_INTEGER_FIELDS = frozenset({"age", *(f"pay_{m}" for m in CREDIT_MONTHS)})


def credit_spec() -> TabularInput:
    fields: list[NumericField | CategoricalField] = [
        NumericField(name="limit_bal", minimum=1.0, maximum=CREDIT_NUMERIC["limit_bal"][3]),
        CategoricalField(name="education", vocabulary=EDUCATION),
    ]
    fields.extend(
        NumericField(name=n, minimum=lo, maximum=hi)
        for n, (_, _, lo, hi) in CREDIT_NUMERIC.items()
        if n != "limit_bal"
    )
    features: tuple[PassthroughFeature | OneHotFeature | RatioFeature, ...] = (
        *(PassthroughFeature(source=n) for n in CREDIT_NUMERIC),
        OneHotFeature(source="education"),
        RatioFeature(name="bill_utilisation", numerator="bill_amt1", denominator="limit_bal"),
    )
    return TabularInput(fields=tuple(fields), features=features)


def credit_features(record: Record) -> list[float]:
    """Reference feature engineering, written independently of the API's ``to_vector``."""
    cols = [float(record[n]) for n in CREDIT_NUMERIC]  # type: ignore[arg-type]
    cols.extend(1.0 if record["education"] == v else 0.0 for v in EDUCATION)
    cols.append(float(record["bill_amt1"]) / float(record["limit_bal"]))  # type: ignore[arg-type]
    return cols


def credit_record(rng: np.random.Generator) -> Record:
    record: Record = {}
    for name, (mean, std, lo, hi) in CREDIT_NUMERIC.items():
        if name == "limit_bal":
            value = float(np.clip(np.exp(rng.normal(11.8, 0.8)), 10000.0, 1_000_000.0))
        else:
            value = float(np.clip(rng.normal(mean, std), lo, hi))
        record[name] = float(np.rint(value)) if name in _INTEGER_FIELDS else round(value, 2)
    record["education"] = EDUCATION[int(rng.integers(len(EDUCATION)))]
    return record


def build_credit(
    root: Path, version: str = DUMMY_VERSION, status: Status = Status.DUMMY, seed: int = 11
) -> Path:
    rng = np.random.default_rng(seed)
    fit = np.asarray([credit_features(credit_record(rng)) for _ in range(2000)])
    layers: tuple[MlpLayer, ...] = (
        NormalizeLayer(key="norm"),
        DenseLayer(key="d0", activation="relu"),
        DenseLayer(key="d1", activation="relu"),
        DenseLayer(key="out", activation="softmax"),
    )
    width = fit.shape[1]
    arrays: Arrays = {
        **_stats("norm", fit.mean(axis=0), fit.var(axis=0)),
        **_dense(rng, "d0", width, 64, 1.0),
        **_dense(rng, "d1", 64, 32, 1.0),
        **_dense(rng, "out", 32, len(CREDIT_LABELS), 1.4),
    }
    records = [credit_record(rng) for _ in range(24)]

    def labels_for(i: int, row: F32) -> str:
        # ~2/3 of the labels agree with the model, so the metric is not trivially 1.0
        pick = int(np.argmax(row))
        return CREDIT_LABELS[pick if i % 3 else (pick + 1) % len(CREDIT_LABELS)]

    return _tabular_artifact(
        out_dir=root / "credit-default-mlp" / version,
        name="credit-default-mlp",
        version=version,
        status=status,
        description=(
            "DUMMY credit card default classifier (UCI 350 schema, binary); random weights. "
            "Educational only: not a real credit decision model."
        ),
        task=Task.CLASSIFICATION,
        spec=credit_spec(),
        output=ClassificationOutput(labels=CREDIT_LABELS),
        layers=layers,
        arrays=arrays,
        records=records,
        featurize=credit_features,
        labels_for=labels_for,
        tolerance=GoldenTolerance(),
        quality=_accuracy_quality(len(CREDIT_LABELS), CREDIT_LABELS),
    )


# ---- Realised volatility, next 5 sessions (regression in log space)

DOW = ("mon", "tue", "wed", "thu", "fri")
VOL_UNIT = "volatilidade anualizada (fração)"
LOG_TARGET_MEAN, LOG_TARGET_STD = float(np.log(0.25)), 0.5
# name: (mean, std, minimum, maximum), in request order (the categorical "dow" comes last).
VOL_NUMERIC: dict[str, tuple[float, float, float, float]] = {
    **{f"ret_{d}d": (0.0, 0.015, -0.4, 0.4) for d in range(1, 6)},
    "rv_5": (0.25, 0.15, 0.0, 10.0),
    "rv_10": (0.25, 0.14, 0.0, 10.0),
    "rv_20": (0.25, 0.13, 1e-6, 10.0),  # > 0: it is a ratio denominator
    "rv_60": (0.26, 0.12, 0.0, 10.0),
    "pk_5": (0.22, 0.12, 0.0, 10.0),
    "pk_20": (0.22, 0.10, 0.0, 10.0),
    "range_5": (0.017, 0.010, 0.0, 1.0),
    "volume_ratio": (1.0, 0.4, 1e-6, 1000.0),
    "log_dollar_volume": (18.5, 1.5, 0.0, 40.0),
    "gap": (0.0005, 0.008, -1.0, 10.0),
    "mom_20": (0.01, 0.07, -1.0, 10.0),
    "mom_60": (0.03, 0.12, -1.0, 10.0),
    "drawdown_60": (-0.06, 0.06, -1.0, 0.0),
    "mkt_rv_20": (0.14, 0.06, 0.0, 5.0),
}


def volatility_spec() -> TabularInput:
    fields: list[NumericField | CategoricalField] = [
        NumericField(name=n, minimum=lo, maximum=hi) for n, (_, _, lo, hi) in VOL_NUMERIC.items()
    ]
    fields.append(CategoricalField(name="dow", vocabulary=DOW))
    features: tuple[PassthroughFeature | OneHotFeature | RatioFeature, ...] = (
        *(PassthroughFeature(source=n) for n in VOL_NUMERIC),
        OneHotFeature(source="dow"),
        RatioFeature(name="rv_ratio", numerator="rv_5", denominator="rv_20"),
    )
    return TabularInput(fields=tuple(fields), features=features)


def volatility_features(record: Record) -> list[float]:
    """Reference feature engineering, written independently of the API's ``to_vector``."""
    cols = [float(record[n]) for n in VOL_NUMERIC]  # type: ignore[arg-type]
    cols.extend(1.0 if record["dow"] == v else 0.0 for v in DOW)
    cols.append(float(record["rv_5"]) / float(record["rv_20"]))  # type: ignore[arg-type]
    return cols


def volatility_record(rng: np.random.Generator) -> Record:
    record: Record = {}
    for name, (mean, std, lo, hi) in VOL_NUMERIC.items():
        floor = lo if lo < 0 else max(lo, 1e-4)  # keep strictly positive values away from 0
        record[name] = round(float(np.clip(rng.normal(mean, std), floor, hi)), 6)
    record["dow"] = DOW[int(rng.integers(len(DOW)))]
    return record


def build_volatility(
    root: Path, version: str = DUMMY_VERSION, status: Status = Status.DUMMY, seed: int = 12
) -> Path:
    rng = np.random.default_rng(seed)
    fit = np.asarray([volatility_features(volatility_record(rng)) for _ in range(2000)])
    # normalize -> dense -> dense -> dense -> denormalize -> exp: the same tail the exported
    # inference clone has, so the NumPy fallback exercises ExpLayer.
    layers: tuple[MlpLayer, ...] = (
        NormalizeLayer(key="norm"),
        DenseLayer(key="d0", activation="relu"),
        DenseLayer(key="d1", activation="relu"),
        DenseLayer(key="out", activation="linear"),
        DenormalizeLayer(key="target"),
        ExpLayer(),
    )
    width = fit.shape[1]
    arrays: Arrays = {
        **_stats("norm", fit.mean(axis=0), fit.var(axis=0)),
        **_dense(rng, "d0", width, 16, 1.0),
        **_dense(rng, "d1", 16, 8, 1.0),
        **_dense(rng, "out", 8, 1, 0.5),
        **_stats("target", [LOG_TARGET_MEAN], [LOG_TARGET_STD**2]),
    }
    records = [volatility_record(rng) for _ in range(24)]

    def labels_for(i: int, row: F32) -> float:
        # a deterministic "ground truth" around the prediction (so RMSE is non-zero)
        return float(row[0] * (1.0 + 0.15 * np.sin(i * 1.7)))

    return _tabular_artifact(
        out_dir=root / "volatility-mlp" / version,
        name="volatility-mlp",
        version=version,
        status=status,
        description=(
            "DUMMY next-5-sessions realised volatility regressor (annualised fraction, "
            "exp of a log-space MLP); random weights. Educational only: not investment advice."
        ),
        task=Task.REGRESSION,
        spec=volatility_spec(),
        output=RegressionOutput(target="realized_volatility_5d", unit=VOL_UNIT),
        layers=layers,
        arrays=arrays,
        records=records,
        featurize=volatility_features,
        labels_for=labels_for,
        tolerance=GoldenTolerance(),
        quality=_rmse_quality,
    )


# --------------------------------------------------------------------------- image model

CANDLE_LABELS = ("baixa", "media", "alta")
IMAGE_SIZE = 64
POOL = 16  # 64 / 16 = 4x4 grid
SESSIONS = 20
# Daily volatility of the synthetic geometric random walk behind each regime.
_DAILY_SIGMA = {"baixa": 0.004, "media": 0.012, "alta": 0.03}
_PRICE_SPAN = 0.12  # +-12 % around the first close spans the plot area


def synthetic_chart(label: str, rng: np.random.Generator) -> NDArray[np.uint8]:
    """A 64x64 RGB chart of 20 synthetic sessions (never derived from real market data).

    Channel 0: OHLC bars (open tick left, close tick right); channel 1: running mean of the
    close; channel 2: volume bars from the bottom edge. Fixed vertical scale, no antialiasing.
    """
    sigma = _DAILY_SIGMA[label]
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, SESSIONS)))
    opens = np.concatenate(([100.0], closes[:-1]))
    wick = np.abs(rng.normal(0.0, sigma * 0.6, (2, SESSIONS)))
    highs = np.maximum(opens, closes) * (1.0 + wick[0])
    lows = np.minimum(opens, closes) * (1.0 - wick[1])
    volume = rng.lognormal(0.0, 0.4, SESSIONS)
    mean_close = np.cumsum(closes) / np.arange(1, SESSIONS + 1)

    def row(price: float) -> int:
        y = 31.5 - (price / 100.0 - 1.0) / _PRICE_SPAN * 31.5
        return int(np.clip(np.rint(y), 0, IMAGE_SIZE - 1))

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i in range(SESSIONS):
        x = 3 + 3 * i
        image[row(highs[i]) : row(lows[i]) + 1, x, 0] = 255
        image[row(opens[i]), x - 1, 0] = 255
        image[row(closes[i]), x + 1, 0] = 255
        image[row(mean_close[i]), x - 1 : x + 2, 1] = 255
        height = int(np.rint(volume[i] / volume.max() * 40))
        if height:
            image[IMAGE_SIZE - height :, x, 2] = 255
    return image


def pooled_features(images: F32) -> F32:
    """Reference for Mul(1/255) -> NCHW -> AveragePool(16) -> Flatten, on NHWC input."""
    x = images / np.float32(255.0)
    n = x.shape[0]
    grid = IMAGE_SIZE // POOL
    pooled = x.reshape(n, grid, POOL, grid, POOL, 3).mean(axis=(2, 4))  # (n, 4, 4, 3)
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
    assert kernel.shape == (n_features, len(CANDLE_LABELS))  # noqa: S101 - generator sanity check
    return _serialize(
        nodes,
        inits,
        [("inputs", ["batch", IMAGE_SIZE, IMAGE_SIZE, 3])],
        ("output_0", ["batch", len(CANDLE_LABELS)]),
    )


def build_candlestick(
    root: Path, version: str = DUMMY_VERSION, status: Status = Status.DUMMY, seed: int = 13
) -> Path:
    rng = np.random.default_rng(seed)
    # "Train" a nearest-centroid classifier on synthetic charts: logits = -k * ||x - c||^2.
    centroids = np.stack(
        [
            pooled_features(
                np.stack([synthetic_chart(label, rng) for _ in range(16)]).astype(np.float32)
            ).mean(axis=0)
            for label in CANDLE_LABELS
        ]
    )
    k = np.float32(40.0)
    kernel = (2 * k * centroids.T).astype(np.float32)  # (48, 3)
    bias = (-k * (centroids**2).sum(axis=1)).astype(np.float32)

    out_dir = root / "candlestick-cnn" / version
    golden_dir = out_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    files, labels = [], []
    for i in range(12):
        label = CANDLE_LABELS[i % len(CANDLE_LABELS)]
        name = f"g{i:02d}_{label}.png"
        Image.fromarray(synthetic_chart(label, rng)).save(golden_dir / name, format="PNG")
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
            [CANDLE_LABELS[int(np.argmax(o))] == y for o, y in zip(outputs, labels, strict=True)]
        )
    )

    golden = ImageGolden(
        model="candlestick-cnn",
        version=version,
        cases=tuple(
            ImageGoldenCase(file=f, output=tuple(_round(v) for v in o), label=y)
            for f, o, y in zip(files, outputs, labels, strict=True)
        ),
    )
    golden_sha = _write(out_dir / "golden.json", (golden.model_dump_json(indent=2) + "\n").encode())
    onnx_sha = _write(out_dir / "model.onnx", onnx_bytes)
    in_spec, out_spec = _tensors((None, IMAGE_SIZE, IMAGE_SIZE, 3), len(CANDLE_LABELS))
    manifest = Manifest(
        name="candlestick-cnn",
        version=version,
        status=status,
        task=Task.IMAGE_CLASSIFICATION,
        description=(
            "DUMMY volatility-regime classifier on 64x64 OHLC chart images (3 classes): "
            "pooling + linear. Educational only: not investment advice."
        ),
        created_at=CREATED_AT,
        input=ImageInput(
            height=IMAGE_SIZE, width=IMAGE_SIZE, formats=("PNG",), max_bytes=1_048_576
        ),
        output=ClassificationOutput(labels=CANDLE_LABELS),
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
        training=_training_info("candlestick-cnn"),
        golden=GoldenRef(file="golden.json", sha256=golden_sha),
    )
    return _finish(out_dir, manifest, _results(manifest))


# --------------------------------------------------------------------------- CLI

BUILDERS: dict[str, Callable[..., Path]] = {
    "credit-default-mlp": build_credit,
    "volatility-mlp": build_volatility,
    "candlestick-cnn": build_candlestick,
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
