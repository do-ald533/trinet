"""Artifact manifest: the contract between the training notebook and the inference API.

The notebook writes ``artifacts/<name>/<version>/manifest.json``; the API validates it
with this model before loading anything. Preprocessing that lives *outside* the graph is
restricted to three closed primitives (constant imputation, one-hot by vocabulary and
A/B ratio) so both sides can implement it identically. Normalisation and rescaling live
*inside* the graph.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

_SLUG = r"^[a-z0-9][a-z0-9-]{0,62}$"
_SEMVER = r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$"
_SHA256 = r"^[0-9a-f]{64}$"
# Plain file name inside the artifact directory: no separators, no traversal, no dotfiles.
_FILE_NAME = r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$"
_IDENT = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"

FileName = Annotated[str, Field(pattern=_FILE_NAME)]
Sha256 = Annotated[str, Field(pattern=_SHA256)]
Identifier = Annotated[str, Field(pattern=_IDENT)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Task(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    IMAGE_CLASSIFICATION = "image_classification"


class Status(StrEnum):
    DUMMY = "dummy"
    CANDIDATE = "candidate"
    RELEASE = "release"


class TensorSpec(_Strict):
    """A graph input/output tensor. ``None`` in ``shape`` marks a dynamic (batch) axis."""

    name: str = Field(min_length=1)
    dtype: Literal["float32"] = "float32"
    shape: tuple[int | None, ...] = Field(min_length=1)


# --------------------------------------------------------------------------- tabular input


class NumericField(_Strict):
    """Raw numeric request field. ``impute`` is used when the value is null/missing."""

    kind: Literal["numeric"] = "numeric"
    name: Identifier
    description: str = ""
    impute: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _bounds(self) -> NumericField:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"field {self.name!r}: minimum > maximum")
        return self


class CategoricalField(_Strict):
    """Raw categorical request field with a closed vocabulary (fitted on the training split)."""

    kind: Literal["categorical"] = "categorical"
    name: Identifier
    description: str = ""
    vocabulary: tuple[str, ...] = Field(min_length=1)
    impute: str | None = None

    @model_validator(mode="after")
    def _vocabulary(self) -> CategoricalField:
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError(f"field {self.name!r}: duplicated vocabulary entries")
        if self.impute is not None and self.impute not in self.vocabulary:
            raise ValueError(f"field {self.name!r}: impute value not in vocabulary")
        return self


InputField = Annotated[NumericField | CategoricalField, Field(discriminator="kind")]


class PassthroughFeature(_Strict):
    """One column: the (imputed) value of a numeric field."""

    kind: Literal["passthrough"] = "passthrough"
    source: Identifier


class OneHotFeature(_Strict):
    """``len(vocabulary)`` columns, in vocabulary order, for a categorical field."""

    kind: Literal["one_hot"] = "one_hot"
    source: Identifier


class RatioFeature(_Strict):
    """One column: ``numerator / denominator`` of two (imputed) numeric fields."""

    kind: Literal["ratio"] = "ratio"
    name: Identifier
    numerator: Identifier
    denominator: Identifier


Feature = Annotated[PassthroughFeature | OneHotFeature | RatioFeature, Field(discriminator="kind")]


class TabularInput(_Strict):
    """Raw named fields -> ordered float32 feature vector (``features`` order = column order)."""

    kind: Literal["tabular"] = "tabular"
    fields: tuple[InputField, ...] = Field(min_length=1)
    features: tuple[Feature, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _references(self) -> TabularInput:
        by_name = {f.name: f for f in self.fields}
        if len(by_name) != len(self.fields):
            raise ValueError("duplicated field names")
        for feature in self.features:
            match feature:
                case PassthroughFeature(source=src):
                    _expect(by_name, src, NumericField)
                case OneHotFeature(source=src):
                    _expect(by_name, src, CategoricalField)
                case RatioFeature(numerator=num, denominator=den):
                    _expect(by_name, num, NumericField)
                    _expect(by_name, den, NumericField)
        names = self.feature_names()
        if len(set(names)) != len(names):
            raise ValueError("duplicated feature column names")
        return self

    def field(self, name: str) -> NumericField | CategoricalField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def feature_names(self) -> list[str]:
        """Column names of the feature vector, in order (one-hot expands to ``src=value``)."""
        names: list[str] = []
        for feature in self.features:
            match feature:
                case PassthroughFeature(source=src):
                    names.append(src)
                case OneHotFeature(source=src):
                    field = self.field(src)
                    assert isinstance(field, CategoricalField)  # noqa: S101 - guaranteed by validator
                    names.extend(f"{src}={value}" for value in field.vocabulary)
                case RatioFeature(name=name):
                    names.append(name)
        return names

    @property
    def width(self) -> int:
        return len(self.feature_names())


def _expect(
    by_name: dict[str, NumericField | CategoricalField],
    name: str,
    kind: type[NumericField] | type[CategoricalField],
) -> None:
    field = by_name.get(name)
    if field is None:
        raise ValueError(f"feature references unknown field {name!r}")
    if not isinstance(field, kind):
        raise ValueError(f"field {name!r} must be {kind.__name__}")


class ImageInput(_Strict):
    """Encoded image -> RGB float32 array in ``value_range`` (rescaling happens in the graph)."""

    kind: Literal["image"] = "image"
    height: int = Field(gt=0, le=4096)
    width: int = Field(gt=0, le=4096)
    channels: Literal[3] = 3
    resample: Literal["bilinear"] = "bilinear"
    value_range: tuple[float, float] = (0.0, 255.0)
    max_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    formats: tuple[Literal["JPEG", "PNG", "WEBP"], ...] = ("JPEG", "PNG")


ModelInput = Annotated[TabularInput | ImageInput, Field(discriminator="kind")]


# --------------------------------------------------------------------------- output


class ClassificationOutput(_Strict):
    kind: Literal["classification"] = "classification"
    labels: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _unique(self) -> ClassificationOutput:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("duplicated labels")
        return self


class RegressionOutput(_Strict):
    kind: Literal["regression"] = "regression"
    target: str = Field(min_length=1)
    unit: str = ""


ModelOutput = Annotated[ClassificationOutput | RegressionOutput, Field(discriminator="kind")]


# --------------------------------------------------------------------------- runtimes
# Ordered by preference in ``Manifest.runtimes``: the API walks the list (fallback ladder).


class OnnxRuntimeSpec(_Strict):
    kind: Literal["onnx"] = "onnx"
    file: FileName
    sha256: Sha256
    opset: int = Field(ge=7)


class NormalizeLayer(_Strict):
    """``(x - mean) / sqrt(variance)``; arrays ``<key>.mean`` and ``<key>.variance``."""

    op: Literal["normalize"] = "normalize"
    key: Identifier


class DenseLayer(_Strict):
    """``activation(x @ kernel + bias)``; arrays ``<key>.kernel`` and ``<key>.bias``."""

    op: Literal["dense"] = "dense"
    key: Identifier
    activation: Literal["relu", "linear", "softmax"]


class DenormalizeLayer(_Strict):
    """``x * sqrt(variance) + mean`` (Keras ``Normalization(invert=True)``)."""

    op: Literal["denormalize"] = "denormalize"
    key: Identifier


class ExpLayer(_Strict):
    """``exp(x)`` (Keras ``Activation("exponential")``): log-space targets back to natural units."""

    op: Literal["exp"] = "exp"


MlpLayer = Annotated[
    NormalizeLayer | DenseLayer | DenormalizeLayer | ExpLayer, Field(discriminator="op")
]


class NumpyMlpRuntimeSpec(_Strict):
    """Pure-NumPy fallback for MLPs: weights in an ``.npz`` (loaded with ``allow_pickle=False``)."""

    kind: Literal["numpy_mlp"] = "numpy_mlp"
    file: FileName
    sha256: Sha256
    layers: tuple[MlpLayer, ...] = Field(min_length=1)


class LiteRtRuntimeSpec(_Strict):
    kind: Literal["litert"] = "litert"
    file: FileName
    sha256: Sha256


RuntimeSpec = Annotated[
    OnnxRuntimeSpec | NumpyMlpRuntimeSpec | LiteRtRuntimeSpec, Field(discriminator="kind")
]


# --------------------------------------------------------------------------- provenance


class TrainingInfo(_Strict):
    dataset: str = Field(min_length=1)
    data_sha256: Sha256
    seed: int
    python: str
    tensorflow: str
    keras: str
    hardware: str = ""


class QualityGate(_Strict):
    """Export-time gate: the notebook refuses to write the artifact if it does not pass."""

    metric: str = Field(min_length=1)
    split: Literal["validation", "test"] = "validation"
    threshold: float
    value: float
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold


class ServingSkew(_Strict):
    """Training (Keras) vs serving (exported runtime + API preprocessing) agreement."""

    samples: int = Field(gt=0)
    max_abs_diff: float = Field(ge=0)
    top1_agreement: float | None = Field(default=None, ge=0, le=1)
    accuracy_delta_pp: float | None = None


class GoldenRef(_Strict):
    file: FileName
    sha256: Sha256


# --------------------------------------------------------------------------- manifest

_TASK_SHAPES: dict[Task, tuple[str, str]] = {
    Task.CLASSIFICATION: ("tabular", "classification"),
    Task.REGRESSION: ("tabular", "regression"),
    Task.IMAGE_CLASSIFICATION: ("image", "classification"),
}


class Manifest(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    name: Annotated[str, Field(pattern=_SLUG)]
    version: Annotated[str, Field(pattern=_SEMVER)]
    status: Status
    task: Task
    description: str = ""
    created_at: datetime
    git_sha: str | None = None
    input: ModelInput
    output: ModelOutput
    inputs: tuple[TensorSpec, ...] = Field(min_length=1, max_length=1)
    outputs: tuple[TensorSpec, ...] = Field(min_length=1, max_length=1)
    runtimes: tuple[RuntimeSpec, ...] = Field(min_length=1)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    quality_gate: QualityGate
    serving_skew: ServingSkew | None = None
    training: TrainingInfo
    golden: GoldenRef | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Manifest:
        expected_input, expected_output = _TASK_SHAPES[self.task]
        if self.input.kind != expected_input or self.output.kind != expected_output:
            raise ValueError(
                f"task {self.task.value!r} requires {expected_input} input "
                f"and {expected_output} output"
            )
        self._check_tensor_shapes()
        if isinstance(self.input, ImageInput) and any(
            isinstance(r, NumpyMlpRuntimeSpec) for r in self.runtimes
        ):
            raise ValueError("numpy_mlp runtime only supports tabular models")
        files = [r.file for r in self.runtimes]
        if len(set(files)) != len(files):
            raise ValueError("runtimes must reference distinct files")
        if self.status is Status.RELEASE and not self.quality_gate.passed:
            raise ValueError("release artifacts must pass the quality gate")
        return self

    def _check_tensor_shapes(self) -> None:
        in_shape = self.inputs[0].shape
        out_shape = self.outputs[0].shape
        if isinstance(self.input, TabularInput):
            if in_shape[-1] != self.input.width:
                raise ValueError(
                    f"input tensor width {in_shape[-1]} != feature width {self.input.width}"
                )
        elif tuple(in_shape[1:]) != (self.input.height, self.input.width, self.input.channels):
            raise ValueError("input tensor shape does not match image input spec")
        n_out = len(self.output.labels) if isinstance(self.output, ClassificationOutput) else 1
        if out_shape[-1] != n_out:
            raise ValueError(f"output tensor width {out_shape[-1]} != {n_out}")


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_bytes())
