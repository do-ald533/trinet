"""HTTP DTOs (distinct from the domain entities)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from trinet_api.domain.entities import (
    ClassificationResult,
    ModelBundle,
    ModelEntry,
    Prediction,
    RegressionResult,
    UnavailableModel,
)
from trinet_contract import Manifest, QualityGate, ServingSkew, Status, Task
from trinet_contract.manifest import ModelInput, ModelOutput, TensorSpec


class _Dto(BaseModel):
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- predictions


class PredictionRequest(_Dto):
    """JSON body of a tabular prediction (``extra`` forbidden: typos must not pass silently)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instances: list[dict[str, object]]


class ClassScoreDto(_Dto):
    label: str
    probability: float


class ClassificationPrediction(_Dto):
    kind: Literal["classification"] = "classification"
    label: str
    probability: float
    probabilities: dict[str, float]


class RegressionPrediction(_Dto):
    kind: Literal["regression"] = "regression"
    value: float
    unit: str
    target: str


class ImagePrediction(_Dto):
    kind: Literal["image_classification"] = "image_classification"
    label: str
    probability: float
    top_k: list[ClassScoreDto]


PredictionDto = Annotated[
    ClassificationPrediction | RegressionPrediction | ImagePrediction, Field(discriminator="kind")
]


class PredictionResponse(_Dto):
    model_name: str
    model_version: str
    request_id: str
    predictions: list[PredictionDto]


def prediction_dto(prediction: Prediction, task: Task) -> PredictionDto:
    match prediction:
        case RegressionResult(value=value, unit=unit, target=target):
            return RegressionPrediction(value=value, unit=unit, target=target)
        case ClassificationResult(label=label, probability=probability, scores=scores):
            if task is Task.IMAGE_CLASSIFICATION:
                top = [ClassScoreDto(label=s.label, probability=s.probability) for s in scores]
                return ImagePrediction(label=label, probability=probability, top_k=top)
            return ClassificationPrediction(
                label=label,
                probability=probability,
                probabilities={s.label: s.probability for s in scores},
            )


# --------------------------------------------------------------------------- models


class ModelSummary(_Dto):
    name: str
    version: str | None
    available: bool
    status: Status | None = None
    task: Task | None = None
    runtime: str | None = None
    reason: str | None = Field(default=None, description="Why the model cannot serve.")
    description: str = ""
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)


class ModelList(_Dto):
    models: list[ModelSummary]


class TrainingPublic(_Dto):
    dataset: str
    data_sha256: str
    seed: int
    python: str
    tensorflow: str
    keras: str


class ModelDetail(_Dto):
    """Public view of a manifest: no file names, no file hashes."""

    name: str
    version: str
    status: Status
    task: Task
    description: str
    created_at: datetime
    runtime: str
    input: ModelInput
    output: ModelOutput
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    metrics: dict[str, dict[str, float]]
    quality_gate: QualityGate
    serving_skew: ServingSkew | None
    training: TrainingPublic


def model_summary(entry: ModelEntry) -> ModelSummary:
    if isinstance(entry, UnavailableModel):
        return ModelSummary(
            name=entry.name, version=entry.version, available=False, reason=entry.reason
        )
    manifest = entry.manifest
    return ModelSummary(
        name=manifest.name,
        version=manifest.version,
        available=True,
        status=manifest.status,
        task=manifest.task,
        runtime=entry.runtime.kind,
        description=manifest.description,
        metrics=manifest.metrics,
    )


def model_detail(bundle: ModelBundle) -> ModelDetail:
    manifest: Manifest = bundle.manifest
    training = manifest.training
    return ModelDetail(
        name=manifest.name,
        version=manifest.version,
        status=manifest.status,
        task=manifest.task,
        description=manifest.description,
        created_at=manifest.created_at,
        runtime=bundle.runtime.kind,
        input=manifest.input,
        output=manifest.output,
        inputs=list(manifest.inputs),
        outputs=list(manifest.outputs),
        metrics=manifest.metrics,
        quality_gate=manifest.quality_gate,
        serving_skew=manifest.serving_skew,
        training=TrainingPublic(
            dataset=training.dataset,
            data_sha256=training.data_sha256,
            seed=training.seed,
            python=training.python,
            tensorflow=training.tensorflow,
            keras=training.keras,
        ),
    )


# --------------------------------------------------------------------------- health


class HealthResponse(_Dto):
    status: Literal["ok"] = "ok"


class ReadyResponse(_Dto):
    status: Literal["ok", "degraded"]
    models: dict[str, bool] = Field(description="Model name -> can serve predictions.")
