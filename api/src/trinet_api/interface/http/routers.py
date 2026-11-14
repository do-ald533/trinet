"""Routes: health, model catalogue and predictions."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, NoReturn

import structlog
from fastapi import APIRouter, Path, Request, Response, Security
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from trinet_api.application.service import ImageRequest, InferenceRequest, TabularRequest
from trinet_api.domain.entities import ClassificationResult, ModelBundle
from trinet_api.domain.errors import (
    InvalidInput,
    Issue,
    NotReady,
    PayloadTooLarge,
    UnsupportedInput,
)
from trinet_api.interface.http.dependencies import MetricsDep, ServiceDep
from trinet_api.interface.http.problems import problem_responses
from trinet_api.interface.http.schemas import (
    HealthResponse,
    ModelDetail,
    ModelList,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
    model_detail,
    model_summary,
    prediction_dto,
)
from trinet_api.interface.http.security import require_api_key
from trinet_contract import ImageInput

log = structlog.get_logger("trinet_api.predictions")

JSON = "application/json"
MULTIPART = "multipart/form-data"
MODEL_VERSION_HEADER = "X-Model-Version"
HEALTH_LIVE_PATH = "/health/live"
HEALTH_READY_PATH = "/health/ready"
_MAX_LOGGED_LABELS = 10

ModelName = Annotated[
    str,
    Path(
        pattern=r"^[a-z0-9][a-z0-9-]{0,62}$",
        description="Model name, as listed by GET /v1/models.",
    ),
]

health_router = APIRouter(tags=["health"])
models_router = APIRouter(prefix="/v1", tags=["models"], dependencies=[Security(require_api_key)])

_VERSION_HEADER_DOC: dict[str, Any] = {
    MODEL_VERSION_HEADER: {
        "schema": {"type": "string"},
        "description": "Version of the model that produced the answer.",
    }
}

#: Request body of the predictions endpoint. FastAPI cannot express "JSON *or* multipart"
#: from one signature, so the document is declared here and the handler parses by hand.
PREDICTION_REQUEST_BODY: dict[str, Any] = {
    "required": True,
    "description": (
        "Tabular models: JSON with raw named fields (see GET /v1/models/{name}). "
        "Image models: multipart/form-data with a single `file` part."
    ),
    "content": {
        JSON: {
            "schema": {
                "type": "object",
                "required": ["instances"],
                "additionalProperties": False,
                "properties": {
                    "instances": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 256,  # replaced by TRINET_MAX_BATCH in the served document
                        "items": {
                            "type": "object",
                            "additionalProperties": {
                                "anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]
                            },
                        },
                    }
                },
            }
        },
        MULTIPART: {
            "schema": {
                "type": "object",
                "required": ["file"],
                "properties": {"file": {"type": "string", "format": "binary"}},
            }
        },
    },
}


# --------------------------------------------------------------------------- health


@health_router.get(HEALTH_LIVE_PATH, operation_id="health_live", summary="Liveness probe")
async def health_live() -> HealthResponse:
    return HealthResponse()


@health_router.get(
    HEALTH_READY_PATH,
    operation_id="health_ready",
    summary="Readiness probe",
    responses=problem_responses(503),
)
async def health_ready(service: ServiceDep) -> ReadyResponse:
    registry = service.registry
    if not registry.ready:
        raise NotReady("no model is loaded and serving")
    models = {entry.name: isinstance(entry, ModelBundle) for entry in registry}
    return ReadyResponse(status="ok" if all(models.values()) else "degraded", models=models)


# --------------------------------------------------------------------------- models


@models_router.get(
    "/models",
    operation_id="list_models",
    summary="List models",
    responses=problem_responses(401, 429),
)
async def list_models(service: ServiceDep) -> ModelList:
    return ModelList(models=[model_summary(entry) for entry in service.registry])


@models_router.get(
    "/models/{name}",
    operation_id="get_model",
    summary="Public manifest of a model",
    responses={
        200: {"headers": _VERSION_HEADER_DOC},
        **problem_responses(401, 404, 422, 429),
    },
)
async def get_model(name: ModelName, service: ServiceDep, response: Response) -> ModelDetail:
    bundle = service.registry.get(name)
    response.headers[MODEL_VERSION_HEADER] = bundle.version
    return model_detail(bundle)


# --------------------------------------------------------------------------- predictions


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {name}")


def _clip(part: str | int) -> str | int:
    return part if isinstance(part, int) else part[:64]


def parse_instances(body: bytes) -> list[object]:
    """JSON body -> the raw ``instances`` (NaN/Infinity and non-objects are rejected)."""
    try:
        document = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise InvalidInput([Issue((), "invalid_json", "the body is not valid JSON")]) from None
    try:
        return list(PredictionRequest.model_validate(document, strict=True).instances)
    except ValidationError as exc:
        raise InvalidInput(
            [
                Issue(tuple(_clip(p) for p in err["loc"]), str(err["type"]), str(err["msg"]))
                for err in exc.errors(include_input=False, include_url=False, include_context=False)
            ]
        ) from None


async def _read_upload(request: Request, max_bytes: int) -> bytes:
    async with request.form(max_files=1, max_fields=1) as form:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise InvalidInput(
                [Issue(("file",), "required", "multipart field 'file' with the image is required")]
            )
        if upload.size is not None and upload.size > max_bytes:
            raise PayloadTooLarge(f"image larger than {max_bytes} bytes")
        data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLarge(f"image larger than {max_bytes} bytes")
    return data


def _media_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


@models_router.post(
    "/models/{name}/predictions",
    operation_id="create_prediction",
    summary="Predict",
    response_model=PredictionResponse,
    openapi_extra={"requestBody": PREDICTION_REQUEST_BODY},
    responses={
        200: {"headers": _VERSION_HEADER_DOC},
        **problem_responses(400, 401, 404, 413, 415, 422, 429, 503),
    },
)
async def create_prediction(
    name: ModelName,
    request: Request,
    response: Response,
    service: ServiceDep,
    metrics: MetricsDep,
) -> PredictionResponse:
    bundle = service.registry.get(name)  # 404 before the body is read
    manifest = bundle.manifest
    expected = MULTIPART if isinstance(manifest.input, ImageInput) else JSON
    if _media_type(request) != expected:
        raise UnsupportedInput(f"model {name!r} expects Content-Type {expected}")

    payload: InferenceRequest
    if isinstance(manifest.input, ImageInput):
        data = await _read_upload(request, manifest.input.max_bytes)
        digest = hashlib.sha256(data).hexdigest()
        payload = ImageRequest(data)
    else:
        body = await request.body()
        digest = hashlib.sha256(body).hexdigest()
        payload = TabularRequest(parse_instances(body))

    outcome = await run_in_threadpool(service.predict, name, payload)

    labels = [p.label for p in outcome.predictions if isinstance(p, ClassificationResult)]
    metrics.observe_prediction(
        model=name,
        runtime=outcome.runtime,
        inference_seconds=outcome.inference_ms / 1000.0,
        labels=labels,
    )
    log.info(
        "prediction",
        model=outcome.model_name,
        version=outcome.model_version,
        runtime=outcome.runtime,
        batch_size=len(outcome.predictions),
        predicted=labels[:_MAX_LOGGED_LABELS],
        latency_ms=round(outcome.latency_ms, 3),
        input_sha256=digest,  # a hash, never the payload
    )
    response.headers[MODEL_VERSION_HEADER] = outcome.model_version
    return PredictionResponse(
        model_name=outcome.model_name,
        model_version=outcome.model_version,
        request_id=request.state.request_id,
        predictions=[prediction_dto(p, manifest.task) for p in outcome.predictions],
    )
