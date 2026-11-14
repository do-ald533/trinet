"""``PredictionService``: one use case, behaviour driven by the manifest (not by subclasses)."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trinet_api.application.postprocessing import classification_result, regression_result
from trinet_api.application.preprocessing import to_image_batch, to_matrix
from trinet_api.domain.entities import ModelBundle, ModelRegistry, Prediction
from trinet_api.domain.errors import InferenceFailed, Overloaded, TriNetError, UnsupportedInput
from trinet_contract import (
    ClassificationOutput,
    ImageInput,
    Manifest,
    RegressionOutput,
    TabularInput,
    Task,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3


@dataclass(frozen=True, slots=True)
class TabularRequest:
    """Raw named records (already JSON-decoded, still unvalidated)."""

    records: Sequence[object]


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """One encoded image (JPEG/PNG/WEBP bytes)."""

    data: bytes


type InferenceRequest = TabularRequest | ImageRequest


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    model_name: str
    model_version: str
    runtime: str
    predictions: tuple[Prediction, ...]
    latency_ms: float
    inference_ms: float


class ConcurrencyLimiter:
    """Bounds concurrent inferences; waiters give up after ``queue_timeout_s`` (-> 503)."""

    def __init__(self, max_concurrency: int, queue_timeout_s: float) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._timeout = queue_timeout_s
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_concurrency = max_concurrency

    @property
    def inflight(self) -> int:
        return self._inflight

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self._semaphore.acquire(timeout=self._timeout):
            raise Overloaded(retry_after_s=max(1, math.ceil(self._timeout)))
        with self._lock:
            self._inflight += 1
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1
            self._semaphore.release()


class PredictionService:
    def __init__(
        self,
        registry: ModelRegistry,
        limiter: ConcurrencyLimiter,
        *,
        max_batch: int = 256,
        top_k: int = DEFAULT_TOP_K,
        max_image_pixels: int = 16_777_216,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._registry = registry
        self._limiter = limiter
        self._max_batch = max_batch
        self._top_k = top_k
        self._max_image_pixels = max_image_pixels
        self._clock = clock

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def limiter(self) -> ConcurrencyLimiter:
        return self._limiter

    def predict(self, name: str, request: InferenceRequest) -> PredictionOutcome:
        """Validate, run the model and shape the answer.

        Raises ``ModelNotFound`` / ``ModelNotAvailable`` (before taking a slot),
        ``InvalidInput`` / ``UnsupportedInput`` / ``PayloadTooLarge`` (bad payload),
        ``Overloaded`` (no slot in time) or ``InferenceFailed``.
        """
        bundle = self._registry.get(name)
        manifest = bundle.manifest
        with self._limiter.slot():
            started = self._clock()
            batch = self._preprocess(manifest, request)
            inference_started = self._clock()
            raw = self._infer(bundle, batch)
            inference_ms = (self._clock() - inference_started) * 1000.0
            predictions = self._postprocess(manifest, raw)
            latency_ms = (self._clock() - started) * 1000.0
        return PredictionOutcome(
            model_name=manifest.name,
            model_version=manifest.version,
            runtime=bundle.runtime.kind,
            predictions=predictions,
            latency_ms=latency_ms,
            inference_ms=inference_ms,
        )

    # ------------------------------------------------------------------ steps

    def _preprocess(self, manifest: Manifest, request: InferenceRequest) -> NDArray[np.float32]:
        match manifest.input, request:
            case TabularInput() as spec, TabularRequest(records=records):
                return to_matrix(records, spec, max_batch=self._max_batch)
            case ImageInput() as spec, ImageRequest(data=data):
                return to_image_batch(data, spec, max_pixels=self._max_image_pixels)
            case _:
                expected = "JSON instances" if manifest.input.kind == "tabular" else "an image file"
                raise UnsupportedInput(f"model {manifest.name!r} expects {expected}")

    def _infer(self, bundle: ModelBundle, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        try:
            raw = bundle.runtime.predict(batch)
        except TriNetError:
            raise
        except Exception as exc:
            logger.exception("runtime %s failed for model %s", bundle.runtime.kind, bundle.name)
            raise InferenceFailed("the model runtime failed") from exc
        if raw.ndim != 2 or raw.shape[0] != batch.shape[0]:
            raise InferenceFailed("model output shape does not match the request batch")
        return raw

    def _postprocess(self, manifest: Manifest, raw: NDArray[np.float32]) -> tuple[Prediction, ...]:
        match manifest.output:
            case ClassificationOutput(labels=labels):
                top_k = self._top_k if manifest.task is Task.IMAGE_CLASSIFICATION else None
                return tuple(classification_result(row, labels, top_k=top_k) for row in raw)
            case RegressionOutput(unit=unit, target=target):
                return tuple(regression_result(row, unit=unit, target=target) for row in raw)
