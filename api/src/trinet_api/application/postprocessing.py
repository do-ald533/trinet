"""Pure post-processing: raw model output rows -> domain predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from trinet_api.domain.entities import ClassificationResult, ClassScore, RegressionResult
from trinet_api.domain.errors import InferenceFailed

PROB_ATOL = 1e-3


def classification_result(
    row: NDArray[np.floating], labels: Sequence[str], *, top_k: int | None = None
) -> ClassificationResult:
    """Probability row (softmax already inside the graph) -> label + sorted class scores.

    ``top_k=None`` keeps every class. Ties break by class index, so results are stable.
    Raises ``InferenceFailed`` if the row is not a finite probability distribution over
    ``labels`` (a model/manifest mismatch must fail loudly, not be papered over).
    """
    probs = np.asarray(row, dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(labels):
        raise InferenceFailed(f"model produced {probs.shape[0]} scores for {len(labels)} labels")
    if not np.all(np.isfinite(probs)) or probs.min() < -PROB_ATOL or probs.max() > 1 + PROB_ATOL:
        raise InferenceFailed("model output is not a valid probability vector")
    if abs(float(probs.sum()) - 1.0) > PROB_ATOL:
        raise InferenceFailed("model output probabilities do not sum to 1")
    order = np.argsort(-probs, kind="stable")
    keep = order if top_k is None else order[:top_k]
    scores = tuple(ClassScore(labels[i], float(probs[i])) for i in keep)
    return ClassificationResult(
        label=scores[0].label, probability=scores[0].probability, scores=scores
    )


def regression_result(row: NDArray[np.floating], *, unit: str, target: str) -> RegressionResult:
    """Single-column output row -> value (already in target units; denormalised in-graph)."""
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.shape[0] != 1 or not np.isfinite(values[0]):
        raise InferenceFailed("model produced a non-finite or multi-valued regression output")
    return RegressionResult(value=float(values[0]), unit=unit, target=target)
