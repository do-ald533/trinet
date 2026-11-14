from __future__ import annotations

import numpy as np
import pytest

from trinet_api.application.postprocessing import classification_result, regression_result
from trinet_api.domain.errors import InferenceFailed

LABELS = ("a", "b", "c", "d")


def test_all_classes_sorted_by_probability() -> None:
    result = classification_result(np.array([0.1, 0.5, 0.3, 0.1], dtype=np.float32), LABELS)
    assert result.label == "b"
    assert result.probability == pytest.approx(0.5)
    assert [s.label for s in result.scores] == ["b", "c", "a", "d"]  # ties keep class order


def test_top_k_truncates() -> None:
    result = classification_result(np.array([0.1, 0.5, 0.3, 0.1]), LABELS, top_k=2)
    assert [s.label for s in result.scores] == ["b", "c"]
    assert sum(s.probability for s in result.scores) < 1.0


@pytest.mark.parametrize(
    "row",
    [
        [0.5, 0.5, 0.5, 0.5],  # does not sum to 1
        [np.nan, 0.5, 0.25, 0.25],
        [np.inf, 0.0, 0.0, 0.0],
        [1.5, -0.5, 0.0, 0.0],  # outside [0, 1]
        [0.5, 0.5],  # wrong number of classes
    ],
)
def test_invalid_probability_vectors_fail_loudly(row: list[float]) -> None:
    with pytest.raises(InferenceFailed):
        classification_result(np.array(row, dtype=np.float32), LABELS)


def test_regression_result() -> None:
    result = regression_result(np.array([123456.5], dtype=np.float32), unit="USD", target="price")
    assert (result.value, result.unit, result.target) == (123456.5, "USD", "price")


@pytest.mark.parametrize("row", [[np.nan], [np.inf], [1.0, 2.0], []])
def test_regression_rejects_bad_rows(row: list[float]) -> None:
    with pytest.raises(InferenceFailed):
        regression_result(np.array(row, dtype=np.float32), unit="USD", target="price")
