"""Hypothesis properties of the pure pre/post-processing and the softmax runtimes.

Profile ``ci`` (50 examples) is registered in ``conftest.py``; ``HYPOTHESIS_PROFILE=dev`` runs 200.
"""

from __future__ import annotations

import io
import math
from typing import Any

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from PIL import Image

from trinet_api.application.postprocessing import classification_result, regression_result
from trinet_api.application.preprocessing import (
    MAX_ISSUES,
    decode_image,
    to_matrix,
    to_vector,
)
from trinet_api.domain.entities import ModelRegistry
from trinet_api.domain.errors import InvalidInput, PayloadTooLarge, UnsupportedInput
from trinet_api.infrastructure.runtimes import static_dims
from trinet_contract import ImageInput, NumericField, TabularInput

CREDIT = dummy.credit_spec()
VOL = dummy.volatility_spec()
DOW_COLUMNS = slice(len(dummy.VOL_NUMERIC), len(dummy.VOL_NUMERIC) + len(dummy.DOW))
VOL_WIDTH = len(dummy.VOL_NUMERIC) + len(dummy.DOW) + 1  # passthroughs, one-hot, one ratio
SAFE = 1e30  # well inside float32, so ratios and sums cannot overflow


def numeric_value(field: NumericField) -> st.SearchStrategy[float]:
    low = -SAFE if field.minimum is None else max(field.minimum, -SAFE)
    high = SAFE if field.maximum is None else min(field.maximum, SAFE)
    return st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False)


def tabular_record(spec: TabularInput) -> st.SearchStrategy[dict[str, Any]]:
    values: dict[str, st.SearchStrategy[Any]] = {}
    for field in spec.fields:
        if isinstance(field, NumericField):
            strategy = numeric_value(field)
            values[field.name] = st.none() | strategy if field.impute is not None else strategy
        else:
            values[field.name] = st.sampled_from(field.vocabulary)
    return st.fixed_dictionaries(values)


JUNK = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(),
    st.text(max_size=20),
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(max_size=3), st.integers(), max_size=2),
)


# --------------------------------------------------------------------------- tabular


@given(tabular_record(VOL))
def test_valid_volatility_records_give_finite_float32_vectors(record: dict[str, Any]) -> None:
    vector = to_vector(record, VOL)
    assert vector.dtype == np.float32
    assert vector.shape == (VOL_WIDTH,)
    assert np.all(np.isfinite(vector))


@given(tabular_record(VOL))
def test_volatility_one_hot_has_exactly_one_hot_bit_and_the_ratio_is_a_ratio(
    record: dict[str, Any],
) -> None:
    vector = to_vector(record, VOL)
    one_hot = vector[DOW_COLUMNS]
    assert one_hot.sum() == 1.0
    assert set(one_hot.tolist()) <= {0.0, 1.0}
    assert dummy.DOW[int(np.argmax(one_hot))] == record["dow"]
    assert math.isclose(
        float(vector[-1]), record["rv_5"] / record["rv_20"], rel_tol=1e-6, abs_tol=1e-37
    )


@given(tabular_record(VOL))
def test_volatility_vectorisation_matches_the_independent_reference(
    record: dict[str, Any],
) -> None:
    expected = np.asarray(dummy.volatility_features(record), dtype=np.float64).astype(np.float32)
    np.testing.assert_allclose(to_vector(record, VOL), expected, rtol=1e-6)


@given(tabular_record(CREDIT))
def test_credit_vectorisation_matches_the_independent_reference(record: dict[str, Any]) -> None:
    expected = np.asarray(dummy.credit_features(record), dtype=np.float64).astype(np.float32)
    np.testing.assert_allclose(to_vector(record, CREDIT), expected, rtol=1e-6)


@given(tabular_record(CREDIT))
def test_passthrough_features_follow_the_feature_order_not_the_field_order(
    record: dict[str, Any],
) -> None:
    vector = to_vector(record, CREDIT)
    numeric = len(dummy.CREDIT_NUMERIC)
    assert vector.shape == (CREDIT.width,)
    expected = np.asarray([record[name] for name in dummy.CREDIT_NUMERIC]).astype(np.float32)
    np.testing.assert_allclose(vector[:numeric], expected, rtol=1e-6)


@given(tabular_record(CREDIT), st.data())
def test_a_bad_value_is_rejected_with_a_loc_pointing_at_it_and_without_echo(
    record: dict[str, Any], data: st.DataObject
) -> None:
    field = data.draw(st.sampled_from(CREDIT.fields))
    choices: list[st.SearchStrategy[Any]] = [
        st.sampled_from([math.nan, math.inf, -math.inf, "x", None, True, [], {}])
    ]
    if isinstance(field, NumericField) and field.minimum is not None:
        minimum = field.minimum
        choices.append(st.floats(min_value=1.0, max_value=1e6).map(lambda d: minimum - d))
    bad = data.draw(st.one_of(*choices))
    record[field.name] = bad
    with pytest.raises(InvalidInput) as raised:
        to_vector(record, CREDIT)
    issues = raised.value.issues
    assert any(field.name in issue.loc for issue in issues)
    for issue in issues:
        assert issue.code
        assert str(bad) not in issue.message or str(bad) in {"x", "", "True", "None", "[]", "{}"}


@given(st.dictionaries(st.text(max_size=12), JUNK, max_size=25))
def test_arbitrary_records_either_vectorise_or_raise_invalid_input(
    record: dict[str, Any],
) -> None:
    issues: tuple[Any, ...] | list[Any] = []
    try:
        vector = to_vector(record, VOL)
    except InvalidInput as exc:
        issues = list(exc.issues)
    else:
        assert np.all(np.isfinite(vector))
    assert len(issues) <= MAX_ISSUES


@given(st.lists(JUNK, max_size=6))
def test_arbitrary_instance_lists_never_raise_anything_but_invalid_input(
    instances: list[Any],
) -> None:
    try:
        matrix = to_matrix(instances, CREDIT, max_batch=4)
    except InvalidInput:
        return
    assert matrix.dtype == np.float32
    assert 1 <= matrix.shape[0] <= 4
    assert np.all(np.isfinite(matrix))


@given(st.lists(tabular_record(CREDIT), min_size=1, max_size=8))
def test_batch_size_is_preserved_and_rows_are_independent(records: list[dict[str, Any]]) -> None:
    matrix = to_matrix(records, CREDIT, max_batch=8)
    assert matrix.shape == (len(records), CREDIT.width)
    for row, record in zip(matrix, records, strict=True):
        np.testing.assert_array_equal(row, to_vector(record, CREDIT))


@given(st.lists(tabular_record(CREDIT), min_size=9, max_size=12))
def test_oversized_batches_are_rejected(records: list[dict[str, Any]]) -> None:
    with pytest.raises(InvalidInput):
        to_matrix(records, CREDIT, max_batch=8)


# --------------------------------------------------------------------------- post-processing


finite_logits = st.lists(
    st.floats(min_value=-50, max_value=50, allow_nan=False), min_size=2, max_size=12
)


def softmax(values: list[float]) -> NDArray[np.float32]:
    x = np.asarray(values, dtype=np.float64)
    e = np.exp(x - x.max())
    return np.asarray(e / e.sum(), dtype=np.float32)


@given(finite_logits, st.integers(min_value=1, max_value=15))
def test_classification_result_is_a_sorted_probability_distribution(
    logits: list[float], top_k: int
) -> None:
    labels = tuple(f"c{i}" for i in range(len(logits)))
    probs = softmax(logits)
    result = classification_result(probs, labels, top_k=top_k)
    scores = [s.probability for s in result.scores]
    assert len(scores) == min(top_k, len(labels))
    assert scores == sorted(scores, reverse=True)
    assert result.label == result.scores[0].label
    assert result.probability == scores[0] == pytest.approx(float(probs.max()))
    assert len({s.label for s in result.scores}) == len(result.scores)
    assert all(0.0 <= p <= 1.0 for p in scores)


@given(finite_logits, st.sampled_from([math.nan, math.inf, -math.inf, -0.5, 1.5]), st.data())
def test_corrupt_probability_vectors_are_never_reported(
    logits: list[float], poison: float, data: st.DataObject
) -> None:
    probs = softmax(logits)
    probs[data.draw(st.integers(0, len(logits) - 1))] = poison
    with pytest.raises(Exception, match=r"."):
        classification_result(probs, tuple(f"c{i}" for i in range(len(logits))))


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, width=32))
def test_regression_result_roundtrips_a_finite_value(value: float) -> None:
    result = regression_result(np.array([value], dtype=np.float32), unit="USD", target="price")
    assert result.value == pytest.approx(float(np.float32(value)))
    assert (result.unit, result.target) == ("USD", "price")


@given(st.sampled_from([math.nan, math.inf, -math.inf]))
def test_regression_result_refuses_non_finite_values(value: float) -> None:
    with pytest.raises(Exception, match=r"."):
        regression_result(np.array([value], dtype=np.float32), unit="USD", target="price")


# --------------------------------------------------------------------------- images

IMAGE_SPEC = ImageInput(
    height=16,
    width=16,
    channels=3,
    resample="bilinear",
    value_range=(0.0, 255.0),
    max_bytes=1_000_000,
    formats=("JPEG", "PNG"),
)


def encode(width: int, height: int, mode: str, fmt: str, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    shape = (height, width) if channels == 1 else (height, width, channels)
    array = rng.integers(0, 256, size=shape, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format=fmt)
    return buffer.getvalue()


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(1, 64),
    st.integers(1, 64),
    st.sampled_from(["L", "RGB", "RGBA"]),
    st.sampled_from(["PNG", "JPEG"]),
    st.integers(0, 2**16),
)
def test_any_supported_image_decodes_to_the_manifest_shape(
    width: int, height: int, mode: str, fmt: str, seed: int
) -> None:
    assume(not (fmt == "JPEG" and mode == "RGBA"))  # JPEG has no alpha channel
    array = decode_image(encode(width, height, mode, fmt, seed), IMAGE_SPEC)
    assert array.shape == (16, 16, 3)
    assert array.dtype == np.float32
    assert 0.0 <= array.min() <= array.max() <= 255.0


@given(st.binary(max_size=2000))
def test_random_bytes_are_never_decoded_and_never_crash(blob: bytes) -> None:
    with pytest.raises((InvalidInput, UnsupportedInput, PayloadTooLarge)):
        decode_image(blob, IMAGE_SPEC)


@given(st.integers(1, 64), st.integers(1, 64), st.integers(30, 200))
def test_truncated_images_are_rejected_cleanly(width: int, height: int, cut: int) -> None:
    data = encode(width, height, "RGB", "PNG", seed=1)
    truncated = data[: max(9, len(data) - cut)]
    if len(truncated) == len(data):
        return
    with pytest.raises(InvalidInput):
        decode_image(truncated, IMAGE_SPEC)


@given(st.sampled_from([3.5e38, -3.5e38, 1e100, -1e300, 10**400]))
def test_values_beyond_the_float32_range_are_rejected(huge: float) -> None:
    record = dummy.credit_record(np.random.default_rng(0))
    record["pay_amt1"] = huge
    with pytest.raises(InvalidInput) as raised:
        to_vector(record, CREDIT)
    assert [issue.loc for issue in raised.value.issues] == [("pay_amt1",)]


# --------------------------------------------------------------------------- runtimes


@pytest.mark.parametrize("model", ["credit-default-mlp", "candlestick-cnn"])
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
def test_classifier_rows_are_distributions_for_any_finite_input(
    registry: ModelRegistry, model: str, data: st.DataObject
) -> None:
    runtime = registry.get(model).runtime
    manifest = registry.get(model).manifest
    trailing, width = static_dims(manifest.inputs[0], manifest.outputs[0])
    rows = data.draw(st.integers(1, 4))
    scale = data.draw(st.sampled_from([1e-6, 1.0, 1e3, 1e6]))
    seed = data.draw(st.integers(0, 2**32 - 1))
    batch = (np.random.default_rng(seed).normal(size=(rows, *trailing)) * scale).astype(np.float32)
    out = runtime.predict(batch)
    assert out.shape == (rows, width)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-4)
