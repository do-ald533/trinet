from __future__ import annotations

import io
import math

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from PIL import Image

from trinet_api.application.preprocessing import (
    MAX_ISSUES,
    decode_image,
    sniff_format,
    to_image_batch,
    to_matrix,
    to_vector,
)
from trinet_api.domain.errors import InvalidInput, PayloadTooLarge, UnsupportedInput
from trinet_contract import (
    CategoricalField,
    ImageInput,
    NumericField,
    OneHotFeature,
    PassthroughFeature,
    RatioFeature,
    TabularInput,
)

VOL = dummy.volatility_spec()
CREDIT = dummy.credit_spec()
# The dummy specs impute nothing, so imputation gets a spec of its own.
IMPUTING = TabularInput(
    fields=(
        NumericField(name="a", minimum=0.0, maximum=10.0, impute=5.0),
        NumericField(name="b", minimum=1.0),
        CategoricalField(name="c", vocabulary=("x", "y")),
    ),
    features=(
        PassthroughFeature(source="a"),
        OneHotFeature(source="c"),
        RatioFeature(name="a_over_b", numerator="a", denominator="b"),
    ),
)
VOL_NUMERIC_COUNT = len(dummy.VOL_NUMERIC)  # 19 passthrough columns before the one-hot block


def vol_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "ret_1d": 0.004,
        "ret_2d": -0.012,
        "ret_3d": 0.001,
        "ret_4d": 0.007,
        "ret_5d": -0.003,
        "rv_5": 0.21,
        "rv_10": 0.19,
        "rv_20": 0.25,
        "rv_60": 0.27,
        "pk_5": 0.2,
        "pk_20": 0.22,
        "range_5": 0.015,
        "volume_ratio": 1.1,
        "log_dollar_volume": 18.4,
        "gap": 0.002,
        "mom_20": 0.03,
        "mom_60": 0.08,
        "drawdown_60": -0.05,
        "mkt_rv_20": 0.13,
        "dow": "wed",
    }
    record.update(overrides)
    return record


def issues_of(spec: TabularInput, record: dict[str, object]) -> dict[tuple[str | int, ...], str]:
    with pytest.raises(InvalidInput) as info:
        to_vector(record, spec)
    return {i.loc: i.code for i in info.value.issues}


# --------------------------------------------------------------------------- tabular


def test_to_vector_applies_the_three_primitives() -> None:
    record = vol_record()
    vector = to_vector(record, VOL)
    assert vector.dtype == np.float32
    assert vector.shape == (VOL.width,) == (25,)
    numeric = [float(record[name]) for name in dummy.VOL_NUMERIC]  # type: ignore[arg-type]
    assert vector[:VOL_NUMERIC_COUNT].tolist() == pytest.approx(numeric, rel=1e-6)
    # one-hot in vocabulary order: mon, tue, wed, thu, fri
    assert vector[VOL_NUMERIC_COUNT : VOL_NUMERIC_COUNT + 5].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert vector[-1] == pytest.approx(0.21 / 0.25, rel=1e-6)  # rv_5 / rv_20


def test_the_credit_spec_ends_with_one_hot_education_and_utilisation() -> None:
    record = dict.fromkeys(dummy.CREDIT_NUMERIC, 1.0) | {"education": "high_school"}
    record["limit_bal"], record["bill_amt1"], record["age"] = 20000.0, 5000.0, 30.0
    vector = to_vector(record, CREDIT)
    assert vector.shape == (CREDIT.width,) == (25,)
    assert vector[20:24].tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]  # graduate, university, high_school, other
    assert vector[24] == pytest.approx(0.25)


@pytest.mark.parametrize("value", [None, "missing"])
def test_null_or_missing_is_imputed_when_the_field_has_impute(value: object) -> None:
    record: dict[str, object] = {"a": value, "b": 2.0, "c": "x"}
    if value == "missing":
        del record["a"]
    vector = to_vector(record, IMPUTING)
    assert vector[0] == 5.0
    assert vector[3] == pytest.approx(2.5)  # the imputed value feeds the ratio too


def test_null_is_rejected_without_impute_and_missing_is_distinguished() -> None:
    nulled = issues_of(VOL, vol_record(rv_5=None))
    assert nulled == {("rv_5",): "null_not_allowed"}
    record = vol_record()
    del record["rv_5"]
    assert issues_of(VOL, record) == {("rv_5",): "required"}


def test_unknown_fields_are_rejected() -> None:
    assert issues_of(VOL, vol_record(extra=1)) == {("extra",): "unknown_field"}


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (-1000.0, "below_minimum"),
        (-0.5, "below_minimum"),
        (0.5, "above_maximum"),
        (math.nan, "type_error"),
        (math.inf, "type_error"),
        (-math.inf, "type_error"),
        ("0.01", "type_error"),
        (True, "type_error"),
        ([0.01], "type_error"),
        (10**400, "type_error"),
        (1e300, "out_of_range"),
    ],
)
def test_numeric_validation(value: object, code: str) -> None:
    assert issues_of(VOL, vol_record(ret_1d=value)) == {("ret_1d",): code}


def test_numeric_upper_bound_and_integer_acceptance() -> None:
    assert issues_of(VOL, vol_record(volume_ratio=2000)) == {("volume_ratio",): "above_maximum"}
    assert to_vector(vol_record(volume_ratio=2), VOL)[12] == 2.0  # int is a valid number


@pytest.mark.parametrize("value", ["sat", "", 3, None, "WED"])
def test_categorical_must_be_in_vocabulary(value: object) -> None:
    issues = issues_of(VOL, vol_record(dow=value))
    assert list(issues) == [("dow",)]


def test_ratio_denominators_have_a_positive_minimum_so_zero_is_a_422() -> None:
    assert issues_of(VOL, vol_record(rv_20=0.0)) == {("rv_20",): "below_minimum"}
    record = dict.fromkeys(dummy.CREDIT_NUMERIC, 1.0) | {"education": "other", "age": 30.0}
    record["limit_bal"] = 0.0
    assert issues_of(CREDIT, record) == {("limit_bal",): "below_minimum"}


def test_error_messages_never_echo_the_value() -> None:
    with pytest.raises(InvalidInput) as info:
        to_vector(vol_record(dow="SECRET-VALUE-123", ret_1d="SECRET-NUM"), VOL)
    text = " ".join(f"{i.code} {i.message}" for i in info.value.issues)
    assert "SECRET" not in text


def test_zero_denominator_and_ratio_overflow() -> None:
    spec = TabularInput(
        fields=(NumericField(name="a"), NumericField(name="b")),
        features=(RatioFeature(name="a_over_b", numerator="a", denominator="b"),),
    )
    assert issues_of(spec, {"a": 1.0, "b": 0.0}) == {("b",): "zero_denominator"}
    assert issues_of(spec, {"a": 3e38, "b": 1e-30}) == {("a_over_b",): "out_of_range"}
    assert to_vector({"a": 3.0, "b": 2.0}, spec).tolist() == [1.5]


def test_to_matrix_validates_every_row_and_reports_the_index() -> None:
    rows = [vol_record(), vol_record(ret_1d=0.9), "not-an-object", vol_record()]
    with pytest.raises(InvalidInput) as info:
        to_matrix(rows, VOL, max_batch=10)
    assert {(i.loc, i.code) for i in info.value.issues} == {
        (("instances", 1, "ret_1d"), "above_maximum"),
        (("instances", 2), "type_error"),
    }


def test_to_matrix_shape_and_batch_limits() -> None:
    matrix = to_matrix([vol_record()] * 3, VOL, max_batch=3)
    assert matrix.shape == (3, 25)
    assert matrix.dtype == np.float32
    with pytest.raises(InvalidInput, match="1 validation"):
        to_matrix([vol_record()] * 4, VOL, max_batch=3)
    with pytest.raises(InvalidInput):
        to_matrix([], VOL, max_batch=3)


def test_issue_list_is_capped() -> None:
    with pytest.raises(InvalidInput) as info:
        to_matrix([{}] * 100, CREDIT, max_batch=100)
    assert len(info.value.issues) == MAX_ISSUES


def test_categorical_only_spec_and_passthrough_roundtrip() -> None:
    spec = TabularInput(
        fields=(CategoricalField(name="c", vocabulary=("x", "y")), NumericField(name="n")),
        features=(OneHotFeature(source="c"), PassthroughFeature(source="n")),
    )
    assert to_vector({"c": "y", "n": 2.5}, spec).tolist() == [0.0, 1.0, 2.5]


# --------------------------------------------------------------------------- image

IMAGE = ImageInput(height=150, width=150)


def encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def solid(
    size: tuple[int, int] = (150, 150), colour: tuple[int, int, int] = (10, 120, 250)
) -> Image.Image:
    return Image.new("RGB", size, colour)


def test_png_roundtrip_is_exact() -> None:
    pixels = np.random.default_rng(0).integers(0, 256, size=(150, 150, 3), dtype=np.uint8)
    array = decode_image(encode(Image.fromarray(pixels), "PNG"), IMAGE)
    assert array.dtype == np.float32
    assert array.shape == (150, 150, 3)
    assert np.array_equal(array, pixels.astype(np.float32))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_supported_formats_are_decoded(fmt: str) -> None:
    array = decode_image(encode(solid(), fmt), IMAGE)
    assert array.shape == (150, 150, 3)
    assert np.abs(array[0, 0] - np.array([10, 120, 250])).max() <= 3  # JPEG is lossy


def test_webp_only_when_the_manifest_allows_it() -> None:
    data = encode(solid(), "WEBP")
    with pytest.raises(UnsupportedInput):
        decode_image(data, IMAGE)
    spec = ImageInput(height=150, width=150, formats=("JPEG", "PNG", "WEBP"))
    assert decode_image(data, spec).shape == (150, 150, 3)


@pytest.mark.parametrize("size", [(300, 200), (60, 60), (150, 151)])
def test_other_sizes_are_resized_with_pillow_bilinear(size: tuple[int, int]) -> None:
    array = decode_image(encode(solid(size), "PNG"), IMAGE)
    assert array.shape == (150, 150, 3)
    assert np.allclose(array, [10, 120, 250], atol=1)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK", "LA"])
def test_other_colour_modes_become_rgb(mode: str) -> None:
    image = solid().convert(mode)
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    assert decode_image(encode(image, fmt), IMAGE).shape == (150, 150, 3)


def test_value_range_rescaling() -> None:
    spec = ImageInput(height=150, width=150, value_range=(0.0, 1.0))
    array = decode_image(encode(solid(colour=(255, 0, 51)), "PNG"), spec)
    assert array[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.2])


def test_batch_has_a_leading_axis() -> None:
    assert to_image_batch(encode(solid(), "PNG"), IMAGE).shape == (1, 150, 150, 3)


def test_magic_bytes_decide_the_format_not_the_content() -> None:
    assert sniff_format(b"\xff\xd8\xff\xe0rest") == "JPEG"
    assert sniff_format(b"\x89PNG\r\n\x1a\nrest") == "PNG"
    assert sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "WEBP"
    assert sniff_format(b"GIF89a....") is None
    assert sniff_format(b"") is None


@pytest.mark.parametrize(
    "payload", [b"", b"GIF89a" + b"\x00" * 64, b"<html>not an image</html>", b"%PDF-1.7"]
)
def test_non_images_are_unsupported(payload: bytes) -> None:
    with pytest.raises(UnsupportedInput):
        decode_image(payload, IMAGE)


def test_truncated_or_corrupt_images_are_invalid_input() -> None:
    data = encode(solid(), "PNG")
    for broken in (data[: len(data) // 2], data[:20], b"\x89PNG\r\n\x1a\n" + b"junk" * 20):
        with pytest.raises(InvalidInput):
            decode_image(broken, IMAGE)


def test_size_limits() -> None:
    data = encode(solid(), "PNG")
    with pytest.raises(PayloadTooLarge):
        decode_image(data, ImageInput(height=150, width=150, max_bytes=100))
    with pytest.raises(InvalidInput) as info:
        decode_image(encode(solid((400, 400)), "PNG"), IMAGE, max_pixels=100_000)
    assert "pixels" in info.value.issues[0].message


def test_png_only_manifests_refuse_jpeg_and_webp() -> None:
    png_only = ImageInput(height=64, width=64, formats=("PNG",))
    assert decode_image(encode(solid((64, 64)), "PNG"), png_only).shape == (64, 64, 3)
    for fmt in ("JPEG", "WEBP"):
        with pytest.raises(UnsupportedInput, match="PNG"):
            decode_image(encode(solid((64, 64)), fmt), png_only)
