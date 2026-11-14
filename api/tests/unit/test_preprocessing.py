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

HOUSING = dummy.housing_spec()
DRYBEAN = dummy.drybean_spec()


def housing_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "longitude": -122.2,
        "latitude": 37.8,
        "housing_median_age": 30.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3,
        "ocean_proximity": "NEAR BAY",
    }
    record.update(overrides)
    return record


def issues_of(spec: TabularInput, record: dict[str, object]) -> dict[tuple[str | int, ...], str]:
    with pytest.raises(InvalidInput) as info:
        to_vector(record, spec)
    return {i.loc: i.code for i in info.value.issues}


# --------------------------------------------------------------------------- tabular


def test_to_vector_applies_the_three_primitives() -> None:
    vector = to_vector(housing_record(), HOUSING)
    assert vector.dtype == np.float32
    assert vector.shape == (HOUSING.width,) == (15,)
    assert vector[:8].tolist() == pytest.approx(
        [-122.2, 37.8, 30.0, 880.0, 129.0, 322.0, 126.0, 8.3], rel=1e-6
    )
    # one-hot in vocabulary order: <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN
    assert vector[8:13].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert vector[13] == pytest.approx(880.0 / 126.0, rel=1e-6)
    assert vector[14] == pytest.approx(322.0 / 126.0, rel=1e-6)


@pytest.mark.parametrize("bedrooms", [None, "missing"])
def test_null_or_missing_is_imputed_when_the_field_has_impute(bedrooms: object) -> None:
    record = housing_record(total_bedrooms=bedrooms)
    if bedrooms == "missing":
        del record["total_bedrooms"]
    assert to_vector(record, HOUSING)[4] == pytest.approx(dummy.BEDROOMS_IMPUTE)


def test_null_is_rejected_without_impute_and_missing_is_distinguished() -> None:
    nulled = issues_of(HOUSING, housing_record(median_income=None))
    assert nulled == {("median_income",): "null_not_allowed"}
    record = housing_record()
    del record["median_income"]
    assert issues_of(HOUSING, record) == {("median_income",): "required"}


def test_unknown_fields_are_rejected() -> None:
    assert issues_of(HOUSING, housing_record(extra=1)) == {("extra",): "unknown_field"}


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (-1000.0, "below_minimum"),
        (-130.0, "below_minimum"),
        (-90.0, "above_maximum"),
        (math.nan, "type_error"),
        (math.inf, "type_error"),
        (-math.inf, "type_error"),
        ("12", "type_error"),
        (True, "type_error"),
        ([1.0], "type_error"),
        (10**400, "type_error"),
        (1e300, "out_of_range"),
    ],
)
def test_numeric_validation(value: object, code: str) -> None:
    assert issues_of(HOUSING, housing_record(longitude=value)) == {("longitude",): code}


def test_numeric_upper_bound_and_integer_acceptance() -> None:
    assert issues_of(HOUSING, housing_record(latitude=99)) == {("latitude",): "above_maximum"}
    assert to_vector(housing_record(latitude=40), HOUSING)[1] == 40.0  # int is a valid number


@pytest.mark.parametrize("value", ["Mars", "", 3, None])
def test_categorical_must_be_in_vocabulary(value: object) -> None:
    issues = issues_of(HOUSING, housing_record(ocean_proximity=value))
    assert list(issues) == [("ocean_proximity",)]


def test_error_messages_never_echo_the_value() -> None:
    with pytest.raises(InvalidInput) as info:
        to_vector(
            housing_record(ocean_proximity="SECRET-VALUE-123", longitude="SECRET-NUM"), HOUSING
        )
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
    rows = [housing_record(), housing_record(latitude=99), "not-an-object", housing_record()]
    with pytest.raises(InvalidInput) as info:
        to_matrix(rows, HOUSING, max_batch=10)
    assert {(i.loc, i.code) for i in info.value.issues} == {
        (("instances", 1, "latitude"), "above_maximum"),
        (("instances", 2), "type_error"),
    }


def test_to_matrix_shape_and_batch_limits() -> None:
    matrix = to_matrix([housing_record()] * 3, HOUSING, max_batch=3)
    assert matrix.shape == (3, 15)
    assert matrix.dtype == np.float32
    with pytest.raises(InvalidInput, match="1 validation"):
        to_matrix([housing_record()] * 4, HOUSING, max_batch=3)
    with pytest.raises(InvalidInput):
        to_matrix([], HOUSING, max_batch=3)


def test_issue_list_is_capped() -> None:
    with pytest.raises(InvalidInput) as info:
        to_matrix([{}] * 100, DRYBEAN, max_batch=100)
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
