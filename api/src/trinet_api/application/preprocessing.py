"""Pure pre-processing: raw request data -> the float32 tensor the model expects.

Tabular models use three closed primitives declared in the manifest (constant imputation,
one-hot by vocabulary and A/B ratio); images are decoded and resized with Pillow. Nothing
here touches I/O, global state or the runtime, so the same code runs in the API and in the
golden/skew tests. Error messages never echo the offending value.
"""

from __future__ import annotations

import io
import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from trinet_api.domain.errors import InvalidInput, Issue, PayloadTooLarge, UnsupportedInput
from trinet_contract import (
    CategoricalField,
    ImageInput,
    NumericField,
    OneHotFeature,
    PassthroughFeature,
    RatioFeature,
    TabularInput,
)

type Record = Mapping[str, object]
type FloatArray = NDArray[np.float32]

MAX_ISSUES = 50
_F32_MAX = float(np.finfo(np.float32).max)
_MAX_CLIP = 64
_MAX_VOCAB_IN_MESSAGE = 10


def _clip(text: str, limit: int = _MAX_CLIP) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- tabular


def _as_finite_float(raw: object) -> float | None:
    """``raw`` as a finite float, or ``None`` if it is not a finite JSON number."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    try:
        value = float(raw)
    except OverflowError:  # an int too large for a double
        return None
    return value if math.isfinite(value) else None


def _check_numeric(
    raw: object, field: NumericField, loc: tuple[str | int, ...]
) -> tuple[float | None, Issue | None]:
    value = _as_finite_float(raw)
    if value is None:
        return None, Issue(loc, "type_error", "must be a finite number")
    if abs(value) > _F32_MAX:
        return None, Issue(loc, "out_of_range", "magnitude exceeds the float32 range")
    if field.minimum is not None and value < field.minimum:
        return None, Issue(loc, "below_minimum", f"must be >= {field.minimum:g}")
    if field.maximum is not None and value > field.maximum:
        return None, Issue(loc, "above_maximum", f"must be <= {field.maximum:g}")
    return value, None


def _check_categorical(
    raw: object, field: CategoricalField, loc: tuple[str | int, ...]
) -> tuple[str | None, Issue | None]:
    if not isinstance(raw, str):
        return None, Issue(loc, "type_error", "must be a string")
    if raw not in field.vocabulary:
        shown = ", ".join(repr(v) for v in field.vocabulary[:_MAX_VOCAB_IN_MESSAGE])
        more = ", ..." if len(field.vocabulary) > _MAX_VOCAB_IN_MESSAGE else ""
        return None, Issue(loc, "not_in_vocabulary", f"must be one of: {shown}{more}")
    return raw, None


def resolve_record(
    record: Record, spec: TabularInput, loc: tuple[str | int, ...] = ()
) -> tuple[dict[str, float | str], list[Issue]]:
    """Validate ``record`` against the raw fields and apply constant imputation.

    ``None`` (explicit null) or a missing key is accepted only when the field has ``impute``.
    Unknown keys are rejected. Returns the resolved values and every issue found.
    """
    issues: list[Issue] = []
    values: dict[str, float | str] = {}
    known = {f.name for f in spec.fields}
    issues.extend(
        Issue((*loc, _clip(str(key))), "unknown_field", "unknown field")
        for key in record
        if key not in known
    )
    for field in spec.fields:
        field_loc = (*loc, field.name)
        raw = record.get(field.name)
        if raw is None:
            if field.impute is None:
                code = "null_not_allowed" if field.name in record else "required"
                message = "null is not allowed" if code == "null_not_allowed" else "field required"
                issues.append(Issue(field_loc, code, message))
            else:
                values[field.name] = field.impute
            continue
        checked: float | str | None
        issue: Issue | None
        if isinstance(field, NumericField):
            checked, issue = _check_numeric(raw, field, field_loc)
        else:
            checked, issue = _check_categorical(raw, field, field_loc)
        if issue is not None:
            issues.append(issue)
        elif checked is not None:
            values[field.name] = checked
    return values, issues


def _ratio(
    feature: RatioFeature, values: Mapping[str, float | str], loc: tuple[str | int, ...]
) -> tuple[float | None, Issue | None]:
    numerator = values.get(feature.numerator)
    denominator = values.get(feature.denominator)
    if not isinstance(numerator, float) or not isinstance(denominator, float):
        return None, None  # an upstream issue was already reported for the missing operand
    if denominator == 0.0:
        return None, Issue((*loc, feature.denominator), "zero_denominator", "must not be zero")
    result = numerator / denominator
    if not math.isfinite(result) or abs(result) > _F32_MAX:
        return None, Issue((*loc, feature.name), "out_of_range", "ratio exceeds the float32 range")
    return result, None


def _columns(
    values: Mapping[str, float | str], spec: TabularInput, loc: tuple[str | int, ...]
) -> tuple[list[float], list[Issue]]:
    """Feature columns in manifest order (computed in float64, cast once by the caller)."""
    columns: list[float] = []
    issues: list[Issue] = []
    for feature in spec.features:
        match feature:
            case PassthroughFeature(source=src):
                value = values.get(src)
                if isinstance(value, float):
                    columns.append(value)
            case OneHotFeature(source=src):
                field = spec.field(src)
                if isinstance(field, CategoricalField) and src in values:
                    columns.extend(1.0 if values[src] == v else 0.0 for v in field.vocabulary)
            case RatioFeature():
                ratio, issue = _ratio(feature, values, loc)
                if issue is not None:
                    issues.append(issue)
                elif ratio is not None:
                    columns.append(ratio)
    return columns, issues


def vectorize(
    record: Record, spec: TabularInput, loc: tuple[str | int, ...] = ()
) -> tuple[list[float], list[Issue]]:
    """``(columns, issues)``; ``columns`` is only complete when ``issues`` is empty."""
    values, issues = resolve_record(record, spec, loc)
    columns, ratio_issues = _columns(values, spec, loc)
    return columns, [*issues, *ratio_issues]


def to_vector(record: Record, spec: TabularInput) -> FloatArray:
    """Raw named fields -> ordered float32 feature vector ``(width,)``.

    Raises ``InvalidInput`` (never silently coerces) when the record violates the manifest.
    """
    columns, issues = vectorize(record, spec)
    if issues:
        raise InvalidInput(issues[:MAX_ISSUES])
    return np.asarray(columns, dtype=np.float32)


def to_matrix(records: Sequence[object], spec: TabularInput, *, max_batch: int) -> FloatArray:
    """A batch of raw records -> ``(n, width)`` float32 matrix, validating every record."""
    if not records:
        raise InvalidInput([Issue(("instances",), "empty_batch", "at least one instance required")])
    if len(records) > max_batch:
        raise InvalidInput(
            [Issue(("instances",), "batch_too_large", f"at most {max_batch} instances per request")]
        )
    rows: list[list[float]] = []
    issues: list[Issue] = []
    for index, record in enumerate(records):
        loc: tuple[str | int, ...] = ("instances", index)
        if not isinstance(record, Mapping):
            issues.append(Issue(loc, "type_error", "must be an object"))
        else:
            columns, record_issues = vectorize(record, spec, loc)
            issues.extend(record_issues)
            rows.append(columns)
        if len(issues) >= MAX_ISSUES:
            break
    if issues:
        raise InvalidInput(issues[:MAX_ISSUES])
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- image


def sniff_format(data: bytes) -> str | None:
    """Image format from the magic bytes (the client-declared type is never trusted)."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def _invalid_image(message: str) -> InvalidInput:
    return InvalidInput([Issue(("file",), "invalid_image", message)])


def decode_image(data: bytes, spec: ImageInput, *, max_pixels: int = 16_777_216) -> FloatArray:
    """Encoded image bytes -> ``(height, width, 3)`` float32 in ``spec.value_range``.

    Pipeline (the same one the notebook measures in ``serving_skew``): magic-byte check ->
    header-only size check (decompression-bomb guard) -> decode -> ``convert("RGB")`` ->
    Pillow bilinear resize when the size differs. EXIF orientation is deliberately ignored.
    """
    if len(data) > spec.max_bytes:
        raise PayloadTooLarge(f"image larger than {spec.max_bytes} bytes")
    fmt = sniff_format(data)
    if fmt is None or fmt not in spec.formats:
        raise UnsupportedInput(f"unsupported image format; allowed: {', '.join(spec.formats)}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[fmt]) as image:
                if image.width * image.height > max_pixels:
                    raise _invalid_image(f"image has more than {max_pixels} pixels")
                image.load()
                rgb = image.convert("RGB")
    except InvalidInput:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise _invalid_image(f"image has more than {max_pixels} pixels") from None
    except Exception:  # Pillow raises many unrelated types for corrupt files
        raise _invalid_image("the file is not a valid, complete image") from None
    size = (spec.width, spec.height)
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.uint8).astype(np.float32)
    low, high = spec.value_range
    if (low, high) != (0.0, 255.0):
        array = array * np.float32((high - low) / 255.0) + np.float32(low)
    return array


def to_image_batch(data: bytes, spec: ImageInput, *, max_pixels: int = 16_777_216) -> FloatArray:
    """One encoded image -> a ``(1, height, width, 3)`` batch."""
    return decode_image(data, spec, max_pixels=max_pixels)[np.newaxis, ...]
