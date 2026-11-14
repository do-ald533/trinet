"""Golden samples: raw inputs + expected outputs produced by the notebook pipeline.

The API test suite replays them to detect training/serving skew:
- tabular: ``record`` (raw, ``None`` = missing) -> ``features`` (pre-graph vector) -> ``output``;
- image: PNG file -> ``output``.
``label`` (ground truth) is present for the labelled evaluation sample re-scored in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from trinet_contract.manifest import FileName


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoldenTolerance(_Strict):
    features_atol: float = Field(default=1e-6, ge=0)
    output_atol: float = Field(default=1e-4, ge=0)
    # Minimum top-1 probability margin for a case to be used in top-1 agreement checks.
    top1_margin: float = Field(default=1e-3, ge=0)


class TabularGoldenCase(_Strict):
    record: dict[str, float | str | None]
    features: tuple[float, ...] = Field(min_length=1)
    output: tuple[float, ...] = Field(min_length=1)
    label: str | float | None = None


class ImageGoldenCase(_Strict):
    file: FileName
    output: tuple[float, ...] = Field(min_length=1)
    label: str | None = None


class TabularGolden(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["tabular"] = "tabular"
    model: str
    version: str
    tolerance: GoldenTolerance = GoldenTolerance()
    cases: tuple[TabularGoldenCase, ...] = Field(min_length=1)


class ImageGolden(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["image"] = "image"
    model: str
    version: str
    tolerance: GoldenTolerance = GoldenTolerance(output_atol=1e-3)
    # Directory (relative to the artifact directory) holding the PNG files.
    directory: FileName = "golden"
    cases: tuple[ImageGoldenCase, ...] = Field(min_length=1)


Golden = Annotated[TabularGolden | ImageGolden, Field(discriminator="kind")]

_GOLDEN_ADAPTER: TypeAdapter[TabularGolden | ImageGolden] = TypeAdapter(Golden)


def load_golden(path: Path) -> TabularGolden | ImageGolden:
    return _GOLDEN_ADAPTER.validate_json(path.read_bytes())


def golden_json_schema() -> dict[str, object]:
    return _GOLDEN_ADAPTER.json_schema()
