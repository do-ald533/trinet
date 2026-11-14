"""Immutable entities. Request/response DTOs live in ``interface/http/schemas.py``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from trinet_api.domain.errors import ModelNotAvailable, ModelNotFound
from trinet_api.domain.ports import ModelRuntime
from trinet_contract import Manifest


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A loaded model: its validated manifest plus the runtime that serves it."""

    manifest: Manifest
    runtime: ModelRuntime

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version


@dataclass(frozen=True, slots=True)
class UnavailableModel:
    """An artifact that is on disk but cannot serve. ``reason`` is a short machine code."""

    name: str
    version: str | None
    reason: str


ModelEntry = ModelBundle | UnavailableModel


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """``scores`` is sorted by probability, highest first (top-k for images, all for tabular)."""

    label: str
    probability: float
    scores: tuple[ClassScore, ...]


@dataclass(frozen=True, slots=True)
class RegressionResult:
    value: float
    unit: str
    target: str


Prediction = ClassificationResult | RegressionResult


class ModelRegistry:
    """Read-only name -> entry mapping (data-driven registry: a new model is a new artifact)."""

    def __init__(self, entries: Mapping[str, ModelEntry]) -> None:
        self._entries: dict[str, ModelEntry] = dict(sorted(entries.items()))

    def __iter__(self) -> Iterator[ModelEntry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def entry(self, name: str) -> ModelEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise ModelNotFound(name) from None

    def get(self, name: str) -> ModelBundle:
        """Return a servable model or raise ``ModelNotFound`` / ``ModelNotAvailable``."""
        entry = self.entry(name)
        if isinstance(entry, UnavailableModel):
            raise ModelNotAvailable(name, entry.reason)
        return entry

    @property
    def available(self) -> tuple[ModelBundle, ...]:
        return tuple(e for e in self._entries.values() if isinstance(e, ModelBundle))

    @property
    def ready(self) -> bool:
        """At least one model can serve (the others may be unavailable)."""
        return bool(self.available)
