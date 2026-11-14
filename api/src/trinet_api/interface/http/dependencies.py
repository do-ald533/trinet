"""FastAPI dependencies. Everything comes from ``app.state`` (set by the composition root),
so tests can swap pieces with ``app.dependency_overrides``."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from trinet_api.application.service import PredictionService
from trinet_api.infrastructure.observability import Metrics
from trinet_api.settings import Settings


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_service(request: Request) -> PredictionService:
    service: PredictionService = request.app.state.service
    return service


def get_metrics(request: Request) -> Metrics:
    metrics: Metrics = request.app.state.metrics
    return metrics


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[PredictionService, Depends(get_service)]
MetricsDep = Annotated[Metrics, Depends(get_metrics)]
