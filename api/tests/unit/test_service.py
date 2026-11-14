from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from pathlib import Path

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from PIL import Image
from tests.fakes import FakeRuntime

from trinet_api.application.service import (
    ConcurrencyLimiter,
    ImageRequest,
    PredictionService,
    TabularRequest,
)
from trinet_api.domain.entities import (
    ClassificationResult,
    ModelBundle,
    ModelRegistry,
    RegressionResult,
    UnavailableModel,
)
from trinet_api.domain.errors import (
    InferenceFailed,
    InvalidInput,
    ModelNotAvailable,
    ModelNotFound,
    Overloaded,
    UnsupportedInput,
)
from trinet_api.domain.ports import ModelRuntime
from trinet_api.infrastructure.loading import load_model
from trinet_contract import Manifest


@pytest.fixture(scope="module")
def manifests(artifacts_root: Path) -> dict[str, Manifest]:
    out = {}
    for name in dummy.MODELS:
        entry = load_model(artifacts_root / name / dummy.DUMMY_VERSION)
        assert isinstance(entry, ModelBundle)
        out[name] = entry.manifest
    return out


def service_with(
    manifests: dict[str, Manifest],
    runtimes: dict[str, ModelRuntime],
    *,
    concurrency: int = 2,
    timeout: float = 0.05,
    **kwargs: object,
) -> PredictionService:
    entries = {n: ModelBundle(manifests[n], rt) for n, rt in runtimes.items()}
    entries["broken-model"] = UnavailableModel("broken-model", "1.0.0", "no_usable_runtime")
    return PredictionService(
        ModelRegistry(entries),
        ConcurrencyLimiter(concurrency, timeout),
        **kwargs,  # type: ignore[arg-type]
    )


def drybean_record(manifests: dict[str, Manifest]) -> dict[str, float]:
    return dict.fromkeys((f.name for f in manifests["drybean-mlp"].input.fields), 1.0)  # type: ignore[union-attr]


def test_tabular_classification_returns_all_class_probabilities(
    manifests: dict[str, Manifest],
) -> None:
    fake = FakeRuntime((16,), 7)
    service = service_with(manifests, {"drybean-mlp": fake})
    outcome = service.predict("drybean-mlp", TabularRequest([drybean_record(manifests)] * 3))
    assert outcome.model_name == "drybean-mlp"
    assert outcome.model_version == "0.0.0-dummy"
    assert outcome.runtime == "fake"
    assert fake.calls == [3]  # one batched call
    assert len(outcome.predictions) == 3
    first = outcome.predictions[0]
    assert isinstance(first, ClassificationResult)
    assert len(first.scores) == 7
    assert sum(s.probability for s in first.scores) == pytest.approx(1.0, abs=1e-5)
    assert outcome.latency_ms >= outcome.inference_ms >= 0


def test_regression_value_and_unit(manifests: dict[str, Manifest]) -> None:
    service = service_with(
        manifests,
        {
            "housing-mlp": FakeRuntime(
                (15,), 1, fn=lambda b: np.full((len(b), 1), 4.5e5, np.float32)
            )
        },
    )
    record = {
        "longitude": -120.0, "latitude": 36.0, "housing_median_age": 20.0, "total_rooms": 1000.0,
        "population": 500.0, "households": 200.0, "median_income": 4.0, "ocean_proximity": "INLAND",
    }  # fmt: skip
    (prediction,) = service.predict("housing-mlp", TabularRequest([record])).predictions
    assert isinstance(prediction, RegressionResult)
    assert (prediction.value, prediction.unit, prediction.target) == (
        4.5e5,
        "USD",
        "median_house_value",
    )


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 90), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_returns_top_k_only(manifests: dict[str, Manifest]) -> None:
    service = service_with(manifests, {"intel-cnn": FakeRuntime((150, 150, 3), 6)}, top_k=2)
    (prediction,) = service.predict("intel-cnn", ImageRequest(png())).predictions
    assert isinstance(prediction, ClassificationResult)
    assert len(prediction.scores) == 2


def test_unknown_and_unavailable_models(manifests: dict[str, Manifest]) -> None:
    service = service_with(manifests, {"drybean-mlp": FakeRuntime((16,), 7)})
    with pytest.raises(ModelNotFound):
        service.predict("nope", TabularRequest([]))
    with pytest.raises(ModelNotAvailable) as info:
        service.predict("broken-model", TabularRequest([]))
    assert info.value.reason == "no_usable_runtime"


def test_payload_kind_must_match_the_manifest(manifests: dict[str, Manifest]) -> None:
    service = service_with(
        manifests,
        {"drybean-mlp": FakeRuntime((16,), 7), "intel-cnn": FakeRuntime((150, 150, 3), 6)},
    )
    with pytest.raises(UnsupportedInput):
        service.predict("drybean-mlp", ImageRequest(png()))
    with pytest.raises(UnsupportedInput):
        service.predict("intel-cnn", TabularRequest([{}]))


def test_invalid_input_never_reaches_the_runtime(manifests: dict[str, Manifest]) -> None:
    fake = FakeRuntime((16,), 7)
    service = service_with(manifests, {"drybean-mlp": fake}, max_batch=2)
    with pytest.raises(InvalidInput):
        service.predict("drybean-mlp", TabularRequest([{"Area": 1}]))
    with pytest.raises(InvalidInput):
        service.predict("drybean-mlp", TabularRequest([drybean_record(manifests)] * 3))
    assert fake.calls == []


def test_runtime_failures_become_inference_failed_without_leaking(
    manifests: dict[str, Manifest],
) -> None:
    service = service_with(
        manifests, {"drybean-mlp": FakeRuntime((16,), 7, error=RuntimeError("secret path /etc/x"))}
    )
    with pytest.raises(InferenceFailed) as info:
        service.predict("drybean-mlp", TabularRequest([drybean_record(manifests)]))
    assert "secret" not in str(info.value)


@pytest.mark.parametrize(
    "fn",
    [
        lambda b: np.zeros((len(b) + 1, 7), np.float32),  # wrong number of rows
        lambda b: np.zeros((len(b),), np.float32),  # wrong rank
        lambda b: np.full((len(b), 7), np.nan, np.float32),  # not probabilities
        lambda b: np.full((len(b), 7), 0.5, np.float32),  # does not sum to 1
    ],
)
def test_bad_runtime_output_is_rejected(
    manifests: dict[str, Manifest], fn: Callable[[np.ndarray], np.ndarray]
) -> None:
    service = service_with(manifests, {"drybean-mlp": FakeRuntime((16,), 7, fn=fn)})
    with pytest.raises(InferenceFailed):
        service.predict("drybean-mlp", TabularRequest([drybean_record(manifests)]))


def test_overload_when_every_slot_is_busy(manifests: dict[str, Manifest]) -> None:
    gate = threading.Event()
    fake = FakeRuntime((16,), 7, gate=gate)
    service = service_with(manifests, {"drybean-mlp": fake}, concurrency=1, timeout=0.05)
    request = TabularRequest([drybean_record(manifests)])
    first = threading.Thread(target=service.predict, args=("drybean-mlp", request))
    first.start()
    deadline = time.monotonic() + 2
    while service.limiter.inflight == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(Overloaded) as info:
        service.predict("drybean-mlp", request)
    assert info.value.retry_after_s >= 1
    gate.set()
    first.join(timeout=5)
    assert service.limiter.inflight == 0
    service.predict("drybean-mlp", request)  # the slot was released


def test_limiter_never_exceeds_max_concurrency() -> None:
    limiter = ConcurrencyLimiter(max_concurrency=3, queue_timeout_s=5)
    active = peak = 0
    lock = threading.Lock()

    def work() -> None:
        nonlocal active, peak
        with limiter.slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1

    threads = [threading.Thread(target=work) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak <= 3
    assert limiter.inflight == 0


def test_limiter_rejects_zero_slots() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        ConcurrencyLimiter(0, 1)


def test_latency_uses_the_injected_clock(manifests: dict[str, Manifest]) -> None:
    ticks = iter([0.0, 0.010, 0.030, 0.050])
    service = service_with(
        manifests, {"drybean-mlp": FakeRuntime((16,), 7)}, clock=lambda: next(ticks)
    )
    outcome = service.predict("drybean-mlp", TabularRequest([drybean_record(manifests)]))
    assert outcome.inference_ms == pytest.approx(20.0)
    assert outcome.latency_ms == pytest.approx(50.0)
