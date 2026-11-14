"""LiteRT adapter: stub interpreter (always) and a real tiny ``.tflite`` (when installed)."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, ClassVar

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from numpy.typing import NDArray

from trinet_api.infrastructure.loading import load_model
from trinet_api.infrastructure.runtimes import RuntimeLoadError, litert
from trinet_api.infrastructure.runtimes.litert import LiteRtRuntime
from trinet_contract import TensorSpec

IN = TensorSpec(name="x", shape=(None, 4))
OUT = TensorSpec(name="y", shape=(None, 3))
F32 = np.dtype(np.float32)


class StubInterpreter:
    """Mimics the parts of the LiteRT ``Interpreter`` API the adapter relies on."""

    instances: ClassVar[list[StubInterpreter]] = []
    out_dtype: Any = np.float32
    produce: Any = None

    def __init__(self, model_content: bytes, num_threads: int) -> None:
        self.model_content = model_content
        self.num_threads = num_threads
        self.rows = 1
        self.resizes: list[list[int]] = []
        self._value = np.zeros((1, 4), np.float32)
        StubInterpreter.instances.append(self)

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self) -> list[dict[str, Any]]:
        return [{"index": 0, "dtype": np.float32, "shape": np.array([self.rows, 4])}]

    def get_output_details(self) -> list[dict[str, Any]]:
        return [{"index": 1, "dtype": StubInterpreter.out_dtype, "shape": np.array([self.rows, 3])}]

    def resize_tensor_input(self, index: int, shape: list[int]) -> None:
        self.resizes.append(shape)
        self.rows = shape[0]

    def set_tensor(self, index: int, value: Any) -> None:
        self._value = value

    def invoke(self) -> None:
        return None

    def get_tensor(self, index: int) -> Any:
        if StubInterpreter.produce is not None:
            return StubInterpreter.produce(self._value)
        return np.tile(np.array([[0.2, 0.3, 0.5]], np.float32), (self._value.shape[0], 1))


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[StubInterpreter]:
    StubInterpreter.instances = []
    StubInterpreter.out_dtype = np.float32
    StubInterpreter.produce = None
    monkeypatch.setattr(litert, "_interpreter_class", lambda: StubInterpreter)
    return StubInterpreter


def test_stub_runtime_resizes_only_when_the_batch_changes(stub: type[StubInterpreter]) -> None:
    runtime = LiteRtRuntime(b"model", IN, OUT, num_threads=3)
    interpreter = stub.instances[0]
    assert interpreter.model_content == b"model"
    assert interpreter.num_threads == 3
    assert runtime.kind == "litert"
    assert runtime.predict(np.zeros((1, 4), np.float32)).shape == (1, 3)
    assert interpreter.resizes == []
    assert runtime.predict(np.zeros((5, 4), np.float32)).shape == (5, 3)
    assert runtime.predict(np.zeros((5, 4), np.float32)).shape == (5, 3)
    assert interpreter.resizes == [[5, 4]]


def test_stub_runtime_rejects_bad_batches(stub: type[StubInterpreter]) -> None:
    runtime = LiteRtRuntime(b"model", IN, OUT)
    bad_batches: list[NDArray[Any]] = [
        np.zeros((2, 4), np.float64),
        np.zeros((2, 5), np.float32),
        np.zeros(4, np.float32),
        np.zeros((0, 4), np.float32),
    ]
    for bad in bad_batches:
        with pytest.raises(ValueError, match="batch"):
            runtime.predict(bad)


def test_stub_output_is_copied_and_validated(stub: type[StubInterpreter]) -> None:
    shared = np.full((1, 3), 1 / 3, np.float32)
    stub.produce = lambda _x: shared
    runtime = LiteRtRuntime(b"model", IN, OUT)
    result = runtime.predict(np.zeros((1, 4), np.float32))
    result[0, 0] = 99.0
    assert shared[0, 0] != 99.0  # callers cannot corrupt the interpreter's buffer

    stub.produce = lambda _x: np.zeros((1, 2), np.float32)
    with pytest.raises(RuntimeError, match="shape"):
        runtime.predict(np.zeros((1, 4), np.float32))
    stub.produce = lambda _x: np.full((1, 3), np.nan, np.float32)
    with pytest.raises(RuntimeError, match="non-finite"):
        runtime.predict(np.zeros((1, 4), np.float32))


def test_non_float32_tensors_are_refused(stub: type[StubInterpreter]) -> None:
    stub.out_dtype = np.uint8
    with pytest.raises(RuntimeLoadError, match="float32"):
        LiteRtRuntime(b"model", IN, OUT)


def test_interpreter_failures_become_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Exploding:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("Could not open model")

    monkeypatch.setattr(litert, "_interpreter_class", lambda: Exploding)
    with pytest.raises(RuntimeLoadError, match="rejected"):
        LiteRtRuntime(b"junk", IN, OUT)


def test_missing_package_is_a_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ai_edge_litert", None)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", None)
    with pytest.raises(RuntimeLoadError, match="not installed"):
        LiteRtRuntime(b"x", IN, OUT)


def test_dynamic_trailing_dimensions_are_refused(stub: type[StubInterpreter]) -> None:
    with pytest.raises(RuntimeLoadError, match="dynamic"):
        LiteRtRuntime(b"x", TensorSpec(name="x", shape=(None, None)), OUT)


def test_calls_are_serialised_by_a_lock(stub: type[StubInterpreter]) -> None:
    active = 0
    peak = 0
    guard = threading.Lock()

    def produce(value: Any) -> Any:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            return np.tile(np.array([[0.2, 0.3, 0.5]], np.float32), (value.shape[0], 1))
        finally:
            with guard:
                active -= 1

    stub.produce = produce
    runtime = LiteRtRuntime(b"model", IN, OUT)
    threads = [
        threading.Thread(target=lambda: [runtime.predict(np.zeros((2, 4), np.float32))] * 20)
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 1


# --------------------------------------------------------------------------- real interpreter


@pytest.fixture(scope="module")
def tflite() -> tuple[bytes, np.ndarray, np.ndarray]:
    pytest.importorskip("ai_edge_litert")
    from tests.litert_model import softmax_classifier

    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(4, 3)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    return softmax_classifier(kernel, bias), kernel, bias


def reference(x: Any, kernel: Any, bias: Any) -> Any:
    logits = x @ kernel + bias
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("rows", [1, 7, 2, 33, 4])
def test_real_interpreter_matches_numpy_for_any_batch_size(
    tflite: tuple[bytes, Any, Any], rows: int
) -> None:
    blob, kernel, bias = tflite
    runtime = LiteRtRuntime(blob, IN, OUT)
    x = np.random.default_rng(rows).normal(size=(rows, 4)).astype(np.float32)
    got = runtime.predict(x)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, reference(x, kernel, bias), atol=1e-5)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-5)


def test_real_interpreter_is_thread_safe(tflite: tuple[bytes, Any, Any]) -> None:
    blob, kernel, bias = tflite
    runtime = LiteRtRuntime(blob, IN, OUT)
    failures: list[str] = []

    def work(seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(25):
            x = rng.normal(size=(int(rng.integers(1, 9)), 4)).astype(np.float32)
            if not np.allclose(runtime.predict(x), reference(x, kernel, bias), atol=1e-5):
                failures.append("mismatch")

    threads = [threading.Thread(target=work, args=(s,)) for s in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_real_interpreter_rejects_garbage() -> None:
    pytest.importorskip("ai_edge_litert")
    with pytest.raises(RuntimeLoadError, match="rejected"):
        LiteRtRuntime(b"definitely not a flatbuffer", IN, OUT)


def test_loader_serves_a_litert_only_manifest(
    artifacts_copy: Path, tflite: tuple[bytes, Any, Any]
) -> None:
    """End to end: a manifest whose only runtime is LiteRT goes live through the loader."""
    _, _, _ = tflite
    from tests.litert_model import softmax_classifier

    rng = np.random.default_rng(1)
    blob = softmax_classifier(
        rng.normal(size=(25, 2)).astype(np.float32), rng.normal(size=2).astype(np.float32)
    )
    directory = artifacts_copy / "credit-default-mlp" / dummy.DUMMY_VERSION
    (directory / "model.tflite").write_bytes(blob)
    manifest_path = directory / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["runtimes"] = [
        {"kind": "litert", "file": "model.tflite", "sha256": hashlib.sha256(blob).hexdigest()}
    ]
    manifest_path.write_text(json.dumps(data))
    bundle = load_model(directory)
    assert bundle.runtime.kind == "litert"  # type: ignore[union-attr]
