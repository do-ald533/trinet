"""Test doubles. Lives outside ``src/`` so it never ships in the image."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from trinet_api.infrastructure.runtimes import check_batch

type Fn = Callable[[NDArray[np.float32]], NDArray[np.float32]]


def softmax_rows(x: NDArray[np.floating]) -> NDArray[np.float32]:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class FakeRuntime:
    """Deterministic ``ModelRuntime`` with no model behind it.

    ``trailing`` is the per-row input shape and ``width`` the output size. The default
    function is a fixed linear map (softmax when ``width > 1``), so outputs depend on the
    input and are reproducible. ``fn`` overrides it; ``error`` makes ``predict`` raise;
    ``delay_s`` / ``gate`` simulate slow inference.
    """

    kind = "fake"

    def __init__(
        self,
        trailing: tuple[int, ...],
        width: int,
        *,
        fn: Fn | None = None,
        error: Exception | None = None,
        delay_s: float = 0.0,
        gate: threading.Event | None = None,
    ) -> None:
        self.trailing = trailing
        self.width = width
        self.error = error
        self.delay_s = delay_s
        self.gate = gate
        self.calls: list[int] = []
        self._lock = threading.Lock()
        size = int(np.prod(trailing))
        self._weights = np.random.default_rng(0).normal(size=(size, width)).astype(np.float32)
        self._fn = fn or self._default

    def _default(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        logits = batch.reshape(batch.shape[0], -1) @ self._weights
        if self.width == 1:
            return np.asarray(logits, dtype=np.float32)
        scale = np.maximum(
            1.0, np.abs(logits).max(axis=1, keepdims=True)
        )  # per row: rows stay independent
        return softmax_rows(logits / scale)

    def predict(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        check_batch(batch, self.trailing)  # same input contract as the real adapters
        with self._lock:
            self.calls.append(batch.shape[0])
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return self._fn(batch.copy())
