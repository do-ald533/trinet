"""Ports: what the application needs from the outside world (two members, on purpose)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ModelRuntime(Protocol):
    """An inference engine bound to one model (ONNX Runtime, pure NumPy, LiteRT, a fake...).

    Contract (checked by the shared runtime contract suite, so every adapter is
    substitutable for every other):

    * ``predict`` takes a C-contiguous ``float32`` batch ``(n, ...)`` and returns a
      ``float32`` array ``(n, outputs)`` with finite values;
    * wrong dtype, rank or trailing dimensions raise ``ValueError``;
    * the input is never mutated, and concurrent calls are safe.
    """

    @property
    def kind(self) -> str:
        """Manifest runtime kind this adapter implements (``onnx``, ``numpy_mlp``...)."""
        ...

    def predict(self, batch: NDArray[np.float32]) -> NDArray[np.float32]: ...
