"""Runtime adapters (Strategy): each one implements ``trinet_api.domain.ports.ModelRuntime``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from trinet_contract import TensorSpec


class RuntimeLoadError(Exception):
    """An artifact cannot be turned into a runtime (the loader moves to the next rung)."""


def static_dims(input_spec: TensorSpec, output_spec: TensorSpec) -> tuple[tuple[int, ...], int]:
    """``(trailing input dims, output width)``; only the batch axis may be dynamic."""
    trailing = input_spec.shape[1:]
    width = output_spec.shape[-1]
    if width is None or len(output_spec.shape) != 2 or any(d is None for d in trailing):
        raise RuntimeLoadError("only the batch axis of the tensors may be dynamic")
    return tuple(int(d) for d in trailing if d is not None), width


def check_batch(batch: NDArray[np.float32], trailing: tuple[int, ...]) -> None:
    """Enforce the ``ModelRuntime`` input contract: float32, ``(n, *trailing)``, ``n >= 1``."""
    if not isinstance(batch, np.ndarray) or batch.dtype != np.float32:
        raise ValueError("batch must be a float32 numpy array")
    if batch.ndim != len(trailing) + 1 or tuple(batch.shape[1:]) != trailing:
        raise ValueError(f"batch must have shape (n, {', '.join(map(str, trailing))})")
    if batch.shape[0] < 1:
        raise ValueError("batch must contain at least one row")


def check_output(output: NDArray[np.float32], rows: int, width: int) -> NDArray[np.float32]:
    """Shape/finite check of what an engine returned; yields a float32 ``(rows, width)`` array."""
    result = np.asarray(output, dtype=np.float32)
    if result.shape != (rows, width):
        raise RuntimeError(f"runtime returned shape {result.shape}, expected {(rows, width)}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("runtime returned non-finite values")
    return result
