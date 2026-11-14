"""Pure-NumPy runtime for the tabular MLPs (rung 2 of the fallback ladder).

Layers come from the manifest (``normalize`` / ``dense`` / ``denormalize`` / ``exp``); weights
come from an ``.npz`` loaded with ``allow_pickle=False``. All maths is float32, like the exported
graph.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trinet_api.infrastructure.runtimes import (
    RuntimeLoadError,
    check_batch,
    check_output,
    static_dims,
)
from trinet_contract import DenormalizeLayer, DenseLayer, ExpLayer, NormalizeLayer, TensorSpec
from trinet_contract.manifest import MlpLayer

# Keras ``Normalization`` divides by ``max(sqrt(variance), keras.backend.epsilon())``.
_EPSILON = np.float32(1e-7)

type _Matrix = NDArray[np.float32]
type _Step = Callable[[_Matrix], _Matrix]


@dataclass(frozen=True, slots=True)
class _Normalize:
    mean: _Matrix
    std: _Matrix

    def __call__(self, x: _Matrix) -> _Matrix:
        return np.asarray((x - self.mean) / self.std, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class _Denormalize:
    mean: _Matrix
    std: _Matrix

    def __call__(self, x: _Matrix) -> _Matrix:
        return np.asarray(x * self.std + self.mean, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class _Exp:
    def __call__(self, x: _Matrix) -> _Matrix:
        return np.asarray(np.exp(x), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class _Dense:
    kernel: _Matrix
    bias: _Matrix
    activation: str

    def __call__(self, x: _Matrix) -> _Matrix:
        y = x @ self.kernel + self.bias
        if self.activation == "relu":
            return np.asarray(np.maximum(y, np.float32(0.0)), dtype=np.float32)
        if self.activation == "softmax":
            e = np.exp(y - y.max(axis=-1, keepdims=True))
            return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=np.float32)
        return np.asarray(y, dtype=np.float32)


def _array(weights: np.lib.npyio.NpzFile, name: str, ndim: int) -> _Matrix:
    if name not in weights.files:
        raise RuntimeLoadError(f"weights file has no array {name!r}")
    array = np.asarray(weights[name])
    if array.dtype.kind != "f":
        raise RuntimeLoadError(f"array {name!r} is not floating point")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise RuntimeLoadError(f"array {name!r} has non-finite values")
    if ndim == 1:
        array = array.reshape(-1)  # Keras stores (1, n) or (n,) statistics
    elif array.ndim != 2:
        raise RuntimeLoadError(f"array {name!r} must be 2-D")
    return np.ascontiguousarray(array)


def _std(variance: _Matrix) -> _Matrix:
    return np.maximum(np.sqrt(np.maximum(variance, np.float32(0.0))), _EPSILON)


def _build_step(layer: MlpLayer, weights: np.lib.npyio.NpzFile, width: int) -> tuple[_Step, int]:
    """Return the step for ``layer`` and the feature width after it."""
    match layer:
        case ExpLayer():
            return _Exp(), width
        case NormalizeLayer(key=key) | DenormalizeLayer(key=key):
            mean = _array(weights, f"{key}.mean", 1)
            std = _std(_array(weights, f"{key}.variance", 1))
            if mean.shape != (width,) or std.shape != (width,):
                raise RuntimeLoadError(f"statistics {key!r} do not match width {width}")
            if isinstance(layer, NormalizeLayer):
                return _Normalize(mean, std), width
            return _Denormalize(mean, std), width
        case DenseLayer(key=key, activation=activation):
            kernel = _array(weights, f"{key}.kernel", 2)
            bias = _array(weights, f"{key}.bias", 1)
            if kernel.shape[0] != width or bias.shape != (kernel.shape[1],):
                raise RuntimeLoadError(f"dense layer {key!r} does not chain from width {width}")
            return _Dense(kernel, bias, activation), kernel.shape[1]
    raise RuntimeLoadError("unsupported layer")  # pragma: no cover - contract union is closed


class NumpyMlpRuntime:
    kind = "numpy_mlp"

    def __init__(
        self,
        weights: bytes,
        layers: tuple[MlpLayer, ...],
        input_spec: TensorSpec,
        output_spec: TensorSpec,
    ) -> None:
        self._trailing, self._width = static_dims(input_spec, output_spec)
        if len(self._trailing) != 1:
            raise RuntimeLoadError("numpy_mlp only supports flat feature vectors")
        try:
            archive = np.load(io.BytesIO(weights), allow_pickle=False)
        except (ValueError, OSError, EOFError) as exc:
            raise RuntimeLoadError(f"not a valid .npz file: {type(exc).__name__}") from exc
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise RuntimeLoadError("weights file must be an .npz archive")
        steps: list[_Step] = []
        width = self._trailing[0]
        with archive:
            try:
                for layer in layers:
                    step, width = _build_step(layer, archive, width)
                    steps.append(step)
            except (ValueError, OSError, EOFError) as exc:  # corrupt member / pickled array
                raise RuntimeLoadError(f"unreadable weights: {type(exc).__name__}") from exc
        if width != self._width:
            raise RuntimeLoadError(f"MLP outputs {width} values, manifest says {self._width}")
        self._steps = tuple(steps)

    def predict(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        check_batch(batch, self._trailing)
        x = batch
        for step in self._steps:
            x = step(x)
        return check_output(x, batch.shape[0], self._width)
