"""LiteRT (``ai-edge-litert``) runtime: rung 3 of the ladder, used for the CNN if ONNX fails.

The package is an optional extra (``trinet-api[litert]``) and is imported lazily, so an image
built without it still starts; a manifest that asks for ``litert`` then simply falls through
to the next rung (or marks the model unavailable). The interpreter is not thread-safe, so
every call holds a lock.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trinet_api.infrastructure.runtimes import (
    RuntimeLoadError,
    check_batch,
    check_output,
    static_dims,
)
from trinet_contract import TensorSpec


def _interpreter_class() -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeLoadError("ai-edge-litert is not installed (extra 'litert')") from exc
    return Interpreter


class LiteRtRuntime:
    kind = "litert"

    def __init__(
        self,
        model: bytes,
        input_spec: TensorSpec,
        output_spec: TensorSpec,
        *,
        num_threads: int = 1,
    ) -> None:
        self._trailing, self._width = static_dims(input_spec, output_spec)
        interpreter_cls = _interpreter_class()
        try:
            self._interpreter = interpreter_cls(model_content=model, num_threads=num_threads)
            self._interpreter.allocate_tensors()
            (self._input,) = self._interpreter.get_input_details()
            (self._output,) = self._interpreter.get_output_details()
        except RuntimeLoadError:
            raise
        except Exception as exc:  # the wrapper raises ValueError/RuntimeError, depending on version
            raise RuntimeLoadError(f"LiteRT rejected the model: {type(exc).__name__}") from exc
        if self._input["dtype"] != np.float32 or self._output["dtype"] != np.float32:
            raise RuntimeLoadError("LiteRT model must use float32 input and output")
        self._lock = threading.Lock()
        self._current_rows = int(self._input["shape"][0])

    def predict(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        check_batch(batch, self._trailing)
        rows = batch.shape[0]
        with self._lock:
            if rows != self._current_rows:
                self._interpreter.resize_tensor_input(self._input["index"], [rows, *self._trailing])
                self._interpreter.allocate_tensors()
                self._current_rows = rows
            self._interpreter.set_tensor(self._input["index"], batch)
            self._interpreter.invoke()
            output = np.array(self._interpreter.get_tensor(self._output["index"]), copy=True)
        return check_output(output, rows, self._width)
