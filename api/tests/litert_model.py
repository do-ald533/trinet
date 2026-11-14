"""Builds a tiny real ``.tflite`` (FULLY_CONNECTED -> SOFTMAX) without TensorFlow.

Uses the FlatBuffers object API that ships inside ``ai-edge-litert``; callers must
``pytest.importorskip("ai_edge_litert")`` first.
"""

from __future__ import annotations

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as fb
from numpy.typing import NDArray


def _buffer(array: NDArray[np.float32] | None = None) -> fb.BufferT:
    buffer = fb.BufferT()
    if array is not None:
        buffer.data = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint8)
    return buffer


def _tensor(
    name: str, shape: list[int], *, buffer: int = 0, dynamic_batch: bool = False
) -> fb.TensorT:
    tensor = fb.TensorT()
    tensor.name = name
    tensor.shape = shape
    tensor.type = fb.TensorType.FLOAT32
    tensor.buffer = buffer
    if dynamic_batch:
        tensor.shapeSignature = [-1, *shape[1:]]
    return tensor


def _operator_code(builtin: int) -> fb.OperatorCodeT:
    code = fb.OperatorCodeT()
    code.deprecatedBuiltinCode = builtin
    code.builtinCode = builtin
    code.version = 1
    return code


def softmax_classifier(kernel: NDArray[np.float32], bias: NDArray[np.float32]) -> bytes:
    """``softmax(x @ kernel + bias)`` with a dynamic batch dimension; kernel is ``(in, out)``."""
    n_in, n_out = kernel.shape
    weights = kernel.T.astype(np.float32)  # FULLY_CONNECTED stores (out, in)

    softmax_options = fb.SoftmaxOptionsT()
    softmax_options.beta = 1.0

    fully_connected = fb.OperatorT()
    fully_connected.opcodeIndex = 0
    fully_connected.inputs = [0, 1, 2]
    fully_connected.outputs = [3]
    fully_connected.builtinOptionsType = fb.BuiltinOptions.FullyConnectedOptions
    fully_connected.builtinOptions = fb.FullyConnectedOptionsT()

    softmax = fb.OperatorT()
    softmax.opcodeIndex = 1
    softmax.inputs = [3]
    softmax.outputs = [4]
    softmax.builtinOptionsType = fb.BuiltinOptions.SoftmaxOptions
    softmax.builtinOptions = softmax_options

    graph = fb.SubGraphT()
    graph.name = "main"
    graph.tensors = [
        _tensor("inputs", [1, n_in], dynamic_batch=True),
        _tensor("kernel", [n_out, n_in], buffer=1),
        _tensor("bias", [n_out], buffer=2),
        _tensor("logits", [1, n_out], dynamic_batch=True),
        _tensor("probabilities", [1, n_out], dynamic_batch=True),
    ]
    graph.inputs = [0]
    graph.outputs = [4]
    graph.operators = [fully_connected, softmax]

    model = fb.ModelT()
    model.version = 3
    model.operatorCodes = [
        _operator_code(fb.BuiltinOperator.FULLY_CONNECTED),
        _operator_code(fb.BuiltinOperator.SOFTMAX),
    ]
    model.subgraphs = [graph]
    model.buffers = [_buffer(), _buffer(weights), _buffer(bias.astype(np.float32))]
    model.description = "trinet test model"

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b"TFL3")
    return bytes(builder.Output())
