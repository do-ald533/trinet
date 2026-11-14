"""ONNX Runtime adapter (CPU).

Security notes (verified against onnxruntime 1.31.1, see ``tests/unit/test_onnx_guard.py``):

* The model is passed to ORT as **bytes**, never as a path, after the loader verified its
  SHA-256. ORT still resolves *external data* files of a bytes-loaded model relative to the
  process working directory (it only rejects paths that escape it), so loading from bytes is
  not enough to refuse external data.
* Therefore ``has_external_data`` walks the protobuf wire format (no ``onnx`` package, which
  had CVE-2026-34447 in this very code path) and the adapter refuses any model with a tensor
  whose ``data_location`` is EXTERNAL or that carries ``external_data`` entries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from trinet_api.infrastructure.runtimes import (
    RuntimeLoadError,
    check_batch,
    check_output,
    static_dims,
)
from trinet_contract import TensorSpec

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- protobuf scan
# Field numbers from onnx.proto (stable since IR v3). Only the messages that can hold a
# TensorProto are traversed; large ``raw_data`` blobs are skipped by length, never copied.

_MODEL_GRAPH = 7
_MODEL_TRAINING_INFO = 20
_MODEL_FUNCTION = 25
_TRAINING_GRAPHS = (1, 2)  # initialization, algorithm
_GRAPH_NODE = 1
_GRAPH_INITIALIZER = 5
_GRAPH_SPARSE_INITIALIZER = 15
_FUNCTION_NODE = 7
_NODE_ATTRIBUTE = 5
_ATTR_TENSOR = 5
_ATTR_GRAPH = 6
_ATTR_TENSORS = 10
_ATTR_GRAPHS = 11
_ATTR_SPARSE = 22
_ATTR_SPARSES = 23
_SPARSE_TENSORS = (1, 2)  # values, indices
_TENSOR_EXTERNAL_DATA = 13
_TENSOR_DATA_LOCATION = 14
_DATA_LOCATION_EXTERNAL = 1
_MAX_DEPTH = 16

_VARINT, _FIXED64, _LEN, _FIXED32 = 0, 1, 2, 5
type _Span = tuple[int, int]


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    for shift in range(0, 70, 7):
        if pos >= len(buf):
            raise ValueError("truncated protobuf")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
    raise ValueError("malformed varint")


def _fields(buf: bytes, span: _Span) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(field, wire_type, a, b)``: a varint value in ``a``, or a LEN payload span a..b."""
    pos, end = span
    while pos < end:
        tag, pos = _varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == _VARINT:
            value, pos = _varint(buf, pos)
            yield field, wire, value, 0
        elif wire == _LEN:
            length, pos = _varint(buf, pos)
            if pos + length > end:
                raise ValueError("truncated protobuf")
            yield field, wire, pos, pos + length
            pos += length
        elif wire in (_FIXED32, _FIXED64):
            size = 4 if wire == _FIXED32 else 8
            if pos + size > end:
                raise ValueError("truncated protobuf")
            pos += size
            yield field, wire, 0, 0
        else:
            raise ValueError("unsupported protobuf wire type")


def _subs(buf: bytes, span: _Span, wanted: tuple[int, ...]) -> Iterator[_Span]:
    for field, wire, a, b in _fields(buf, span):
        if wire == _LEN and field in wanted:
            yield a, b


def _tensor_is_external(buf: bytes, span: _Span) -> bool:
    for field, wire, a, _b in _fields(buf, span):
        if field == _TENSOR_EXTERNAL_DATA and wire == _LEN:
            return True
        if field == _TENSOR_DATA_LOCATION and wire == _VARINT and a == _DATA_LOCATION_EXTERNAL:
            return True
    return False


def _attribute_external(buf: bytes, span: _Span, depth: int) -> bool:
    tensors = (*_subs(buf, span, (_ATTR_TENSOR, _ATTR_TENSORS)),)
    if any(_tensor_is_external(buf, t) for t in tensors):
        return True
    sparse = (*_subs(buf, span, (_ATTR_SPARSE, _ATTR_SPARSES)),)
    if any(_sparse_external(buf, s) for s in sparse):
        return True
    return any(
        _graph_external(buf, g, depth + 1) for g in _subs(buf, span, (_ATTR_GRAPH, _ATTR_GRAPHS))
    )


def _sparse_external(buf: bytes, span: _Span) -> bool:
    return any(_tensor_is_external(buf, t) for t in _subs(buf, span, _SPARSE_TENSORS))


def _nodes_external(buf: bytes, nodes: Iterator[_Span], depth: int) -> bool:
    return any(
        _attribute_external(buf, attr, depth)
        for node in nodes
        for attr in _subs(buf, node, (_NODE_ATTRIBUTE,))
    )


def _graph_external(buf: bytes, span: _Span, depth: int) -> bool:
    if depth > _MAX_DEPTH:
        raise ValueError("ONNX graph nesting too deep")
    if any(_tensor_is_external(buf, t) for t in _subs(buf, span, (_GRAPH_INITIALIZER,))):
        return True
    if any(_sparse_external(buf, s) for s in _subs(buf, span, (_GRAPH_SPARSE_INITIALIZER,))):
        return True
    return _nodes_external(buf, _subs(buf, span, (_GRAPH_NODE,)), depth)


def has_external_data(model: bytes) -> bool:
    """True if any tensor of the serialized ``ModelProto`` lives outside the file.

    Raises ``ValueError`` if the bytes are not a well-formed protobuf (treated as unusable).
    """
    whole: _Span = (0, len(model))
    for field, wire, a, b in _fields(model, whole):
        if wire != _LEN:
            continue
        if field == _MODEL_GRAPH and _graph_external(model, (a, b), 0):
            return True
        if field == _MODEL_FUNCTION and _nodes_external(
            model, _subs(model, (a, b), (_FUNCTION_NODE,)), 0
        ):
            return True
        if field == _MODEL_TRAINING_INFO and any(
            _graph_external(model, g, 1) for g in _subs(model, (a, b), _TRAINING_GRAPHS)
        ):
            return True
    return False


# --------------------------------------------------------------------------- runtime


def _check_io(
    session: ort.InferenceSession, input_spec: TensorSpec, output_spec: TensorSpec
) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeLoadError("ONNX graph must have exactly one input and one output")
    for actual, expected, role in (
        (inputs[0], input_spec, "input"),
        (outputs[0], output_spec, "output"),
    ):
        if actual.name != expected.name:
            raise RuntimeLoadError(
                f"ONNX {role} is named {actual.name!r}, manifest says {expected.name!r}"
            )
        if actual.type != "tensor(float)":
            raise RuntimeLoadError(f"ONNX {role} type is {actual.type}, expected tensor(float)")
        shape = list(actual.shape)
        if len(shape) != len(expected.shape) or any(
            isinstance(a, int) and b is not None and a != b
            for a, b in zip(shape, expected.shape, strict=True)
        ):
            raise RuntimeLoadError(f"ONNX {role} shape {shape} does not match the manifest")


class OnnxRuntime:
    """One shared ``InferenceSession`` per model (``run`` is thread-safe and releases the GIL)."""

    kind = "onnx"

    def __init__(
        self,
        model: bytes,
        input_spec: TensorSpec,
        output_spec: TensorSpec,
        *,
        intra_op_threads: int = 1,
    ) -> None:
        try:
            external = has_external_data(model)
        except ValueError as exc:
            raise RuntimeLoadError(f"not a valid ONNX file: {exc}") from exc
        if external:
            raise RuntimeLoadError("ONNX model uses external data; refused")
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        try:
            self._session = ort.InferenceSession(
                model, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime raises its own, version-dependent types
            raise RuntimeLoadError(f"onnxruntime rejected the model: {type(exc).__name__}") from exc
        _check_io(self._session, input_spec, output_spec)
        self._input_name = input_spec.name
        self._output_name = output_spec.name
        self._trailing, self._width = static_dims(input_spec, output_spec)

    def predict(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        check_batch(batch, self._trailing)
        (output,) = self._session.run([self._output_name], {self._input_name: batch})
        return check_output(output, batch.shape[0], self._width)
