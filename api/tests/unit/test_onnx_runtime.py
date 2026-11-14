"""ONNX adapter: external-data refusal (protobuf scan) and manifest/graph consistency."""

from __future__ import annotations

from pathlib import Path

import make_dummy_artifacts as dummy
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from trinet_api.infrastructure.runtimes import RuntimeLoadError
from trinet_api.infrastructure.runtimes.onnx import OnnxRuntime, has_external_data
from trinet_contract import TensorSpec

IN = TensorSpec(name="x", shape=(None, 4))
OUT = TensorSpec(name="y", shape=(None, 3))


def external_tensor(name: str = "W") -> TensorProto:
    tensor = numpy_helper.from_array(np.ones((4, 3), np.float32), name=name)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", "w.bin"), ("offset", "0"), ("length", "48")):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    return tensor


def inline_tensor(name: str = "W") -> TensorProto:
    return numpy_helper.from_array(np.ones((4, 3), np.float32), name=name)


def graph_with(
    initializers: list[TensorProto], nodes: list[onnx.NodeProto] | None = None
) -> onnx.GraphProto:
    return helper.make_graph(
        nodes or [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
        initializer=initializers,
    )


def model_of(graph: onnx.GraphProto) -> onnx.ModelProto:
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def serialize(model: onnx.ModelProto) -> bytes:
    return bytes(model.SerializeToString())


def test_inline_model_is_not_external_and_loads() -> None:
    data = serialize(model_of(graph_with([inline_tensor()])))
    assert has_external_data(data) is False
    runtime = OnnxRuntime(data, IN, OUT)
    assert runtime.predict(np.ones((2, 4), np.float32)).tolist() == [[4.0] * 3] * 2


def test_external_initializer_is_detected_and_refused() -> None:
    data = serialize(model_of(graph_with([external_tensor()])))
    assert has_external_data(data) is True
    with pytest.raises(RuntimeLoadError, match="external data"):
        OnnxRuntime(data, IN, OUT)


def test_data_location_alone_is_enough() -> None:
    tensor = inline_tensor()
    tensor.data_location = TensorProto.EXTERNAL
    assert has_external_data(serialize(model_of(graph_with([tensor])))) is True


def test_external_data_hidden_in_a_constant_node_attribute() -> None:
    constant = helper.make_node("Constant", [], ["W"], value=external_tensor("c"))
    graph = graph_with([], [constant, helper.make_node("MatMul", ["x", "W"], ["y"])])
    assert has_external_data(serialize(model_of(graph))) is True


def test_external_data_in_a_subgraph() -> None:
    branch = helper.make_graph([], "then", [], [], initializer=[external_tensor()])
    other = helper.make_graph([], "else", [], [])
    node = helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=other)
    assert has_external_data(serialize(model_of(graph_with([], [node])))) is True


def test_external_data_in_sparse_initializers_and_functions_and_training_info() -> None:
    sparse_model = model_of(graph_with([inline_tensor()]))
    sparse = sparse_model.graph.sparse_initializer.add()
    sparse.values.CopyFrom(external_tensor("s"))
    assert has_external_data(serialize(sparse_model)) is True

    function_model = model_of(graph_with([inline_tensor()]))
    function = function_model.functions.add()
    function.name, function.domain = "f", "custom"
    function.node.add().CopyFrom(
        helper.make_node("Constant", [], ["o"], value=external_tensor("f"))
    )
    assert has_external_data(serialize(function_model)) is True

    training_model = model_of(graph_with([inline_tensor()]))
    training_model.training_info.add().initialization.CopyFrom(graph_with([external_tensor()]))
    assert has_external_data(serialize(training_model)) is True


def test_raw_data_that_looks_like_protobuf_is_not_misread() -> None:
    # bytes that would decode as "field 14 = EXTERNAL" if raw_data were (wrongly) descended into
    tensor = helper.make_tensor("W", TensorProto.UINT8, [4], b"p\x01p\x01", raw=True)
    assert has_external_data(serialize(model_of(graph_with([tensor])))) is False


@pytest.mark.parametrize(
    "blob", [b"\xff\xff\xff\xff\xff", b"\x3a\x10abc", b"\x0b", b"\x39" + b"\x00" * 3]
)
def test_malformed_protobuf_is_an_error_not_a_pass(blob: bytes) -> None:
    with pytest.raises(ValueError, match=r"protobuf|varint|wire"):
        has_external_data(blob)
    with pytest.raises(RuntimeLoadError, match="not a valid ONNX"):
        OnnxRuntime(blob, IN, OUT)


def test_pathological_nesting_is_rejected() -> None:
    graph = graph_with([inline_tensor()])
    for _ in range(20):
        node = helper.make_node("If", ["c"], ["y"], then_branch=graph, else_branch=graph_with([]))
        graph = graph_with([], [node])
    with pytest.raises(ValueError, match="nesting"):
        has_external_data(serialize(model_of(graph)))


def test_real_external_data_file_is_refused_even_when_cwd_could_resolve_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """onnxruntime resolves external data of a bytes-loaded model against the CWD."""
    model = model_of(graph_with([inline_tensor()]))
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path, save_as_external_data=True, location="w.bin", size_threshold=0)
    assert (tmp_path / "w.bin").exists()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeLoadError, match="external data"):
        OnnxRuntime(path.read_bytes(), IN, OUT)


def test_all_dummy_models_pass_the_scan(artifacts_root: Path) -> None:
    for onnx_file in artifacts_root.glob("*/*/model.onnx"):
        assert has_external_data(onnx_file.read_bytes()) is False


@pytest.mark.parametrize(
    ("input_spec", "output_spec", "message"),
    [
        (TensorSpec(name="other", shape=(None, 4)), OUT, "input is named"),
        (IN, TensorSpec(name="other", shape=(None, 3)), "output is named"),
        (TensorSpec(name="x", shape=(None, 5)), OUT, "shape"),
        (IN, TensorSpec(name="y", shape=(None, 9)), "shape"),
        (TensorSpec(name="x", shape=(None, None)), OUT, "dynamic"),
    ],
)
def test_graph_must_match_the_manifest_tensors(
    input_spec: TensorSpec, output_spec: TensorSpec, message: str
) -> None:
    data = serialize(model_of(graph_with([inline_tensor()])))
    with pytest.raises(RuntimeLoadError, match=message):
        OnnxRuntime(data, input_spec, output_spec)


def test_non_float_graphs_and_unparseable_graphs_are_rejected() -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.INT64, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, ["batch", 4])],
    )
    with pytest.raises(RuntimeLoadError, match=r"tensor\(float\)"):
        OnnxRuntime(serialize(model_of(graph)), IN, TensorSpec(name="y", shape=(None, 4)))
    bad_op = helper.make_graph(
        [helper.make_node("NoSuchOperator", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
    )
    with pytest.raises(RuntimeLoadError, match="rejected"):
        OnnxRuntime(serialize(model_of(bad_op)), IN, OUT)


def test_session_is_configured_for_a_single_cpu_thread(artifacts_root: Path) -> None:
    onnx_bytes = (
        artifacts_root / "credit-default-mlp" / dummy.DUMMY_VERSION / "model.onnx"
    ).read_bytes()
    runtime = OnnxRuntime(
        onnx_bytes,
        TensorSpec(name="inputs", shape=(None, 25)),
        TensorSpec(name="output_0", shape=(None, 2)),
    )
    options = runtime._session.get_session_options()
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (1, 1)
    assert str(options.execution_mode) == "ExecutionMode.ORT_SEQUENTIAL"
    assert runtime._session.get_providers() == ["CPUExecutionProvider"]
