from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from trinet_api.domain.entities import ModelBundle, UnavailableModel
from trinet_api.infrastructure.loading import (
    ArtifactIntegrityError,
    RuntimeOptions,
    latest_versions,
    load_model,
    load_registry,
    read_verified,
    semver_key,
)
from trinet_contract import Status

DRYBEAN = "drybean-mlp"
INTEL = "intel-cnn"


# --------------------------------------------------------------------------- helpers


def model_dir(root: Path, name: str, version: str = dummy.DUMMY_VERSION) -> Path:
    return root / name / version


def edit_manifest(directory: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def replace_runtime_file(directory: Path, kind: str, payload: bytes) -> None:
    """Overwrite a runtime file *and* its manifest digest (so only the content is wrong)."""

    def mutate(data: dict[str, Any]) -> None:
        for runtime in data["runtimes"]:
            if runtime["kind"] == kind:
                (directory / runtime["file"]).write_bytes(payload)
                runtime["sha256"] = hashlib.sha256(payload).hexdigest()

    edit_manifest(directory, mutate)


def entry_of(root: Path, name: str) -> Any:
    return load_model(model_dir(root, name))


# --------------------------------------------------------------------------- versions


def test_semver_precedence_follows_the_spec() -> None:
    ordered = [
        "0.0.0-dummy", "0.0.0", "0.1.0-alpha", "0.1.0-alpha.1", "0.1.0-alpha.beta",
        "0.1.0-beta", "0.1.0-beta.2", "0.1.0-beta.11", "0.1.0-rc.1", "0.1.0",
        "1.0.0-rc.1", "1.0.0", "1.0.1", "1.2.0", "1.10.0", "2.0.0",
    ]  # fmt: skip
    keys = [semver_key(v) for v in ordered]
    assert None not in keys
    assert keys == sorted(keys)  # type: ignore[type-var]
    assert len(set(keys)) == len(keys)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["1.0", "v1.0.0", "1.0.0+build", "latest", "", "1.0.x", "01..0"])
def test_non_semver_names_are_not_versions(text: str) -> None:
    assert semver_key(text) is None


def test_release_beats_dummy_and_highest_version_wins(tmp_path: Path) -> None:
    dummy.build_all(tmp_path, models=[DRYBEAN])
    dummy.build_drybean(tmp_path, "1.0.0", Status.CANDIDATE)
    dummy.build_drybean(tmp_path, "1.0.0-rc.1", Status.CANDIDATE)
    dummy.build_drybean(tmp_path, "0.9.0", Status.CANDIDATE)
    assert latest_versions(tmp_path) == {DRYBEAN: tmp_path / DRYBEAN / "1.0.0"}
    entry = load_registry(tmp_path).get(DRYBEAN)
    assert entry.version == "1.0.0"
    assert entry.manifest.status is Status.CANDIDATE


def test_dummy_is_used_when_it_is_all_there_is(artifacts_root: Path) -> None:
    registry = load_registry(artifacts_root)
    assert [e.name for e in registry] == ["drybean-mlp", "housing-mlp", "intel-cnn"]
    assert {e.version for e in registry} == {dummy.DUMMY_VERSION}
    assert registry.ready
    assert len(registry.available) == 3


def test_stray_files_and_odd_directories_are_ignored(tmp_path: Path) -> None:
    dummy.build_all(tmp_path, models=[DRYBEAN])
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "UPPER-CASE" / "1.0.0").mkdir(parents=True)
    (tmp_path / ".hidden" / "1.0.0").mkdir(parents=True)
    (tmp_path / DRYBEAN / "latest").mkdir()
    (tmp_path / DRYBEAN / "notes.txt").write_text("x")
    (tmp_path / "empty-model").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / DRYBEAN, target_is_directory=True)
    assert set(latest_versions(tmp_path)) == {DRYBEAN}


def test_missing_artifacts_directory_gives_an_empty_not_ready_registry(tmp_path: Path) -> None:
    registry = load_registry(tmp_path / "nope")
    assert len(registry) == 0
    assert not registry.ready


# --------------------------------------------------------------------------- happy path


def test_first_runtime_of_the_ladder_is_used(artifacts_root: Path) -> None:
    bundle = entry_of(artifacts_root, DRYBEAN)
    assert isinstance(bundle, ModelBundle)
    assert bundle.runtime.kind == "onnx"
    assert bundle.name == DRYBEAN


# --------------------------------------------------------------------------- the ladder


def test_corrupt_onnx_falls_back_to_numpy(artifacts_copy: Path) -> None:
    onnx_path = model_dir(artifacts_copy, DRYBEAN) / "model.onnx"
    onnx_path.write_bytes(onnx_path.read_bytes() + b"tamper")  # digest no longer matches
    bundle = entry_of(artifacts_copy, DRYBEAN)
    assert isinstance(bundle, ModelBundle)
    assert bundle.runtime.kind == "numpy_mlp"


def test_garbage_onnx_with_a_valid_digest_also_falls_back(artifacts_copy: Path) -> None:
    replace_runtime_file(model_dir(artifacts_copy, DRYBEAN), "onnx", b"this is not an onnx model")
    bundle = entry_of(artifacts_copy, DRYBEAN)
    assert isinstance(bundle, ModelBundle)
    assert bundle.runtime.kind == "numpy_mlp"


def test_external_data_model_is_skipped_by_the_loader(artifacts_copy: Path) -> None:
    tensor = numpy_helper.from_array(np.ones((16, 7), np.float32), name="W")
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "weights.bin"
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["inputs", "W"], ["output_0"])],
        "g",
        [helper.make_tensor_value_info("inputs", TensorProto.FLOAT, ["batch", 16])],
        [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch", 7])],
        initializer=[tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    replace_runtime_file(
        model_dir(artifacts_copy, DRYBEAN), "onnx", bytes(model.SerializeToString())
    )
    bundle = entry_of(artifacts_copy, DRYBEAN)
    assert isinstance(bundle, ModelBundle)
    assert bundle.runtime.kind == "numpy_mlp"


def test_warm_up_rejects_a_model_that_is_not_a_classifier(artifacts_copy: Path) -> None:
    """An export whose last layer is not softmax must not go live (zero batch is checked)."""
    directory = model_dir(artifacts_copy, DRYBEAN)
    arrays = {
        k: np.load(directory / "weights.npz")[k] for k in np.load(directory / "weights.npz").files
    }
    layers = dummy.build_drybean  # noqa: F841 - documents where the layer list comes from
    from trinet_contract import DenseLayer, NormalizeLayer

    linear_layers = (
        NormalizeLayer(key="norm"),
        DenseLayer(key="d0", activation="relu"),
        DenseLayer(key="d1", activation="relu"),
        DenseLayer(key="out", activation="linear"),
    )
    replace_runtime_file(directory, "onnx", dummy.mlp_onnx(linear_layers, arrays, 16, 7))
    bundle = entry_of(artifacts_copy, DRYBEAN)
    assert isinstance(bundle, ModelBundle)
    assert bundle.runtime.kind == "numpy_mlp"


def test_every_runtime_corrupt_means_unavailable_with_integrity_reason(
    artifacts_copy: Path,
) -> None:
    directory = model_dir(artifacts_copy, DRYBEAN)
    for name in ("model.onnx", "weights.npz"):
        (directory / name).write_bytes(b"corrupted")
    entry = entry_of(artifacts_copy, DRYBEAN)
    assert entry == UnavailableModel(DRYBEAN, dummy.DUMMY_VERSION, "integrity_check_failed")


def test_cnn_without_a_fallback_is_unavailable_but_the_others_keep_serving(
    artifacts_copy: Path,
) -> None:
    replace_runtime_file(model_dir(artifacts_copy, INTEL), "onnx", b"nope")
    registry = load_registry(artifacts_copy)
    assert registry.entry(INTEL) == UnavailableModel(
        INTEL, dummy.DUMMY_VERSION, "no_usable_runtime"
    )
    assert registry.get(DRYBEAN).runtime.kind == "onnx"
    assert registry.get("housing-mlp").runtime.kind == "onnx"
    assert registry.ready


def test_litert_without_the_package_is_just_another_failed_rung(
    artifacts_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = model_dir(artifacts_copy, INTEL)
    payload = b"fake tflite"

    def mutate(data: dict[str, Any]) -> None:
        (directory / "model.tflite").write_bytes(payload)
        data["runtimes"] = [
            {
                "kind": "litert",
                "file": "model.tflite",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]

    edit_manifest(directory, mutate)
    monkeypatch.setitem(sys.modules, "ai_edge_litert", None)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", None)
    assert entry_of(artifacts_copy, INTEL).reason == "no_usable_runtime"


# --------------------------------------------------------------------------- manifest problems


def test_invalid_manifests_are_unavailable_not_fatal(artifacts_copy: Path) -> None:
    for name, content in (
        (DRYBEAN, "{not json"),
        ("housing-mlp", json.dumps({"schema_version": "1.0", "name": "housing-mlp"})),
    ):
        (model_dir(artifacts_copy, name) / "manifest.json").write_text(content)
    registry = load_registry(artifacts_copy)
    assert registry.entry(DRYBEAN).reason == "invalid_manifest"  # type: ignore[union-attr]
    assert registry.entry("housing-mlp").reason == "invalid_manifest"  # type: ignore[union-attr]
    assert registry.get(INTEL).name == INTEL


def test_missing_manifest_and_extra_fields(artifacts_copy: Path) -> None:
    (model_dir(artifacts_copy, DRYBEAN) / "manifest.json").unlink()
    assert entry_of(artifacts_copy, DRYBEAN).reason == "invalid_manifest"
    edit_manifest(model_dir(artifacts_copy, INTEL), lambda d: d.update(surprise=1))
    assert entry_of(artifacts_copy, INTEL).reason == "invalid_manifest"


def test_manifest_must_agree_with_its_directory(artifacts_copy: Path) -> None:
    edit_manifest(model_dir(artifacts_copy, DRYBEAN), lambda d: d.update(version="9.9.9"))
    assert entry_of(artifacts_copy, DRYBEAN).reason == "invalid_manifest"
    edit_manifest(model_dir(artifacts_copy, INTEL), lambda d: d.update(name="drybean-mlp"))
    assert entry_of(artifacts_copy, INTEL).reason == "invalid_manifest"


def test_golden_digest_is_verified_too(artifacts_copy: Path) -> None:
    golden = model_dir(artifacts_copy, DRYBEAN) / "golden.json"
    golden.write_text(golden.read_text() + "\n")
    assert entry_of(artifacts_copy, DRYBEAN).reason == "integrity_check_failed"


# --------------------------------------------------------------------------- file access


def test_read_verified_checks_digest_and_location(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"data")
    good = hashlib.sha256(b"data").hexdigest()
    assert read_verified(tmp_path, "a.bin", good) == b"data"
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        read_verified(tmp_path, "a.bin", "0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        read_verified(tmp_path, "missing.bin", good)


def test_symlinks_leaving_the_artifact_directory_are_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "secret.bin"
    outside.parent.mkdir()
    outside.write_bytes(b"data")
    inside = tmp_path / "artifact"
    inside.mkdir()
    (inside / "link.bin").symlink_to(outside)
    with pytest.raises(ArtifactIntegrityError, match="outside"):
        read_verified(inside, "link.bin", hashlib.sha256(b"data").hexdigest())
    with pytest.raises(ArtifactIntegrityError, match="outside"):
        read_verified(inside, "../outside/secret.bin", hashlib.sha256(b"data").hexdigest())


def test_oversized_artifacts_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trinet_api.infrastructure.loading.MAX_ARTIFACT_BYTES", 3)
    (tmp_path / "big.bin").write_bytes(b"12345")
    with pytest.raises(ArtifactIntegrityError, match="larger"):
        read_verified(tmp_path, "big.bin", hashlib.sha256(b"12345").hexdigest())


def test_runtime_options_reach_onnxruntime(artifacts_root: Path) -> None:
    bundle = load_model(model_dir(artifacts_root, DRYBEAN), RuntimeOptions(intra_op_threads=2))
    assert isinstance(bundle, ModelBundle)
    assert bundle.runtime._session.get_session_options().intra_op_num_threads == 2  # type: ignore[attr-defined]
