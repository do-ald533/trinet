"""Artifact discovery and loading: manifest validation, SHA-256, fallback ladder, warm-up.

``artifacts/<name>/<version>/manifest.json`` -> one ``ModelEntry`` per model name. Only the
highest semver version of each name is considered (a prerelease is lower than the release,
so ``1.0.0`` wins over ``0.0.0-dummy``). If that version cannot be served, the model is
registered as unavailable instead of silently falling back to an older version.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from trinet_api.application.postprocessing import classification_result
from trinet_api.domain.entities import ModelBundle, ModelEntry, ModelRegistry, UnavailableModel
from trinet_api.domain.ports import ModelRuntime
from trinet_api.infrastructure.runtimes import RuntimeLoadError, static_dims
from trinet_api.infrastructure.runtimes.litert import LiteRtRuntime
from trinet_api.infrastructure.runtimes.numpy_mlp import NumpyMlpRuntime
from trinet_api.infrastructure.runtimes.onnx import OnnxRuntime
from trinet_contract import (
    ClassificationOutput,
    LiteRtRuntimeSpec,
    Manifest,
    NumpyMlpRuntimeSpec,
    OnnxRuntimeSpec,
    load_manifest,
)
from trinet_contract.manifest import RuntimeSpec

logger = logging.getLogger(__name__)

MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class ArtifactIntegrityError(Exception):
    """A referenced file is missing, escapes the artifact directory or fails its SHA-256."""


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    intra_op_threads: int = 1


# --------------------------------------------------------------------------- versions

type SemverKey = tuple[int, int, int, tuple[object, ...]]


def semver_key(text: str) -> SemverKey | None:
    """Sort key following semver 2.0 precedence, or ``None`` if ``text`` is not a version."""
    match = _SEMVER.match(text)
    if match is None:
        return None
    major, minor, patch, pre = match.groups()
    if pre is None:
        return int(major), int(minor), int(patch), (1,)  # a release outranks any prerelease
    identifiers = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return int(major), int(minor), int(patch), (0, identifiers)


def latest_versions(artifacts_dir: Path) -> dict[str, Path]:
    """Model name -> directory of its highest semver version (symlinks are ignored)."""
    latest: dict[str, Path] = {}
    if not artifacts_dir.is_dir():
        return latest
    for model_dir in sorted(artifacts_dir.iterdir()):
        if model_dir.is_symlink() or not model_dir.is_dir() or not _SLUG.match(model_dir.name):
            continue
        candidates = [
            (key, child)
            for child in model_dir.iterdir()
            if child.is_dir() and not child.is_symlink() and (key := semver_key(child.name))
        ]
        if candidates:
            latest[model_dir.name] = max(candidates, key=lambda item: item[0])[1]
    return latest


# --------------------------------------------------------------------------- files


def read_verified(directory: Path, file_name: str, sha256: str) -> bytes:
    """Read ``directory/file_name`` fully into memory and check its digest (no TOCTOU)."""
    root = directory.resolve()
    path = (directory / file_name).resolve()
    if path.parent != root or not path.is_file():
        raise ArtifactIntegrityError(f"{file_name}: missing or outside the artifact directory")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ArtifactIntegrityError(f"{file_name}: larger than {MAX_ARTIFACT_BYTES} bytes")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ArtifactIntegrityError(f"{file_name}: SHA-256 mismatch")
    return data


# --------------------------------------------------------------------------- runtimes


def _build_runtime(
    spec: RuntimeSpec, manifest: Manifest, directory: Path, options: RuntimeOptions
) -> ModelRuntime:
    data = read_verified(directory, spec.file, spec.sha256)
    input_spec, output_spec = manifest.inputs[0], manifest.outputs[0]
    match spec:
        case OnnxRuntimeSpec():
            return OnnxRuntime(
                data, input_spec, output_spec, intra_op_threads=options.intra_op_threads
            )
        case NumpyMlpRuntimeSpec(layers=layers):
            return NumpyMlpRuntime(data, layers, input_spec, output_spec)
        case LiteRtRuntimeSpec():
            return LiteRtRuntime(
                data, input_spec, output_spec, num_threads=options.intra_op_threads
            )
    raise RuntimeLoadError(f"unknown runtime kind {spec.kind!r}")  # pragma: no cover


def warm_up(runtime: ModelRuntime, manifest: Manifest) -> None:
    """Run a zero batch and check the answer is shaped like the manifest says."""
    trailing, width = static_dims(manifest.inputs[0], manifest.outputs[0])
    out = runtime.predict(np.zeros((1, *trailing), dtype=np.float32))
    if out.shape != (1, width) or not np.all(np.isfinite(out)):
        raise RuntimeLoadError(f"warm-up produced shape {out.shape} or non-finite values")
    if isinstance(manifest.output, ClassificationOutput):
        classification_result(out[0], manifest.output.labels)  # must be a probability vector


def _manifest_checks(manifest: Manifest, name: str, version: str, directory: Path) -> None:
    if manifest.name != name or manifest.version != version:
        raise ValueError("manifest name/version do not match the directory")
    if manifest.golden is not None:
        read_verified(directory, manifest.golden.file, manifest.golden.sha256)


def load_model(version_dir: Path, options: RuntimeOptions | None = None) -> ModelEntry:
    """Load one ``artifacts/<name>/<version>`` directory (never raises for bad artifacts)."""
    options = options or RuntimeOptions()
    name, version = version_dir.parent.name, version_dir.name
    try:
        manifest_path = version_dir / "manifest.json"
        if manifest_path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("manifest too large")
        manifest = load_manifest(manifest_path)
        _manifest_checks(manifest, name, version, version_dir)
    except ArtifactIntegrityError as exc:
        logger.error("model %s %s: integrity check failed: %s", name, version, exc)
        return UnavailableModel(name, version, "integrity_check_failed")
    except (OSError, ValueError, ValidationError) as exc:
        logger.error("model %s %s: invalid manifest: %s", name, version, type(exc).__name__)
        return UnavailableModel(name, version, "invalid_manifest")

    integrity_only = True
    for spec in manifest.runtimes:
        try:
            runtime = _build_runtime(spec, manifest, version_dir, options)
            warm_up(runtime, manifest)
        except ArtifactIntegrityError as exc:
            logger.error("model %s %s: runtime %s: %s", name, version, spec.kind, exc)
        except Exception as exc:  # any failure of this rung moves to the next one
            integrity_only = False
            logger.warning(
                "model %s %s: runtime %s unusable (%s: %s)",
                name,
                version,
                spec.kind,
                type(exc).__name__,
                exc,
            )
        else:
            logger.info("model %s %s: serving with runtime %s", name, version, spec.kind)
            return ModelBundle(manifest, runtime)
    reason = "integrity_check_failed" if integrity_only else "no_usable_runtime"
    logger.error("model %s %s: unavailable (%s)", name, version, reason)
    return UnavailableModel(name, version, reason)


def load_registry(artifacts_dir: Path, options: RuntimeOptions | None = None) -> ModelRegistry:
    """Scan ``artifacts_dir`` and load the latest version of every model found."""
    if not artifacts_dir.is_dir():
        logger.error("artifacts directory %s does not exist", artifacts_dir)
    entries = {
        name: load_model(version_dir, options)
        for name, version_dir in latest_versions(artifacts_dir).items()
    }
    return ModelRegistry(entries)
