"""Validate the model artifacts committed under ``artifacts/`` (CI gate and release gate).

Layout checked: ``artifacts/<name>/<version>/manifest.json`` plus the files the manifest references
(runtime models, golden samples). Only the standard library and the frozen ``trinet-contract``
package are used, so the check runs with ``uv run --package trinet-contract`` (no API deps).

Checks, for every ``<name>/<version>`` directory:

* the manifest validates against the contract (``trinet_contract.load_manifest``) and its
  ``name`` / ``version`` match the directory names;
* every runtime file and the golden file exist, are regular files (symlinks are refused) and match
  the SHA-256 recorded in the manifest;
* the golden file validates (``trinet_contract.load_golden``), refers to the same model/version and
  is dimensionally consistent with the manifest (feature width, output width, PNG files present);
* no unsafe serialisation formats are present (``.keras``, ``.h5``, pickle, ...): the API never
  deserialises them (PLANO §3) and they must never enter git;
* size guard per file and for the whole tree (limits below).

``--release`` additionally applies the release gate (PLANO §8.2, item 7) to the *latest* version of
each model: it must not be a ``dummy``, its version must be >= 1.0.0 (no pre-release) and its export
quality gate must have passed. Missing golden samples / image serving skew are reported as warnings.

Size limits (override with ``--max-file-mib`` / ``--max-total-mib``): GitHub rejects blobs over
100 MiB and warns above 50 MiB, and every re-export adds a new blob to the history. The largest
legitimate file is the CNN (~1.9 M float32 parameters = ~8 MiB as ONNX, the same again as an
optional LiteRT file), so 20 MiB per file leaves 2.5x headroom while still catching an accidental
optimizer-state or fp64 export. 64 MiB for the whole tree keeps the image layer and the clone small
for a repository that holds three models in a handful of versions.

Exit status: 0 = all good, 1 = validation errors (including a missing or empty artifacts directory),
2 = command-line usage error (argparse).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from trinet_contract import (
    ImageGolden,
    ImageInput,
    Manifest,
    Status,
    TabularGolden,
    TabularInput,
    load_golden,
    load_manifest,
    sha256_file,
)

MIB = 1024 * 1024
DEFAULT_MAX_FILE_MIB = 20
DEFAULT_MAX_TOTAL_MIB = 64
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"

# Formats that execute code on load or that the API never reads (PLANO §3).
FORBIDDEN_SUFFIXES = frozenset(
    {".keras", ".h5", ".hdf5", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".pb"}
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MANIFEST_NAME = "manifest.json"
EXTRA_ALLOWED = frozenset({MANIFEST_NAME, "results.json"})

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
RELEASE_MIN_KEY = (1, 0, 0, 1, "")

SemverKey = tuple[int, int, int, int, str]


@dataclass
class Report:
    """Collected findings; printed once at the end."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")

    def info(self, message: str) -> None:
        self.infos.append(message)


def semver_key(version: str) -> SemverKey:
    """Sort key: a pre-release (``1.0.0-rc1``) orders before its release (``1.0.0``)."""
    match = _SEMVER.match(version)
    if match is None:
        return (-1, -1, -1, -1, version)
    major, minor, patch, pre = match.groups()
    return (int(major), int(minor), int(patch), 0 if pre else 1, pre or "")


# --------------------------------------------------------------------------- file level


def check_file(report: Report, where: str, path: Path, expected_sha256: str | None = None) -> bool:
    """File must exist, be a regular non-symlink file and (optionally) match its SHA-256."""
    if path.is_symlink():
        report.error(where, f"{path.name} is a symlink (refused)")
        return False
    if not path.is_file():
        report.error(where, f"referenced file {path.name} is missing")
        return False
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            report.error(
                where,
                f"{path.name} SHA-256 mismatch (manifest {expected_sha256[:12]}..., "
                f"file {actual[:12]}...)",
            )
            return False
    return True


def check_sizes(report: Report, root: Path, max_file: int, max_total: int) -> None:
    total = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_symlink():
            report.error(str(rel), "symlinks are not allowed inside artifacts/")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > max_file:
            report.error(
                str(rel),
                f"{size / MIB:.1f} MiB exceeds the per-file limit of {max_file / MIB:.0f} MiB",
            )
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            report.error(str(rel), f"forbidden format {path.suffix!r} (never committed or served)")
    report.info(f"artifacts tree: {total / MIB:.2f} MiB (limit {max_total / MIB:.0f} MiB)")
    if total > max_total:
        report.error(
            str(root), f"{total / MIB:.1f} MiB exceeds the total limit of {max_total / MIB:.0f} MiB"
        )


# --------------------------------------------------------------------------- golden


def check_tabular_golden(
    report: Report, where: str, manifest: Manifest, golden: TabularGolden
) -> None:
    if not isinstance(manifest.input, TabularInput):
        report.error(where, "tabular golden file for a non-tabular model")
        return
    width = manifest.input.width
    out_width = manifest.outputs[0].shape[-1]
    field_names = {f.name for f in manifest.input.fields}
    for index, case in enumerate(golden.cases):
        if len(case.features) != width:
            report.error(
                where, f"golden case {index}: {len(case.features)} features, expected {width}"
            )
        if out_width is not None and len(case.output) != out_width:
            report.error(
                where, f"golden case {index}: {len(case.output)} outputs, expected {out_width}"
            )
        unknown = set(case.record) - field_names
        if unknown:
            report.error(where, f"golden case {index}: unknown record fields {sorted(unknown)}")
    report.info(f"{where}: {len(golden.cases)} tabular golden cases")


def check_image_golden(
    report: Report, where: str, version_dir: Path, manifest: Manifest, golden: ImageGolden
) -> list[Path]:
    """Validate an image golden file; returns the PNG paths it references."""
    if not isinstance(manifest.input, ImageInput):
        report.error(where, "image golden file for a non-image model")
        return []
    out_width = manifest.outputs[0].shape[-1]
    pngs: list[Path] = []
    for index, case in enumerate(golden.cases):
        if out_width is not None and len(case.output) != out_width:
            report.error(
                where, f"golden case {index}: {len(case.output)} outputs, expected {out_width}"
            )
        png = version_dir / golden.directory / case.file
        pngs.append(png)
        if not check_file(report, where, png):
            continue
        with png.open("rb") as fh:
            if fh.read(len(PNG_MAGIC)) != PNG_MAGIC:
                report.error(where, f"golden image {case.file} is not a PNG")
    report.info(f"{where}: {len(golden.cases)} image golden cases")
    return pngs


def check_golden(report: Report, where: str, version_dir: Path, manifest: Manifest) -> list[Path]:
    """Validate the golden file referenced by the manifest; returns every file it pulls in."""
    if manifest.golden is None:
        return []
    golden_path = version_dir / manifest.golden.file
    if not check_file(report, where, golden_path, manifest.golden.sha256):
        return []
    try:
        golden = load_golden(golden_path)
    except ValueError as exc:
        report.error(where, f"invalid golden file {golden_path.name}: {_first_line(exc)}")
        return [golden_path]
    if golden.model != manifest.name or golden.version != manifest.version:
        report.error(
            where,
            f"golden file is for {golden.model}@{golden.version}, "
            f"manifest is {manifest.name}@{manifest.version}",
        )
    if isinstance(golden, TabularGolden):
        check_tabular_golden(report, where, manifest, golden)
        return [golden_path]
    return [golden_path, *check_image_golden(report, where, version_dir, manifest, golden)]


# --------------------------------------------------------------------------- version level


def check_version(report: Report, model_dir: Path, version_dir: Path) -> Manifest | None:
    where = f"{model_dir.name}/{version_dir.name}"
    manifest_path = version_dir / MANIFEST_NAME
    if not check_file(report, where, manifest_path):
        return None
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        report.error(where, f"invalid {MANIFEST_NAME}: {_first_line(exc)}")
        return None
    if manifest.name != model_dir.name or manifest.version != version_dir.name:
        report.error(
            where, f"manifest says {manifest.name}@{manifest.version}, directory is {where}"
        )
    referenced = {manifest_path}
    for runtime in manifest.runtimes:
        path = version_dir / runtime.file
        check_file(report, where, path, runtime.sha256)
        referenced.add(path)
    referenced.update(check_golden(report, where, version_dir, manifest))
    for path in sorted(version_dir.rglob("*")):
        if path.is_file() and path not in referenced and path.name not in EXTRA_ALLOWED:
            report.warn(where, f"unreferenced file {path.relative_to(version_dir)}")
    report.info(
        f"{where}: status={manifest.status.value}, runtimes={[r.kind for r in manifest.runtimes]}"
    )
    return manifest


# --------------------------------------------------------------------------- release gate


def check_release(report: Report, manifests: dict[str, dict[str, Manifest]]) -> None:
    """Release gate: the latest version of every model must be a real, passing export."""
    for name, versions in sorted(manifests.items()):
        latest_version = max(versions, key=semver_key)
        manifest = versions[latest_version]
        where = f"{name}/{latest_version}"
        if manifest.status is Status.DUMMY:
            report.error(where, "latest version is a dummy artifact - refusing release")
        if semver_key(latest_version) < RELEASE_MIN_KEY:
            report.error(where, "latest version must be >= 1.0.0 (no pre-release) for a release")
        if not manifest.quality_gate.passed:
            report.error(where, "export quality gate did not pass")
        if manifest.golden is None:
            report.warn(where, "no golden samples (training/serving skew is untested)")
        if isinstance(manifest.input, ImageInput) and manifest.serving_skew is None:
            report.warn(where, "image model without measured serving_skew")
        if manifest.status is Status.CANDIDATE:
            report.warn(where, "status is 'candidate', not 'release'")


# --------------------------------------------------------------------------- driver


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__


def discover(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pairs.extend((model_dir, v) for v in sorted(p for p in model_dir.iterdir() if p.is_dir()))
    return pairs


def run(root: Path, *, release: bool, min_models: int, max_file: int, max_total: int) -> Report:
    report = Report()
    if not root.is_dir():
        report.error(str(root), "artifacts directory does not exist (run `make dummy`)")
        return report
    manifests: dict[str, dict[str, Manifest]] = {}
    for model_dir, version_dir in discover(root):
        manifest = check_version(report, model_dir, version_dir)
        if manifest is not None:
            manifests.setdefault(manifest.name, {})[manifest.version] = manifest
    if len(manifests) < min_models:
        report.error(
            str(root), f"found {len(manifests)} valid model(s), need at least {min_models}"
        )
    check_sizes(report, root, max_file, max_total)
    if release:
        check_release(report, manifests)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="artifacts directory")
    parser.add_argument("--release", action="store_true", help="also apply the release gate")
    parser.add_argument("--min-models", type=int, default=1, help="minimum number of models")
    parser.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    parser.add_argument("--max-total-mib", type=float, default=DEFAULT_MAX_TOTAL_MIB)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.root,
        release=args.release,
        min_models=args.min_models,
        max_file=int(args.max_file_mib * MIB),
        max_total=int(args.max_total_mib * MIB),
    )
    for line in report.infos:
        print(f"info:    {line}")
    for line in report.warnings:
        print(f"warning: {line}")
    for line in report.errors:
        print(f"error:   {line}", file=sys.stderr)
    if report.errors:
        mode = "release gate" if args.release else "artifact check"
        print(f"\n{mode} FAILED: {len(report.errors)} error(s)", file=sys.stderr)
        return 1
    print("\nartifact check passed" + (" (release gate included)" if args.release else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
