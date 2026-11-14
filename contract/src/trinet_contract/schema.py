"""Generate the JSON Schemas of the contract (CI fails on drift: ``git diff --exit-code``).

Usage: ``uv run python -m trinet_contract.schema [output_dir]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trinet_contract.golden import golden_json_schema
from trinet_contract.manifest import Manifest

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "schema"


def write_schemas(out_dir: Path = DEFAULT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = {
        "manifest.schema.json": Manifest.model_json_schema(),
        "golden.schema.json": golden_json_schema(),
    }
    written = []
    for name, schema in schemas.items():
        path = out_dir / name
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    for p in write_schemas(target):
        print(p)
