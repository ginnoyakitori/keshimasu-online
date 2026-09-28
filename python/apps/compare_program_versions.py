# python/apps/compare_program_versions.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERSION_ROOT = ROOT / "data" / "program_versions"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def files_by_path(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        file["path"]: file
        for file in manifest.get("files", [])
        if isinstance(file, dict) and "path" in file
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two saved generator/solver program versions."
    )

    parser.add_argument(
        "a",
        help="Path to first snapshot directory or manifest.json.",
    )

    parser.add_argument(
        "b",
        help="Path to second snapshot directory or manifest.json.",
    )

    return parser.parse_args()


def manifest_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = (ROOT / path).resolve()

    if path.is_dir():
        path = path / "manifest.json"

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def main() -> None:
    args = parse_args()

    a_path = manifest_path(args.a)
    b_path = manifest_path(args.b)

    a = read_json(a_path)
    b = read_json(b_path)

    a_files = files_by_path(a)
    b_files = files_by_path(b)

    a_set = set(a_files.keys())
    b_set = set(b_files.keys())

    added = sorted(b_set - a_set)
    removed = sorted(a_set - b_set)

    common = sorted(a_set & b_set)

    changed: List[str] = []
    unchanged: List[str] = []

    for path in common:
        if a_files[path].get("sha256") != b_files[path].get("sha256"):
            changed.append(path)
        else:
            unchanged.append(path)

    print("=== COMPARE PROGRAM VERSIONS ===")
    print("A:", a.get("snapshotName"), a_path)
    print("B:", b.get("snapshotName"), b_path)
    print()
    print("Kind A:", a.get("kind"))
    print("Kind B:", b.get("kind"))
    print("Generation A:", a.get("generation"))
    print("Generation B:", b.get("generation"))
    print()
    print("Changed :", len(changed))
    print("Added   :", len(added))
    print("Removed :", len(removed))
    print("Same    :", len(unchanged))

    if changed:
        print()
        print("Changed files:")
        for path in changed:
            print("  -", path)

    if added:
        print()
        print("Added files:")
        for path in added:
            print("  +", path)

    if removed:
        print()
        print("Removed files:")
        for path in removed:
            print("  -", path)


if __name__ == "__main__":
    main()