# python/apps/snapshot_program_version.py

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_ROOT = ROOT / "data" / "program_versions"


GENERATOR_FILES = [
    "python/src/keshimasu_py/ga_reverse.py",
    "python/apps/build_numpy_dataset.py",
    "python/apps/train_policy.py",
    "python/apps/export_policy_onnx.py",
]

SOLVER_FILES = [
    "python/src/keshimasu_py/core_solver.py",
    "python/src/keshimasu_py/torch_policy.py",
    "python/src/keshimasu_py/policy_inference.py",
    "python/apps/solver_policy.py",
    "src/solver/solver.js",
    "src/solver/solver-policy.js",
    "src/solver/word-search.js",
    "src/solver/gravity.js",
    "src/neural/onnx-policy.js",
    "src/neural/inference.js",
]

MODEL_FILES = [
    "data/models/torch_policy/policy_model.pt",
    "data/models/torch_policy/policy_model.onnx",
    "data/models/torch_policy/metadata.json",
    "data/models/torch_policy/onnx_metadata.json",
]

BENCHMARK_FILES = [
    "data/cleanup/benchmark_onnx_policy_result.json",
    "data/models/torch_policy/benchmark_f3_50.json",
    "data/models/torch_policy/benchmark_f2_50.json",
]


def now_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def run_git_command(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_git_info() -> Dict[str, Any]:
    return {
        "commit": run_git_command(["rev-parse", "HEAD"]),
        "branch": run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "statusShort": run_git_command(["status", "--short"]),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file_preserve_relative(
    *,
    rel_path: str,
    dest_root: Path,
) -> Optional[Dict[str, Any]]:
    src = ROOT / rel_path

    if not src.exists():
        return None

    dest = dest_root / rel_path
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)

    return {
        "path": rel_path.replace("\\", "/"),
        "size": src.stat().st_size,
        "sha256": sha256_file(src),
    }


def next_generation_number(kind_dir: Path) -> int:
    if not kind_dir.exists():
        return 1

    nums: List[int] = []

    for child in kind_dir.iterdir():
        if not child.is_dir():
            continue

        name = child.name

        if not name.startswith("gen-"):
            continue

        parts = name.split("-")

        if len(parts) < 2:
            continue

        try:
            nums.append(int(parts[1]))
        except Exception:
            pass

    if not nums:
        return 1

    return max(nums) + 1


def sanitize_name(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ["-", "_"] else "-"
        for ch in name.strip()
    )

    cleaned = "-".join(part for part in cleaned.split("-") if part)

    return cleaned or "snapshot"


def build_file_list(
    *,
    kind: str,
    include_model: bool,
    include_benchmarks: bool,
    extra_files: List[str],
) -> List[str]:
    files: List[str] = []

    if kind in ["generator", "both"]:
        files.extend(GENERATOR_FILES)

    if kind in ["solver", "both"]:
        files.extend(SOLVER_FILES)

    if include_model:
        files.extend(MODEL_FILES)

    if include_benchmarks:
        files.extend(BENCHMARK_FILES)

    files.extend(extra_files)

    # 重複除去、順序維持
    seen = set()
    out: List[str] = []

    for file in files:
        normalized = file.replace("\\", "/")

        if normalized in seen:
            continue

        seen.add(normalized)
        out.append(normalized)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot generator/solver program versions."
    )

    parser.add_argument(
        "--kind",
        choices=["generator", "solver", "both"],
        required=True,
        help="Which program group to snapshot.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Human-readable version name.",
    )

    parser.add_argument(
        "--note",
        default="",
        help="Free text note.",
    )

    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    parser.add_argument(
        "--include-model",
        action="store_true",
        help="Copy current policy model files.",
    )

    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help="Copy benchmark JSON files if present.",
    )

    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Additional relative file path to include. Can be specified multiple times.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kind = args.kind
    name = sanitize_name(args.name)
    output_root = Path(args.output_root).resolve()

    kind_dir = output_root / kind
    generation_no = next_generation_number(kind_dir)

    snapshot_name = f"gen-{generation_no:06d}-{name}"
    snapshot_dir = kind_dir / snapshot_name
    files_dir = snapshot_dir / "files"

    ensure_dir(files_dir)

    file_list = build_file_list(
        kind=kind,
        include_model=bool(args.include_model),
        include_benchmarks=bool(args.include_benchmarks),
        extra_files=list(args.extra_file or []),
    )

    copied_files: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    for rel_path in file_list:
        info = copy_file_preserve_relative(
            rel_path=rel_path,
            dest_root=files_dir,
        )

        if info is None:
            missing_files.append(rel_path)
        else:
            copied_files.append(info)

    manifest = {
        "format": "keshimasu-program-version-v1",
        "kind": kind,
        "generation": generation_no,
        "name": name,
        "snapshotName": snapshot_name,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
        "root": str(ROOT),
        "git": collect_git_info(),
        "files": copied_files,
        "missingFiles": missing_files,
        "options": {
            "includeModel": bool(args.include_model),
            "includeBenchmarks": bool(args.include_benchmarks),
            "extraFiles": list(args.extra_file or []),
        },
    }

    manifest_path = snapshot_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    latest_path = kind_dir / "LATEST.txt"
    latest_path.write_text(snapshot_name, encoding="utf-8")

    print("=== SNAPSHOT CREATED ===")
    print("Kind       :", kind)
    print("Generation :", generation_no)
    print("Name       :", snapshot_name)
    print("Directory  :", snapshot_dir)
    print("Files      :", len(copied_files))
    print("Missing    :", len(missing_files))
    print("Manifest   :", manifest_path)

    if missing_files:
        print()
        print("Missing files:")
        for file in missing_files:
            print("  -", file)


if __name__ == "__main__":
    main()
