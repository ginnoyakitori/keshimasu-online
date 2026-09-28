# python/apps/clean_duplicate_word_replays.py

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from keshimasu_py.config import PROJECT_ROOT


DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "cleanup"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_solved_moves(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    既存の複数形式に対応して solvedMoves を取得する。
    """
    solver = replay.get("solver", {})

    moves = solver.get("solvedMoves")
    if moves:
        return moves

    moves = replay.get("solverSolutionMoves")
    if moves:
        return moves

    stats = replay.get("stats", {})
    moves = stats.get("solvedMoves")
    if moves:
        return moves

    return []


def find_duplicate_words_in_moves(
    moves: List[Dict[str, Any]],
) -> Tuple[bool, List[str], Dict[str, int]]:
    words: List[str] = []

    for move in moves:
        word = move.get("word")

        if isinstance(word, str) and word:
            words.append(word)

    counter = Counter(words)

    duplicates = [
        word for word, count in counter.items()
        if count >= 2
    ]

    duplicate_counts = {
        word: int(counter[word])
        for word in duplicates
    }

    return len(duplicates) > 0, duplicates, duplicate_counts


def replay_has_duplicate_words(path: Path) -> Dict[str, Any]:
    replay = load_json(path)
    moves = get_solved_moves(replay)

    has_duplicate, duplicate_words, duplicate_counts = find_duplicate_words_in_moves(
        moves
    )

    return {
        "file": str(path),
        "hasDuplicate": has_duplicate,
        "duplicateWords": duplicate_words,
        "duplicateCounts": duplicate_counts,
        "moves": len(moves),
        "id": replay.get("id", path.stem),
        "targetWildcards": replay.get("targetWildcards"),
    }


def collect_json_files(root: Path) -> List[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        return [root]

    if not root.exists():
        return []

    return sorted(root.rglob("*.json"))


def ensure_report_dir() -> Path:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_REPORT_DIR


def save_reports(rows: List[Dict[str, Any]]) -> Dict[str, Path]:
    report_dir = ensure_report_dir()

    json_path = report_dir / "duplicate_word_replays_report.json"
    csv_path = report_dir / "duplicate_word_replays_report.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "file",
        "hasDuplicate",
        "duplicateWords",
        "duplicateCounts",
        "moves",
        "id",
        "targetWildcards",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out = dict(row)
            out["duplicateWords"] = json.dumps(
                row["duplicateWords"],
                ensure_ascii=False,
            )
            out["duplicateCounts"] = json.dumps(
                row["duplicateCounts"],
                ensure_ascii=False,
            )
            writer.writerow(out)

    return {
        "json": json_path,
        "csv": csv_path,
    }


def archive_file(
    file: Path,
    *,
    archive_root: Path,
) -> Path:
    """
    PROJECT_ROOT からの相対パスを保って archive に移動する。
    """
    try:
        rel = file.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        rel = Path(file.name)

    dest = archive_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(file), str(dest))

    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find/archive/delete replay files that use the same word more than once."
    )

    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "Root directory or json file to scan. "
            "Can be specified multiple times. "
            "Default: data/training/selfplay"
        ),
    )

    parser.add_argument(
        "--archive",
        default=None,
        help=(
            "Archive duplicate files into this directory instead of deleting. "
            "Example: _archive/duplicate_word_replays"
        ),
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete duplicate files. Use carefully.",
    )

    parser.add_argument(
        "--oves",
        action="store_true",
        help="Also report files with no solvedMoves. They are not treated as duplicate by default.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for --delete. Prevents accidental deletion.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.root:
        roots = [
            Path(r).resolve()
            for r in args.root
        ]
    else:
        roots = [
            (PROJECT_ROOT / "data" / "training" / "selfplay").resolve()
        ]

    if args.delete and args.archive:
        raise ValueError("Use either --delete or --archive, not both.")

    if args.delete and not args.yes:
        raise ValueError("Delete mode requires --yes.")

    archive_root = None

    if args.archive:
        archive_root = Path(args.archive).resolve()

    all_files: List[Path] = []

    for root in roots:
        files = collect_json_files(root)
        all_files.extend(files)

    all_files = sorted(set(all_files))

    print("=== CLEAN DUPLICATE WORD REPLAYS ===")
    print("Project root:", PROJECT_ROOT)
    print("Scan roots:")

    for root in roots:
        print(" ", root)

    print("JSON files:", len(all_files))

    rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []
    no_moves_rows: List[Dict[str, Any]] = []

    for idx, file in enumerate(all_files, start=1):
        try:
            row = replay_has_duplicate_words(file)
        except Exception as e:
            row = {
                "file": str(file),
                "hasDuplicate": False,
                "duplicateWords": [],
                "duplicateCounts": {},
                "moves": 0,
                "id": file.stem,
                "targetWildcards": None,
                "error": str(e),
            }

        rows.append(row)

        if row["moves"] == 0:
            no_moves_rows.append(row)

        if row["hasDuplicate"]:
            duplicate_rows.append(row)

            print(
                f"[DUP {len(duplicate_rows)}] "
                f"{file} "
                f"duplicates={row['duplicateCounts']}"
            )

    reports = save_reports(rows)

    print("\n=== SUMMARY ===")
    print("Scanned files:", len(all_files))
    print("Duplicate files:", len(duplicate_rows))
    print("No solvedMoves files:", len(no_moves_rows))
    print("Report JSON:", reports["json"])
    print("Report CSV :", reports["csv"])

    if not duplicate_rows:
        print("\nNo duplicate-word replay files found.")
        return

    if archive_root is None and not args.delete:
        print("\nDRY RUN only. No files were moved or deleted.")
        print("To archive:")
        print(
            "  python python/apps/clean_duplicate_word_replays.py "
            "--archive _archive/duplicate_word_replays"
        )
        print("To delete:")
        print(
            "  python python/apps/clean_duplicate_word_replays.py "
            "--delete --yes"
        )
        return

    if archive_root is not None:
        print("\n=== ARCHIVING ===")
        print("Archive root:", archive_root)

        for row in duplicate_rows:
            file = Path(row["file"])

            if not file.exists():
                print("SKIP missing:", file)
                continue

            dest = archive_file(
                file,
                archive_root=archive_root,
            )

            print("MOVED:", file, "->", dest)

        print("\nArchive completed.")
        return

    if args.delete:
        print("\n=== DELETING ===")

        for row in duplicate_rows:
            file = Path(row["file"])

            if not file.exists():
                print("SKIP missing:", file)
                continue

            file.unlink()
            print("DELETED:", file)

        print("\nDelete completed.")


if __name__ == "__main__":
    main()