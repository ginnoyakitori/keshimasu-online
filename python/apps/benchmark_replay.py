# python/apps/benchmark_replay.py

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from keshimasu_py.config import PROJECT_ROOT, MAX_WORD_LENGTH
from keshimasu_py.codec import (
    build_codec_for_board_and_words,
    encode_board,
    encode_words,
)
from keshimasu_py.core_solver import solve_numba
from keshimasu_py.replay_io import (
    list_replay_files,
    load_replay,
    replay_board,
    replay_words,
    replay_target_f,
    replay_solver_stats,
    replay_solver_solved,
)


BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def summarize_branch_hist(hist: Dict[Any, Any]) -> Dict[str, int]:
    if not hist:
        return {
            "maxBranch": 0,
            "multiBranchCount": 0,
        }

    max_branch = 0
    multi = 0

    for k, v in hist.items():
        kk = safe_int(k)
        vv = safe_int(v)

        max_branch = max(max_branch, kk)

        if kk >= 2:
            multi += vv

    return {
        "maxBranch": max_branch,
        "multiBranchCount": multi,
    }


def solve_replay(
    replay: Dict[str, Any],
    *,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_word_mask: bool,
) -> Dict[str, Any]:
    board_raw = replay_board(replay)
    words_raw = replay_words(replay)

    if not words_raw:
        raise ValueError("Replay has no words")

    codec = build_codec_for_board_and_words(words_raw, board_raw)

    word_array, word_lengths, filtered_words = encode_words(
        words_raw,
        codec,
        max_word_length=MAX_WORD_LENGTH,
    )

    board = encode_board(board_raw, codec)

    result = solve_numba(
        board,
        word_array,
        word_lengths,
        codec.wild_id,
        filtered_words,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_word_mask=use_word_mask,
    )

    return {
        "result": result,
        "codec": codec,
        "words": filtered_words,
    }


def benchmark(
    files: List[Path],
    *,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_word_mask: bool,
    warmup: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    if not files:
        return {
            "rows": [],
            "summary": {
                "files": 0,
            },
        }

    if warmup:
        print("Warming up Numba JIT...")
        try:
            first = load_replay(files[0])
            _ = solve_replay(
                first,
                max_nodes=max_nodes,
                max_depth=max_depth,
                max_paths_per_word=max_paths_per_word,
                use_word_mask=use_word_mask,
            )
            print("Warmup done.")
        except Exception as e:
            print("Warmup failed:", e)

    solved_count = 0
    failed_count = 0

    total_js_nodes = 0
    total_py_nodes = 0
    total_js_backtracks = 0
    total_py_backtracks = 0
    total_js_dead_ends = 0
    total_py_dead_ends = 0
    total_js_elapsed = 0.0
    total_py_elapsed = 0.0

    by_f: Dict[int, Dict[str, Any]] = {}

    for idx, file in enumerate(files, start=1):
        replay = load_replay(file)
        target_f = replay_target_f(replay)
        js_stats = replay_solver_stats(replay)

        try:
            pack = solve_replay(
                replay,
                max_nodes=max_nodes,
                max_depth=max_depth,
                max_paths_per_word=max_paths_per_word,
                use_word_mask=use_word_mask,
            )

            py_result = pack["result"]
            py_stats = py_result["stats"]
            py_solved = bool(py_result["solved"])

            if py_solved:
                solved_count += 1
            else:
                failed_count += 1

            js_branch = summarize_branch_hist(
                js_stats.get("branchingHistogram", {})
            )
            py_branch = summarize_branch_hist(
                py_stats.get("branchingHistogram", {})
            )

            row = {
                "file": str(file),
                "league": f"f{target_f}",
                "targetWildcards": target_f,

                "jsSolved": replay_solver_solved(replay),
                "pySolved": py_solved,

                "jsNodes": safe_int(js_stats.get("exploredNodes")),
                "pyNodes": safe_int(py_stats.get("exploredNodes")),

                "jsBacktracks": safe_int(js_stats.get("backtracks")),
                "pyBacktracks": safe_int(py_stats.get("backtracks")),

                "jsDeadEnds": safe_int(js_stats.get("deadEnds")),
                "pyDeadEnds": safe_int(py_stats.get("deadEnds")),

                "jsForcedMoves": safe_int(js_stats.get("forcedMoves")),
                "pyForcedMoves": safe_int(py_stats.get("forcedMoves")),

                "jsMaxDepth": safe_int(js_stats.get("maxDepth")),
                "pyMaxDepth": safe_int(py_stats.get("maxDepth")),

                "jsElapsedMs": safe_float(js_stats.get("elapsedMs")),
                "pyElapsedMs": safe_float(py_stats.get("elapsedMs")),

                "jsMaxBranch": js_branch["maxBranch"],
                "pyMaxBranch": py_branch["maxBranch"],

                "jsMultiBranchCount": js_branch["multiBranchCount"],
                "pyMultiBranchCount": py_branch["multiBranchCount"],
            }

            row["nodesDiff"] = row["jsNodes"] - row["pyNodes"]
            row["backtracksDiff"] = row["jsBacktracks"] - row["pyBacktracks"]
            row["deadEndsDiff"] = row["jsDeadEnds"] - row["pyDeadEnds"]
            row["elapsedMsDiff"] = row["jsElapsedMs"] - row["pyElapsedMs"]

            rows.append(row)

            total_js_nodes += row["jsNodes"]
            total_py_nodes += row["pyNodes"]
            total_js_backtracks += row["jsBacktracks"]
            total_py_backtracks += row["pyBacktracks"]
            total_js_dead_ends += row["jsDeadEnds"]
            total_py_dead_ends += row["pyDeadEnds"]
            total_js_elapsed += row["jsElapsedMs"]
            total_py_elapsed += row["pyElapsedMs"]

            if target_f not in by_f:
                by_f[target_f] = {
                    "count": 0,
                    "pySolved": 0,
                    "jsNodes": 0,
                    "pyNodes": 0,
                    "jsBacktracks": 0,
                    "pyBacktracks": 0,
                    "jsDeadEnds": 0,
                    "pyDeadEnds": 0,
                    "jsElapsedMs": 0.0,
                    "pyElapsedMs": 0.0,
                }

            bf = by_f[target_f]
            bf["count"] += 1
            bf["pySolved"] += 1 if py_solved else 0
            bf["jsNodes"] += row["jsNodes"]
            bf["pyNodes"] += row["pyNodes"]
            bf["jsBacktracks"] += row["jsBacktracks"]
            bf["pyBacktracks"] += row["pyBacktracks"]
            bf["jsDeadEnds"] += row["jsDeadEnds"]
            bf["pyDeadEnds"] += row["pyDeadEnds"]
            bf["jsElapsedMs"] += row["jsElapsedMs"]
            bf["pyElapsedMs"] += row["pyElapsedMs"]

            print(
                f"[{idx}/{len(files)}] {file.parent.name}/{file.name} "
                f"F={target_f} solved={py_solved} "
                f"JS nodes={row['jsNodes']} PY nodes={row['pyNodes']} "
                f"JS ms={row['jsElapsedMs']:.3f} PY ms={row['pyElapsedMs']:.3f}"
            )

        except Exception as e:
            failed_count += 1

            row = {
                "file": str(file),
                "league": f"f{target_f}",
                "targetWildcards": target_f,
                "error": str(e),
            }

            rows.append(row)

            print(f"[{idx}/{len(files)}] ERROR {file}: {e}")

    n = max(1, len(files))

    summary_by_f: Dict[str, Any] = {}

    for f, data in sorted(by_f.items()):
        count = max(1, data["count"])

        summary_by_f[f"f{f}"] = {
            "count": data["count"],
            "solveRate": data["pySolved"] / count,

            "avgJsNodes": data["jsNodes"] / count,
            "avgPyNodes": data["pyNodes"] / count,

            "avgJsBacktracks": data["jsBacktracks"] / count,
            "avgPyBacktracks": data["pyBacktracks"] / count,

            "avgJsDeadEnds": data["jsDeadEnds"] / count,
            "avgPyDeadEnds": data["pyDeadEnds"] / count,

            "avgJsElapsedMs": data["jsElapsedMs"] / count,
            "avgPyElapsedMs": data["pyElapsedMs"] / count,
        }

    summary = {
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": len(files),
        "solved": solved_count,
        "failed": failed_count,
        "solveRate": solved_count / n,

        "avgJsNodes": total_js_nodes / n,
        "avgPyNodes": total_py_nodes / n,

        "avgJsBacktracks": total_js_backtracks / n,
        "avgPyBacktracks": total_py_backtracks / n,

        "avgJsDeadEnds": total_js_dead_ends / n,
        "avgPyDeadEnds": total_py_dead_ends / n,

        "avgJsElapsedMs": total_js_elapsed / n,
        "avgPyElapsedMs": total_py_elapsed / n,

        "byF": summary_by_f,
    }

    return {
        "rows": rows,
        "summary": summary,
    }


def save_result(result: Dict[str, Any]) -> Dict[str, str]:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    json_file = BENCHMARK_DIR / "numba_replay_benchmark.json"
    csv_file = BENCHMARK_DIR / "numba_replay_benchmark.csv"

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    rows = result.get("rows", [])

    if rows:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))

        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return {
        "json": str(json_file),
        "csv": str(csv_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--league", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--max-depth", type=int, default=120)
    parser.add_argument("--max-paths-per-word", type=int, default=80)
    parser.add_argument("--use-word-mask", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = list_replay_files(
        league=args.league,
        limit=args.limit if args.limit > 0 else None,
    )

    print("=== NUMBA REPLAY BENCHMARK ===")
    print("League:", args.league)
    print("Files:", len(files))

    if not files:
        print("No replay files found.")
        return

    result = benchmark(
        files,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        max_paths_per_word=args.max_paths_per_word,
        use_word_mask=args.use_word_mask,
        warmup=not args.no_warmup,
    )

    saved = save_result(result)

    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(" JSON:", saved["json"])
    print(" CSV :", saved["csv"])


if __name__ == "__main__":
    main()