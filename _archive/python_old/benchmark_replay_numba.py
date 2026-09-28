# python/benchmark_replay_numba.py
#
# JS Replay Buffer を Python / Numba Solver で再解答してベンチマークする
#
# 入力:
#   data/training/selfplay/f0/*.json
#   data/training/selfplay/f1/*.json
#   data/training/selfplay/f2/*.json
#
# 出力:
#   data/benchmarks/numba_replay_benchmark.json
#   data/benchmarks/numba_replay_benchmark.csv
#
# 実行例:
#   python python/benchmark_replay_numba.py
#   python python/benchmark_replay_numba.py --league f2
#   python python/benchmark_replay_numba.py --league all --limit 50
#   python python/benchmark_replay_numba.py --max-nodes 500000
#

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# ------------------------------------------------------------
# Import core_solver_numba.py
# ------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from core_solver_numba import (  # noqa: E402
    ROWS,
    COLS,
    MAX_WORD_LENGTH,
    build_codec,
    encode_words,
    encode_board,
    board_to_string,
    solve_numba,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

def get_replay_root() -> Path:
    return PROJECT_ROOT / "data" / "training" / "selfplay"


def get_benchmark_dir() -> Path:
    return PROJECT_ROOT / "data" / "benchmarks"


# ------------------------------------------------------------
# Replay loading
# ------------------------------------------------------------

def list_replay_files(
    replay_root: Path,
    league: str = "all",
    limit: Optional[int] = None,
) -> List[Path]:
    """
    replay JSON の一覧を取得する。

    league:
      "all" -> f0/f1/f2... 全部
      "f0"  -> f0 だけ
      "f2"  -> f2 だけ
      "2"   -> f2 として扱う
    """
    if not replay_root.exists():
        return []

    if league == "all":
        league_dirs = [
            p for p in replay_root.iterdir()
            if p.is_dir() and p.name.startswith("f")
        ]
    else:
        if league.startswith("f"):
            league_name = league
        else:
            league_name = f"f{league}"

        league_dirs = [replay_root / league_name]

    files: List[Path] = []

    for d in sorted(league_dirs):
        if not d.exists():
            continue

        for file in sorted(d.glob("*.json")):
            files.append(file)

    if limit is not None and limit > 0:
        files = files[:limit]

    return files


def load_json(file: Path) -> Dict[str, Any]:
    with file.open("r", encoding="utf-8") as f:
        return json.load(f)


def replay_board(replay: Dict[str, Any]) -> List[List[Any]]:
    """
    Replay の board を取得する。
    """
    board = replay.get("board")

    if board is None:
        raise ValueError("Replay has no board")

    return board


def replay_words(replay: Dict[str, Any]) -> List[str]:
    """
    Replay に保存されている words を使う。
    なければ空リスト。
    """
    words = replay.get("words")

    if words is None:
        return []

    # 念のため 2〜5文字だけ
    return [
        w for w in words
        if isinstance(w, str)
        and 2 <= len(list(w)) <= MAX_WORD_LENGTH
    ]


def replay_target_wildcards(replay: Dict[str, Any]) -> int:
    return int(replay.get("targetWildcards", 0))


def replay_solver_stats(replay: Dict[str, Any]) -> Dict[str, Any]:
    """
    JS Solver の stats を取得。
    """
    solver = replay.get("solver", {})
    stats = solver.get("stats", {})

    # 古い形式対応
    if not stats:
        stats = replay.get("stats", {})

    return stats


def replay_solver_solved(replay: Dict[str, Any]) -> bool:
    solver = replay.get("solver", {})

    if "solved" in solver:
        return bool(solver["solved"])

    return True


# ------------------------------------------------------------
# Codec helpers
# ------------------------------------------------------------

def collect_board_chars(board: List[List[Any]]) -> str:
    chars: List[str] = []

    for row in board:
        for cell in row:
            if cell is None:
                continue

            if cell == "・":
                continue

            if cell == "Ｆ":
                chars.append("F")
            else:
                chars.append(str(cell))

    return "".join(chars)


def build_codec_for_replay(
    words: List[str],
    board: List[List[Any]],
):
    """
    codec は words の文字 + board 上の文字を含めて作る。

    これにより、board にだけ存在する文字があっても encode_board で落ちない。
    """
    board_chars = collect_board_chars(board)

    codec_source_words = list(words)

    if board_chars:
        codec_source_words.append(board_chars)

    return build_codec(codec_source_words, wildcard_char="F")


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

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


def summarize_branch_hist(hist: Dict[Any, Any]) -> Dict[str, Any]:
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

        if kk > max_branch:
            max_branch = kk

        if kk >= 2:
            multi += vv

    return {
        "maxBranch": max_branch,
        "multiBranchCount": multi,
    }


def diff_stats(
    js_stats: Dict[str, Any],
    py_stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "nodesDiff": safe_int(js_stats.get("exploredNodes")) - safe_int(py_stats.get("exploredNodes")),
        "backtracksDiff": safe_int(js_stats.get("backtracks")) - safe_int(py_stats.get("backtracks")),
        "deadEndsDiff": safe_int(js_stats.get("deadEnds")) - safe_int(py_stats.get("deadEnds")),
        "forcedMovesDiff": safe_int(js_stats.get("forcedMoves")) - safe_int(py_stats.get("forcedMoves")),
        "maxDepthDiff": safe_int(js_stats.get("maxDepth")) - safe_int(py_stats.get("maxDepth")),
        "elapsedMsDiff": safe_float(js_stats.get("elapsedMs")) - safe_float(py_stats.get("elapsedMs")),
    }


# ------------------------------------------------------------
# Benchmark one replay
# ------------------------------------------------------------

def solve_replay_with_numba(
    replay: Dict[str, Any],
    *,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_word_mask: bool,
) -> Dict[str, Any]:
    board_raw = replay_board(replay)
    words_raw = replay_words(replay)

    if len(words_raw) == 0:
        raise ValueError("Replay has no words")

    codec = build_codec_for_replay(words_raw, board_raw)

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


def warmup_numba_from_replay(
    replay: Dict[str, Any],
    *,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_word_mask: bool,
) -> None:
    """
    初回JITコンパイルをベンチマークから除外するためのウォームアップ。
    """
    _ = solve_replay_with_numba(
        replay,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_word_mask=use_word_mask,
    )


# ------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------
def benchmark_replays(
    files: List[Path],
    *,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_word_mask: bool,
    warmup: bool,
    print_boards: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    if len(files) == 0:
        return {
            "rows": [],
            "summary": {
                "files": 0,
            },
        }

    print(f"Replay files: {len(files)}")

    if warmup:
        print("Warming up Numba JIT with first replay...")
        first = load_json(files[0])

        try:
            warmup_numba_from_replay(
                first,
                max_nodes=max_nodes,
                max_depth=max_depth,
                max_paths_per_word=max_paths_per_word,
                use_word_mask=use_word_mask,
            )
            print("Warmup done.")
        except Exception as e:
            print("Warmup failed:", e)

    total_py_nodes = 0
    total_py_backtracks = 0
    total_py_dead_ends = 0
    total_py_elapsed = 0.0

    total_js_nodes = 0
    total_js_backtracks = 0
    total_js_dead_ends = 0
    total_js_elapsed = 0.0

    solved_count = 0
    failed_count = 0

    by_f: Dict[int, Dict[str, Any]] = {}

    for idx, file in enumerate(files, start=1):
        replay = load_json(file)

        target_f = replay_target_wildcards(replay)
        js_stats = replay_solver_stats(replay)

        try:
            solve_pack = solve_replay_with_numba(
                replay,
                max_nodes=max_nodes,
                max_depth=max_depth,
                max_paths_per_word=max_paths_per_word,
                use_word_mask=use_word_mask,
            )

            py_result = solve_pack["result"]
            codec = solve_pack["codec"]

            py_stats = py_result["stats"]
            py_solved = bool(py_result["solved"])

            if py_solved:
                solved_count += 1
            else:
                failed_count += 1

            d = diff_stats(js_stats, py_stats)

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

                **d,
            }

            rows.append(row)

            total_js_nodes += row["jsNodes"]
            total_js_backtracks += row["jsBacktracks"]
            total_js_dead_ends += row["jsDeadEnds"]
            total_js_elapsed += row["jsElapsedMs"]

            total_py_nodes += row["pyNodes"]
            total_py_backtracks += row["pyBacktracks"]
            total_py_dead_ends += row["pyDeadEnds"]
            total_py_elapsed += row["pyElapsedMs"]

            if target_f not in by_f:
                by_f[target_f] = {
                    "count": 0,
                    "pySolved": 0,

                    "pyNodes": 0,
                    "pyBacktracks": 0,
                    "pyDeadEnds": 0,
                    "pyElapsedMs": 0.0,

                    "jsNodes": 0,
                    "jsBacktracks": 0,
                    "jsDeadEnds": 0,
                    "jsElapsedMs": 0.0,
                }

            bf = by_f[target_f]

            bf["count"] += 1
            bf["pySolved"] += 1 if py_solved else 0

            bf["pyNodes"] += row["pyNodes"]
            bf["pyBacktracks"] += row["pyBacktracks"]
            bf["pyDeadEnds"] += row["pyDeadEnds"]
            bf["pyElapsedMs"] += row["pyElapsedMs"]

            bf["jsNodes"] += row["jsNodes"]
            bf["jsBacktracks"] += row["jsBacktracks"]
            bf["jsDeadEnds"] += row["jsDeadEnds"]
            bf["jsElapsedMs"] += row["jsElapsedMs"]

            print(
                f"[{idx}/{len(files)}] {file.parent.name}/{file.name} "
                f"F={target_f} "
                f"solved={py_solved} "
                f"JS nodes={row['jsNodes']} PY nodes={row['pyNodes']} "
                f"JS ms={row['jsElapsedMs']:.2f} PY ms={row['pyElapsedMs']:.2f}"
            )

            if print_boards and idx <= 3:
                print("\nFinal Board:")
                print(
                    board_to_string(
                        py_result["board"],
                        codec,
                    )
                )
                print()

        except Exception as e:
            failed_count += 1

            row = {
                "file": str(file),
                "league": f"f{target_f}",
                "targetWildcards": target_f,
                "error": str(e),
            }

            rows.append(row)

            print(
                f"[{idx}/{len(files)}] ERROR {file}: {e}"
            )

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
# ------------------------------------------------------------
# Save benchmark result
# ------------------------------------------------------------

def save_benchmark_result(result: Dict[str, Any]) -> Dict[str, str]:
    out_dir = get_benchmark_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_file = out_dir / "numba_replay_benchmark.json"
    csv_file = out_dir / "numba_replay_benchmark.csv"

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    rows = result.get("rows", [])

    if rows:
        fieldnames = sorted(
            set().union(*(r.keys() for r in rows))
        )

        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                writer.writerow(row)
    else:
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            f.write("no rows\n")

    return {
        "json": str(json_file),
        "csv": str(csv_file),
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark JS replay buffer with Python/Numba solver"
    )

    parser.add_argument(
        "--league",
        default="all",
        help="all, f0, f1, f2, f3, or numeric like 2",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of replay files. 0 means no limit.",
    )

    parser.add_argument(
        "--max-nodes",
        type=int,
        default=500_000,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--max-paths-per-word",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--use-word-mask",
        action="store_true",
        help="Use 64-bit word usage mask. Only safe if words <= 64.",
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable Numba warmup.",
    )

    parser.add_argument(
        "--print-boards",
        action="store_true",
        help="Print final boards for first few files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    replay_root = get_replay_root()

    files = list_replay_files(
        replay_root,
        league=args.league,
        limit=args.limit if args.limit > 0 else None,
    )

    print("=== NUMBA REPLAY BENCHMARK ===")
    print("Replay root:", replay_root)
    print("League:", args.league)
    print("Files:", len(files))
    print("maxNodes:", args.max_nodes)
    print("maxDepth:", args.max_depth)
    print("maxPathsPerWord:", args.max_paths_per_word)
    print("useWordMask:", args.use_word_mask)

    if len(files) == 0:
        print("No replay files found.")
        return

    result = benchmark_replays(
        files,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        max_paths_per_word=args.max_paths_per_word,
        use_word_mask=args.use_word_mask,
        warmup=not args.no_warmup,
        print_boards=args.print_boards,
    )

    saved = save_benchmark_result(result)

    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(" JSON:", saved["json"])
    print(" CSV :", saved["csv"])


if __name__ == "__main__":
    main()