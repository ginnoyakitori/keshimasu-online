# python/src/keshimasu_py/replay_io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT, MAX_WORD_LENGTH


DEFAULT_REPLAY_ROOT = PROJECT_ROOT / "data" / "training" / "selfplay"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_replay(path: Path) -> Dict[str, Any]:
    return load_json(path)


def list_replay_files(
    replay_root: Path = DEFAULT_REPLAY_ROOT,
    league: str = "all",
    limit: Optional[int] = None,
) -> List[Path]:
    """
    Replay JSON ファイル一覧を取得する。

    Parameters
    ----------
    replay_root:
        data/training/selfplay のパス。

    league:
        "all" の場合は f0, f1, f2... をすべて読む。
        "f2" または "2" の場合は f2 のみ読む。

    limit:
        読み込むファイル数の上限。
        None または 0 以下なら制限なし。

    Returns
    -------
    List[Path]
        Replay JSON ファイルのパス一覧。
    """
    if not replay_root.exists():
        return []

    if league == "all":
        dirs = [
            p for p in replay_root.iterdir()
            if p.is_dir() and p.name.startswith("f")
        ]
    else:
        league_name = league if league.startswith("f") else f"f{league}"
        dirs = [replay_root / league_name]

    files: List[Path] = []

    for d in sorted(dirs):
        if not d.exists():
            continue

        files.extend(sorted(d.glob("*.json")))

    if limit is not None and limit > 0:
        files = files[:limit]

    return files


def replay_board(replay: Dict[str, Any]) -> List[List[Any]]:
    """
    Replay から board を取得する。
    """
    board = replay.get("board")

    if board is None:
        raise ValueError("Replay has no board")

    return board


def replay_words(replay: Dict[str, Any]) -> List[str]:
    """
    Replay に保存された words を取得する。

    2〜MAX_WORD_LENGTH 文字の単語だけ返す。
    """
    words = replay.get("words", [])

    return [
        word for word in words
        if isinstance(word, str)
        and 2 <= len(list(word)) <= MAX_WORD_LENGTH
    ]


def replay_target_f(replay: Dict[str, Any]) -> int:
    """
    Replay の targetWildcards を取得する。
    """
    return int(replay.get("targetWildcards", 0))


def replay_actual_f(replay: Dict[str, Any]) -> int:
    """
    Replay の actualWildcards を取得する。
    なければ targetWildcards を返す。
    """
    if "actualWildcards" in replay:
        return int(replay["actualWildcards"])

    generator = replay.get("generator", {})

    if "actualWildcards" in generator:
        return int(generator["actualWildcards"])

    return replay_target_f(replay)


def replay_solver_stats(replay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replay の solver stats を取得する。
    """
    solver = replay.get("solver", {})
    stats = solver.get("stats")

    if stats:
        return stats

    # 古い保存形式対応
    stats = replay.get("stats")

    if stats:
        return stats

    return {}


def replay_solver_solved(replay: Dict[str, Any]) -> bool:
    """
    Solver が解けたかどうか。
    """
    solver = replay.get("solver", {})

    if "solved" in solver:
        return bool(solver["solved"])

    if "solved" in replay:
        return bool(replay["solved"])

    # 古い形式では solved がない場合もあるので、
    # solvedMoves があれば True 扱い。
    return len(replay_solved_moves(replay)) > 0


def replay_solved_moves(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Solver の解答手順を取得する。

    対応形式:
      replay["solver"]["solvedMoves"]
      replay["solverSolutionMoves"]
      replay["stats"]["solvedMoves"]
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


def replay_generator_moves(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generator 側の保証解を取得する。
    """
    generator = replay.get("generator", {})
    moves = generator.get("solutionMoves")

    if moves:
        return moves

    moves = replay.get("generatorSolutionMoves")

    if moves:
        return moves

    return []


def replay_id(replay: Dict[str, Any], fallback: str = "") -> str:
    """
    Replay ID を取得する。
    """
    return str(replay.get("id") or fallback)


def replay_epoch(replay: Dict[str, Any], default: int = -1) -> int:
    """
    Replay の epoch を取得する。
    """
    try:
        return int(replay.get("epoch", default))
    except Exception:
        return default


def replay_summary(replay: Dict[str, Any], fallback_id: str = "") -> Dict[str, Any]:
    """
    Replay の概要をまとめて返す。
    デバッグや一覧表示用。
    """
    stats = replay_solver_stats(replay)

    return {
        "id": replay_id(replay, fallback=fallback_id),
        "epoch": replay_epoch(replay),
        "targetWildcards": replay_target_f(replay),
        "actualWildcards": replay_actual_f(replay),
        "solverSolved": replay_solver_solved(replay),
        "moves": len(replay_solved_moves(replay)),
        "exploredNodes": int(stats.get("exploredNodes", 0) or 0),
        "backtracks": int(stats.get("backtracks", 0) or 0),
        "deadEnds": int(stats.get("deadEnds", 0) or 0),
        "forcedMoves": int(stats.get("forcedMoves", 0) or 0),
        "elapsedMs": float(stats.get("elapsedMs", 0.0) or 0.0),
    }