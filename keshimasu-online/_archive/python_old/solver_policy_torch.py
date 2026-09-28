# python/solver_policy_torch.py
#
# Policy-guided DFS Solver using:
# - Numba core functions for move generation / gravity / remove
# - PyTorch policy network for move ordering
#
# 目的:
#   通常 Numba Solver と Torch Policy Solver を比較する
#
# 実行例:
#   python python/solver_policy_torch.py
#   python python/solver_policy_torch.py --league f2 --index 0
#   python python/solver_policy_torch.py --replay data/training/selfplay/f2/xxx.json
#   python python/solver_policy_torch.py --benchmark --league f2 --limit 20
#
# 注意:
# - PyTorch は Numba の中には入れない
# - DFS 制御は Python 側
# - 候補列挙・消去・重力だけ Numba を使う
#

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Import local modules
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from core_solver_numba import (  # noqa: E402
    ROWS,
    COLS,
    MAX_WORD_LENGTH,
    DIR_H,
    DIR_V,
    _find_all_moves,
    _sort_moves_by_score,
    _remove_move_and_apply_gravity,
    solve_numba,
)

from inference_policy_torch import TorchPolicy  # noqa: E402


# ============================================================
# Paths
# ============================================================

DEFAULT_REPLAY_ROOT = PROJECT_ROOT / "data" / "training" / "selfplay"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"


# ============================================================
# Replay utilities
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_replay_files(
    replay_root: Path,
    league: str = "f2",
    limit: Optional[int] = None,
) -> List[Path]:
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
    board = replay.get("board")

    if board is None:
        raise ValueError("Replay has no board")

    return board


def replay_target_f(replay: Dict[str, Any]) -> int:
    return int(replay.get("targetWildcards", 0))


def normalize_cell(cell: Any) -> Optional[str]:
    if cell is None:
        return None

    if cell == "・":
        return None

    if cell == "Ｆ":
        return "F"

    return str(cell)


# ============================================================
# Metadata / Encoding
# ============================================================

def get_dataset_metadata_from_policy(policy: TorchPolicy) -> Dict[str, Any]:
    """
    train_policy_torch.py で checkpoint に保存した dataset metadata を取得する。

    checkpoint["metadata"] に以下が入っている想定:
      words
      charToId
      idToChar
      wildId
      emptyId
    """
    if policy.checkpoint is None:
        raise RuntimeError("Policy checkpoint is not loaded.")

    dataset_metadata = policy.checkpoint.get("metadata")

    if not dataset_metadata:
        raise RuntimeError(
            "checkpoint['metadata'] is missing. "
            "Please retrain with train_policy_torch.py."
        )

    return dataset_metadata


def encode_board_with_metadata(
    board_raw: List[List[Any]],
    metadata: Dict[str, Any],
) -> np.ndarray:
    char_to_id = metadata["charToId"]

    board = np.zeros((ROWS, COLS), dtype=np.int16)

    if len(board_raw) != ROWS:
        raise ValueError(f"board must have {ROWS} rows")

    for r in range(ROWS):
        row = board_raw[r]

        if len(row) != COLS:
            raise ValueError(f"row {r} must have {COLS} cells")

        for c in range(COLS):
            cell = normalize_cell(row[c])

            if cell is None:
                board[r, c] = 0
            else:
                if cell not in char_to_id:
                    raise KeyError(f"unknown char in metadata charToId: {cell}")

                board[r, c] = int(char_to_id[cell])

    return board


def board_to_string_with_metadata(
    board: np.ndarray,
    metadata: Dict[str, Any],
) -> str:
    id_to_char = metadata["idToChar"]

    lines: List[str] = []

    for r in range(ROWS):
        cells: List[str] = []

        for c in range(COLS):
            value = int(board[r, c])

            if value == 0:
                cells.append("・")
                continue

            ch = id_to_char.get(str(value), "?")

            if ch == "F":
                cells.append("Ｆ")
            else:
                cells.append(ch)

        lines.append(" ".join(cells))

    return "\n".join(lines)


def build_context_from_replay_and_policy(
    replay: Dict[str, Any],
    policy: TorchPolicy,
) -> Dict[str, Any]:
    metadata = get_dataset_metadata_from_policy(policy)

    board_raw = replay_board(replay)

    board = encode_board_with_metadata(
        board_raw,
        metadata,
    )

    word_array = policy.word_array
    word_lengths = policy.word_lengths

    if word_array is None or word_lengths is None:
        raise RuntimeError("Policy checkpoint has no word_array/word_lengths.")

    words_text = metadata["words"]

    wild_id = int(metadata["wildId"])

    return {
        "board": np.ascontiguousarray(board.astype(np.int16)),
        "word_array": np.ascontiguousarray(word_array.astype(np.int16)),
        "word_lengths": np.ascontiguousarray(word_lengths.astype(np.int16)),
        "words_text": words_text,
        "wild_id": wild_id,
        "metadata": metadata,
    }


# ============================================================
# Move utilities
# ============================================================

def encoded_move_to_path(
    move: np.ndarray,
    word_lengths: np.ndarray,
) -> List[Tuple[int, int]]:
    wid = int(move[0])
    row = int(move[1])
    col = int(move[2])
    direction = int(move[3])

    length = int(word_lengths[wid])

    path: List[Tuple[int, int]] = []

    for i in range(length):
        if direction == DIR_H:
            path.append((row, col + i))
        else:
            path.append((row + i, col))

    return path


def encoded_move_to_dict(
    move: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    policy_score: Optional[float] = None,
    heuristic_score: Optional[float] = None,
) -> Dict[str, Any]:
    wid = int(move[0])
    direction = int(move[3])

    out = {
        "word_id": wid,
        "word": words_text[wid],
        "row": int(move[1]),
        "col": int(move[2]),
        "direction": "H" if direction == DIR_H else "V",
        "path": encoded_move_to_path(move, word_lengths),
    }

    if policy_score is not None:
        out["policyScore"] = float(policy_score)

    if heuristic_score is not None:
        out["heuristicScore"] = float(heuristic_score)

    return out


def board_hash(board: np.ndarray) -> bytes:
    """
    Python側 visited 用。
    速度優先なら将来 Zobrist hash に差し替え。
    """
    return board.tobytes()


def is_board_empty_py(board: np.ndarray) -> bool:
    return not np.any(board)


def count_filled_py(board: np.ndarray) -> int:
    return int(np.count_nonzero(board))


def find_moves_encoded(
    board: np.ndarray,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    max_paths_per_word: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba _find_all_moves から
    moves[N,4], scores[N] を返す。
    """
    moves, scores, count = _find_all_moves(
        board,
        word_array,
        word_lengths,
        np.int16(wild_id),
        max_paths_per_word,
        False,
        np.uint64(0),
    )

    if count <= 0:
        return (
            np.zeros((0, 4), dtype=np.int16),
            np.zeros((0,), dtype=np.float64),
        )

    moves = moves[:count].copy()
    scores = scores[:count].copy()

    return moves, scores


def heuristic_order_moves(
    moves: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if moves.shape[0] == 0:
        return moves, scores

    # Numba版 insertion sort をそのまま使う
    tmp_moves = moves.copy()
    tmp_scores = scores.copy()

    _sort_moves_by_score(
        tmp_moves,
        tmp_scores,
        tmp_moves.shape[0],
    )

    return tmp_moves, tmp_scores


# ============================================================
# Stats
# ============================================================

def create_stats() -> Dict[str, Any]:
    return {
        "exploredNodes": 0,
        "backtracks": 0,
        "deadEnds": 0,
        "forcedMoves": 0,
        "maxDepth": 0,
        "elapsedMs": 0.0,
        "branchingHistogram": {},
    }


def record_branch(stats: Dict[str, Any], branch: int) -> None:
    hist = stats["branchingHistogram"]
    key = str(branch)
    hist[key] = hist.get(key, 0) + 1


# ============================================================
# Policy-guided DFS
# ============================================================

class PolicySolver:
    def __init__(
        self,
        policy: TorchPolicy,
        *,
        max_nodes: int = 500_000,
        max_depth: int = 120,
        max_paths_per_word: int = 80,
        use_policy: bool = True,
        fallback_to_heuristic: bool = True,
        use_visited: bool = True,
        verbose: bool = False,
    ) -> None:
        self.policy = policy

        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_paths_per_word = max_paths_per_word

        self.use_policy = use_policy
        self.fallback_to_heuristic = fallback_to_heuristic
        self.use_visited = use_visited
        self.verbose = verbose

    def order_moves(
        self,
        board: np.ndarray,
        moves: np.ndarray,
        heuristic_scores: np.ndarray,
        *,
        word_array: np.ndarray,
        word_lengths: np.ndarray,
        target_f: int,
        depth: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if moves.shape[0] == 0:
            return moves, heuristic_scores.astype(np.float32)

        if not self.use_policy:
            ordered, scores = heuristic_order_moves(moves, heuristic_scores)
            return ordered, scores.astype(np.float32)

        try:
            ordered_moves, policy_scores = self.policy.order_encoded_moves(
                board,
                moves,
                heuristic_scores.astype(np.float32),
                word_array=word_array,
                word_lengths=word_lengths,
                target_f=target_f,
                depth=depth,
            )

            return ordered_moves, policy_scores.astype(np.float32)

        except Exception as e:
            if self.verbose:
                print("[PolicySolver] policy ordering failed:", e)

            if not self.fallback_to_heuristic:
                raise

            ordered, scores = heuristic_order_moves(moves, heuristic_scores)
            return ordered, scores.astype(np.float32)

    def solve(
        self,
        board: np.ndarray,
        word_array: np.ndarray,
        word_lengths: np.ndarray,
        words_text: List[str],
        wild_id: int,
        target_f: int = 0,
    ) -> Dict[str, Any]:
        stats = create_stats()
        visited = set()

        start = time.perf_counter()

        solved, final_board, solved_moves = self._dfs(
            board=np.ascontiguousarray(board.astype(np.int16)),
            word_array=np.ascontiguousarray(word_array.astype(np.int16)),
            word_lengths=np.ascontiguousarray(word_lengths.astype(np.int16)),
            words_text=words_text,
            wild_id=wild_id,
            target_f=target_f,
            depth=0,
            visited=visited,
            stats=stats,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats["elapsedMs"] = elapsed_ms

        return {
            "solved": solved,
            "board": final_board,
            "stats": stats,
            "solvedMoves": solved_moves,
            "usedPolicy": self.use_policy,
        }

    def _dfs(
        self,
        *,
        board: np.ndarray,
        word_array: np.ndarray,
        word_lengths: np.ndarray,
        words_text: List[str],
        wild_id: int,
        target_f: int,
        depth: int,
        visited: set,
        stats: Dict[str, Any],
    ) -> Tuple[bool, np.ndarray, List[Dict[str, Any]]]:
        stats["exploredNodes"] += 1
        stats["maxDepth"] = max(stats["maxDepth"], depth)

        if stats["exploredNodes"] >= self.max_nodes:
            stats["deadEnds"] += 1
            return False, board, []

        if depth > self.max_depth:
            stats["deadEnds"] += 1
            return False, board, []

        if is_board_empty_py(board):
            return True, board, []

        h = board_hash(board)

        if self.use_visited:
            if h in visited:
                stats["deadEnds"] += 1
                return False, board, []

            visited.add(h)

        moves, heuristic_scores = find_moves_encoded(
            board,
            word_array,
            word_lengths,
            wild_id,
            self.max_paths_per_word,
        )

        branch = int(moves.shape[0])
        record_branch(stats, branch)

        if branch == 0:
            stats["deadEnds"] += 1
            return False, board, []

        if branch == 1:
            stats["forcedMoves"] += 1

        ordered_moves, policy_scores = self.order_moves(
            board,
            moves,
            heuristic_scores,
            word_array=word_array,
            word_lengths=word_lengths,
            target_f=target_f,
            depth=depth,
        )

        before_filled = count_filled_py(board)

        for i in range(ordered_moves.shape[0]):
            move = ordered_moves[i]

            next_board = _remove_move_and_apply_gravity(
                board,
                move,
                word_lengths,
            )

            after_filled = count_filled_py(next_board)

            if after_filled >= before_filled:
                continue

            child_visited = set(visited) if self.use_visited else visited

            solved, final_board, child_moves = self._dfs(
                board=next_board,
                word_array=word_array,
                word_lengths=word_lengths,
                words_text=words_text,
                wild_id=wild_id,
                target_f=target_f,
                depth=depth + 1,
                visited=child_visited,
                stats=stats,
            )

            if solved:
                move_dict = encoded_move_to_dict(
                    move,
                    word_lengths,
                    words_text,
                    policy_score=float(policy_scores[i]) if i < len(policy_scores) else None,
                    heuristic_score=None,
                )

                return True, final_board, [move_dict] + child_moves

        stats["backtracks"] += 1

        return False, board, []


# ============================================================
# Comparison
# ============================================================

def compare_on_replay(
    replay_file: Path,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str = "auto",
    max_nodes: int = 500_000,
    max_depth: int = 120,
    max_paths_per_word: int = 80,
    use_policy: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    replay = load_json(replay_file)

    target_f = replay_target_f(replay)

    policy = TorchPolicy(
        model_dir=model_dir,
        device=device,
    )
    policy.load()

    ctx = build_context_from_replay_and_policy(
        replay,
        policy,
    )

    board = ctx["board"]
    word_array = ctx["word_array"]
    word_lengths = ctx["word_lengths"]
    words_text = ctx["words_text"]
    wild_id = ctx["wild_id"]
    metadata = ctx["metadata"]

    # 通常 Numba Solver
    normal = solve_numba(
        board,
        word_array,
        word_lengths,
        wild_id,
        words_text,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_word_mask=False,
    )

    # Policy Solver
    solver = PolicySolver(
        policy,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_policy=use_policy,
        fallback_to_heuristic=True,
        use_visited=True,
        verbose=verbose,
    )

    policy_result = solver.solve(
        board,
        word_array,
        word_lengths,
        words_text,
        wild_id,
        target_f=target_f,
    )

    improvement = {
        "nodes": normal["stats"]["exploredNodes"] - policy_result["stats"]["exploredNodes"],
        "backtracks": normal["stats"]["backtracks"] - policy_result["stats"]["backtracks"],
        "deadEnds": normal["stats"]["deadEnds"] - policy_result["stats"]["deadEnds"],
        "elapsedMs": normal["stats"]["elapsedMs"] - policy_result["stats"]["elapsedMs"],
    }

    return {
        "replayFile": str(replay_file),
        "targetF": target_f,
        "boardText": board_to_string_with_metadata(board, metadata),
        "normal": normal,
        "policy": policy_result,
        "improvement": improvement,
    }

def print_compare_result(result: Dict[str, Any]) -> None:
    print("\n=== BOARD ===\n")
    print(result["boardText"])

    normal = result["normal"]
    policy = result["policy"]
    imp = result["improvement"]

    normal_stats = normal["stats"]
    policy_stats = policy["stats"]

    print("\n=== NORMAL NUMBA SOLVER ===\n")
    print("Solved:", normal["solved"])
    print("Nodes:", normal_stats["exploredNodes"])
    print("Backtracks:", normal_stats["backtracks"])
    print("DeadEnds:", normal_stats["deadEnds"])
    print("Forced:", normal_stats["forcedMoves"])
    print("Depth:", normal_stats["maxDepth"])
    print("ElapsedMs:", f"{normal_stats['elapsedMs']:.4f}")

    print("\n=== TORCH POLICY SOLVER ===\n")
    print("Solved:", policy["solved"])
    print("UsedPolicy:", policy["usedPolicy"])
    print("Nodes:", policy_stats["exploredNodes"])
    print("Backtracks:", policy_stats["backtracks"])
    print("DeadEnds:", policy_stats["deadEnds"])
    print("Forced:", policy_stats["forcedMoves"])
    print("Depth:", policy_stats["maxDepth"])
    print("ElapsedMs:", f"{policy_stats['elapsedMs']:.4f}")

    print("\n=== IMPROVEMENT normal - policy ===\n")
    print("Nodes:", imp["nodes"])
    print("Backtracks:", imp["backtracks"])
    print("DeadEnds:", imp["deadEnds"])
    print("ElapsedMs:", f"{imp['elapsedMs']:.4f}")

    print("\n=== POLICY SOLVED MOVES ===\n")

    for i, move in enumerate(policy["solvedMoves"], start=1):
        print(
            f"{i:02d}. {move['word']} "
            f"{move['direction']} "
            f"({move['row']},{move['col']}) "
            f"path={move['path']} "
            f"policy={move.get('policyScore', 0):.4f}"
        )


# ============================================================
# Benchmark
# ============================================================

def benchmark_policy_solver(
    files: List[Path],
    *,
    model_dir: Path,
    device: str,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_policy: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    total_normal_nodes = 0
    total_policy_nodes = 0
    total_normal_backtracks = 0
    total_policy_backtracks = 0
    total_normal_deadends = 0
    total_policy_deadends = 0
    total_normal_ms = 0.0
    total_policy_ms = 0.0

    solved_count = 0

    by_f: Dict[str, Dict[str, Any]] = {}

    # モデルは毎回ロードしないよう、先に1回だけロード
    policy = TorchPolicy(
        model_dir=model_dir,
        device=device,
    )
    policy.load()

    for idx, replay_file in enumerate(files, start=1):
        replay = load_json(replay_file)
        target_f = replay_target_f(replay)

        ctx = build_context_from_replay_and_policy(
            replay,
            policy,
        )

        board = ctx["board"]
        word_array = ctx["word_array"]
        word_lengths = ctx["word_lengths"]
        words_text = ctx["words_text"]
        wild_id = ctx["wild_id"]

        normal = solve_numba(
            board,
            word_array,
            word_lengths,
            wild_id,
            words_text,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_paths_per_word=max_paths_per_word,
            use_word_mask=False,
        )

        solver = PolicySolver(
            policy,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_paths_per_word=max_paths_per_word,
            use_policy=use_policy,
            fallback_to_heuristic=True,
            use_visited=True,
            verbose=False,
        )

        policy_result = solver.solve(
            board,
            word_array,
            word_lengths,
            words_text,
            wild_id,
            target_f=target_f,
        )

        if policy_result["solved"]:
            solved_count += 1

        row = {
            "file": str(replay_file),
            "league": f"f{target_f}",
            "targetF": target_f,

            "normalSolved": bool(normal["solved"]),
            "policySolved": bool(policy_result["solved"]),

            "normalNodes": int(normal["stats"]["exploredNodes"]),
            "policyNodes": int(policy_result["stats"]["exploredNodes"]),

            "normalBacktracks": int(normal["stats"]["backtracks"]),
            "policyBacktracks": int(policy_result["stats"]["backtracks"]),

            "normalDeadEnds": int(normal["stats"]["deadEnds"]),
            "policyDeadEnds": int(policy_result["stats"]["deadEnds"]),

            "normalElapsedMs": float(normal["stats"]["elapsedMs"]),
            "policyElapsedMs": float(policy_result["stats"]["elapsedMs"]),
        }

        row["nodesImprovement"] = row["normalNodes"] - row["policyNodes"]
        row["backtracksImprovement"] = row["normalBacktracks"] - row["policyBacktracks"]
        row["deadEndsImprovement"] = row["normalDeadEnds"] - row["policyDeadEnds"]
        row["elapsedMsImprovement"] = row["normalElapsedMs"] - row["policyElapsedMs"]

        rows.append(row)

        total_normal_nodes += row["normalNodes"]
        total_policy_nodes += row["policyNodes"]
        total_normal_backtracks += row["normalBacktracks"]
        total_policy_backtracks += row["policyBacktracks"]
        total_normal_deadends += row["normalDeadEnds"]
        total_policy_deadends += row["policyDeadEnds"]
        total_normal_ms += row["normalElapsedMs"]
        total_policy_ms += row["policyElapsedMs"]

        key = f"f{target_f}"

        if key not in by_f:
            by_f[key] = {
                "count": 0,
                "policySolved": 0,
                "normalNodes": 0,
                "policyNodes": 0,
                "normalBacktracks": 0,
                "policyBacktracks": 0,
                "normalDeadEnds": 0,
                "policyDeadEnds": 0,
                "normalElapsedMs": 0.0,
                "policyElapsedMs": 0.0,
            }

        bf = by_f[key]
        bf["count"] += 1
        bf["policySolved"] += 1 if policy_result["solved"] else 0
        bf["normalNodes"] += row["normalNodes"]
        bf["policyNodes"] += row["policyNodes"]
        bf["normalBacktracks"] += row["normalBacktracks"]
        bf["policyBacktracks"] += row["policyBacktracks"]
        bf["normalDeadEnds"] += row["normalDeadEnds"]
        bf["policyDeadEnds"] += row["policyDeadEnds"]
        bf["normalElapsedMs"] += row["normalElapsedMs"]
        bf["policyElapsedMs"] += row["policyElapsedMs"]

        print(
            f"[{idx}/{len(files)}] "
            f"{replay_file.parent.name}/{replay_file.name} "
            f"F={target_f} "
            f"N nodes={row['normalNodes']} "
            f"P nodes={row['policyNodes']} "
            f"diff={row['nodesImprovement']} "
            f"N ms={row['normalElapsedMs']:.3f} "
            f"P ms={row['policyElapsedMs']:.3f}"
        )

    n = max(1, len(files))

    summary_by_f: Dict[str, Any] = {}

    for key, data in sorted(by_f.items()):
        count = max(1, data["count"])

        summary_by_f[key] = {
            "count": data["count"],
            "solveRate": data["policySolved"] / count,

            "avgNormalNodes": data["normalNodes"] / count,
            "avgPolicyNodes": data["policyNodes"] / count,
            "avgNodesImprovement": (
                data["normalNodes"] - data["policyNodes"]
            ) / count,

            "avgNormalBacktracks": data["normalBacktracks"] / count,
            "avgPolicyBacktracks": data["policyBacktracks"] / count,
            "avgBacktracksImprovement": (
                data["normalBacktracks"] - data["policyBacktracks"]
            ) / count,

            "avgNormalDeadEnds": data["normalDeadEnds"] / count,
            "avgPolicyDeadEnds": data["policyDeadEnds"] / count,
            "avgDeadEndsImprovement": (
                data["normalDeadEnds"] - data["policyDeadEnds"]
            ) / count,

            "avgNormalElapsedMs": data["normalElapsedMs"] / count,
            "avgPolicyElapsedMs": data["policyElapsedMs"] / count,
        }

    summary = {
        "files": len(files),
        "policySolved": solved_count,
        "policySolveRate": solved_count / n,

        "avgNormalNodes": total_normal_nodes / n,
        "avgPolicyNodes": total_policy_nodes / n,
        "avgNodesImprovement": (
            total_normal_nodes - total_policy_nodes
        ) / n,

        "avgNormalBacktracks": total_normal_backtracks / n,
        "avgPolicyBacktracks": total_policy_backtracks / n,
        "avgBacktracksImprovement": (
            total_normal_backtracks - total_policy_backtracks
        ) / n,

        "avgNormalDeadEnds": total_normal_deadends / n,
        "avgPolicyDeadEnds": total_policy_deadends / n,
        "avgDeadEndsImprovement": (
            total_normal_deadends - total_policy_deadends
        ) / n,

        "avgNormalElapsedMs": total_normal_ms / n,
        "avgPolicyElapsedMs": total_policy_ms / n,

        "byF": summary_by_f,
    }

    return {
        "rows": rows,
        "summary": summary,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torch policy-guided solver"
    )

    parser.add_argument(
        "--replay",
        default=None,
        help="Path to replay json",
    )

    parser.add_argument(
        "--league",
        default="f2",
        help="f0, f1, f2, f3, all, or numeric like 2",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark multiple replays",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
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
        "--no-policy",
        action="store_true",
        help="Disable policy ordering and use heuristic ordering",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()

    if args.benchmark:
        files = list_replay_files(
            DEFAULT_REPLAY_ROOT,
            league=args.league,
            limit=args.limit if args.limit > 0 else None,
        )

        if not files:
            raise FileNotFoundError(
                f"No replay files found: league={args.league}"
            )

        print("=== BENCHMARK TORCH POLICY SOLVER ===")
        print("Files:", len(files))
        print("League:", args.league)
        print("Model:", model_dir)

        result = benchmark_policy_solver(
            files,
            model_dir=model_dir,
            device=args.device,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            max_paths_per_word=args.max_paths_per_word,
            use_policy=not args.no_policy,
        )

        print("\n=== SUMMARY ===")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

        return

    if args.replay:
        replay_file = Path(args.replay).resolve()
    else:
        files = list_replay_files(
            DEFAULT_REPLAY_ROOT,
            league=args.league,
        )

        if not files:
            raise FileNotFoundError(
                f"No replay files found: league={args.league}"
            )

        index = max(0, min(args.index, len(files) - 1))
        replay_file = files[index]

    print("=== COMPARE SOLVERS ON REPLAY ===")
    print("Replay:", replay_file)
    print("Model:", model_dir)

    result = compare_on_replay(
        replay_file,
        model_dir=model_dir,
        device=args.device,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        max_paths_per_word=args.max_paths_per_word,
        use_policy=not args.no_policy,
        verbose=args.verbose,
    )

    print_compare_result(result)


if __name__ == "__main__":
    main()