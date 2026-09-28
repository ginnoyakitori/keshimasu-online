# python/apps/solver_policy.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from keshimasu_py.config import (
    PROJECT_ROOT,
    ROWS,
    COLS,
)

from keshimasu_py.core_solver import (
    _find_all_moves,
    _sort_moves_by_score,
    _remove_move_and_apply_gravity,
    solve_numba,
)

from keshimasu_py.policy_inference import TorchPolicy

from keshimasu_py.replay_io import (
    list_replay_files,
    load_replay,
    replay_board,
    replay_target_f,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"

DIR_H = 0
DIR_V = 1


def normalize_cell(cell: Any) -> Optional[str]:
    if cell is None:
        return None

    if cell == "・":
        return None

    if cell == "Ｆ":
        return "F"

    return str(cell)


def encode_board_with_metadata(
    board_raw: List[List[Any]],
    metadata: Dict[str, Any],
) -> np.ndarray:
    char_to_id = metadata["charToId"]

    board = np.zeros((ROWS, COLS), dtype=np.int16)

    for r in range(ROWS):
        for c in range(COLS):
            cell = normalize_cell(board_raw[r][c])

            if cell is None:
                board[r, c] = 0
            else:
                if cell not in char_to_id:
                    raise KeyError(f"unknown char in metadata: {cell}")

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
            else:
                ch = id_to_char.get(str(value), "?")
                cells.append("Ｆ" if ch == "F" else ch)

        lines.append(" ".join(cells))

    return "\n".join(lines)


def build_context_from_replay_and_policy(
    replay: Dict[str, Any],
    policy: TorchPolicy,
) -> Dict[str, Any]:
    if policy.checkpoint is None:
        raise RuntimeError("Policy checkpoint is not loaded.")

    metadata = policy.checkpoint.get("metadata")

    if not metadata:
        raise RuntimeError("checkpoint['metadata'] is missing.")

    board = encode_board_with_metadata(
        replay_board(replay),
        metadata,
    )

    if policy.word_array is None or policy.word_lengths is None:
        raise RuntimeError("Policy checkpoint has no word_array/word_lengths.")

    return {
        "board": np.ascontiguousarray(board.astype(np.int16)),
        "word_array": np.ascontiguousarray(policy.word_array.astype(np.int16)),
        "word_lengths": np.ascontiguousarray(policy.word_lengths.astype(np.int16)),
        "words_text": metadata["words"],
        "wild_id": int(metadata["wildId"]),
        "metadata": metadata,
    }


def is_board_empty(board: np.ndarray) -> bool:
    return not np.any(board)


def count_filled(board: np.ndarray) -> int:
    return int(np.count_nonzero(board))


def state_hash(
    board: np.ndarray,
    used_word_ids: set[int],
) -> tuple[bytes, tuple[int, ...]]:
    """
    visited 用の状態キー。

    同じ盤面でも、使用済み単語が違えば別状態として扱う。
    """
    return (
        board.tobytes(),
        tuple(sorted(used_word_ids)),
    )


def find_moves_encoded(
    board: np.ndarray,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    max_paths_per_word: int,
) -> Tuple[np.ndarray, np.ndarray]:
    moves, scores, count = _find_all_moves(
        board,
        word_array,
        word_lengths,
        int(wild_id),
        int(max_paths_per_word),
        False,
        np.uint64(0),
    )

    if count <= 0:
        return (
            np.zeros((0, 4), dtype=np.int16),
            np.zeros((0,), dtype=np.float64),
        )

    return moves[:count].copy(), scores[:count].copy()


def filter_used_words(
    moves: np.ndarray,
    scores: np.ndarray,
    used_word_ids: set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用済み word_id を候補手から除外する。
    """
    if moves.shape[0] == 0:
        return moves, scores

    if not used_word_ids:
        return moves, scores

    keep_indices: List[int] = []

    for i in range(moves.shape[0]):
        wid = int(moves[i, 0])

        if wid not in used_word_ids:
            keep_indices.append(i)

    if not keep_indices:
        return (
            np.zeros((0, 4), dtype=np.int16),
            np.zeros((0,), dtype=np.float64),
        )

    idx = np.array(keep_indices, dtype=np.int64)

    return moves[idx], scores[idx]


def heuristic_order_moves(
    moves: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if moves.shape[0] == 0:
        return moves, scores

    ordered_moves = moves.copy()
    ordered_scores = scores.copy()

    _sort_moves_by_score(
        ordered_moves,
        ordered_scores,
        ordered_moves.shape[0],
    )

    return ordered_moves, ordered_scores


def ensure_moves_2d(moves: np.ndarray) -> np.ndarray:
    """
    encoded moves を必ず shape=(N, 4) の int16 配列に揃える。

    Policy inference 側から shape=(N, 1, 4) や
    shape=(1, 4) が返ってきても、Numba solver に渡せる形に直す。
    """
    arr = np.asarray(moves, dtype=np.int16)

    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int16)

    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError(f"invalid move shape: {arr.shape}")

        return arr.reshape(1, 4).copy()

    return arr.reshape(-1, 4).copy()


def ensure_scores_1d(
    scores: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    scores を必ず shape=(N,) の float32 配列に揃える。
    """
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)

    if arr.shape[0] == n:
        return arr.copy()

    if arr.shape[0] > n:
        return arr[:n].copy()

    out = np.zeros((n,), dtype=np.float32)
    out[:arr.shape[0]] = arr

    return out


def encoded_move_to_path(
    move: np.ndarray,
    word_lengths: np.ndarray,
) -> List[Tuple[int, int]]:
    move = np.asarray(move, dtype=np.int16).reshape(4)

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
) -> Dict[str, Any]:
    move = np.asarray(move, dtype=np.int16).reshape(4)

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

    return out


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
        policy_min_branch: int = 3,
        policy_weight: float = 0.75,
        heuristic_weight: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self.policy = policy
        self.max_nodes = int(max_nodes)
        self.max_depth = int(max_depth)
        self.max_paths_per_word = int(max_paths_per_word)
        self.use_policy = bool(use_policy)
        self.fallback_to_heuristic = bool(fallback_to_heuristic)
        self.use_visited = bool(use_visited)
        self.policy_min_branch = int(policy_min_branch)
        self.policy_weight = float(policy_weight)
        self.heuristic_weight = float(heuristic_weight)
        self.verbose = bool(verbose)

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
        moves = ensure_moves_2d(moves)

        heuristic_scores = ensure_scores_1d(
            heuristic_scores,
            moves.shape[0],
        )

        if moves.shape[0] == 0:
            return moves, heuristic_scores.astype(np.float32)

        if not self.use_policy or moves.shape[0] < self.policy_min_branch:
            ordered, scores = heuristic_order_moves(
                moves,
                heuristic_scores,
            )

            ordered = ensure_moves_2d(ordered)
            scores = ensure_scores_1d(scores, ordered.shape[0])

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
                policy_weight=self.policy_weight,
                heuristic_weight=self.heuristic_weight,
            )

            ordered_moves = ensure_moves_2d(ordered_moves)

            policy_scores = ensure_scores_1d(
                policy_scores,
                ordered_moves.shape[0],
            )

            return ordered_moves, policy_scores.astype(np.float32)

        except Exception as e:
            if self.verbose:
                print("[PolicySolver] policy ordering failed:", e)

            if not self.fallback_to_heuristic:
                raise

            ordered, scores = heuristic_order_moves(
                moves,
                heuristic_scores,
            )

            ordered = ensure_moves_2d(ordered)
            scores = ensure_scores_1d(scores, ordered.shape[0])

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
        used_word_ids: set[int] = set()

        start = time.perf_counter()

        solved, final_board, solved_moves = self._dfs(
            board=np.ascontiguousarray(board.astype(np.int16)),
            word_array=np.ascontiguousarray(word_array.astype(np.int16)),
            word_lengths=np.ascontiguousarray(word_lengths.astype(np.int16)),
            words_text=words_text,
            wild_id=int(wild_id),
            target_f=int(target_f),
            depth=0,
            visited=visited,
            used_word_ids=used_word_ids,
            stats=stats,
        )

        stats["elapsedMs"] = (time.perf_counter() - start) * 1000.0

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
        used_word_ids: set[int],
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

        if is_board_empty(board):
            return True, board, []

        h = state_hash(board, used_word_ids)

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

        moves, heuristic_scores = filter_used_words(
            moves,
            heuristic_scores,
            used_word_ids,
        )

        moves = ensure_moves_2d(moves)
        heuristic_scores = ensure_scores_1d(
            heuristic_scores,
            moves.shape[0],
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

        ordered_moves = ensure_moves_2d(ordered_moves)

        policy_scores = ensure_scores_1d(
            policy_scores,
            ordered_moves.shape[0],
        )

        before_filled = count_filled(board)

        for i in range(ordered_moves.shape[0]):
            move = np.ascontiguousarray(
                ordered_moves[i].reshape(4).astype(np.int16)
            )

            next_board = _remove_move_and_apply_gravity(
                board,
                move,
                word_lengths,
            )

            if count_filled(next_board) >= before_filled:
                continue

            child_visited = set(visited) if self.use_visited else visited

            child_used_word_ids = set(used_word_ids)
            child_used_word_ids.add(int(move[0]))

            solved, final_board, child_moves = self._dfs(
                board=next_board,
                word_array=word_array,
                word_lengths=word_lengths,
        words_text=words_text,
                wild_id=wild_id,
                target_f=target_f,
                depth=depth + 1,
                visited=child_visited,
                used_word_ids=child_used_word_ids,
                stats=stats,
            )

            if solved:
                score = (
                    float(policy_scores[i])
                    if i < len(policy_scores)
                    else None
                )

                move_dict = encoded_move_to_dict(
                    move,
                    word_lengths,
                    words_text,
                    policy_score=score,
                )

                return True, final_board, [move_dict] + child_moves

        stats["backtracks"] += 1

        return False, board, []


def compare_on_replay(
    replay_file: Path,
    *,
    model_dir: Path,
    device: str,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_policy: bool,
    policy_min_branch: int,
    policy_weight: float,
    heuristic_weight: float,
    verbose: bool,
) -> Dict[str, Any]:
    replay = load_replay(replay_file)
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

    normal = solve_numba(
        board,
        word_array,
        word_lengths,
        wild_id,
        words_text,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_word_mask=True,
    )

    solver = PolicySolver(
        policy,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_policy=use_policy,
        policy_min_branch=policy_min_branch,
        policy_weight=policy_weight,
        heuristic_weight=heuristic_weight,
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

    return {
        "replayFile": str(replay_file),
        "targetF": target_f,
        "boardText": board_to_string_with_metadata(board, metadata),
        "normal": normal,
        "policy": policy_result,
        "improvement": {
            "nodes": normal["stats"]["exploredNodes"]
            - policy_result["stats"]["exploredNodes"],
            "backtracks": normal["stats"]["backtracks"]
            - policy_result["stats"]["backtracks"],
            "deadEnds": normal["stats"]["deadEnds"]
            - policy_result["stats"]["deadEnds"],
            "elapsedMs": normal["stats"]["elapsedMs"]
            - policy_result["stats"]["elapsedMs"],
        },
    }


def benchmark_policy_solver(
    files: List[Path],
    *,
    model_dir: Path,
    device: str,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_policy: bool,
    policy_min_branch: int,
    policy_weight: float,
    heuristic_weight: float,
) -> Dict[str, Any]:
    policy = TorchPolicy(
        model_dir=model_dir,
        device=device,
    )
    policy.load()

    rows: List[Dict[str, Any]] = []

    totals = {
        "normalSolved": 0,
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

    by_f: Dict[str, Dict[str, Any]] = {}

    for idx, replay_file in enumerate(files, start=1):
        replay = load_replay(replay_file)
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
            use_word_mask=True,
        )

        solver = PolicySolver(
            policy,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_paths_per_word=max_paths_per_word,
            use_policy=use_policy,
            policy_min_branch=policy_min_branch,
            policy_weight=policy_weight,
            heuristic_weight=heuristic_weight,
        )

        policy_result = solver.solve(
            board,
            word_array,
            word_lengths,
            words_text,
            wild_id,
            target_f=target_f,
        )

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
        row["backtracksImprovement"] = (
            row["normalBacktracks"] - row["policyBacktracks"]
        )
        row["deadEndsImprovement"] = (
            row["normalDeadEnds"] - row["policyDeadEnds"]
        )
        row["elapsedMsImprovement"] = (
            row["normalElapsedMs"] - row["policyElapsedMs"]
        )

        rows.append(row)

        totals["normalSolved"] += 1 if row["normalSolved"] else 0
        totals["policySolved"] += 1 if row["policySolved"] else 0

        for key in [
            "normalNodes",
            "policyNodes",
            "normalBacktracks",
            "policyBacktracks",
            "normalDeadEnds",
            "policyDeadEnds",
            "normalElapsedMs",
            "policyElapsedMs",
        ]:
            totals[key] += row[key]

        fkey = f"f{target_f}"

        if fkey not in by_f:
            by_f[fkey] = {
                "count": 0,

                "normalSolved": 0,
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

        bf = by_f[fkey]

        bf["count"] += 1

        bf["normalSolved"] += 1 if row["normalSolved"] else 0
        bf["policySolved"] += 1 if row["policySolved"] else 0

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
            f"N solved={row['normalSolved']} "
            f"P solved={row['policySolved']} "
            f"N nodes={row['normalNodes']} "
            f"P nodes={row['policyNodes']} "
            f"diff={row['nodesImprovement']} "
            f"N ms={row['normalElapsedMs']:.3f} "
            f"P ms={row['policyElapsedMs']:.3f}"
        )

    n = max(1, len(files))

    summary_by_f: Dict[str, Any] = {}

    for key, data in sorted(by_f.items()):
        count = max(1, int(data["count"]))

        summary_by_f[key] = {
            "count": int(data["count"]),

            "normalSolved": int(data["normalSolved"]),
            "normalSolveRate": data["normalSolved"] / count,

            "policySolved": int(data["policySolved"]),
            "policySolveRate": data["policySolved"] / count,

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
            "avgElapsedMsImprovement": (
                data["normalElapsedMs"] - data["policyElapsedMs"]
            ) / count,
        }

    summary = {
        "files": len(files),

        "normalSolved": int(totals["normalSolved"]),
        "normalSolveRate": totals["normalSolved"] / n,

        "policySolved": int(totals["policySolved"]),
        "policySolveRate": totals["policySolved"] / n,

        "avgNormalNodes": totals["normalNodes"] / n,
        "avgPolicyNodes": totals["policyNodes"] / n,
        "avgNodesImprovement": (
            totals["normalNodes"] - totals["policyNodes"]
        ) / n,

        "avgNormalBacktracks": totals["normalBacktracks"] / n,
        "avgPolicyBacktracks": totals["policyBacktracks"] / n,
        "avgBacktracksImprovement": (
            totals["normalBacktracks"] - totals["policyBacktracks"]
        ) / n,

        "avgNormalDeadEnds": totals["normalDeadEnds"] / n,
        "avgPolicyDeadEnds": totals["policyDeadEnds"] / n,
        "avgDeadEndsImprovement": (
            totals["normalDeadEnds"] - totals["policyDeadEnds"]
        ) / n,

        "avgNormalElapsedMs": totals["normalElapsedMs"] / n,
        "avgPolicyElapsedMs": totals["policyElapsedMs"] / n,
        "avgElapsedMsImprovement": (
            totals["normalElapsedMs"] - totals["policyElapsedMs"]
        ) / n,

        "byF": summary_by_f,
    }

    return {
        "rows": rows,
        "summary": summary,
    }


def print_compare_result(result: Dict[str, Any]) -> None:
    print("\n=== BOARD ===\n")
    print(result["boardText"])

    normal = result["normal"]
    policy = result["policy"]
    imp = result["improvement"]

    ns = normal["stats"]
    ps = policy["stats"]

    print("\n=== NORMAL NUMBA SOLVER ===\n")
    print("Solved:", normal["solved"])
    print("Nodes:", ns["exploredNodes"])
    print("Backtracks:", ns["backtracks"])
    print("DeadEnds:", ns["deadEnds"])
    print("Forced:", ns["forcedMoves"])
    print("Depth:", ns["maxDepth"])
    print("ElapsedMs:", f"{ns['elapsedMs']:.4f}")

    print("\n=== TORCH POLICY SOLVER ===\n")
    print("Solved:", policy["solved"])
    print("UsedPolicy:", policy["usedPolicy"])
    print("Nodes:", ps["exploredNodes"])
    print("Backtracks:", ps["backtracks"])
    print("DeadEnds:", ps["deadEnds"])
    print("Forced:", ps["forcedMoves"])
    print("Depth:", ps["maxDepth"])
    print("ElapsedMs:", f"{ps['elapsedMs']:.4f}")

    print("\n=== IMPROVEMENT normal - policy ===\n")
    print("Nodes:", imp["nodes"])
    print("Backtracks:", imp["backtracks"])
    print("DeadEnds:", imp["deadEnds"])
    print("ElapsedMs:", f"{imp['elapsedMs']:.4f}")

    print("\n=== POLICY SOLVED MOVES ===\n")

    for i, move in enumerate(policy["solvedMoves"], start=1):
        print(
            f"{i:02d}. "
            f"{move['word']} "
            f"{move['direction']} "
            f"({move['row']},{move['col']}) "
            f"path={move['path']} "
            f"policy={move.get('policyScore', 0):.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torch policy-guided solver"
    )

    parser.add_argument("--replay", default=None)
    parser.add_argument("--league", default="f2")
    parser.add_argument("--index", type=int, default=0)

    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--limit", type=int, default=20)

    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--max-depth", type=int, default=120)
    parser.add_argument("--max-paths-per-word", type=int, default=80)

    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--policy-min-branch", type=int, default=3)
    parser.add_argument("--policy-weight", type=float, default=0.75)
    parser.add_argument("--heuristic-weight", type=float, default=0.25)

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()

    if args.benchmark:
        files = list_replay_files(
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
        print("Rule: same word cannot be used more than once")
        print("Visited key: board + used_word_ids")

        result = benchmark_policy_solver(
            files,
            model_dir=model_dir,
            device=args.device,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            max_paths_per_word=args.max_paths_per_word,
            use_policy=not args.no_policy,
            policy_min_branch=args.policy_min_branch,
            policy_weight=args.policy_weight,
            heuristic_weight=args.heuristic_weight,
        )

        print("\n=== SUMMARY ===")
        print(
            json.dumps(
                result["summary"],
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    if args.replay:
        replay_file = Path(args.replay).resolve()
    else:
        files = list_replay_files(
            league=args.league,
        )

        if not files:
            raise FileNotFoundError(
                f"No replay files found: league={args.league}"
            )

        index = max(
            0,
            min(args.index, len(files) - 1),
        )

        replay_file = files[index]

    print("=== COMPARE SOLVERS ON REPLAY ===")
    print("Replay:", replay_file)
    print("Model:", model_dir)
    print("Rule: same word cannot be used more than once")
    print("Visited key: board + used_word_ids")

    result = compare_on_replay(
        replay_file,
        model_dir=model_dir,
        device=args.device,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        max_paths_per_word=args.max_paths_per_word,
        use_policy=not args.no_policy,
        policy_min_branch=args.policy_min_branch,
        policy_weight=args.policy_weight,
        heuristic_weight=args.heuristic_weight,
        verbose=args.verbose,
    )

    print_compare_result(result)


if __name__ == "__main__":
    main()