# python/src/keshimasu_py/core_solver.py

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from numba import njit

from .config import (
    ROWS,
    COLS,
    PLAYABLE_START_ROW,
    MAX_WORD_LENGTH,
)

# Numba のグローバル定数焼き付けを安全にするため、
# 方向定数は core_solver.py 内で固定する。
DIR_H = 0
DIR_V = 1


@njit(cache=True)
def _is_board_empty(board: np.ndarray) -> bool:
    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != 0:
                return False

    return True


@njit(cache=True)
def _count_filled(board: np.ndarray) -> int:
    count = 0

    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != 0:
                count += 1

    return count


@njit(cache=True)
def _cell_matches(
    cell: int,
    target: int,
    wild_id: int,
) -> bool:
    return cell == target or cell == wild_id


@njit(cache=True)
def _can_place_horizontal(
    board: np.ndarray,
    word: np.ndarray,
    length: int,
    row: int,
    col: int,
    wild_id: int,
) -> bool:
    if row < PLAYABLE_START_ROW:
        return False

    if col + length > COLS:
        return False

    for i in range(length):
        cell = int(board[row, col + i])
        target = int(word[i])

        if not _cell_matches(cell, target, wild_id):
            return False

    return True


@njit(cache=True)
def _can_place_vertical(
    board: np.ndarray,
    word: np.ndarray,
    length: int,
    row: int,
    col: int,
    wild_id: int,
) -> bool:
    if row < PLAYABLE_START_ROW:
        return False

    if row + length > ROWS:
        return False

    for i in range(length):
        cell = int(board[row + i, col])
        target = int(word[i])

        if not _cell_matches(cell, target, wild_id):
            return False

    return True


@njit(cache=True)
def _move_score(
    board: np.ndarray,
    word_id: int,
    row: int,
    col: int,
    direction: int,
    length: int,
    wild_id: int,
) -> np.float64:
    length_score = length * 20.0

    if direction == DIR_H:
        avg_row = row
    else:
        avg_row = row + (length - 1) * 0.5

    lower_score = avg_row * 2.0
    horizontal_bonus = 2.0 if direction == DIR_H else 0.0

    wildcard_count = 0

    for i in range(length):
        if direction == DIR_H:
            rr = row
            cc = col + i
        else:
            rr = row + i
            cc = col

        if int(board[rr, cc]) == wild_id:
            wildcard_count += 1

    wildcard_penalty = wildcard_count * 3.0

    return (
        length_score
        + lower_score
        + horizontal_bonus
        - wildcard_penalty
    )


@njit(cache=True)
def _find_all_moves(
    board: np.ndarray,
    words: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    max_paths_per_word: int,
    use_word_mask: bool,
    used_mask: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    全候補を列挙する低レベル関数。

    注意:
      use_word_mask / used_mask は旧互換用。
      64語を超える語彙では不十分なので、DFS本体では used_words 配列で再使用禁止する。
    """
    num_words = words.shape[0]
    max_moves = num_words * max_paths_per_word

    moves = np.zeros((max_moves, 4), dtype=np.int16)
    scores = np.zeros((max_moves,), dtype=np.float64)

    count = 0

    for wid in range(num_words):
        if use_word_mask:
            if wid >= 64:
                continue

            bit = np.uint64(1) << np.uint64(wid)

            if (used_mask & bit) != 0:
                continue

        length = int(word_lengths[wid])

        if length < 2 or length > MAX_WORD_LENGTH:
            continue

        per_word_count = 0

        for r in range(PLAYABLE_START_ROW, ROWS):
            for c in range(COLS):
                if per_word_count < max_paths_per_word:
                    if _can_place_horizontal(
                        board,
                        words[wid],
                        length,
                        r,
                        c,
                        wild_id,
                    ):
                        if count < max_moves:
                            moves[count, 0] = wid
                            moves[count, 1] = r
                            moves[count, 2] = c
                            moves[count, 3] = DIR_H

                            scores[count] = _move_score(
                                board,
                                wid,
                                r,
                                c,
                                DIR_H,
                                length,
                                wild_id,
                            )

                            count += 1
                            per_word_count += 1

                if per_word_count < max_paths_per_word:
                    if _can_place_vertical(
                        board,
                        words[wid],
                        length,
                        r,
                        c,
                        wild_id,
                    ):
                        if count < max_moves:
                            moves[count, 0] = wid
                            moves[count, 1] = r
                            moves[count, 2] = c
                            moves[count, 3] = DIR_V

                            scores[count] = _move_score(
                                board,
                                wid,
                                r,
                                c,
                                DIR_V,
                                length,
                                wild_id,
                            )

                            count += 1
                            per_word_count += 1

    return moves, scores, count


@njit(cache=True)
def _sort_moves_by_score(
    moves: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> None:
    for i in range(1, count):
        key_score = scores[i]

        key0 = moves[i, 0]
        key1 = moves[i, 1]
        key2 = moves[i, 2]
        key3 = moves[i, 3]

        j = i - 1

        while j >= 0 and scores[j] < key_score:
            scores[j + 1] = scores[j]

            moves[j + 1, 0] = moves[j, 0]
            moves[j + 1, 1] = moves[j, 1]
            moves[j + 1, 2] = moves[j, 2]
            moves[j + 1, 3] = moves[j, 3]

            j -= 1

        scores[j + 1] = key_score

        moves[j + 1, 0] = key0
        moves[j + 1, 1] = key1
        moves[j + 1, 2] = key2
        moves[j + 1, 3] = key3


@njit(cache=True)
def _apply_gravity(board: np.ndarray) -> np.ndarray:
    out = np.zeros((ROWS, COLS), dtype=np.int16)

    for c in range(COLS):
        write_r = ROWS - 1

        for r in range(ROWS - 1, -1, -1):
            v = board[r, c]

            if v != 0:
                out[write_r, c] = v
                write_r -= 1

    return out


@njit(cache=True)
def _remove_move_and_apply_gravity(
    board: np.ndarray,
    move: np.ndarray,
    word_lengths: np.ndarray,
) -> np.ndarray:
    next_board = board.copy()

    wid = int(move[0])
    row = int(move[1])
    col = int(move[2])
    direction = int(move[3])

    length = int(word_lengths[wid])

    for i in range(length):
        if direction == DIR_H:
            rr = row
            cc = col + i
        else:
            rr = row + i
            cc = col

        next_board[rr, cc] = 0

    return _apply_gravity(next_board)


@njit(cache=True)
def _dfs_solve(
    board: np.ndarray,
    words: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    enforce_no_repeat: bool,
    used_words: np.ndarray,
    depth: int,
    stats: np.ndarray,
    branch_hist: np.ndarray,
    solution: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    """
    stats:
      0 exploredNodes
      1 backtracks
      2 deadEnds
      3 forcedMoves
      4 maxDepth
    """
    stats[0] += 1

    if depth > stats[4]:
        stats[4] = depth

    if stats[0] >= max_nodes:
        stats[2] += 1
        return False, board

    if depth > max_depth:
        stats[2] += 1
        return False, board

    if _is_board_empty(board):
        return True, board

    moves, scores, move_count = _find_all_moves(
        board,
        words,
        word_lengths,
        wild_id,
        max_paths_per_word,
        False,
        np.uint64(0),
    )

    legal_count = 0

    for i in range(move_count):
        wid = int(moves[i, 0])

        if enforce_no_repeat and used_words[wid] != 0:
            continue

        legal_count += 1

    hist_index = legal_count

    if hist_index >= branch_hist.shape[0]:
        hist_index = branch_hist.shape[0] - 1

    branch_hist[hist_index] += 1

    if legal_count == 0:
        stats[2] += 1
        return False, board

    if legal_count == 1:
        stats[3] += 1

    _sort_moves_by_score(moves, scores, move_count)

    before_filled = _count_filled(board)

    for i in range(move_count):
        move = moves[i]
        wid = int(move[0])

        if enforce_no_repeat and used_words[wid] != 0:
            continue

        next_board = _remove_move_and_apply_gravity(
            board,
            move,
            word_lengths,
        )

        after_filled = _count_filled(next_board)

        if after_filled >= before_filled:
            continue

        next_used_words = used_words.copy()

        if enforce_no_repeat:
            next_used_words[wid] = 1

        solved, final_board = _dfs_solve(
            next_board,
            words,
            word_lengths,
            wild_id,
            max_nodes,
            max_depth,
            max_paths_per_word,
            enforce_no_repeat,
            next_used_words,
            depth + 1,
            stats,
            branch_hist,
            solution,
        )

        if solved:
            solution[depth, 0] = move[0]
            solution[depth, 1] = move[1]
            solution[depth, 2] = move[2]
            solution[depth, 3] = move[3]

            return True, final_board

    stats[1] += 1

    return False, board


def move_to_path(
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


def move_to_dict(
    move: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
) -> Dict[str, Any]:
    wid = int(move[0])
    direction = int(move[3])

    return {
        "word_id": wid,
        "word": words_text[wid],
        "row": int(move[1]),
        "col": int(move[2]),
        "direction": "H" if direction == DIR_H else "V",
        "path": move_to_path(move, word_lengths),
    }


def solve_numba(
    board: np.ndarray,
    words: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    words_text: List[str] | None = None,
    max_nodes: int = 500_000,
    max_depth: int = 120,
    max_paths_per_word: int = 80,
    use_word_mask: bool = True,
) -> Dict[str, Any]:
    """
    use_word_mask:
      True の場合、同じ word_id の再使用を禁止する。
      旧名との互換のため use_word_mask という名前を残している。
    """
    if words_text is None:
        words_text = [str(i) for i in range(words.shape[0])]

    board = np.ascontiguousarray(board.astype(np.int16))
    words = np.ascontiguousarray(words.astype(np.int16))
    word_lengths = np.ascontiguousarray(word_lengths.astype(np.int16))

    stats = np.zeros((5,), dtype=np.int64)
    branch_hist = np.zeros((2048,), dtype=np.int64)

    solution = np.zeros((max_depth + 1, 4), dtype=np.int16)
    solution[:, :] = -1

    used_words = np.zeros((words.shape[0],), dtype=np.uint8)

    start = time.perf_counter()

    solved, final_board = _dfs_solve(
        board,
        words,
        word_lengths,
        int(wild_id),
        int(max_nodes),
        int(max_depth),
        int(max_paths_per_word),
        bool(use_word_mask),
        used_words,
        0,
        stats,
        branch_hist,
        solution,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    solved_moves: List[Dict[str, Any]] = []

    if solved:
        for d in range(max_depth + 1):
            if solution[d, 0] < 0:
                break

            solved_moves.append(
                move_to_dict(
                    solution[d],
                    word_lengths,
                    words_text,
                )
            )

    hist_dict: Dict[int, int] = {}

    for i in range(branch_hist.shape[0]):
        if branch_hist[i] != 0:
            hist_dict[i] = int(branch_hist[i])

    return {
        "solved": bool(solved),
        "board": final_board,
        "stats": {
            "exploredNodes": int(stats[0]),
            "backtracks": int(stats[1]),
            "deadEnds": int(stats[2]),
            "forcedMoves": int(stats[3]),
            "maxDepth": int(stats[4]),
            "elapsedMs": elapsed_ms,
            "branchingHistogram": hist_dict,
        },
        "solvedMoves": solved_moves,
    }


def find_all_moves_py(
    board: np.ndarray,
    words: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    words_text: List[str],
    max_paths_per_word: int = 80,
    used_word_ids: set[int] | None = None,
) -> List[Dict[str, Any]]:
    moves, scores, count = _find_all_moves(
        np.ascontiguousarray(board.astype(np.int16)),
        np.ascontiguousarray(words.astype(np.int16)),
        np.ascontiguousarray(word_lengths.astype(np.int16)),
        int(wild_id),
        int(max_paths_per_word),
        False,
        np.uint64(0),
    )

    _sort_moves_by_score(moves, scores, count)

    if used_word_ids is None:
        used_word_ids = set()

    out: List[Dict[str, Any]] = []

    for i in range(count):
        wid = int(moves[i, 0])

        if wid in used_word_ids:
            continue

        move = move_to_dict(
            moves[i],
            word_lengths,
            words_text,
        )

        move["score"] = float(scores[i])
        out.append(move)

    return out