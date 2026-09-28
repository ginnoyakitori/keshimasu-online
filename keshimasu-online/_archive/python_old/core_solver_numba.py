# python/core_solver_numba.py
#
# Python / NumPy / Numba core solver
#
# 仕様:
# - 盤面: 8 x 5
# - 消去可能エリア: 下5段のみ row 3〜7
# - 単語方向: 左→右 / 上→下 のみ
# - 単語長: 最大5文字
# - EMPTY = 0
# - WILD = "F" を任意文字として扱う
#
# 内部:
# - board: np.int16[8,5]
# - words: np.int16[num_words,5]
# - word_lengths: np.int16[num_words]
# - move: [word_id, row, col, direction]
#   direction 0 = 横 左→右
#   direction 1 = 縦 上→下
#
# 実行:
#   python python/core_solver_numba.py
#

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
from numba import njit


# ============================================================
# Constants
# ============================================================

ROWS = 8
COLS = 5

PLAYABLE_START_ROW = 3
PLAYABLE_ROWS = ROWS - PLAYABLE_START_ROW

MAX_WORD_LENGTH = 5

EMPTY = np.int16(0)

DIR_H = np.int16(0)  # 左 -> 右
DIR_V = np.int16(1)  # 上 -> 下

# Numba側では wildcard id を整数で扱う
# 実際の値は codec 作成時に決まる
# _solve_dfs などには wild_id として渡す


# ============================================================
# Codec
# ============================================================

@dataclass
class Codec:
    char_to_id: Dict[str, int]
    id_to_char: Dict[int, str]
    wild_id: int


def normalize_cell_for_encoding(cell: Any) -> Any:
    """
    表示用文字を内部用に戻す。
    """
    if cell is None:
        return None

    if cell == "・":
        return None

    if cell == "Ｆ":
        return "F"

    return cell


def build_codec(words: List[str], wildcard_char: str = "F") -> Codec:
    """
    文字 -> 整数ID の辞書を作る。

    0 は EMPTY 固定。
    wildcard_char も必ず登録する。
    """
    chars = set()

    for word in words:
        for ch in list(word):
            chars.add(ch)

    chars.add(wildcard_char)

    char_to_id: Dict[str, int] = {}
    id_to_char: Dict[int, str] = {}

    next_id = 1

    for ch in sorted(chars):
        char_to_id[ch] = next_id
        id_to_char[next_id] = ch
        next_id += 1

    wild_id = char_to_id[wildcard_char]

    return Codec(
        char_to_id=char_to_id,
        id_to_char=id_to_char,
        wild_id=wild_id,
    )


def encode_words(
    words: List[str],
    codec: Codec,
    max_word_length: int = MAX_WORD_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    単語を int16 配列へ変換する。

    戻り値:
      word_array: shape=(num_words, 5)
      word_lengths: shape=(num_words,)
      filtered_words: 実際に使った単語文字列
    """
    filtered = [
        w for w in words
        if 2 <= len(list(w)) <= max_word_length
    ]

    word_array = np.zeros(
        (len(filtered), max_word_length),
        dtype=np.int16,
    )

    word_lengths = np.zeros(
        (len(filtered),),
        dtype=np.int16,
    )

    for wi, word in enumerate(filtered):
        chars = list(word)
        word_lengths[wi] = len(chars)

        for i, ch in enumerate(chars):
            word_array[wi, i] = codec.char_to_id[ch]

    return word_array, word_lengths, filtered


def encode_board(
    rows: List[Any],
    codec: Codec,
) -> np.ndarray:
    """
    盤面を int16[8,5] に変換する。

    rows は以下のどちらでもOK:
      ["ア イ ウ エ オ", ...]
      [["ア","イ","ウ","エ","オ"], ...]
    """
    board = np.zeros((ROWS, COLS), dtype=np.int16)

    if len(rows) != ROWS:
        raise ValueError(f"board must have {ROWS} rows")

    for r in range(ROWS):
        row = rows[r]

        if isinstance(row, str):
            cells = row.split()
        else:
            cells = list(row)

        if len(cells) != COLS:
            raise ValueError(
                f"row {r} must have {COLS} cells: {cells}"
            )

        for c in range(COLS):
            cell = normalize_cell_for_encoding(cells[c])

            if cell is None:
                board[r, c] = EMPTY
            else:
                if cell not in codec.char_to_id:
                    raise ValueError(f"unknown char: {cell}")

                board[r, c] = codec.char_to_id[cell]

    return board


def decode_cell(
    value: int,
    codec: Codec,
    display: bool = True,
) -> str:
    if value == 0:
        return "・" if display else ""

    ch = codec.id_to_char.get(int(value), "?")

    if display and ch == "F":
        return "Ｆ"

    return ch


def board_to_string(board: np.ndarray, codec: Codec) -> str:
    lines = []

    for r in range(board.shape[0]):
        cells = [
            decode_cell(int(board[r, c]), codec, display=True)
            for c in range(board.shape[1])
        ]
        lines.append(" ".join(cells))

    return "\n".join(lines)


# ============================================================
# Numba core
# ============================================================

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
def _cell_matches(cell: np.int16, target: np.int16, wild_id: np.int16) -> bool:
    return cell == target or cell == wild_id


@njit(cache=True)
def _can_place_horizontal(
    board: np.ndarray,
    word: np.ndarray,
    length: np.int16,
    row: int,
    col: int,
    wild_id: np.int16,
) -> bool:
    if row < PLAYABLE_START_ROW:
        return False

    if col + length > COLS:
        return False

    for i in range(length):
        cell = board[row, col + i]
        target = word[i]

        if not _cell_matches(cell, target, wild_id):
            return False

    return True


@njit(cache=True)
def _can_place_vertical(
    board: np.ndarray,
    word: np.ndarray,
    length: np.int16,
    row: int,
    col: int,
    wild_id: np.int16,
) -> bool:
    if row < PLAYABLE_START_ROW:
        return False

    if row + length > ROWS:
        return False

    for i in range(length):
        cell = board[row + i, col]
        target = word[i]

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
    wild_id: np.int16,
) -> np.float64:
    """
    JS側の scoreMove に近い簡易ヒューリスティック。
    """
    length_score = length * 20.0

    # 平均row
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

        if board[rr, cc] == wild_id:
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
    wild_id: np.int16,
    max_paths_per_word: int,
    use_word_mask: bool,
    used_mask: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    全合法 move を列挙する。

    moves[count] = [word_id, row, col, direction]
    scores[count] = heuristic score
    """
    num_words = words.shape[0]

    max_moves = num_words * max_paths_per_word

    moves = np.zeros((max_moves, 4), dtype=np.int16)
    scores = np.zeros((max_moves,), dtype=np.float64)

    count = 0

    for wid in range(num_words):
        # optional: 64単語までの bit mask
        if use_word_mask:
            if wid >= 64:
                # 64以上はこの簡易maskでは扱わない
                continue

            bit = np.uint64(1) << np.uint64(wid)

            if (used_mask & bit) != 0:
                continue

        length = word_lengths[wid]

        if length < 2 or length > MAX_WORD_LENGTH:
            continue

        per_word_count = 0

        for r in range(PLAYABLE_START_ROW, ROWS):
            for c in range(COLS):
                # 横
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

                # 縦
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
    """
    descending insertion sort.
    count は通常そこまで巨大ではないためこれで十分。
    """
    for i in range(1, count):
        key_score = scores[i]
        key_move0 = moves[i, 0]
        key_move1 = moves[i, 1]
        key_move2 = moves[i, 2]
        key_move3 = moves[i, 3]

        j = i - 1

        while j >= 0 and scores[j] < key_score:
            scores[j + 1] = scores[j]
            moves[j + 1, 0] = moves[j, 0]
            moves[j + 1, 1] = moves[j, 1]
            moves[j + 1, 2] = moves[j, 2]
            moves[j + 1, 3] = moves[j, 3]
            j -= 1

        scores[j + 1] = key_score
        moves[j + 1, 0] = key_move0
        moves[j + 1, 1] = key_move1
        moves[j + 1, 2] = key_move2
        moves[j + 1, 3] = key_move3


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
    wild_id: np.int16,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
    use_word_mask: bool,
    used_mask: np.uint64,
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
        use_word_mask,
        used_mask,
    )

    hist_index = move_count

    if hist_index >= branch_hist.shape[0]:
        hist_index = branch_hist.shape[0] - 1

    branch_hist[hist_index] += 1

    if move_count == 0:
        stats[2] += 1
        return False, board

    if move_count == 1:
        stats[3] += 1

    _sort_moves_by_score(moves, scores, move_count)

    before_filled = _count_filled(board)

    for i in range(move_count):
        move = moves[i]

        next_board = _remove_move_and_apply_gravity(
            board,
            move,
            word_lengths,
        )

        after_filled = _count_filled(next_board)

        if after_filled >= before_filled:
            continue

        next_mask = used_mask

        if use_word_mask:
            wid = int(move[0])

            if wid < 64:
                next_mask = used_mask | (
                    np.uint64(1) << np.uint64(wid)
                )

        solved, final_board = _dfs_solve(
            next_board,
            words,
            word_lengths,
            wild_id,
            max_nodes,
            max_depth,
            max_paths_per_word,
            use_word_mask,
            next_mask,
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

# ============================================================
# Public solver wrapper
# ============================================================

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
    row = int(move[1])
    col = int(move[2])
    direction = int(move[3])

    return {
        "word_id": wid,
        "word": words_text[wid],
        "row": row,
        "col": col,
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
    use_word_mask: bool = False,
) -> Dict[str, Any]:
    """
    Numba core solver の Python wrapper。
    """
    if words_text is None:
        words_text = [str(i) for i in range(words.shape[0])]

    board = np.ascontiguousarray(board.astype(np.int16))
    words = np.ascontiguousarray(words.astype(np.int16))
    word_lengths = np.ascontiguousarray(word_lengths.astype(np.int16))

    stats = np.zeros((5,), dtype=np.int64)

    # branch数ヒストグラム
    # index が branch count
    # 最後のindexは overflow用
    branch_hist = np.zeros((2048,), dtype=np.int64)

    solution = np.zeros((max_depth + 1, 4), dtype=np.int16)
    solution[:, :] = -1

    start = time.perf_counter()

    solved, final_board = _dfs_solve(
        board,
        words,
        word_lengths,
        np.int16(wild_id),
        max_nodes,
        max_depth,
        max_paths_per_word,
        use_word_mask,
        np.uint64(0),
        0,
        stats,
        branch_hist,
        solution,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    solved_moves: List[Dict[str, Any]] = []

    if solved:
        # solution[depth] に move が順番通り入っている
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

    # branch_hist を dict 化
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
) -> List[Dict[str, Any]]:
    """
    デバッグ用: Numba候補列挙を Python dict に変換。
    """
    moves, scores, count = _find_all_moves(
        np.ascontiguousarray(board.astype(np.int16)),
        np.ascontiguousarray(words.astype(np.int16)),
        np.ascontiguousarray(word_lengths.astype(np.int16)),
        np.int16(wild_id),
        max_paths_per_word,
        False,
        np.uint64(0),
    )

    _sort_moves_by_score(moves, scores, count)

    out = []

    for i in range(count):
        m = move_to_dict(
            moves[i],
            word_lengths,
            words_text,
        )
        m["score"] = float(scores[i])
        out.append(m)

    return out


# ============================================================
# Demo
# ============================================================
def demo() -> None:
    words = [
        "パキスタン",
        "コスタリカ",
        "ニジェール",
        "エクアドル",
        "カンボジア",
        "リトアニア",
        "グアテマラ",
        "スリランカ",
        "アメリカ",
        "イラク",
        "リビア",
        "マルタ",
    ]

    codec = build_codec(words, wildcard_char="F")

    word_array, word_lengths, filtered_words = encode_words(
        words,
        codec,
    )

    # 全8行が左→右で消せる簡単なテスト盤面
    # 下段から消え、上段が落ちてくる。
    board_rows = [
        "パ キ ス タ ン",
        "コ ス タ リ カ",
        "ニ ジ ェ ー ル",
        "エ ク ア ド ル",
        "カ ン ボ ジ ア",
        "リ ト ア ニ ア",
        "グ ア テ マ ラ",
        "ス リ ラ ン カ",
    ]

    board = encode_board(board_rows, codec)

    print("=== INPUT BOARD ===")
    print(board_to_string(board, codec))

    print("\n=== CANDIDATES AT START ===")

    moves = find_all_moves_py(
        board,
        word_array,
        word_lengths,
        codec.wild_id,
        filtered_words,
    )

    for i, m in enumerate(moves[:20], start=1):
        print(
            f"{i:02d}. {m['word']} {m['direction']} "
            f"({m['row']},{m['col']}) path={m['path']} score={m['score']:.2f}"
        )

    print("\n=== SOLVE ===")

    result = solve_numba(
        board,
        word_array,
        word_lengths,
        codec.wild_id,
        filtered_words,
        max_nodes=100_000,
        max_depth=120,
        max_paths_per_word=80,
        use_word_mask=False,
    )

    print("Solved:", result["solved"])
    print("Stats:", result["stats"])

    print("\n=== SOLVED MOVES ===")

    for i, move in enumerate(result["solvedMoves"], start=1):
        print(
            f"{i:02d}. {move['word']} {move['direction']} "
            f"({move['row']},{move['col']}) path={move['path']}"
        )

    print("\n=== FINAL BOARD ===")
    print(board_to_string(result["board"], codec))


if __name__ == "__main__":
    demo()