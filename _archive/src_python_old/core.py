# src/core.py

import numpy as np
from numba import njit, types
from numba.typed import Dict
import time

# ======================================================
# 定数
# ======================================================

EMPTY = 0
WILD_F = 1

ROWS = 8
COLS = 5

BOARD_STATE_TYPE = types.UniTuple(types.uint64, 6)


# utils.py から必要な関数を読み込み
from utils import (
    VISIBLE_START_ROW,
    ID_TO_CHAR,
    CHAR_TO_ID,
    encode_board,
    prepare_words_matrix,
)

# ======================================================
# 重力処理
# ======================================================

@njit(cache=True, fastmath=True)
def apply_gravity_in_place(board):
    """
    消去後の重力処理
    """

    for c in range(COLS):

        chars = np.zeros(ROWS, dtype=np.int8)
        char_idx = 0

        # 下から収集
        for r in range(ROWS - 1, -1, -1):

            if board[r, c] != EMPTY:
                chars[char_idx] = board[r, c]
                char_idx += 1

        # 下から再配置
        idx = 0

        for r in range(ROWS - 1, -1, -1):

            if idx < char_idx:
                board[r, c] = chars[idx]
                idx += 1
            else:
                board[r, c] = EMPTY


# ======================================================
# 手の適用
# ======================================================

@njit(cache=True, fastmath=True)
def apply_move(board, cand):
    """
    cand:
    [w_idx, r1, c1, r2, c2, direction, word_len]
    """

    next_board = board.copy()

    r1 = cand[1]
    c1 = cand[2]
    r2 = cand[3]
    c2 = cand[4]

    direction = cand[5]

    # 横
    if direction == 0:

        for c in range(c1, c2 + 1):
            next_board[r1, c] = EMPTY

    # 縦
    else:

        for r in range(r1, r2 + 1):
            next_board[r, c1] = EMPTY

    apply_gravity_in_place(next_board)

    return next_board


# ======================================================
# 候補列挙
# ======================================================

@njit(cache=True, fastmath=True)
def get_raw_candidates(board, words_matrix, word_lengths):

    cands = np.zeros((1000, 7), dtype=np.int32)
    cand_idx = 0

    num_words = words_matrix.shape[0]

    # ==================================================
    # 横探索
    # ==================================================

    for r in range(VISIBLE_START_ROW, ROWS):

        for c in range(COLS):

            for w_idx in range(num_words):

                w_len = word_lengths[w_idx]

                if c + w_len > COLS:
                    continue

                match = True

                for i in range(w_len):

                    board_char = board[r, c + i]
                    word_char = words_matrix[w_idx, i]

                    if (
                        board_char != word_char
                        and board_char != WILD_F
                        and word_char != WILD_F
                    ):
                        match = False
                        break

                if match:

                    if cand_idx < 1000:

                        cands[cand_idx, 0] = w_idx
                        cands[cand_idx, 1] = r
                        cands[cand_idx, 2] = c
                        cands[cand_idx, 3] = r
                        cands[cand_idx, 4] = c + w_len - 1
                        cands[cand_idx, 5] = 0
                        cands[cand_idx, 6] = w_len

                        cand_idx += 1

    # ==================================================
    # 縦探索
    # ==================================================

    for c in range(COLS):

        for r in range(VISIBLE_START_ROW, ROWS):

            for w_idx in range(num_words):

                w_len = word_lengths[w_idx]

                if r + w_len > ROWS:
                    continue

                match = True

                for i in range(w_len):

                    board_char = board[r + i, c]
                    word_char = words_matrix[w_idx, i]

                    if (
                        board_char != word_char
                        and board_char != WILD_F
                        and word_char != WILD_F
                    ):
                        match = False
                        break

                if match:

                    if cand_idx < 1000:

                        cands[cand_idx, 0] = w_idx
                        cands[cand_idx, 1] = r
                        cands[cand_idx, 2] = c
                        cands[cand_idx, 3] = r + w_len - 1
                        cands[cand_idx, 4] = c
                        cands[cand_idx, 5] = 1
                        cands[cand_idx, 6] = w_len

                        cand_idx += 1

    return cands[:cand_idx]


# ======================================================
# 盤面ハッシュ
# ======================================================

@njit(cache=True)
def compute_board_hash(board):

    vals = np.zeros(6, dtype=np.uint64)

    for c in range(COLS):

        val = np.uint64(0)

        for r in range(ROWS):
            val = (val << np.uint64(6)) | np.uint64(board[r, c])

        vals[c] = val

    return (
        vals[0],
        vals[1],
        vals[2],
        vals[3],
        vals[4],
        vals[5],
    )


# ======================================================
# 評価関数
# ======================================================

@njit(cache=True, fastmath=True)
def evaluate_board(board, weights):

    score = 0.0

    # ==================================================
    # 空列ボーナス
    # ==================================================

    empty_cols = 0

    for c in range(COLS):

        is_empty = True

        for r in range(ROWS):

            if board[r, c] != EMPTY:
                is_empty = False
                break

        if is_empty:
            empty_cols += 1

    score += empty_cols * weights[0]

    # ==================================================
    # 表面凹凸
    # ==================================================

    heights = np.zeros(COLS, dtype=np.int32)

    for c in range(COLS):

        h = 0

        for r in range(ROWS):

            if board[r, c] != EMPTY:
                h = ROWS - r
                break

        heights[c] = h

    surface_diff = 0

    for c in range(COLS - 1):
        surface_diff += abs(heights[c] - heights[c + 1])

    score -= surface_diff * weights[1]

    return score


# ======================================================
# DFS ソルバー
# ======================================================
@njit(cache=True)
def dfs_solver(
    initial_board,
    words_matrix,
    word_lengths,
    weights,
    max_nodes=100000,
    lookahead=100,
):

    visited = Dict.empty(
        key_type=BOARD_STATE_TYPE,
        value_type=types.boolean,
    )    # ==================================================
    # スタック
    # ==================================================

    stack_board = np.zeros((40, ROWS, COLS), dtype=np.int8)
    stack_mask = np.zeros(40, dtype=np.uint64)
    stack_depth = np.zeros(40, dtype=np.int32)
    stack_path_len = np.zeros(40, dtype=np.int32)

    history_paths = np.zeros((40, 20, 7), dtype=np.int32)

    # 初期状態
    stack_board[0] = initial_board
    stack_mask[0] = 0
    stack_depth[0] = 0
    stack_path_len[0] = 0

    stack_p = 1

    node_count = 0

    solved = False

    result_path = np.zeros((20, 7), dtype=np.int32)
    result_len = 0

    # ==================================================
    # DFS
    # ==================================================

    while stack_p > 0:

        node_count += 1

        if node_count > max_nodes:
            break

        # ポップ
        stack_p -= 1

        curr_board = stack_board[stack_p]
        curr_mask = stack_mask[stack_p]
        curr_depth = stack_depth[stack_p]
        curr_path_len = stack_path_len[stack_p]

        # ==================================================
        # 全消し判定
        # ==================================================

        is_empty = True

        for r in range(ROWS):

            for c in range(COLS):

                if curr_board[r, c] != EMPTY:
                    is_empty = False
                    break

            if not is_empty:
                break

        if is_empty:

            solved = True

            result_len = curr_path_len

            for i in range(curr_path_len):
                result_path[i] = history_paths[stack_p, i]

            break

        # ==================================================
        # visited 判定
        # ==================================================

        h_key = compute_board_hash(curr_board)

        final_key = (
            h_key[0],
            h_key[1],
            h_key[2],
            h_key[3],
            h_key[4],
            np.uint64(curr_mask),
        )

        if final_key in visited:
            continue

        visited[final_key] = True

        # ==================================================
        # 候補生成
        # ==================================================

        cands = get_raw_candidates(
            curr_board,
            words_matrix,
            word_lengths,
        )

        if len(cands) == 0:
            continue

        # ==================================================
        # 評価
        # ==================================================

        scores = np.zeros(len(cands), dtype=np.float64)

        for i in range(len(cands)):

            w_idx = cands[i, 0]

            # 使用済み単語
            if (curr_mask & (np.uint64(1) << np.uint64(w_idx))) != 0:
                scores[i] = -999999.0
                continue

            nb = apply_move(curr_board, cands[i])

            scores[i] = evaluate_board(nb, weights)

        sorted_indices = np.argsort(scores)

        start_loop = 0

        if len(sorted_indices) > lookahead:
            start_loop = len(sorted_indices) - lookahead

        # ==================================================
        # 子ノード生成
        # ==================================================

        for s_i in range(start_loop, len(sorted_indices)):

            idx = sorted_indices[s_i]

            if scores[idx] < -900000.0:
                continue

            w_idx = cands[idx, 0]

            next_board = apply_move(curr_board, cands[idx])

            next_mask = curr_mask | (
                np.uint64(1) << np.uint64(w_idx)
            )

            if stack_p >= 40:
                continue

            stack_board[stack_p] = next_board
            stack_mask[stack_p] = next_mask
            stack_depth[stack_p] = curr_depth + 1
            stack_path_len[stack_p] = curr_path_len + 1

            # 履歴コピー
            for p_idx in range(curr_path_len):
                history_paths[stack_p, p_idx] = (
                    history_paths[stack_p - 1, p_idx]
                )

            history_paths[stack_p, curr_path_len] = cands[idx]

            stack_p += 1

    return solved, node_count, result_path[:result_len]


# ======================================================
# ラッパークラス
# ======================================================

class KeshimasuSolver:

    def __init__(self):
        pass

    def solve(
        self,
        board_strs,
        active_words,
        weights_np,
        max_nodes=50000,
        lookahead=100,
    ):

        start_time = time.time()

        # ==================================================
        # エンコード
        # ==================================================

        board_np = encode_board(board_strs)

        words_matrix, word_lengths = prepare_words_matrix(
            active_words
        )

        # ==================================================
        # Solver
        # ==================================================

        solved, nodes, path_arr = dfs_solver(
            board_np,
            words_matrix,
            word_lengths,
            weights_np,
            max_nodes=max_nodes,
            lookahead=lookahead,
        )

        elapsed = (time.time() - start_time) * 1000.0

        # ==================================================
        # decode
        # ==================================================

        solution = []

        if solved:

            for step in path_arr:

                w_idx = int(step[0])

                solution.append({
                    "word": active_words[w_idx],
                    "row1": int(step[1]),
                    "col1": int(step[2]),
                    "row2": int(step[3]),
                    "col2": int(step[4]),
                    "direction": "H" if step[5] == 0 else "V",
                })

        return {
            "solved": solved,
            "nodes": int(nodes),
            "time_ms": elapsed,
            "solution": solution,
        }

# ======================================================
# テスト
# ======================================================

if __name__ == "__main__":

    print("=== Numba Solver Test ===")

    test_board_strs = [
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['フ', '.', 'シ', 'ル', '.'],
        ['ラ', 'ン', 'ス', 'ウ', 'イ'],
        ['イ', 'ン', 'ド', '.', 'ジ'],
        ['ド', 'イ', 'ツ', '.', 'ー'],
        ['ロ', 'シ', 'ア', 'イ', 'ロ']
    ]

    test_active_words = [
        'フランス',
        'インド',
        'ドイツ',
        'ロシア',
        'フィジー',
        'カイロ'
    ]

    test_weights = np.array(
        [500.0, 4.0, 80.0, -10000.0],
        dtype=np.float64,
    )

    solver = KeshimasuSolver()

    print("JIT compiling...")

    result = solver.solve(
        test_board_strs,
        test_active_words,
        test_weights,
        max_nodes=10000,
        lookahead=50,
    )

    print("\n=== RESULT ===")

    print("Solved :", result["solved"])
    print("Nodes  :", result["nodes"])
    print(
        "Time   :",
        f"{result['time_ms']:.2f} ms"
    )

    if result["solved"]:

        print("\nSolution:")

        for i, step in enumerate(
            result["solution"],
            1,
        ):

            print(
                f"[{i}] "
                f"{step['word']} "
                f"({step['row1']},{step['col1']}) "
                f"-> "
                f"({step['row2']},{step['col2']}) "
                f"{step['direction']}"
            )

    else:

        print("No solution found.")