# python/apps/build_numpy_dataset.py

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
PY_SRC = ROOT / "python" / "src"

if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))


from keshimasu_py.config import (  # noqa: E402
    PROJECT_ROOT,
    ROWS,
    COLS,
    MAX_WORD_LENGTH,
    EMPTY_ID,
)

from keshimasu_py.codec import (  # noqa: E402
    build_codec,
    encode_words,
)

try:
    from keshimasu_py.core_solver import (  # noqa: E402
        find_all_moves_py,
        _remove_move_and_apply_gravity,
    )
except Exception:
    find_all_moves_py = None  # type: ignore

    from keshimasu_py.core_solver import (  # type: ignore  # noqa: E402
        _find_all_moves,
        _remove_move_and_apply_gravity,
    )


FEATURE_DIM = 54
DIR_H = 0
DIR_V = 1
WILDCARD_TEXT = "F"


DEFAULT_REPLAY_ROOT = PROJECT_ROOT / "data" / "training" / "selfplay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "numpy_dataset" / "policy"


# ---------------------------------------------------------------------
# Basic JSON / file helpers
# ---------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def list_json_files(path: Path) -> List[Path]:
    if not path.exists():
        return []

    out: List[Path] = []

    for child in path.iterdir():
        if child.is_dir():
            out.extend(list_json_files(child))
        elif child.is_file() and child.suffix.lower() == ".json":
            out.append(child)

    return sorted(out)


def list_replay_files(
    replay_root: Path,
    league: str,
    limit: int,
) -> List[Path]:
    if league == "all":
        files: List[Path] = []

        for f in ["f0", "f1", "f2", "f3"]:
            files.extend(list_json_files(replay_root / f))
    else:
        files = list_json_files(replay_root / league)

    files = [
        file
        for file in sorted(files)
        if file.name.startswith("epoch-") and file.name.endswith(".json")
    ]

    if limit > 0:
        files = files[:limit]

    return files


# ---------------------------------------------------------------------
# Replay schema helpers
# ---------------------------------------------------------------------


def normalize_cell(cell: Any) -> Optional[str]:
    if cell is None:
        return None

    if cell == "":
        return None

    if cell == "・":
        return None

    if cell == "Ｆ":
        return WILDCARD_TEXT

    return str(cell)


def replay_board(replay: Dict[str, Any]) -> Optional[List[List[Any]]]:
    candidates = [
        replay.get("board"),
        replay.get("initialBoard"),
        replay.get("puzzle", {}).get("board") if isinstance(replay.get("puzzle"), dict) else None,
        replay.get("game", {}).get("board") if isinstance(replay.get("game"), dict) else None,
        replay.get("result", {}).get("board") if isinstance(replay.get("result"), dict) else None,
    ]

    for board in candidates:
        if isinstance(board, list) and board and isinstance(board[0], list):
            return board

    return None


def replay_solution_moves(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [
        replay.get("solvedMoves"),
        replay.get("solutionMoves"),
        replay.get("solverSolutionMoves"),
        replay.get("solver", {}).get("solvedMoves") if isinstance(replay.get("solver"), dict) else None,
        replay.get("stats", {}).get("solvedMoves") if isinstance(replay.get("stats"), dict) else None,
        replay.get("result", {}).get("solvedMoves") if isinstance(replay.get("result"), dict) else None,
    ]

    for moves in candidates:
        if isinstance(moves, list) and moves:
            return [
                move
                for move in moves
                if isinstance(move, dict)
            ]

    return []


def replay_target_f(replay: Dict[str, Any], file: Path) -> int:
    candidates = [
        replay.get("targetWildcards"),
        replay.get("targetF"),
        replay.get("actualWildcards"),
        replay.get("puzzle", {}).get("targetWildcards") if isinstance(replay.get("puzzle"), dict) else None,
        replay.get("puzzle", {}).get("targetF") if isinstance(replay.get("puzzle"), dict) else None,
        replay.get("config", {}).get("targetWildcards") if isinstance(replay.get("config"), dict) else None,
    ]

    for value in candidates:
        if value is None:
            continue

        try:
            return int(value)
        except Exception:
            pass

    parent = file.parent.name

    if parent.startswith("f") and parent[1:].isdigit():
        return int(parent[1:])

    return 0


def replay_words(replay: Dict[str, Any]) -> List[str]:
    candidates = [
        replay.get("words"),
        replay.get("puzzle", {}).get("words") if isinstance(replay.get("puzzle"), dict) else None,
        replay.get("config", {}).get("words") if isinstance(replay.get("config"), dict) else None,
    ]

    for words in candidates:
        if isinstance(words, list) and words:
            return [
                str(word)
                for word in words
                if str(word).strip()
            ]

    return []


def load_words_from_file(path: Path) -> List[str]:
    if not path.exists():
        return []

    data = read_json(path)

    if isinstance(data, list):
        return [
            str(v)
            for v in data
            if str(v).strip()
        ]

    if isinstance(data, dict):
        for key in ["words", "countries", "data"]:
            value = data.get(key)

            if isinstance(value, list):
                return [
                    str(v)
                    for v in value
                    if str(v).strip()
                ]

    return []


def collect_words(
    replay_files: List[Path],
    words_file: Optional[Path],
) -> List[str]:
    words: List[str] = []

    if words_file is not None and words_file.exists():
        words.extend(load_words_from_file(words_file))

    # 既存 metadata があれば優先的に取り込む
    existing_metadata = DEFAULT_OUTPUT_DIR / "metadata.json"

    if existing_metadata.exists():
        try:
            metadata = read_json(existing_metadata)

            if isinstance(metadata, dict) and isinstance(metadata.get("words"), list):
                words.extend(str(w) for w in metadata["words"])
        except Exception:
            pass

    for file in replay_files:
        try:
            replay = read_json(file)
            words.extend(replay_words(replay))
        except Exception:
            pass

    fallback_files = [
        PROJECT_ROOT / "data" / "numpy_dataset" / "policy" / "metadata.json",
        PROJECT_ROOT / "data" / "words" / "countries.json",
        PROJECT_ROOT / "data" / "country_words.json",
        PROJECT_ROOT / "country_words.json",
        PROJECT_ROOT / "country_words_tmp.json",
    ]

    for file in fallback_files:
        if file.exists():
            try:
                data = read_json(file)

                if isinstance(data, dict) and isinstance(data.get("words"), list):
                    words.extend(str(w) for w in data["words"])
                elif isinstance(data, list):
                    words.extend(str(w) for w in data)
            except Exception:
                pass

    words = [
        str(word)
        for word in words
        if 2 <= len(list(str(word))) <= MAX_WORD_LENGTH
    ]

    words = sorted(set(words))

    if not words:
        raise RuntimeError("No words found. Provide --words-file.")

    return words


# ---------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------


def codec_char_to_id(codec: Any) -> Dict[str, int]:
    char_to_id = getattr(codec, "char_to_id", None)

    if isinstance(char_to_id, dict):
        return {
            str(k): int(v)
            for k, v in char_to_id.items()
        }

    char_to_id = getattr(codec, "charToId", None)

    if isinstance(char_to_id, dict):
        return {
            str(k): int(v)
            for k, v in char_to_id.items()
        }

    raise AttributeError("Codec has no char_to_id / charToId")


def codec_id_to_char(codec: Any) -> Dict[str, str]:
    id_to_char = getattr(codec, "id_to_char", None)

    if isinstance(id_to_char, dict):
        return {
            str(k): str(v)
            for k, v in id_to_char.items()
        }

    id_to_char = getattr(codec, "idToChar", None)

    if isinstance(id_to_char, dict):
        return {
            str(k): str(v)
            for k, v in id_to_char.items()
        }

    return {}


def codec_wild_id(codec: Any) -> int:
    if hasattr(codec, "wild_id"):
        return int(getattr(codec, "wild_id"))

    char_to_id = codec_char_to_id(codec)

    if WILDCARD_TEXT in char_to_id:
        return int(char_to_id[WILDCARD_TEXT])

    if "Ｆ" in char_to_id:
        return int(char_to_id["Ｆ"])

    raise KeyError("wild id not found")


def encode_board(
    board_raw: List[List[Any]],
    codec: Any,
) -> np.ndarray:
    char_to_id = codec_char_to_id(codec)

    board = np.full(
        (ROWS, COLS),
        int(EMPTY_ID),
        dtype=np.int16,
    )

    if len(board_raw) != ROWS:
        raise ValueError(f"board rows mismatch: got={len(board_raw)}, expected={ROWS}")

    for r in range(ROWS):
        if len(board_raw[r]) != COLS:
            raise ValueError(f"board cols mismatch at row {r}: got={len(board_raw[r])}, expected={COLS}")

        for c in range(COLS):
            cell = normalize_cell(board_raw[r][c])

            if cell is None:
                board[r, c] = int(EMPTY_ID)
            else:
                if cell not in char_to_id:
                    raise KeyError(f"unknown char: {cell}")

                board[r, c] = int(char_to_id[cell])

    return board


def move_direction_from_path(path: List[List[int]]) -> int:
    if len(path) < 2:
        return DIR_H

    r0, c0 = int(path[0][0]), int(path[0][1])
    r1, c1 = int(path[1][0]), int(path[1][1])

    if r1 == r0 + 1 and c1 == c0:
        return DIR_V

    return DIR_H


def normalize_move_dict(move: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(move)

    path = out.get("path")

    if not isinstance(path, list) or not path:
        row = int(out.get("row", 0))
        col = int(out.get("col", 0))
        direction = out.get("direction", "H")

        if direction in [DIR_V, "V", "v"]:
            d = DIR_V
        else:
            d = DIR_H

        word = str(out.get("word", ""))
        length = len(list(word))

        if d == DIR_H:
            path = [
                [row, col + i]
                for i in range(length)
            ]
        else:
            path = [
                [row + i, col]
                for i in range(length)
            ]

        out["path"] = path

    direction = out.get("direction")

    if direction in [DIR_H, "H", "h"]:
        out["direction"] = "H"
    elif direction in [DIR_V, "V", "v"]:
        out["direction"] = "V"
    else:
        out["direction"] = "V" if move_direction_from_path(out["path"]) == DIR_V else "H"

    if "row" not in out or "col" not in out:
        out["row"] = int(out["path"][0][0])
        out["col"] = int(out["path"][0][1])

    return out


def solution_move_to_encoded(
    move: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> Optional[np.ndarray]:
    move = normalize_move_dict(move)

    word = str(move.get("word", ""))

    if word not in word_to_id:
        return None

    row = int(move["row"])
    col = int(move["col"])
    direction = DIR_V if move.get("direction") == "V" else DIR_H

    return np.array(
        [
            int(word_to_id[word]),
            row,
            col,
            direction,
        ],
        dtype=np.int16,
    )


def encoded_move_key(move: np.ndarray) -> Tuple[int, int, int, int]:
    arr = np.asarray(move, dtype=np.int16).reshape(4)

    return (
        int(arr[0]),
        int(arr[1]),
        int(arr[2]),
        int(arr[3]),
    )


def candidate_dict_to_encoded(
    candidate: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> Optional[np.ndarray]:
    candidate = normalize_move_dict(candidate)

    word = str(candidate.get("word", ""))

    if word not in word_to_id:
        return None

    row = int(candidate["row"])
    col = int(candidate["col"])
    direction = DIR_V if candidate.get("direction") == "V" else DIR_H

    return np.array(
        [
            int(word_to_id[word]),
            row,
            col,
            direction,
        ],
        dtype=np.int16,
    )


# ---------------------------------------------------------------------
# Move generation wrapper
# ---------------------------------------------------------------------


def find_candidates_encoded(
    board: np.ndarray,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    wild_id: int,
    max_paths_per_word: int,
    used_word_ids: set[int],
    word_to_id: Dict[str, int],
) -> np.ndarray:
    encoded_moves: List[np.ndarray] = []

    if find_all_moves_py is not None:
        try:
            candidates = find_all_moves_py(
                board,
                word_array,
                word_lengths,
                int(wild_id),
                words_text,
                max_paths_per_word=max_paths_per_word,
                used_word_ids=used_word_ids,
            )
        except TypeError:
            candidates = find_all_moves_py(
                board,
                word_array,
                word_lengths,
                int(wild_id),
                words_text,
                max_paths_per_word=max_paths_per_word,
            )

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            move = candidate_dict_to_encoded(
                candidate,
                word_to_id,
            )

            if move is None:
                continue

            if int(move[0]) in used_word_ids:
                continue

            encoded_moves.append(move)

    else:
        moves, _scores, count = _find_all_moves(  # type: ignore[name-defined]
            board,
            word_array,
            word_lengths,
            int(wild_id),
            int(max_paths_per_word),
            False,
            np.uint64(0),
        )

        arr = moves[:count].copy().astype(np.int16)

        for move in arr:
            if int(move[0]) in used_word_ids:
                continue

            encoded_moves.append(
                np.asarray(move, dtype=np.int16).reshape(4)
            )

    if not encoded_moves:
        return np.zeros((0, 4), dtype=np.int16)

    return np.stack(encoded_moves, axis=0).astype(np.int16)


# ---------------------------------------------------------------------
# Feature building: X.npy / Y.npy
# ---------------------------------------------------------------------


def normalize_board_flat(
    boards: np.ndarray,
    *,
    char_vocab_size: int,
) -> np.ndarray:
    boards_f = boards.astype(np.float32)

    denom = float(max(1, char_vocab_size))

    return boards_f.reshape((boards_f.shape[0], -1)) / denom


def compute_word_lengths_for_moves(
    moves: np.ndarray,
    word_lengths: np.ndarray,
) -> np.ndarray:
    if moves.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    word_ids = moves[:, 0].astype(np.int64)

    return word_lengths[word_ids].astype(np.float32)


def compute_move_end_positions(
    moves: np.ndarray,
    word_lengths_for_moves: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = moves[:, 1].astype(np.float32)
    cols = moves[:, 2].astype(np.float32)
    dirs = moves[:, 3].astype(np.float32)

    length_minus_1 = np.maximum(
        0.0,
        word_lengths_for_moves.astype(np.float32) - 1.0,
    )

    end_rows = rows + length_minus_1 * dirs
    end_cols = cols + length_minus_1 * (1.0 - dirs)

    return end_rows, end_cols


def build_feature_matrix(
    *,
    boards: np.ndarray,
    moves: np.ndarray,
    branch_counts: np.ndarray,
    target_f: np.ndarray,
    depths: np.ndarray,
    word_lengths: np.ndarray,
    char_vocab_size: int,
    num_words: int,
    rows: int = ROWS,
    cols: int = COLS,
    max_depth_norm: float = 120.0,
    max_branch_norm: float = 256.0,
    max_target_f_norm: float = 3.0,
) -> np.ndarray:
    n = int(boards.shape[0])

    if n == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    board_features = normalize_board_flat(
        boards,
        char_vocab_size=char_vocab_size,
    )

    moves_i16 = moves.astype(np.int16)
    moves_f32 = moves.astype(np.float32)

    word_ids = moves_f32[:, 0]
    move_rows = moves_f32[:, 1]
    move_cols = moves_f32[:, 2]
    directions = moves_f32[:, 3]

    word_len_for_moves = compute_word_lengths_for_moves(
        moves_i16,
        word_lengths.astype(np.int16),
    )

    end_rows, end_cols = compute_move_end_positions(
        moves_i16,
        word_len_for_moves,
    )

    word_id_norm = word_ids / float(max(1, num_words - 1))
    row_norm = move_rows / float(max(1, rows - 1))
    col_norm = move_cols / float(max(1, cols - 1))

    direction_norm = directions
    is_horizontal = (directions == DIR_H).astype(np.float32)
    is_vertical = (directions == DIR_V).astype(np.float32)

    word_len_norm = word_len_for_moves / float(max(1, MAX_WORD_LENGTH))

    target_f_norm = target_f.astype(np.float32) / float(max_target_f_norm)
    depth_norm = depths.astype(np.float32) / float(max_depth_norm)

    branch_norm = (
        np.log1p(branch_counts.astype(np.float32)) /
        math.log1p(max_branch_norm)
    )

    start_index = move_rows * float(cols) + move_cols
    start_index_norm = start_index / float(max(1, rows * cols - 1))

    end_row_norm = end_rows / float(max(1, rows - 1))
    end_col_norm = end_cols / float(max(1, cols - 1))

    filled_ratio = (
        np.count_nonzero(boards != int(EMPTY_ID), axis=(1, 2)).astype(np.float32) /
        float(rows * cols)
    )

    # 【修正箇所1】削れていた4つの変数を正しくリストに並べる
    extra = np.stack(
        [
            word_id_norm,
            row_norm,
            col_norm,
            direction_norm,
            is_horizontal,    # 正しく追加
            is_vertical,      # 正しく追加
            word_len_norm,    # 正しく追加
            target_f_norm,    # 正しく追加
            depth_norm,
            branch_norm,
            start_index_norm,
            end_row_norm,
            end_col_norm,
            filled_ratio,
        ],
        axis=1,
    ).astype(np.float32)

    # 【修正箇所2】消えていた、または名前が変わっていた x の定義を確実に記述する
    x = np.concatenate(
        [
            board_features.astype(np.float32),
            extra,
        ],
        axis=1,
    ).astype(np.float32)

    if x.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Feature dim mismatch: got={x.shape[1]}, expected={FEATURE_DIM}"
        )

    return x

def build_xy_from_arrays(
    *,
    boards: np.ndarray,
    moves: np.ndarray,
    labels: np.ndarray,
    branch_counts: np.ndarray,
    target_f: np.ndarray,
    depths: np.ndarray,
    word_lengths: np.ndarray,
    char_vocab_size: int,
    num_words: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = build_feature_matrix(
        boards=boards,
        moves=moves,
        branch_counts=branch_counts,
        target_f=target_f,
        depths=depths,
        word_lengths=word_lengths,
        char_vocab_size=char_vocab_size,
        num_words=num_words,
    )

    y = labels.astype(np.float32).reshape((-1, 1))

    return x.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------


def process_replay_file(
    *,
    file: Path,
    codec: Any,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    word_to_id: Dict[str, int],
    wild_id: int,
    max_paths_per_word: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], List[int], Dict[str, Any]]:
    replay = read_json(file)

    board_raw = replay_board(replay)
    solution_moves = replay_solution_moves(replay)

    if board_raw is None:
        raise ValueError("missing board")

    if not solution_moves:
        raise ValueError("missing solution moves")

    target_f = replay_target_f(replay, file)

    board = encode_board(
        board_raw,
        codec,
    )

    boards_out: List[np.ndarray] = []
    moves_out: List[np.ndarray] = []
    labels_out: List[int] = []
    branch_counts_out: List[int] = []
    target_f_out: List[int] = []
    depths_out: List[int] = []

    used_word_ids: set[int] = set()

    positive_count = 0
    move_example_count = 0

    for depth, solution_move in enumerate(solution_moves):
        solution_encoded = solution_move_to_encoded(
            solution_move,
            word_to_id,
        )

        if solution_encoded is None:
            raise ValueError(f"solution word not in dictionary: {solution_move.get('word')}")

        if int(solution_encoded[0]) in used_word_ids:
            raise ValueError(f"duplicate word_id in solution: {int(solution_encoded[0])}")

        candidates = find_candidates_encoded(
            board=board,
            word_array=word_array,
            word_lengths=word_lengths,
            words_text=words_text,
            wild_id=wild_id,
            max_paths_per_word=max_paths_per_word,
            used_word_ids=used_word_ids,
            word_to_id=word_to_id,
        )

        solution_key = encoded_move_key(solution_encoded)

        candidate_keys = {
            encoded_move_key(move)
            for move in candidates
        }

        if solution_key not in candidate_keys:
            candidates = np.concatenate(
                [
                    candidates,
                    solution_encoded.reshape(1, 4),
                ],
                axis=0,
            ).astype(np.int16)

        branch_count = int(candidates.shape[0])

        for candidate in candidates:
            candidate_key = encoded_move_key(candidate)
            label = 1 if candidate_key == solution_key else 0

            boards_out.append(board.copy())
            moves_out.append(np.asarray(candidate, dtype=np.int16).reshape(4))
            labels_out.append(label)
            branch_counts_out.append(branch_count)
            target_f_out.append(int(target_f))
            depths_out.append(int(depth))

            move_example_count += 1

            if label == 1:
                positive_count += 1

        # Advance board by the actual solution move.
        next_board = _remove_move_and_apply_gravity(
            board,
            solution_encoded.reshape(4).astype(np.int16),
            word_lengths,
        )

        board = np.asarray(next_board, dtype=np.int16)
        used_word_ids.add(int(solution_encoded[0]))

    info = {
        "file": str(file),
        "targetF": int(target_f),
        "moveExamples": int(move_example_count),
        "policyExamples": int(positive_count),
        "positiveMoves": int(positive_count),
        "solutionSteps": int(len(solution_moves)),
    }

    return (
        boards_out,
        moves_out,
        labels_out,
        branch_counts_out,
        target_f_out,
        depths_out,
        info,
    )


def build_dataset(
    *,
    replay_root: Path,
    output_dir: Path,
    league: str,
    limit: int,
    words_file: Optional[Path],
    max_paths_per_word: int,
) -> Dict[str, Any]:
    files = list_replay_files(
        replay_root,
        league,
        limit,
    )

    if not files:
        raise FileNotFoundError(
            f"No replay files found: root={replay_root}, league={league}"
        )

    words = collect_words(
        files,
        words_file,
    )

    codec = build_codec(words)

    word_array, word_lengths, filtered_words = encode_words(
        words,
        codec,
        max_word_length=MAX_WORD_LENGTH,
    )

    word_array = word_array.astype(np.int16)
    word_lengths = word_lengths.astype(np.int16)
    words_text = list(filtered_words)

    word_to_id = {
        str(word): i
        for i, word in enumerate(words_text)
    }

    char_to_id = codec_char_to_id(codec)
    id_to_char = codec_id_to_char(codec)
    wild_id = codec_wild_id(codec)
    char_vocab_size = int(len(char_to_id) + 1)

    all_boards: List[np.ndarray] = []
    all_moves: List[np.ndarray] = []
    all_labels: List[int] = []
    all_branch_counts: List[int] = []
    all_target_f: List[int] = []
    all_depths: List[int] = []

    skipped: List[Dict[str, str]] = []
    by_f: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "replays": 0,
            "moveExamples": 0,
            "policyExamples": 0,
            "positiveMoves": 0,
        }
    )

    print("=== BUILD NUMPY POLICY DATASET ===")
    print("Replay root:", replay_root)
    print("Output dir :", output_dir)
    print("League     :", league)
    print("Files      :", len(files))
    print("Words      :", len(words_text))
    print("Feature dim:", FEATURE_DIM)

    for idx, file in enumerate(files, start=1):
        try:
            (
                boards,
                moves,
                labels,
                branch_counts,
                target_f,
                depths,
                info,
            ) = process_replay_file(
                file=file,
                codec=codec,
                word_array=word_array,
                word_lengths=word_lengths,
                words_text=words_text,
                word_to_id=word_to_id,
                wild_id=wild_id,
                max_paths_per_word=max_paths_per_word,
            )

            all_boards.extend(boards)
            all_moves.extend(moves)
            all_labels.extend(labels)
            all_branch_counts.extend(branch_counts)
            all_target_f.extend(target_f)
            all_depths.extend(depths)

            fkey = f"f{info['targetF']}"

            by_f[fkey]["replays"] += 1
            by_f[fkey]["moveExamples"] += int(info["moveExamples"])
            by_f[fkey]["policyExamples"] += int(info["policyExamples"])
            by_f[fkey]["positiveMoves"] += int(info["positiveMoves"])

            print(
                f"[{idx}/{len(files)}] OK "
                f"{file.parent.name}/{file.name} "
                f"F={info['targetF']} "
                f"moves={info['moveExamples']} "
                f"pos={info['positiveMoves']}"
            )

        except Exception as e:
            skipped.append(
                {
                    "file": str(file),
                    "reason": str(e),
                }
            )

            print(
                f"[{idx}/{len(files)}] SKIP "
                f"{file.parent.name}/{file.name} "
                f"reason={e}"
            )

    if not all_boards:
        raise RuntimeError("No training examples were built.")

    output_dir.mkdir(parents=True, exist_ok=True)

    boards_np = np.stack(all_boards, axis=0).astype(np.int16)
    moves_np = np.stack(all_moves, axis=0).astype(np.int16)
    labels_np = np.asarray(all_labels, dtype=np.int8)
    branch_counts_np = np.asarray(all_branch_counts, dtype=np.int16)
    target_f_np = np.asarray(all_target_f, dtype=np.int16)
    depths_np = np.asarray(all_depths, dtype=np.int16)

    print()
    print("Saving raw arrays...")

    np.save(output_dir / "boards.npy", boards_np)
    np.save(output_dir / "moves.npy", moves_np)
    np.save(output_dir / "labels.npy", labels_np)
    np.save(output_dir / "branch_counts.npy", branch_counts_np)
    np.save(output_dir / "target_f.npy", target_f_np)
    np.save(output_dir / "depths.npy", depths_np)
    np.save(output_dir / "word_array.npy", word_array)
    np.save(output_dir / "word_lengths.npy", word_lengths)

    print("Building preprocessed X.npy / Y.npy ...")

    x_all, y_all = build_xy_from_arrays(
        boards=boards_np,
        moves=moves_np,
        labels=labels_np,
        branch_counts=branch_counts_np,
        target_f=target_f_np,
        depths=depths_np,
        word_lengths=word_lengths,
        char_vocab_size=char_vocab_size,
        num_words=int(word_array.shape[0]),
    )

    np.save(output_dir / "X.npy", x_all.astype(np.float32))
    np.save(output_dir / "Y.npy", y_all.astype(np.float32))

    summary = {
        "inputReplayFiles": len(files),
        "skippedFiles": len(skipped),
        "moveExamples": int(labels_np.shape[0]),
        "policyExamples": int(np.count_nonzero(labels_np == 1)),
        "positiveMoveExamples": int(np.count_nonzero(labels_np == 1)),
        "negativeMoveExamples": int(np.count_nonzero(labels_np == 0)),
        "rows": ROWS,
        "cols": COLS,
        "maxWordLength": MAX_WORD_LENGTH,
        "moveFormat": [
            "word_id",
            "row",
            "col",
            "direction",
        ],
        "direction": {
            "H": DIR_H,
            "V": DIR_V,
        },
        "featureFormat": {
            "X": "float32",
            "Y": "float32",
            "inputSize": FEATURE_DIM,
            "featureDim": FEATURE_DIM,
            "description": "precomputed policy move features",
        },
        "byF": {
            key: dict(value)
            for key, value in sorted(by_f.items())
        },
        "skipped": skipped,
        "wordReuseRule": "same word_id cannot be used more than once per puzzle",
    }

    metadata = {
        "summary": summary,
        "words": words_text,
        "charToId": {
            str(k): int(v)
            for k, v in char_to_id.items()
        },
        "idToChar": {
            str(k): str(v)
            for k, v in id_to_char.items()
        },
        "wildId": int(wild_id),
        "emptyId": int(EMPTY_ID),
        "wordLengths": [
            int(v)
            for v in word_lengths.tolist()
        ],
        "charVocabSize": int(char_vocab_size),
        "numWords": int(word_array.shape[0]),
    }

    write_json(
        output_dir / "metadata.json",
        metadata,
    )

    print()
    print("Saved:", output_dir / "boards.npy", boards_np.shape, boards_np.dtype)
    print("Saved:", output_dir / "moves.npy", moves_np.shape, moves_np.dtype)
    print("Saved:", output_dir / "labels.npy", labels_np.shape, labels_np.dtype)
    print("Saved:", output_dir / "X.npy", x_all.shape, x_all.dtype)
    print("Saved:", output_dir / "Y.npy", y_all.shape, y_all.dtype)
    print("Saved:", output_dir / "metadata.json")

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preprocessed NumPy policy dataset with X.npy / Y.npy."
    )

    parser.add_argument(
        "--replay-root",
        default=str(DEFAULT_REPLAY_ROOT),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--league",
        default="all",
        help="f0, f1, f2, f3, or all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max replay files to load. 0 means no limit.",
    )

    parser.add_argument(
        "--words-file",
        default="",
        help="Optional words JSON file.",
    )

    parser.add_argument(
        "--max-paths-per-word",
        type=int,
        default=80,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    words_file: Optional[Path]

    if args.words_file:
        words_file = Path(args.words_file).resolve()
    else:
        words_file = None

    summary = build_dataset(
        replay_root=Path(args.replay_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        league=args.league,
        limit=int(args.limit),
        words_file=words_file,
        max_paths_per_word=int(args.max_paths_per_word),
    )

    print()
    print("=== SUMMARY ===")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()