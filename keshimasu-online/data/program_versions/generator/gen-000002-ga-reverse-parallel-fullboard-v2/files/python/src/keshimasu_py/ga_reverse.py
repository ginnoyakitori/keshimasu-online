# python/src/keshimasu_py/ga_reverse.py

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    PROJECT_ROOT,
    ROWS,
    COLS,
    PLAYABLE_START_ROW,
    MAX_WORD_LENGTH,
    EMPTY_ID,
)

from .codec import (
    Codec,
    build_codec,
    encode_words,
    board_to_string,
)

from .core_solver import solve_numba

try:
    from .core_solver import find_all_moves_py
except Exception:
    find_all_moves_py = None  # type: ignore

try:
    from .policy_inference import TorchPolicy
except Exception:
    TorchPolicy = None  # type: ignore


DIR_H = 0
DIR_V = 1
WILDCARD_TEXT = "F"
BOARD_CELLS = ROWS * COLS


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class PlacementGene:
    word: str
    direction: int
    start_col: int
    wild_mask: List[bool] = field(default_factory=list)

    def clone(self) -> "PlacementGene":
        return PlacementGene(
            word=str(self.word),
            direction=int(self.direction),
            start_col=int(self.start_col),
            wild_mask=list(self.wild_mask),
        )


@dataclass
class PuzzleChromosome:
    genes: List[PlacementGene]
    target_wildcards: int = 0
    generation: int = 0

    fitness: float = -1.0e18
    solved: bool = False
    board: Optional[np.ndarray] = None
    result: Optional[Dict[str, Any]] = None
    policy_trap_bonus: float = 0.0
    policy_trap: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "PuzzleChromosome":
        out = PuzzleChromosome(
            genes=[gene.clone() for gene in self.genes],
            target_wildcards=int(self.target_wildcards),
            generation=int(self.generation),
        )

        out.fitness = float(self.fitness)
        out.solved = bool(self.solved)
        out.board = None if self.board is None else self.board.copy()
        out.result = self.result
        out.policy_trap_bonus = float(self.policy_trap_bonus)
        out.policy_trap = dict(self.policy_trap)

        return out


@dataclass
class PolicyTrapConfig:
    enabled: bool = False

    top_wrong_bonus: float = 300.0
    rank_bonus: float = 100.0
    confidence_gap_bonus: float = 500.0
    low_policy_correct_bonus: float = 300.0

    low_policy_threshold: float = 0.25
    confidence_gap_threshold: float = 0.40

    max_candidates_to_score: int = 64


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def _word_len(word: str) -> int:
    return len(list(str(word)))


def _split_word(word: str) -> List[str]:
    return list(str(word))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)

        if math.isfinite(n):
            return n

        return default
    except Exception:
        return default


def _wild_id(codec: Codec) -> int:
    if hasattr(codec, "wild_id"):
        return int(getattr(codec, "wild_id"))

    char_to_id = getattr(codec, "char_to_id", {})

    if WILDCARD_TEXT in char_to_id:
        return int(char_to_id[WILDCARD_TEXT])

    if "Ｆ" in char_to_id:
        return int(char_to_id["Ｆ"])

    raise KeyError("wild_id not found in codec")


def _id_to_char(codec: Codec, value: int) -> str:
    if int(value) == int(EMPTY_ID):
        return "・"

    id_to_char = getattr(codec, "id_to_char", {})

    if value in id_to_char:
        ch = str(id_to_char[value])
    elif str(value) in id_to_char:
        ch = str(id_to_char[str(value)])
    else:
        ch = "?"

    if ch == WILDCARD_TEXT:
        return "Ｆ"

    return ch


def board_to_rows(board: np.ndarray, codec: Codec) -> List[List[str]]:
    rows: List[List[str]] = []

    for r in range(board.shape[0]):
        row: List[str] = []

        for c in range(board.shape[1]):
            row.append(_id_to_char(codec, int(board[r, c])))

        rows.append(row)

    return rows


def board_key(board: np.ndarray, codec: Codec) -> str:
    return "/".join("".join(row) for row in board_to_rows(board, codec))


def count_f_on_board(board: np.ndarray, wild_id: int) -> int:
    return int(np.count_nonzero(board == int(wild_id)))


def count_filled(board: np.ndarray) -> int:
    return int(np.count_nonzero(board != int(EMPTY_ID)))


def count_empty(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == int(EMPTY_ID)))


def is_full_board(board: np.ndarray) -> bool:
    return count_filled(board) == BOARD_CELLS


# ---------------------------------------------------------------------
# Words / codec
# ---------------------------------------------------------------------


def load_words_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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

    raise ValueError(f"Unsupported words file format: {path}")


def default_words() -> List[str]:
    metadata_path = (
        PROJECT_ROOT
        / "data"
        / "numpy_dataset"
        / "policy"
        / "metadata.json"
    )

    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            words = metadata.get("words", [])

            if isinstance(words, list):
                words = [
                    str(w)
                    for w in words
                    if 2 <= _word_len(str(w)) <= MAX_WORD_LENGTH
                ]

                if words:
                    return sorted(set(words))
        except Exception:
            pass

    candidates = [
        PROJECT_ROOT / "data" / "words" / "countries.json",
        PROJECT_ROOT / "data" / "country_words.json",
        PROJECT_ROOT / "country_words.json",
        PROJECT_ROOT / "country_words_tmp.json",
    ]

    for file in candidates:
        if not file.exists():
            continue

        try:
            words = load_words_file(file)
            words = [
                str(w)
                for w in words
                if 2 <= _word_len(str(w)) <= MAX_WORD_LENGTH
            ]

            if words:
                return sorted(set(words))
        except Exception:
            pass

    return sorted(
        set(
            [
                "アメリカ",
                "イギリス",
                "フランス",
                "ドイツ",
                "イタリア",
                "カナダ",
                "インド",
                "タイ",
                "チリ",
                "マリ",
                "ペルー",
                "トルコ",
                "オマーン",
                "ヨルダン",
                "モンゴル",
                "ルワンダ",
                "ガーナ",
                "ケニア",
                "ナミビア",
                "ソマリア",
                "アルバニア",
                "エストニア",
                "リトアニア",
                "スロバキア",
                "スロベニア",
                "クロアチア",
                "ジョージア",
                "アルメニア",
                "インドネシア",
            ]
        )
    )


def build_words_and_codec(
    words: List[str],
) -> Tuple[Codec, np.ndarray, np.ndarray, List[str]]:
    words = [
        str(w)
        for w in words
        if 2 <= _word_len(str(w)) <= MAX_WORD_LENGTH
    ]

    words = sorted(set(words))

    codec = build_codec(words)

    word_array, word_lengths, filtered_words = encode_words(
        words,
        codec,
        max_word_length=MAX_WORD_LENGTH,
    )

    return (
        codec,
        word_array.astype(np.int16),
        word_lengths.astype(np.int16),
        list(filtered_words),
    )


# ---------------------------------------------------------------------
# Reverse gravity construction
# ---------------------------------------------------------------------


def _word_to_ids(word: str, codec: Codec) -> Optional[List[int]]:
    char_to_id = getattr(codec, "char_to_id", {})

    ids: List[int] = []

    for ch in _split_word(word):
        if ch not in char_to_id:
            return None

        ids.append(int(char_to_id[ch]))

    return ids


def _can_push_up_one(board: np.ndarray, col: int) -> bool:
    return int(board[0, col]) == int(EMPTY_ID)


def _push_column_up_one_until_row(
    board: np.ndarray,
    col: int,
    target_row: int,
) -> None:
    for r in range(0, target_row):
        board[r, col] = board[r + 1, col]

    board[target_row, col] = int(EMPTY_ID)


def _can_push_up_block(
    board: np.ndarray,
    col: int,
    block_len: int,
) -> bool:
    if block_len <= 0:
        return False

    if block_len > ROWS:
        return False

    for r in range(0, block_len):
        if int(board[r, col]) != int(EMPTY_ID):
            return False

    return True


def _push_column_up_block_until_row(
    board: np.ndarray,
    col: int,
    target_row: int,
    block_len: int,
) -> None:
    insert_start = target_row - block_len + 1

    for r in range(0, insert_start):
        board[r, col] = board[r + block_len, col]

    for r in range(insert_start, target_row + 1):
        board[r, col] = int(EMPTY_ID)


def _apply_wild_mask(
    ids: List[int],
    wild_mask: List[bool],
    wild_id: int,
) -> List[int]:
    out: List[int] = []

    for i, value in enumerate(ids):
        if i < len(wild_mask) and wild_mask[i]:
            out.append(int(wild_id))
        else:
            out.append(int(value))

    return out


def simulate_reverse_gravity_build(
    chromosome: PuzzleChromosome,
    codec: Codec,
) -> Optional[np.ndarray]:
    board = np.full(
        (ROWS, COLS),
        int(EMPTY_ID),
        dtype=np.int16,
    )

    wild_id = _wild_id(codec)

    for gene in chromosome.genes:
        ids = _word_to_ids(gene.word, codec)

        if ids is None:
            return None

        ids = _apply_wild_mask(
            ids,
            gene.wild_mask,
            wild_id,
        )

        length = len(ids)

        if length < 2 or length > MAX_WORD_LENGTH:
            return None

        if gene.direction == DIR_H:
            if gene.start_col < 0 or gene.start_col + length > COLS:
                return None

            target_row = ROWS - 1

            for i in range(length):
                col = gene.start_col + i

                if not _can_push_up_one(board, col):
                    return None

            for i in range(length):
                col = gene.start_col + i

                _push_column_up_one_until_row(
                    board,
                    col,
                    target_row,
                )

                board[target_row, col] = ids[i]

        else:
            col = int(gene.start_col)

            if col < 0 or col >= COLS:
                return None

            if not _can_push_up_block(board, col, length):
                return None

            target_row = ROWS - 1
            insert_start = target_row - length + 1

            _push_column_up_block_until_row(
                board,
                col,
                target_row,
                length,
            )

            for i, value in enumerate(ids):
                board[insert_start + i, col] = value

    return board


# ---------------------------------------------------------------------
# Solver result helpers
# ---------------------------------------------------------------------


def _stats_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    stats = result.get("stats", {})

    if not isinstance(stats, dict):
        return {}

    return stats


def _result_solved(result: Dict[str, Any]) -> bool:
    return bool(result.get("solved", False))


def _result_solved_moves(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = [
        result.get("solvedMoves"),
        result.get("solutionMoves"),
        result.get("moves"),
        (
            result.get("stats", {}).get("solvedMoves")
            if isinstance(result.get("stats"), dict)
            else None
        ),
    ]

    for moves in candidates:
        if isinstance(moves, list) and moves:
            return moves

    return []


def _planned_solution_moves(
    chromosome: PuzzleChromosome,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for gene in reversed(chromosome.genes):
        length = _word_len(gene.word)

        if gene.direction == DIR_H:
            path = [
                [ROWS - 1, gene.start_col + i]
                for i in range(length)
            ]
            direction = "H"
        else:
            start_row = ROWS - length
            path = [
                [start_row + i, gene.start_col]
                for i in range(length)
            ]
            direction = "V"

        out.append(
            {
                "word": gene.word,
                "direction": direction,
                "path": path,
            }
        )

    return out


def _extract_nodes(result: Dict[str, Any]) -> float:
    stats = _stats_from_result(result)
    return _safe_float(stats.get("exploredNodes", stats.get("nodes", 0.0)))


def _extract_backtracks(result: Dict[str, Any]) -> float:
    return _safe_float(_stats_from_result(result).get("backtracks", 0.0))


def _extract_dead_ends(result: Dict[str, Any]) -> float:
    return _safe_float(_stats_from_result(result).get("deadEnds", 0.0))


def _extract_max_depth(result: Dict[str, Any]) -> float:
    return _safe_float(_stats_from_result(result).get("maxDepth", 0.0))


# ---------------------------------------------------------------------
# Policy trap fitness
# ---------------------------------------------------------------------


def _move_key_from_candidate(
    move: Dict[str, Any],
) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    word = str(move.get("word", ""))
    path = tuple(
        (int(p[0]), int(p[1]))
        for p in move.get("path", [])
    )
    return word, path


def _move_key_from_solution(
    move: Dict[str, Any],
) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    word = str(move.get("word", ""))
    path = tuple(
        (int(p[0]), int(p[1]))
        for p in move.get("path", [])
    )
    return word, path


def _candidate_to_encoded(
    candidate: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> Optional[np.ndarray]:
    word = str(candidate.get("word", ""))

    if word not in word_to_id:
        return None

    path = candidate.get("path", [])

    if not path:
        return None

    row = int(path[0][0])
    col = int(path[0][1])
    direction = DIR_H

    if len(path) >= 2:
        r0, c0 = int(path[0][0]), int(path[0][1])
        r1, c1 = int(path[1][0]), int(path[1][1])

        if r1 == r0 + 1 and c1 == c0:
            direction = DIR_V

    return np.array(
        [
            int(word_to_id[word]),
            row,
            col,
            int(direction),
        ],
        dtype=np.int16,
    )


def _call_policy_order_encoded_moves(
    *,
    policy: Any,
    board: np.ndarray,
    encoded_moves: np.ndarray,
    heuristic_scores: np.ndarray,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    target_f: int,
) -> Tuple[np.ndarray, np.ndarray]:
    result = policy.order_encoded_moves(
        board,
        encoded_moves,
        heuristic_scores,
        word_array=word_array,
        word_lengths=word_lengths,
        target_f=target_f,
        depth=0,
        policy_weight=1.0,
        heuristic_weight=0.0,
    )

    if isinstance(result, tuple) and len(result) >= 2:
        ordered_moves = result[0]
        scores = result[1]

        return (
            np.asarray(ordered_moves, dtype=np.int16).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32).reshape(-1),
        )

    if isinstance(result, dict):
        ordered_moves = result.get("moves", encoded_moves)

        scores = (
            result.get("policyScores")
            if result.get("policyScores") is not None
            else result.get("scores")
            if result.get("scores") is not None
            else result.get("combinedScores")
            if result.get("combinedScores") is not None
            else heuristic_scores
        )

        return (
            np.asarray(ordered_moves, dtype=np.int16).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32).reshape(-1),
        )

    raise TypeError(
        f"Unsupported policy.order_encoded_moves return type: {type(result)}"
    )


def _score_policy_trap(
    *,
    policy: Any,
    board: np.ndarray,
    solution_moves: List[Dict[str, Any]],
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    wild_id: int,
    target_f: int,
    max_paths_per_word: int,
    config: PolicyTrapConfig,
) -> Tuple[float, Dict[str, Any]]:
    if not config.enabled:
        return 0.0, {"enabled": False}

    if policy is None:
        return 0.0, {"enabled": False, "reason": "policy not loaded"}

    if find_all_moves_py is None:
        return 0.0, {
            "enabled": False,
            "reason": "find_all_moves_py unavailable",
        }

    if not solution_moves:
        return 0.0, {
            "enabled": True,
            "reason": "missing solution_moves",
        }

    try:
        candidates = find_all_moves_py(
            board,
            word_array,
            word_lengths,
            int(wild_id),
            words_text,
            max_paths_per_word=max_paths_per_word,
            used_word_ids=set(),
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

    if not candidates:
        return 0.0, {
            "enabled": True,
            "reason": "no initial candidates",
        }

    candidates = candidates[: config.max_candidates_to_score]

    word_to_id = {
        str(word): i
        for i, word in enumerate(words_text)
    }

    encoded: List[np.ndarray] = []
    encoded_candidates: List[Dict[str, Any]] = []

    for candidate in candidates:
        move = _candidate_to_encoded(
            candidate,
            word_to_id,
        )

        if move is None:
            continue

        encoded.append(move)
        encoded_candidates.append(candidate)

    if not encoded:
        return 0.0, {
            "enabled": True,
            "reason": "no encodable candidates",
        }

    encoded_moves = np.stack(
        encoded,
        axis=0,
    ).astype(np.int16)

    heuristic_scores = np.zeros(
        (encoded_moves.shape[0],),
        dtype=np.float32,
    )

    try:
        ordered_moves, scores = _call_policy_order_encoded_moves(
            policy=policy,
            board=board,
            encoded_moves=encoded_moves,
            heuristic_scores=heuristic_scores,
            word_array=word_array,
            word_lengths=word_lengths,
            target_f=target_f,
        )
    except Exception as e:
        return 0.0, {
            "enabled": True,
            "reason": "policy scoring failed",
            "error": str(e),
        }

    ordered_moves = np.asarray(
        ordered_moves,
        dtype=np.int16,
    ).reshape(-1, 4)

    scores = np.asarray(
        scores,
        dtype=np.float32,
    ).reshape(-1)

    max_n = len(encoded_candidates)

    if ordered_moves.shape[0] > max_n:
        ordered_moves = ordered_moves[:max_n]

    if scores.shape[0] > max_n:
        scores = scores[:max_n]

    encoded_key_to_candidate: Dict[
        Tuple[int, int, int, int],
        Dict[str, Any],
    ] = {}

    for move, candidate in zip(encoded_moves, encoded_candidates):
        key = tuple(
            int(v)
            for v in move.tolist()
        )
        encoded_key_to_candidate[key] = candidate

    ordered_candidates: List[Dict[str, Any]] = []
    ordered_scores: List[float] = []

    for i, move in enumerate(ordered_moves):
        key = tuple(
            int(v)
            for v in move.tolist()
        )

        candidate = encoded_key_to_candidate.get(key)

        if candidate is None:
            continue

        ordered_candidates.append(candidate)

        if i < len(scores):
            ordered_scores.append(float(scores[i]))
        else:
            ordered_scores.append(0.0)

    if not ordered_candidates:
        return 0.0, {
            "enabled": True,
            "reason": "ordered candidate mapping failed",
        }

    correct_key = _move_key_from_solution(solution_moves[0])

    top_candidate = ordered_candidates[0]
    top_score = float(ordered_scores[0]) if ordered_scores else 0.0

    correct_rank = -1
    correct_score = 0.0

    for i, candidate in enumerate(ordered_candidates):
        if _move_key_from_candidate(candidate) == correct_key:
            correct_rank = i

            if i < len(ordered_scores):
                correct_score = float(ordered_scores[i])
            else:
                correct_score = 0.0

            break

    top_is_correct = (
        _move_key_from_candidate(top_candidate) ==
        correct_key
    )

    confidence_gap = top_score - correct_score

    bonus = 0.0

    if not top_is_correct:
        bonus += config.top_wrong_bonus

    if correct_rank >= 1:
        bonus += config.rank_bonus * float(correct_rank)

    if confidence_gap >= config.confidence_gap_threshold:
        bonus += config.confidence_gap_bonus * float(confidence_gap)

    if (
        0 <= correct_rank and
        correct_score <= config.low_policy_threshold
    ):
        bonus += (
            config.low_policy_correct_bonus *
            (1.0 - correct_score)
        )

    details = {
        "enabled": True,
        "candidateCount": len(candidates),
        "encodedCandidateCount": len(encoded_candidates),
        "scoredCandidateCount": len(ordered_candidates),
        "topPolicyScore": float(top_score),
        "correctPolicyScore": float(correct_score),
        "confidenceGap": float(confidence_gap),
        "correctRank": int(correct_rank),
        "topPolicyIsCorrect": bool(top_is_correct),
        "topMove": top_candidate,
        "correctFirstMove": solution_moves[0],
        "bonus": float(bonus),
    }

    return float(bonus), details

# ---------------------------------------------------------------------
# Parallel worker context
# ---------------------------------------------------------------------


_WORKER: Dict[str, Any] = {}


def _init_worker_context(
    words: List[str],
    target_wildcards: int,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
) -> None:
    codec, word_array, word_lengths, words_text = build_words_and_codec(words)

    _WORKER.clear()

    _WORKER.update(
        {
            "codec": codec,
            "word_array": word_array,
            "word_lengths": word_lengths,
            "words_text": words_text,
            "wild_id": _wild_id(codec),
            "target_wildcards": int(target_wildcards),
            "max_nodes": int(max_nodes),
            "max_depth": int(max_depth),
            "max_paths_per_word": int(max_paths_per_word),
        }
    )


def _evaluate_chromosome_worker(
    chromosome: PuzzleChromosome,
) -> PuzzleChromosome:
    codec: Codec = _WORKER["codec"]
    word_array: np.ndarray = _WORKER["word_array"]
    word_lengths: np.ndarray = _WORKER["word_lengths"]
    words_text: List[str] = _WORKER["words_text"]

    wild_id = int(_WORKER["wild_id"])
    target_wildcards = int(_WORKER["target_wildcards"])
    max_nodes = int(_WORKER["max_nodes"])
    max_depth = int(_WORKER["max_depth"])
    max_paths_per_word = int(_WORKER["max_paths_per_word"])

    board = simulate_reverse_gravity_build(
        chromosome,
        codec,
    )

    chromosome.board = board
    chromosome.policy_trap_bonus = 0.0
    chromosome.policy_trap = {
        "enabled": False,
        "reason": "worker evaluation only",
    }

    if board is None:
        chromosome.fitness = -1.0e12
        chromosome.solved = False
        return chromosome

    filled = count_filled(board)
    empty = count_empty(board)

    if not is_full_board(board):
        chromosome.fitness = -1.0e10 - float(empty) * 100000.0
        chromosome.solved = False
        chromosome.policy_trap = {
            "enabled": False,
            "reason": "board is not full",
            "filledCount": int(filled),
            "emptyCount": int(empty),
            "requiredFilledCount": int(BOARD_CELLS),
        }
        return chromosome

    actual_f = count_f_on_board(
        board,
        wild_id,
    )

    if actual_f != target_wildcards:
        chromosome.fitness = (
            -1.0e9
            - abs(actual_f - target_wildcards) * 10000.0
        )
        chromosome.solved = False
        chromosome.policy_trap = {
            "enabled": False,
            "reason": "wildcard count mismatch",
            "actualWildcards": int(actual_f),
            "targetWildcards": int(target_wildcards),
        }
        return chromosome

    try:
        result = solve_numba(
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
    except TypeError:
        result = solve_numba(
            board,
            word_array,
            word_lengths,
            wild_id,
            words_text,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_paths_per_word=max_paths_per_word,
        )

    chromosome.result = result
    chromosome.solved = _result_solved(result)

    if not chromosome.solved:
        nodes = _extract_nodes(result)
        chromosome.fitness = -100000.0 + nodes * 0.01
        return chromosome

    nodes = _extract_nodes(result)
    backtracks = _extract_backtracks(result)
    dead_ends = _extract_dead_ends(result)
    max_depth_value = _extract_max_depth(result)

    base_fitness = (
        nodes
        + backtracks * 2.0
        + dead_ends * 3.0
        + max_depth_value * 10.0
        + len(chromosome.genes) * 5.0
        + target_wildcards * 25.0
    )

    chromosome.fitness = float(base_fitness)

    return chromosome


# ---------------------------------------------------------------------
# GA generator
# ---------------------------------------------------------------------


class KeshimasuGAPuzzleGenerator:
    def __init__(
        self,
        *,
        words: List[str],
        target_wildcards: int = 3,
        steps: int = 12,
        population_size: int = 60,
        generations: int = 30,
        elite_count: int = 8,
        mutation_rate: float = 0.25,
        max_nodes: int = 300000,
        max_depth: int = 130,
        max_paths_per_word: int = 80,
        seed: Optional[int] = None,
        use_policy_trap: bool = False,
        policy_trap_weight: float = 0.5,
        policy_trap_top_k: int = 20,
        workers: int = 1,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
        **_ignored_kwargs: Any,
    ) -> None:
        self.words_raw = [
            str(w)
            for w in words
            if 2 <= _word_len(str(w)) <= MAX_WORD_LENGTH
        ]

        if not self.words_raw:
            raise ValueError("No playable words.")

        (
            self.codec,
            self.word_array,
            self.word_lengths,
            self.words_text,
        ) = build_words_and_codec(self.words_raw)

        self.words_by_length: Dict[int, List[str]] = {}

        for word in self.words_text:
            self.words_by_length.setdefault(_word_len(word), []).append(word)

        self.target_wildcards = int(target_wildcards)
        self.steps = int(steps)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elite_count = int(elite_count)
        self.mutation_rate = float(mutation_rate)

        self.max_nodes = int(max_nodes)
        self.max_depth = int(max_depth)
        self.max_paths_per_word = int(max_paths_per_word)

        self.random = random.Random(seed)
        self.seed = seed

        self.use_policy_trap = bool(use_policy_trap)
        self.policy_trap_weight = float(policy_trap_weight)
        self.policy_trap_top_k = int(policy_trap_top_k)

        self.policy_trap_config = PolicyTrapConfig(enabled=bool(use_policy_trap))
        self.policy: Any = None

        self.workers = max(1, int(workers))

        self.output_dir = output_dir or (PROJECT_ROOT / "data" / "ga_reverse")
        self.verbose = bool(verbose)

        self.wild_id = _wild_id(self.codec)

        if self.use_policy_trap:
            self._load_policy()

    def _load_policy(self) -> None:
        if TorchPolicy is None:
            print("[ga_reverse] TorchPolicy unavailable. Policy trap disabled.")
            self.use_policy_trap = False
            self.policy_trap_config.enabled = False
            return

        try:
            policy = TorchPolicy()
            policy.load()
            self.policy = policy
            print("[ga_reverse] Policy loaded for trap fitness.")
        except Exception as e:
            print("[ga_reverse] Failed to load policy. Policy trap disabled:", e)
            self.policy = None
            self.use_policy_trap = False
            self.policy_trap_config.enabled = False

    def _random_word_with_length(
        self,
        length: int,
        used: Optional[set[str]] = None,
    ) -> Optional[str]:
        used = used or set()

        candidates = [
            word
            for word in self.words_by_length.get(length, [])
            if word not in used
        ]

        if not candidates:
            return None

        return self.random.choice(candidates)

    def _random_word(
        self,
        used: Optional[set[str]] = None,
    ) -> str:
        used = used or set()

        candidates = [
            word
            for word in self.words_text
            if word not in used
        ]

        if not candidates:
            return self.random.choice(self.words_text)

        return self.random.choice(candidates)

    def _make_gene_for_word_direction_col(
        self,
        word: str,
        direction: int,
        start_col: int,
    ) -> PlacementGene:
        length = _word_len(word)

        return PlacementGene(
            word=word,
            direction=int(direction),
            start_col=int(start_col),
            wild_mask=[
                False
                for _ in range(length)
            ],
        )

    def _make_gene_for_word(self, word: str) -> PlacementGene:
        length = _word_len(word)
        direction = self.random.choice([DIR_H, DIR_V])

        if direction == DIR_H:
            start_col = self.random.randint(0, max(0, COLS - length))
        else:
            start_col = self.random.randint(0, COLS - 1)

        return self._make_gene_for_word_direction_col(
            word,
            direction,
            start_col,
        )

    def _target_lengths(self) -> List[int]:
        min_total = self.steps * 2
        max_total = self.steps * MAX_WORD_LENGTH

        if not (min_total <= BOARD_CELLS <= max_total):
            raise ValueError(
                f"steps={self.steps} cannot fill {BOARD_CELLS} cells "
                f"with word length range 2..{MAX_WORD_LENGTH}"
            )

        lengths = [2 for _ in range(self.steps)]
        remaining = BOARD_CELLS - sum(lengths)

        indices = list(range(self.steps))
        self.random.shuffle(indices)

        while remaining > 0:
            progressed = False
            self.random.shuffle(indices)

            for i in indices:
                if remaining <= 0:
                    break

                if lengths[i] < MAX_WORD_LENGTH:
                    lengths[i] += 1
                    remaining -= 1
                    progressed = True

            if not progressed:
                break

        self.random.shuffle(lengths)

        return lengths

    def assign_wildcards(self, genes: List[PlacementGene]) -> None:
        positions: List[Tuple[int, int]] = []

        for gi, gene in enumerate(genes):
            gene.wild_mask = [False for _ in range(_word_len(gene.word))]

            for ci in range(_word_len(gene.word)):
                positions.append((gi, ci))

        self.random.shuffle(positions)

        for gi, ci in positions[: self.target_wildcards]:
            if gi < len(genes) and ci < len(genes[gi].wild_mask):
                genes[gi].wild_mask[ci] = True

    def _partition_eight_by_count(
        self,
        count: int,
    ) -> Optional[List[int]]:
        patterns = {
            2: [
                [4, 4],
                [3, 5],
                [5, 3],
            ],
            3: [
                [2, 3, 3],
                [3, 2, 3],
                [3, 3, 2],
                [2, 2, 4],
                [2, 4, 2],
                [4, 2, 2],
            ],
            4: [
                [2, 2, 2, 2],
            ],
        }

        choices = patterns.get(count, [])

        if not choices:
            return None

        result = list(self.random.choice(choices))
        self.random.shuffle(result)

        return result

    def _choose_column_counts_for_steps(self) -> Optional[List[int]]:
        counts: List[int] = []

        def rec(col: int, remaining: int) -> bool:
            if col == COLS:
                return remaining == 0

            rest_cols = COLS - col - 1
            choices = [2, 3, 4]
            self.random.shuffle(choices)

            for count in choices:
                min_rest = rest_cols * 2
                max_rest = rest_cols * 4

                if remaining - count < min_rest:
                    continue

                if remaining - count > max_rest:
                    continue

                counts.append(count)

                if rec(col + 1, remaining - count):
                    return True

                counts.pop()

            return False

        if not rec(0, self.steps):
            return None

        return counts

    def _pick_unused_word_of_length(
        self, length: int, used: set[str],
    ) -> Optional[str]:
        candidates = [
            word for word in self.words_by_length.get(length, []) if word not in used
        ]

        if not candidates:
            return None

        return self.random.choice(candidates)

    def _build_vertical_full_genes(self) -> Optional[List[PlacementGene]]:
        column_counts = self._choose_column_counts_for_steps()

        if column_counts is None:
            return None

        genes: List[PlacementGene] = []
        used: set[str] = set()

        for col, count in enumerate(column_counts):
            lengths = self._partition_eight_by_count(count)

            if lengths is None:
                return None

            for length in lengths:
                word = self._pick_unused_word_of_length(length, used)

                if word is None:
                    return None

                used.add(word)

                genes.append(
                    PlacementGene(
                        word=word,
                        direction=DIR_V,
                        start_col=col,
                        wild_mask=[
                            False
                            for _ in range(length)
                        ],
                    )
                )

        if len(genes) != self.steps:
            return None

        total_len = sum(_word_len(gene.word) for gene in genes)

        if total_len != BOARD_CELLS:
            return None

        self.random.shuffle(genes)
        self.assign_wildcards(genes)

        return genes

    def _build_balanced_full_genes(self) -> Optional[List[PlacementGene]]:
        for _ in range(300):
            genes = self._build_vertical_full_genes()

            if genes is not None:
                return genes

        return None

    def repair_lengths_to_full_board(
        self,
        genes: List[PlacementGene],
    ) -> List[PlacementGene]:
        _ = genes

        repaired = self._build_balanced_full_genes()

        if repaired is not None:
            return repaired

        used: set[str] = set()
        out: List[PlacementGene] = []

        for length in self._target_lengths():
            word = self._random_word_with_length(length, used)

            if word is None:
                word = self._random_word(used)

            used.add(word)
            out.append(self._make_gene_for_word(word))

        self.assign_wildcards(out)

        return out

    def random_chromosome(self) -> PuzzleChromosome:
        genes = self._build_balanced_full_genes()

        if genes is None:
            used: set[str] = set()
            genes = []

            for length in self._target_lengths():
                word = self._random_word_with_length(length, used)

                if word is None:
                    word = self._random_word(used)

                used.add(word)
                genes.append(self._make_gene_for_word(word))

            self.assign_wildcards(genes)

        return PuzzleChromosome(
            genes=genes,
            target_wildcards=self.target_wildcards,
        )

    def mutate(self, chromosome: PuzzleChromosome) -> PuzzleChromosome:
        child = chromosome.clone()

        for gene in child.genes:
            if self.random.random() >= self.mutation_rate:
                continue

            action = self.random.choice(["word", "col"])

            if action == "word":
                length = _word_len(gene.word)
                used = {
                    g.word
                    for g in child.genes
                    if g is not gene
                }

                replacement = self._random_word_with_length(length, used)

                if replacement is not None:
                    gene.word = replacement
                    gene.wild_mask = [False for _ in range(length)]

            elif action == "col":
                if gene.direction == DIR_H:
                    length = _word_len(gene.word)
                    gene.start_col = self.random.randint(0, max(0, COLS - length))
                else:
                    gene.start_col = min(max(0, gene.start_col), COLS - 1)

        self.assign_wildcards(child.genes)

        return child

    def crossover(
        self,
        a: PuzzleChromosome,
        b: PuzzleChromosome,
    ) -> PuzzleChromosome:
        if len(a.genes) != len(b.genes):
            return a.clone()

        cut = self.random.randint(1, max(1, len(a.genes) - 1))

        genes = [
            gene.clone()
            for gene in a.genes[:cut]
        ] + [
            gene.clone()
            for gene in b.genes[cut:]
        ]

        total = sum(_word_len(g.word) for g in genes)

        if total != BOARD_CELLS:
            genes = self.repair_lengths_to_full_board(genes)

        self.assign_wildcards(genes)

        return PuzzleChromosome(
            genes=genes,
            target_wildcards=self.target_wildcards,
        )

    def evaluate(self, chromosome: PuzzleChromosome) -> PuzzleChromosome:
        return _evaluate_chromosome_worker(chromosome)

    def _apply_policy_trap_to_chromosome(
        self,
        chromosome: PuzzleChromosome,
    ) -> PuzzleChromosome:
        if not self.use_policy_trap:
            return chromosome

        if self.policy is None:
            return chromosome

        if chromosome.board is None:
            return chromosome

        if not chromosome.solved:
            return chromosome

        solved_moves = _result_solved_moves(chromosome.result or {})

        if not solved_moves:
            solved_moves = _planned_solution_moves(chromosome)

        trap_bonus, trap_info = _score_policy_trap(
            policy=self.policy,
            board=chromosome.board,
            solution_moves=solved_moves,
            word_array=self.word_array,
            word_lengths=self.word_lengths,
            words_text=self.words_text,
            wild_id=int(self.wild_id),
            target_f=self.target_wildcards,
            max_paths_per_word=self.max_paths_per_word,
            config=self.policy_trap_config,
        )

        chromosome.policy_trap_bonus = float(trap_bonus)
        chromosome.policy_trap = trap_info
        chromosome.fitness = float(chromosome.fitness) + self.policy_trap_weight * float(trap_bonus)

        return chromosome

    def evaluate_population(
        self, population: List[PuzzleChromosome],
    ) -> List[PuzzleChromosome]:
        if self.workers <= 1:
            evaluated = [
                self.evaluate(chromosome) for chromosome in population
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_worker_context,
                initargs=(
                    self.words_raw,
                    self.target_wildcards,
                    self.max_nodes,
                    self.max_depth,
                    self.max_paths_per_word,
                ),
            ) as executor:
                evaluated = list(executor.map(_evaluate_chromosome_worker, population))

        evaluated.sort(key=lambda x: x.fitness, reverse=True)

        if self.use_policy_trap and self.policy is not None:
            top_k = min(self.policy_trap_top_k, len(evaluated))

            for i in range(top_k):
                evaluated[i] = self._apply_policy_trap_to_chromosome(evaluated[i])

            evaluated.sort(key=lambda x: x.fitness, reverse=True)

        return evaluated

    def select_parent(
        self,
        population: List[PuzzleChromosome],
    ) -> PuzzleChromosome:
        k = min(4, len(population))
        candidates = self.random.sample(population, k)
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        return candidates[0]

    def run(self) -> PuzzleChromosome:
        _init_worker_context(
            self.words_raw,
            self.target_wildcards,
            self.max_nodes,
            self.max_depth,
            self.max_paths_per_word,
        )

        population = [
            self.random_chromosome()
            for _ in range(self.population_size)
        ]

        best: Optional[PuzzleChromosome] = None

        for generation in range(self.generations):
            start = time.perf_counter()

            evaluated = self.evaluate_population(population)

            if best is None or evaluated[0].fitness > best.fitness:
                best = evaluated[0].clone()

            elapsed = time.perf_counter() - start
            solved_count = sum(1 for c in evaluated if c.solved)
            valid_board_count = sum(1 for c in evaluated if c.board is not None)
            full_board_count = sum(
                1
                for c in evaluated
                if c.board is not None and is_full_board(c.board)
            )

            print(
                f"gen={generation:03d} "
                f"best={evaluated[0].fitness:.2f} "
                f"global={best.fitness:.2f} "
                f"boards={valid_board_count}/{len(evaluated)} "
                f"full={full_board_count}/{len(evaluated)} "
                f"solved={solved_count}/{len(evaluated)} "
                f"trap={evaluated[0].policy_trap_bonus:.2f} "
                f"time={elapsed:.2f}s"
            )

            elites = [
                c.clone()
                for c in evaluated[: self.elite_count]
            ]

            next_population: List[PuzzleChromosome] = elites

            while len(next_population) < self.population_size:
                p1 = self.select_parent(evaluated)
                p2 = self.select_parent(evaluated)

                child = self.crossover(p1, p2)
                child = self.mutate(child)
                child.generation = generation + 1

                next_population.append(child)

            population = next_population

        if best is None:
            raise RuntimeError("GA failed to produce any chromosome.")

        return self.evaluate(best)

    def save(
        self,
        chromosome: PuzzleChromosome,
        output_name: Optional[str] = None,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if chromosome.board is None:
            raise ValueError("Cannot save chromosome without board.")

        if not is_full_board(chromosome.board):
            raise ValueError(
                "Cannot save chromosome because board is not full: "
                f"filled={count_filled(chromosome.board)}, "
                f"empty={count_empty(chromosome.board)}, "
                f"required={BOARD_CELLS}"
            )

        result = chromosome.result or {}
        stats = _stats_from_result(result)

        solved_moves = _result_solved_moves(result)

        if not solved_moves:
            solved_moves = _planned_solution_moves(chromosome)

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if output_name is None:
            output_name = f"ga-f{self.target_wildcards}-{timestamp}.json"

        out_path = self.output_dir / output_name

        data = {
            "format": "keshimasu-ga-reverse-full-board-v2-parallel",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "versions": collect_current_program_versions(),
            "targetWildcards": self.target_wildcards,
            "actualWildcards": count_f_on_board(chromosome.board, self.wild_id),
            "fullBoardRequired": True,
            "filledCount": count_filled(chromosome.board),
            "emptyCount": count_empty(chromosome.board),
            "fitness": chromosome.fitness,
            "policyTrapBonus": chromosome.policy_trap_bonus,
            "policyTrapWeight": self.policy_trap_weight,
            "policyTrap": chromosome.policy_trap,
            "solved": chromosome.solved,
            "score": chromosome.fitness,
            "board": board_to_rows(chromosome.board, self.codec),
            "boardText": board_to_string(chromosome.board, self.codec),
            "boardKey": board_key(chromosome.board, self.codec),
            "solutionMoves": solved_moves,
            "solverStats": stats,
            "genes": [
                {
                    "word": gene.word,
                    "direction": "H" if gene.direction == DIR_H else "V",
                    "startCol": gene.start_col,
                    "length": _word_len(gene.word),
                    "wildMask": gene.wild_mask,
                }
                for gene in chromosome.genes
            ],
            "config": {
                "rows": ROWS,
                "cols": COLS,
                "boardCells": BOARD_CELLS,
                "playableStartRow": PLAYABLE_START_ROW,
                "reverseBuildTopRow": 0,
                "emptyId": int(EMPTY_ID),
                "maxWordLength": MAX_WORD_LENGTH,
                "steps": self.steps,
                "populationSize": self.population_size,
                "generations": self.generations,
                "eliteCount": self.elite_count,
                "mutationRate": self.mutation_rate,
                "maxNodes": self.max_nodes,
                "maxDepth": self.max_depth,
                "maxPathsPerWord": self.max_paths_per_word,
                "usePolicyTrap": self.use_policy_trap,
                "policyTrapWeight": self.policy_trap_weight,
                "policyTrapTopK": self.policy_trap_top_k,
                "workers": self.workers,
                "seed": self.seed,
            },
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return out_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GA reverse generator using keshimasu_py.solve_numba"
    )

    parser.add_argument("--words-file", default="")
    parser.add_argument("--target-f", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)

    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--mutation", type=float, default=0.25)

    parser.add_argument("--max-nodes", type=int, default=300000)
    parser.add_argument("--max-depth", type=int, default=130)
    parser.add_argument("--max-paths-per-word", type=int, default=80)

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--policy-trap-top-k", type=int, default=20)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "ga_reverse"),
    )
    parser.add_argument("--output-name", default="")

    parser.add_argument("--use-policy-trap", action="store_true")
    parser.add_argument("--policy-trap-weight", type=float, default=0.5)

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def _create_dummy_words_file() -> Path:
    path = PROJECT_ROOT / "country_words_tmp.json"

    if path.exists():
        return path

    words = default_words()

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            words,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path


def main() -> None:
    args = parse_args()

    if args.words_file:
        words_file = Path(args.words_file).resolve()
    else:
        words_file = _create_dummy_words_file()

    words = load_words_file(words_file)

    if args.seed == 0:
        seed = None
    else:
        seed = int(args.seed)

    workers = int(args.workers)

    if workers <= 0:
        workers = os.cpu_count() or 1

    print("=== GA REVERSE GENERATOR ===")
    print("Words file:", words_file)
    print("Words:", len(words))
    print("Target F:", args.target_f)
    print("Steps:", args.steps)
    print("Population:", args.population)
    print("Generations:", args.generations)
    print("Use policy trap:", args.use_policy_trap)
    print("Policy trap weight:", args.policy_trap_weight)
    print("Policy trap top K:", args.policy_trap_top_k)
    print("Workers:", workers)
    print("Full board required:", True)
    print("Required cells:", BOARD_CELLS)

    generator = KeshimasuGAPuzzleGenerator(
        words=words,
        target_wildcards=args.target_f,
        steps=args.steps,
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elite,
        mutation_rate=args.mutation,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        max_paths_per_word=args.max_paths_per_word,
        seed=seed,
        use_policy_trap=args.use_policy_trap,
        policy_trap_weight=args.policy_trap_weight,
        policy_trap_top_k=args.policy_trap_top_k,
        workers=workers,
        output_dir=Path(args.output_dir).resolve(),
        verbose=args.verbose,
    )

    best = generator.run()

    print("")
    print("=== BEST ===")
    print("Solved:", best.solved)
    print("Fitness:", best.fitness)
    print("Policy trap bonus:", best.policy_trap_bonus)

    if best.board is None:
        print("")
        print("NO VALID BOARD GENERATED.")
        print("No file was saved because the best chromosome has no board.")
        return

    if not is_full_board(best.board):
        print("")
        print("NO FULL BOARD GENERATED.")
        print(
            f"filled={count_filled(best.board)}, "
            f"empty={count_empty(best.board)}, "
            f"required={BOARD_CELLS}"
        )
        print("No file was saved because full board is required.")
        return

    out_path = generator.save(
        best,
        output_name=args.output_name or None,
    )

    print("Saved:", out_path)

    print("")
    print("=== BOARD ===")
    print("")
    print(board_to_string(best.board, generator.codec))


if __name__ == "__main__":
    main()