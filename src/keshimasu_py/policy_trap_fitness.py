# python/src/keshimasu_py/policy_trap_fitness.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .policy_inference import TorchPolicy
from .core_solver import (
    find_all_moves_py,
    _remove_move_and_apply_gravity,
    solve_numba,
)


@dataclass
class PolicyTrapConfig:
    enabled: bool = True

    # trap bonus weights
    top_wrong_bonus: float = 300.0
    top_deadend_bonus: float = 500.0
    rank_bonus: float = 100.0
    confidence_gap_bonus: float = 500.0
    low_policy_correct_bonus: float = 300.0

    # thresholds
    low_policy_threshold: float = 0.25
    confidence_gap_threshold: float = 0.40

    # solver check for top policy move
    deadend_max_nodes: int = 3000
    deadend_max_depth: int = 120

    # too expensive if every GA candidate runs full checks
    max_candidates_to_score: int = 64
    max_policy_branch: int = 80

    # if true, only initial position trap is evaluated
    initial_only: bool = True


@dataclass
class PolicyTrapResult:
    trap_bonus: float
    top_policy_score: float
    correct_policy_score: float
    confidence_gap: float
    correct_rank: int
    top_policy_is_correct: bool
    top_policy_deadends: bool
    initial_candidate_count: int
    details: Dict[str, Any]


def move_key_from_candidate(move: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    word = str(move["word"])
    path = tuple((int(r), int(c)) for r, c in move["path"])
    return word, path


def move_key_from_solution_move(move: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    word = str(move["word"])
    path = tuple((int(r), int(c)) for r, c in move["path"])
    return word, path


def candidate_to_encoded_move(
    candidate: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> np.ndarray:
    word = str(candidate["word"])

    if word not in word_to_id:
        raise KeyError(f"Unknown word: {word}")

    wid = int(word_to_id[word])
    path = candidate["path"]

    row = int(path[0][0])
    col = int(path[0][1])

    direction = 0

    if len(path) >= 2:
        r0, c0 = int(path[0][0]), int(path[0][1])
        r1, c1 = int(path[1][0]), int(path[1][1])

        if r1 == r0 + 1 and c1 == c0:
            direction = 1
        else:
            direction = 0

    return np.array(
        [wid, row, col, direction],
        dtype=np.int16,
    )


def build_word_to_id(words_text: List[str]) -> Dict[str, int]:
    return {
        str(word): i
        for i, word in enumerate(words_text)
    }


def score_candidates_with_policy(
    *,
    policy: TorchPolicy,
    board: np.ndarray,
    candidates: List[Dict[str, Any]],
    word_to_id: Dict[str, int],
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    target_f: int,
    depth: int,
    max_candidates: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    if not candidates:
        return [], np.zeros((0, 4), dtype=np.int16), np.zeros((0,), dtype=np.float32)

    candidates = candidates[:max_candidates]

    encoded_moves = np.stack(
        [
            candidate_to_encoded_move(candidate, word_to_id)
            for candidate in candidates
        ],
        axis=0,
    ).astype(np.int16)

    scores = policy.score_encoded_moves(
        board=board,
        moves=encoded_moves,
        word_array=word_array,
        word_lengths=word_lengths,
        target_f=target_f,
        depth=depth,
        branch_count=len(candidates),
    )

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    order = np.argsort(-scores)

    ordered_candidates = [candidates[int(i)] for i in order]
    ordered_encoded = encoded_moves[order]
    ordered_scores = scores[order]

    return ordered_candidates, ordered_encoded, ordered_scores


def check_move_deadends(
    *,
    board: np.ndarray,
    encoded_move: np.ndarray,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    wild_id: int,
    words_text: List[str],
    used_word_ids: Optional[set[int]] = None,
    max_nodes: int,
    max_depth: int,
    max_paths_per_word: int,
) -> bool:
    """
    top policy move を1手進めた後に解けるか確認する。
    解けなければ deadend 扱い。
    """
    if used_word_ids is None:
        used_word_ids = set()

    wid = int(encoded_move[0])

    if wid in used_word_ids:
        return True

    next_board = _remove_move_and_apply_gravity(
        board,
        encoded_move,
        word_lengths,
    )

    next_used = set(used_word_ids)
    next_used.add(wid)

    result = solve_numba(
        next_board,
        word_array,
        word_lengths,
        wild_id,
        words_text,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_paths_per_word=max_paths_per_word,
        use_word_mask=True,
        initial_used_word_ids=next_used,
    )

    return not bool(result["solved"])


def evaluate_initial_policy_trap(
    *,
    board: np.ndarray,
    solution_moves: List[Dict[str, Any]],
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    wild_id: int,
    policy: TorchPolicy,
    target_f: int = 0,
    max_paths_per_word: int = 80,
    config: Optional[PolicyTrapConfig] = None,
) -> PolicyTrapResult:
    """
    初期盤面に対して Policy Trap 度を評価する。

    必要なもの:
      - board: encoded board
      - solution_moves: solver が見つけた正解手順
      - word_array / word_lengths / words_text
      - trained TorchPolicy
    """
    if config is None:
        config = PolicyTrapConfig()

    if not config.enabled:
        return PolicyTrapResult(
            trap_bonus=0.0,
            top_policy_score=0.0,
            correct_policy_score=0.0,
            confidence_gap=0.0,
            correct_rank=-1,
            top_policy_is_correct=False,
            top_policy_deadends=False,
            initial_candidate_count=0,
            details={},
        )

    if not solution_moves:
        return PolicyTrapResult(
            trap_bonus=0.0,
            top_policy_score=0.0,
            correct_policy_score=0.0,
            confidence_gap=0.0,
            correct_rank=-1,
            top_policy_is_correct=False,
            top_policy_deadends=False,
            initial_candidate_count=0,
            details={"reason": "missing solution_moves"},
        )

    candidates = find_all_moves_py(
        board,
        word_array,
        word_lengths,
        int(wild_id),
        words_text,
        max_paths_per_word=max_paths_per_word,
        used_word_ids=set(),
    )

    if not candidates:
        return PolicyTrapResult(
            trap_bonus=0.0,
            top_policy_score=0.0,
            correct_policy_score=0.0,
            confidence_gap=0.0,
            correct_rank=-1,
            top_policy_is_correct=False,
            top_policy_deadends=False,
            initial_candidate_count=0,
            details={"reason": "no candidates"},
        )

    word_to_id = build_word_to_id(words_text)

    ordered_candidates, ordered_encoded, ordered_scores = score_candidates_with_policy(
        policy=policy,
        board=board,
        candidates=candidates,
        word_to_id=word_to_id,
        word_array=word_array,
        word_lengths=word_lengths,
        target_f=target_f,
        depth=0,
        max_candidates=min(
            config.max_candidates_to_score,
            config.max_policy_branch,
        ),
    )

    correct_key = move_key_from_solution_move(solution_moves[0])

    correct_rank = -1
    correct_policy_score = 0.0

    for i, candidate in enumerate(ordered_candidates):
        if move_key_from_candidate(candidate) == correct_key:
            correct_rank = i
            correct_policy_score = float(ordered_scores[i])
            break

    top_policy_score = float(ordered_scores[0]) if len(ordered_scores) else 0.0
    top_candidate = ordered_candidates[0]
    top_encoded = ordered_encoded[0]

    top_policy_is_correct = move_key_from_candidate(top_candidate) == correct_key

    top_policy_deadends = False

    if not top_policy_is_correct:
        try:
            top_policy_deadends = check_move_deadends(
                board=board,
                encoded_move=top_encoded,
                word_array=word_array,
                word_lengths=word_lengths,
                wild_id=wild_id,
                words_text=words_text,
                used_word_ids=set(),
                max_nodes=config.deadend_max_nodes,
                max_depth=config.deadend_max_depth,
                max_paths_per_word=max_paths_per_word,
            )
        except TypeError:
            # solve_numba が initial_used_word_ids 非対応の場合の fallback。
            # この場合 deadend 判定は使わない。
            top_policy_deadends = False

    confidence_gap = top_policy_score - correct_policy_score

    trap_bonus = 0.0

    if not top_policy_is_correct:
        trap_bonus += config.top_wrong_bonus

    if top_policy_deadends:
        trap_bonus += config.top_deadend_bonus

    if correct_rank >= 1:
        trap_bonus += config.rank_bonus * float(correct_rank)

    if confidence_gap >= config.confidence_gap_threshold:
        trap_bonus += config.confidence_gap_bonus * float(confidence_gap)

    if 0 <= correct_rank and correct_policy_score <= config.low_policy_threshold:
        trap_bonus += config.low_policy_correct_bonus * (1.0 - correct_policy_score)

    details = {
        "topMove": top_candidate,
        "topPolicyScore": top_policy_score,
        "correctFirstMove": solution_moves[0],
        "correctPolicyScore": correct_policy_score,
        "correctRank": correct_rank,
        "confidenceGap": confidence_gap,
        "topPolicyIsCorrect": top_policy_is_correct,
        "topPolicyDeadends": top_policy_deadends,
        "candidateCount": len(candidates),
        "scoredCandidateCount": len(ordered_candidates),
    }

    return PolicyTrapResult(
        trap_bonus=float(trap_bonus),
        top_policy_score=float(top_policy_score),
        correct_policy_score=float(correct_policy_score),
        confidence_gap=float(confidence_gap),
        correct_rank=int(correct_rank),
        top_policy_is_correct=bool(top_policy_is_correct),
        top_policy_deadends=bool(top_policy_deadends),
        initial_candidate_count=int(len(candidates)),
        details=details,
    )