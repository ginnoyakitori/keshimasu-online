# python/apps/inference_policy.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from keshimasu_py.config import (
    PROJECT_ROOT,
    MAX_WORD_LENGTH,
    DIR_H,
    DIR_V,
)
from keshimasu_py.codec import (
    build_codec_for_board_and_words,
    encode_board,
    encode_words,
    board_to_string,
)
from keshimasu_py.core_solver import find_all_moves_py
from keshimasu_py.policy_inference import TorchPolicy
from keshimasu_py.replay_io import (
    list_replay_files,
    load_replay,
    replay_board,
    replay_words,
    replay_target_f,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"


def direction_from_path(path: List[Any]) -> int:
    if len(path) < 2:
        return int(DIR_H)

    r0, c0 = int(path[0][0]), int(path[0][1])
    r1, c1 = int(path[1][0]), int(path[1][1])

    if r1 == r0 and c1 == c0 + 1:
        return int(DIR_H)

    if r1 == r0 + 1 and c1 == c0:
        return int(DIR_V)

    raise ValueError(f"invalid path direction: {path}")


def candidate_to_encoded_move(
    candidate: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> np.ndarray:
    word = str(candidate["word"])
    path = candidate["path"]

    return np.array(
        [
            word_to_id[word],
            int(path[0][0]),
            int(path[0][1]),
            direction_from_path(path),
        ],
        dtype=np.int16,
    )


def encoded_move_to_path(move: np.ndarray, word_lengths: np.ndarray) -> List[Tuple[int, int]]:
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


def main() -> None:
    args = parse_args()

    if args.replay:
        replay_file = Path(args.replay).resolve()
    else:
        files = list_replay_files(
            league=args.league,
        )

        if not files:
            raise FileNotFoundError(f"No replay files found: {args.league}")

        index = max(0, min(args.index, len(files) - 1))
        replay_file = files[index]

    replay = load_replay(replay_file)

    words = replay_words(replay)
    board_raw = replay_board(replay)

    codec = build_codec_for_board_and_words(words, board_raw)

    word_array, word_lengths, words_text = encode_words(
        words,
        codec,
        max_word_length=MAX_WORD_LENGTH,
    )

    board = encode_board(board_raw, codec)

    print("=== TORCH POLICY INFERENCE ===")
    print("Replay:", replay_file)
    print("Model dir:", args.model_dir)

    print("\n=== BOARD ===")
    print(board_to_string(board, codec))

    candidates = find_all_moves_py(
        board,
        word_array,
        word_lengths,
        codec.wild_id,
        words_text,
        max_paths_per_word=args.max_paths_per_word,
    )

    print("\nCandidate count:", len(candidates))

    if not candidates:
        return

    word_to_id = {
        word: i for i, word in enumerate(words_text)
    }

    moves = np.stack(
        [
            candidate_to_encoded_move(cand, word_to_id)
            for cand in candidates
        ],
        axis=0,
    ).astype(np.int16)

    heuristic_scores = np.array(
        [float(cand.get("score", 0.0)) for cand in candidates],
        dtype=np.float32,
    )

    policy = TorchPolicy(
        model_dir=Path(args.model_dir),
        device=args.device,
    )
    policy.load()

    ordered_moves, policy_scores = policy.order_encoded_moves(
        board,
        moves,
        heuristic_scores,
        word_array=word_array,
        word_lengths=word_lengths,
        target_f=replay_target_f(replay),
        depth=0,
        policy_weight=args.policy_weight,
        heuristic_weight=args.heuristic_weight,
    )

    print("\n=== TOP MOVES BY POLICY ===")

    top_k = min(args.top_k, ordered_moves.shape[0])

    for i in range(top_k):
        move = ordered_moves[i]
        wid = int(move[0])

        hscore = 0.0

        for j in range(moves.shape[0]):
            if np.array_equal(moves[j], move):
                hscore = float(heuristic_scores[j])
                break

        direction = "H" if int(move[3]) == DIR_H else "V"

        print(
            f"{i + 1:02d}. "
            f"{words_text[wid]} "
            f"{direction} "
            f"({int(move[1])},{int(move[2])}) "
            f"path={encoded_move_to_path(move, word_lengths)} "
            f"policy={float(policy_scores[i]):.4f} "
            f"heuristic={hscore:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--replay", default=None)
    parser.add_argument("--league", default="f2")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-paths-per-word", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--policy-weight", type=float, default=0.75)
    parser.add_argument("--heuristic-weight", type=float, default=0.25)

    return parser.parse_args()


if __name__ == "__main__":
    main()