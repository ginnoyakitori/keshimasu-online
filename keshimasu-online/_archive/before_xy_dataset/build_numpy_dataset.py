# python/apps/build_numpy_dataset.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from keshimasu_py.config import (
    PROJECT_ROOT,
    ROWS,
    COLS,
    MAX_WORD_LENGTH,
    DIR_H,
    DIR_V,
)
from keshimasu_py.codec import (
    build_codec,
    encode_board,
    encode_words,
    collect_board_chars,
)
from keshimasu_py.core_solver import (
    find_all_moves_py,
    _remove_move_and_apply_gravity,
)
from keshimasu_py.replay_io import (
    list_replay_files,
    load_replay,
    replay_board,
    replay_words,
    replay_target_f,
    replay_solved_moves,
    replay_id,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "numpy_dataset" / "policy"


def collect_global_words_and_chars(files: List[Path]) -> Tuple[List[str], List[str]]:
    word_set = set()
    board_chars: List[str] = []

    for file in files:
        replay = load_replay(file)

        for word in replay_words(replay):
            word_set.add(word)

        board_chars.extend(collect_board_chars(replay_board(replay)))

    return sorted(word_set), board_chars


def build_global_codec(words: List[str], board_chars: List[str]):
    source = list(words)

    if board_chars:
        source.append("".join(board_chars))

    return build_codec(source)


def path_key(path: List[Any]) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(p[0]), int(p[1])) for p in path)


def candidate_key(candidate: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    return str(candidate["word"]), path_key(candidate["path"])


def solved_move_key(move: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    return str(move["word"]), path_key(move["path"])


def direction_from_path(path: List[Any]) -> int:
    if len(path) < 2:
        return int(DIR_H)

    r0, c0 = int(path[0][0]), int(path[0][1])
    r1, c1 = int(path[1][0]), int(path[1][1])

    if r1 == r0 and c1 == c0 + 1:
        return int(DIR_H)

    if r1 == r0 + 1 and c1 == c0:
        return int(DIR_V)

    return -1


def candidate_to_encoded_move(
    candidate: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> np.ndarray:
    word = str(candidate["word"])

    if word not in word_to_id:
        raise KeyError(f"word not found: {word}")

    wid = word_to_id[word]
    path = candidate["path"]

    row = int(path[0][0])
    col = int(path[0][1])
    direction = direction_from_path(path)

    if direction < 0:
        raise ValueError(f"invalid direction: {candidate}")

    return np.array([wid, row, col, direction], dtype=np.int16)


def find_label_index(
    candidates: List[Dict[str, Any]],
    chosen_move: Dict[str, Any],
) -> int:
    key = solved_move_key(chosen_move)

    for i, cand in enumerate(candidates):
        if candidate_key(cand) == key:
            return i

    return -1


def build_examples_from_replay(
    replay: Dict[str, Any],
    *,
    puzzle_index: int,
    puzzle_id: str,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    word_to_id: Dict[str, int],
    codec: Any,
    max_paths_per_word: int,
    skip_on_missing_label: bool,
) -> Dict[str, Any]:
    board_raw = replay_board(replay)
    solved_moves = replay_solved_moves(replay)

    if not solved_moves:
        return {
            "skipped": True,
            "reason": "missing solvedMoves",
            "move_examples": [],
            "policy_examples": [],
        }

    board = encode_board(board_raw, codec)
    target_f = replay_target_f(replay)

    move_examples: List[Dict[str, Any]] = []
    policy_examples: List[Dict[str, Any]] = []

    # 同じ単語は2回以上使えない
    used_word_ids: set[int] = set()

    for depth, chosen_move in enumerate(solved_moves):
        candidates = find_all_moves_py(
            board,
            word_array,
            word_lengths,
            int(codec.wild_id),
            words_text,
            max_paths_per_word=max_paths_per_word,
            used_word_ids=used_word_ids,
        )

        label_index = find_label_index(candidates, chosen_move)

        if label_index < 0:
            reason = (
                f"chosen move not found at depth={depth}, "
                f"word={chosen_move.get('word')}"
            )

            if skip_on_missing_label:
                return {
                    "skipped": True,
                    "reason": reason,
                    "move_examples": [],
                    "policy_examples": [],
                }

            print("WARNING:", reason)
            break

        branch_count = len(candidates)

        policy_examples.append({
            "board": board.copy(),
            "label_index": label_index,
            "branch_count": branch_count,
            "target_f": target_f,
            "depth": depth,
            "puzzle_index": puzzle_index,
            "puzzle_id": puzzle_id,
        })

        encoded_candidates: List[np.ndarray] = []

        for i, cand in enumerate(candidates):
            encoded = candidate_to_encoded_move(cand, word_to_id)
            encoded_candidates.append(encoded)

            move_examples.append({
                "board": board.copy(),
                "move": encoded,
                "label": 1 if i == label_index else 0,
                "branch_count": branch_count,
                "target_f": target_f,
                "depth": depth,
                "puzzle_index": puzzle_index,
                "puzzle_id": puzzle_id,
            })

        chosen_encoded = encoded_candidates[label_index]

        used_word_ids.add(int(chosen_encoded[0]))

        board = _remove_move_and_apply_gravity(
            board,
            chosen_encoded,
            word_lengths,
        )

    return {
        "skipped": False,
        "reason": "",
        "move_examples": move_examples,
        "policy_examples": policy_examples,
    }


def build_dataset(
    files: List[Path],
    *,
    max_paths_per_word: int,
    skip_on_missing_label: bool,
) -> Dict[str, Any]:
    if not files:
        raise ValueError("No replay files found.")

    print("Collecting global words/chars...")

    global_words, board_chars = collect_global_words_and_chars(files)

    if not global_words:
        raise ValueError("No words found.")

    codec = build_global_codec(global_words, board_chars)

    word_array, word_lengths, filtered_words = encode_words(
        global_words,
        codec,
        max_word_length=MAX_WORD_LENGTH,
    )

    word_to_id = {
        word: i for i, word in enumerate(filtered_words)
    }

    print("Global words:", len(filtered_words))
    print("Char vocab size:", len(codec.char_to_id) + 1)

    move_examples_all: List[Dict[str, Any]] = []
    policy_examples_all: List[Dict[str, Any]] = []

    skipped: List[Dict[str, Any]] = []
    puzzle_ids: List[str] = []

    by_f: Dict[str, Dict[str, Any]] = {}

    for puzzle_index, file in enumerate(files):
        replay = load_replay(file)
        pid = replay_id(replay, fallback=file.stem)
        puzzle_ids.append(pid)

        target_f = replay_target_f(replay)

        result = build_examples_from_replay(
            replay,
            puzzle_index=puzzle_index,
            puzzle_id=pid,
            word_array=word_array,
            word_lengths=word_lengths,
            words_text=filtered_words,
            word_to_id=word_to_id,
            codec=codec,
            max_paths_per_word=max_paths_per_word,
            skip_on_missing_label=skip_on_missing_label,
        )

        if result["skipped"]:
            skipped.append({
                "file": str(file),
                "reason": result["reason"],
            })

            print(
                f"[{puzzle_index + 1}/{len(files)}] SKIP {file.name}: {result['reason']}"
            )
            continue

        move_examples = result["move_examples"]
        policy_examples = result["policy_examples"]

        move_examples_all.extend(move_examples)
        policy_examples_all.extend(policy_examples)

        key = f"f{target_f}"

        if key not in by_f:
            by_f[key] = {
                "replays": 0,
                "moveExamples": 0,
                "policyExamples": 0,
                "positiveMoves": 0,
            }

        by_f[key]["replays"] += 1
        by_f[key]["moveExamples"] += len(move_examples)
        by_f[key]["policyExamples"] += len(policy_examples)
        by_f[key]["positiveMoves"] += sum(int(ex["label"]) for ex in move_examples)

        print(
            f"[{puzzle_index + 1}/{len(files)}] OK {file.parent.name}/{file.name} "
            f"policy={len(policy_examples)} moves={len(move_examples)}"
        )

    if not move_examples_all:
        raise ValueError("No move examples generated.")

    if not policy_examples_all:
        raise ValueError("No policy examples generated.")

    boards = np.stack([ex["board"] for ex in move_examples_all], axis=0).astype(np.int16)
    moves = np.stack([ex["move"] for ex in move_examples_all], axis=0).astype(np.int16)
    labels = np.array([ex["label"] for ex in move_examples_all], dtype=np.int8)
    branch_counts = np.array([ex["branch_count"] for ex in move_examples_all], dtype=np.int16)
    target_f = np.array([ex["target_f"] for ex in move_examples_all], dtype=np.int16)
    depths = np.array([ex["depth"] for ex in move_examples_all], dtype=np.int16)
    puzzle_indices = np.array([ex["puzzle_index"] for ex in move_examples_all], dtype=np.int32)

    policy_boards = np.stack([ex["board"] for ex in policy_examples_all], axis=0).astype(np.int16)
    policy_labels = np.array([ex["label_index"] for ex in policy_examples_all], dtype=np.int16)
    policy_branch_counts = np.array([ex["branch_count"] for ex in policy_examples_all], dtype=np.int16)
    policy_target_f = np.array([ex["target_f"] for ex in policy_examples_all], dtype=np.int16)
    policy_depths = np.array([ex["depth"] for ex in policy_examples_all], dtype=np.int16)
    policy_puzzle_indices = np.array([ex["puzzle_index"] for ex in policy_examples_all], dtype=np.int32)

    summary = {
        "inputReplayFiles": len(files),
        "skippedFiles": len(skipped),
        "moveExamples": int(labels.shape[0]),
        "policyExamples": int(policy_labels.shape[0]),
        "positiveMoveExamples": int(labels.sum()),
        "negativeMoveExamples": int(labels.shape[0] - labels.sum()),
        "rows": ROWS,
        "cols": COLS,
        "maxWordLength": MAX_WORD_LENGTH,
        "moveFormat": ["word_id", "row", "col", "direction"],
        "direction": {
            "H": int(DIR_H),
            "V": int(DIR_V),
        },
        "byF": by_f,
        "skipped": skipped[:100],
        "wordReuseRule": "same word_id cannot be used more than once per puzzle",
    }

    metadata = {
        "summary": summary,
        "words": filtered_words,
        "wordLengths": [int(x) for x in word_lengths.tolist()],
        "charToId": {str(k): int(v) for k, v in codec.char_to_id.items()},
        "idToChar": {str(k): str(v) for k, v in codec.id_to_char.items()},
        "wildId": int(codec.wild_id),
        "emptyId": 0,
        "puzzleIds": puzzle_ids,
    }

    return {
        "arrays": {
            "boards": boards,
            "moves": moves,
            "labels": labels,
            "branch_counts": branch_counts,
            "target_f": target_f,
            "depths": depths,
            "puzzle_indices": puzzle_indices,

            "policy_boards": policy_boards,
            "policy_labels": policy_labels,
            "policy_branch_counts": policy_branch_counts,
            "policy_target_f": policy_target_f,
            "policy_depths": policy_depths,
            "policy_puzzle_indices": policy_puzzle_indices,

            "word_array": word_array.astype(np.int16),
            "word_lengths": word_lengths.astype(np.int16),
        },
        "metadata": metadata,
    }


def save_dataset(dataset: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    for name, arr in dataset["arrays"].items():
        file = output_dir / f"{name}.npy"
        np.save(file, arr)
        paths[name] = str(file)

    metadata_file = output_dir / "metadata.json"

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(dataset["metadata"], f, ensure_ascii=False, indent=2)

    paths["metadata"] = str(metadata_file)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--league", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-paths-per-word", type=int, default=80)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--keep-going", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = list_replay_files(
        league=args.league,
        limit=args.limit if args.limit > 0 else None,
    )

    output_dir = Path(args.output_dir).resolve()

    print("=== BUILD NUMPY DATASET ===")
    print("League:", args.league)
    print("Files:", len(files))
    print("Output:", output_dir)
    print("Rule: same word cannot be used more than once per puzzle")

    if not files:
        print("No replay files found.")
        return

    dataset = build_dataset(
        files,
        max_paths_per_word=args.max_paths_per_word,
        skip_on_missing_label=not args.keep_going,
    )

    paths = save_dataset(dataset, output_dir)

    print("\n=== DATASET SAVED ===")
    for name, file in paths.items():
        print(f"{name}: {file}")

    print("\n=== SUMMARY ===")
    print(
        json.dumps(
            dataset["metadata"]["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()