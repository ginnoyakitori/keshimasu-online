# python/build_numpy_dataset_from_replay.py
#
# Replay Buffer -> NumPy Dataset converter
#
# 入力:
#   data/training/selfplay/f0/*.json
#   data/training/selfplay/f1/*.json
#   data/training/selfplay/f2/*.json
#   ...
#
# 出力:
#   data/numpy_dataset/policy/
#     boards.npy              [N, 8, 5] int16
#     moves.npy               [N, 4] int16
#     labels.npy              [N] int8
#     branch_counts.npy       [N] int16
#     target_f.npy            [N] int16
#     depths.npy              [N] int16
#     puzzle_indices.npy      [N] int32
#
#     policy_boards.npy       [S, 8, 5] int16
#     policy_labels.npy       [S] int16
#     policy_branch_counts.npy[S] int16
#     policy_target_f.npy     [S] int16
#     policy_depths.npy       [S] int16
#
#     metadata.json
#
# move形式:
#   [word_id, row, col, direction]
#
# direction:
#   0 = H 横 左→右
#   1 = V 縦 上→下
#
# 実行:
#   python python/build_numpy_dataset_from_replay.py
#   python python/build_numpy_dataset_from_replay.py --league f2
#   python python/build_numpy_dataset_from_replay.py --league all --limit 100
#

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------
# Import core_solver_numba.py
# ------------------------------------------------------------

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
    build_codec,
    encode_board,
    encode_words,
    find_all_moves_py,
    _remove_move_and_apply_gravity,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

def get_replay_root() -> Path:
    return PROJECT_ROOT / "data" / "training" / "selfplay"


def get_output_dir() -> Path:
    return PROJECT_ROOT / "data" / "numpy_dataset" / "policy"


# ------------------------------------------------------------
# Replay loading
# ------------------------------------------------------------

def list_replay_files(
    replay_root: Path,
    league: str = "all",
    limit: Optional[int] = None,
) -> List[Path]:
    if not replay_root.exists():
        return []

    if league == "all":
        league_dirs = [
            p for p in replay_root.iterdir()
            if p.is_dir() and p.name.startswith("f")
        ]
    else:
        league_name = league if league.startswith("f") else f"f{league}"
        league_dirs = [replay_root / league_name]

    files: List[Path] = []

    for d in sorted(league_dirs):
        if not d.exists():
            continue

        for file in sorted(d.glob("*.json")):
            files.append(file)

    if limit is not None and limit > 0:
        files = files[:limit]

    return files


def load_json(file: Path) -> Dict[str, Any]:
    with file.open("r", encoding="utf-8") as f:
        return json.load(f)


def replay_board(replay: Dict[str, Any]) -> List[List[Any]]:
    board = replay.get("board")

    if board is None:
        raise ValueError("Replay has no board")

    return board


def replay_words(replay: Dict[str, Any]) -> List[str]:
    words = replay.get("words")

    if not words:
        return []

    return [
        w for w in words
        if isinstance(w, str)
        and 2 <= len(list(w)) <= MAX_WORD_LENGTH
    ]


def replay_target_f(replay: Dict[str, Any]) -> int:
    return int(replay.get("targetWildcards", 0))


def replay_solved_moves(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    solver = replay.get("solver", {})

    moves = solver.get("solvedMoves")

    if moves:
        return moves

    moves = replay.get("solverSolutionMoves")

    if moves:
        return moves

    return []


def replay_id(replay: Dict[str, Any], fallback: str) -> str:
    return str(replay.get("id") or fallback)


# ------------------------------------------------------------
# Codec / vocabulary
# ------------------------------------------------------------

def normalize_cell(cell: Any) -> Any:
    if cell is None:
        return None

    if cell == "・":
        return None

    if cell == "Ｆ":
        return "F"

    return cell


def collect_board_chars(board: List[List[Any]]) -> List[str]:
    chars: List[str] = []

    for row in board:
        for cell in row:
            cell = normalize_cell(cell)

            if cell is None:
                continue

            chars.append(str(cell))

    return chars


def collect_global_words_and_chars(
    files: List[Path],
) -> Tuple[List[str], List[str]]:
    word_set = set()
    board_chars: List[str] = []

    for file in files:
        replay = load_json(file)

        for word in replay_words(replay):
            word_set.add(word)

        board_chars.extend(
            collect_board_chars(replay_board(replay))
        )

    words = sorted(word_set)

    return words, board_chars


def build_global_codec(
    words: List[str],
    board_chars: List[str],
):
    codec_source = list(words)

    if board_chars:
        codec_source.append("".join(board_chars))

    return build_codec(codec_source, wildcard_char="F")


# ------------------------------------------------------------
# Move utilities
# ------------------------------------------------------------

def path_to_tuple(path: List[Any]) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(p[0]), int(p[1])) for p in path)


def move_path_from_encoded(
    move: np.ndarray,
    word_lengths: np.ndarray,
) -> Tuple[Tuple[int, int], ...]:
    wid = int(move[0])
    row = int(move[1])
    col = int(move[2])
    direction = int(move[3])

    length = int(word_lengths[wid])

    cells: List[Tuple[int, int]] = []

    for i in range(length):
        if direction == DIR_H:
            cells.append((row, col + i))
        else:
            cells.append((row + i, col))

    return tuple(cells)


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
    word = candidate["word"]

    if word not in word_to_id:
        raise KeyError(f"word not found in word_to_id: {word}")

    wid = word_to_id[word]

    path = candidate["path"]

    row = int(path[0][0])
    col = int(path[0][1])
    direction = direction_from_path(path)

    if direction < 0:
        raise ValueError(f"invalid candidate direction: {candidate}")

    return np.array(
        [wid, row, col, direction],
        dtype=np.int16,
    )


def solved_move_to_key(move: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    word = str(move["word"])
    path = path_to_tuple(move["path"])

    return word, path


def candidate_to_key(candidate: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[int, int], ...]]:
    word = str(candidate["word"])
    path = path_to_tuple(candidate["path"])

    return word, path


def find_label_index(
    candidates: List[Dict[str, Any]],
    chosen_move: Dict[str, Any],
) -> int:
    chosen_key = solved_move_to_key(chosen_move)

    for i, cand in enumerate(candidates):
        if candidate_to_key(cand) == chosen_key:
            return i

    return -1


# ------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------

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

    for depth, chosen_move in enumerate(solved_moves):
        candidates = find_all_moves_py(
            board,
            word_array,
            word_lengths,
            codec.wild_id,
            words_text,
            max_paths_per_word=max_paths_per_word,
        )

        label_index = find_label_index(
            candidates,
            chosen_move,
        )

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

        # policy example:
        # 1 state -> label candidate index
        policy_examples.append({
            "board": board.copy(),
            "label_index": label_index,
            "branch_count": branch_count,
            "target_f": target_f,
            "depth": depth,
            "puzzle_index": puzzle_index,
            "puzzle_id": puzzle_id,
        })

        # move examples:
        # each candidate -> binary label
        encoded_candidates: List[np.ndarray] = []

        for i, cand in enumerate(candidates):
            encoded = candidate_to_encoded_move(
                cand,
                word_to_id,
            )

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

        # advance board by chosen move
        chosen_encoded = encoded_candidates[label_index]

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

def build_numpy_dataset(
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
        raise ValueError("No words found in replay files.")

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
        replay = load_json(file)
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
        by_f[key]["positiveMoves"] += sum(
            int(ex["label"]) for ex in move_examples
        )

        print(
            f"[{puzzle_index + 1}/{len(files)}] OK {file.parent.name}/{file.name} "
            f"policy={len(policy_examples)} moves={len(move_examples)}"
        )

    if not move_examples_all:
        raise ValueError("No move examples generated.")

    if not policy_examples_all:
        raise ValueError("No policy examples generated.")

    # ------------------------------------------------------------
    # Move-level arrays
    # board + candidateMove -> label
    # ------------------------------------------------------------

    boards = np.stack(
        [ex["board"] for ex in move_examples_all],
        axis=0,
    ).astype(np.int16)

    moves = np.stack(
        [ex["move"] for ex in move_examples_all],
        axis=0,
    ).astype(np.int16)

    labels = np.array(
        [ex["label"] for ex in move_examples_all],
        dtype=np.int8,
    )

    branch_counts = np.array(
        [ex["branch_count"] for ex in move_examples_all],
        dtype=np.int16,
    )

    target_f = np.array(
        [ex["target_f"] for ex in move_examples_all],
        dtype=np.int16,
    )

    depths = np.array(
        [ex["depth"] for ex in move_examples_all],
        dtype=np.int16,
    )

    puzzle_indices = np.array(
        [ex["puzzle_index"] for ex in move_examples_all],
        dtype=np.int32,
    )

    # ------------------------------------------------------------
    # Policy-level arrays
    # board -> labelIndex
    # ------------------------------------------------------------

    policy_boards = np.stack(
        [ex["board"] for ex in policy_examples_all],
        axis=0,
    ).astype(np.int16)

    policy_labels = np.array(
        [ex["label_index"] for ex in policy_examples_all],
        dtype=np.int16,
    )

    policy_branch_counts = np.array(
        [ex["branch_count"] for ex in policy_examples_all],
        dtype=np.int16,
    )

    policy_target_f = np.array(
        [ex["target_f"] for ex in policy_examples_all],
        dtype=np.int16,
    )

    policy_depths = np.array(
        [ex["depth"] for ex in policy_examples_all],
        dtype=np.int16,
    )

    policy_puzzle_indices = np.array(
        [ex["puzzle_index"] for ex in policy_examples_all],
        dtype=np.int32,
    )

    # ------------------------------------------------------------
    # Summary / Metadata
    # ------------------------------------------------------------

    positive_move_examples = int(labels.sum())
    negative_move_examples = int(labels.shape[0] - labels.sum())

    summary = {
        "inputReplayFiles": len(files),
        "skippedFiles": len(skipped),

        "moveExamples": int(labels.shape[0]),
        "policyExamples": int(policy_labels.shape[0]),

        "positiveMoveExamples": positive_move_examples,
        "negativeMoveExamples": negative_move_examples,

        "rows": ROWS,
        "cols": COLS,
        "maxWordLength": MAX_WORD_LENGTH,

        "moveFormat": ["word_id", "row", "col", "direction"],

        "direction": {
            "H": int(DIR_H),
            "V": int(DIR_V),
        },

        "byF": by_f,

        # 長くなりすぎないように先頭100件だけ保存
        "skipped": skipped[:100],
    }

    metadata = {
        "summary": summary,

        "words": filtered_words,
        "wordLengths": [
            int(x) for x in word_lengths.tolist()
        ],

        "charToId": {
            str(k): int(v)
            for k, v in codec.char_to_id.items()
        },

        "idToChar": {
            str(k): str(v)
            for k, v in codec.id_to_char.items()
        },

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


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

def save_numpy_dataset(
    dataset: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = dataset["arrays"]
    metadata = dataset["metadata"]

    paths: Dict[str, str] = {}

    for name, arr in arrays.items():
        file = output_dir / f"{name}.npy"
        np.save(file, arr)
        paths[name] = str(file)

    metadata_file = output_dir / "metadata.json"

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    paths["metadata"] = str(metadata_file)

    return paths
# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NumPy dataset from JS replay buffer"
    )

    parser.add_argument(
        "--league",
        default="all",
        help="all, f0, f1, f2, f3, or numeric like 2",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit replay files. 0 means no limit.",
    )

    parser.add_argument(
        "--max-paths-per-word",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: data/numpy_dataset/policy",
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Do not skip entire replay on missing label. Useful for debugging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    replay_root = PROJECT_ROOT / "data" / "training" / "selfplay"

    files = list_replay_files(
        replay_root,
        league=args.league,
        limit=args.limit if args.limit > 0 else None,
    )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else get_output_dir()
    )

    print("=== BUILD NUMPY DATASET FROM REPLAY ===")
    print("Replay root:", replay_root)
    print("League:", args.league)
    print("Files:", len(files))
    print("Output:", output_dir)
    print("Max paths per word:", args.max_paths_per_word)

    if not files:
        print("No replay files found.")
        return

    dataset = build_numpy_dataset(
        files,
        max_paths_per_word=args.max_paths_per_word,
        skip_on_missing_label=not args.keep_going,
    )

    paths = save_numpy_dataset(dataset, output_dir)

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