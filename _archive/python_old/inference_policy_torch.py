# python/inference_policy_torch.py
#
# PyTorch Policy Inference
#
# 目的:
# - 学習済み PyTorch policy_model.pt を読み込む
# - board + candidateMove に score を付ける
# - score順に候補moveを並べ替える
#
# 入力:
#   data/models/torch_policy/policy_model.pt
#   data/models/torch_policy/metadata.json
#
# 主な用途:
#   1. Numba候補列挙
#   2. PyTorch policy score
#   3. score順に探索
#
# 実行例:
#   python python/inference_policy_torch.py
#   python python/inference_policy_torch.py --replay data/training/selfplay/f2/xxx.json
#   python python/inference_policy_torch.py --league f2 --index 0
#

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Import core_solver_numba.py
# ============================================================

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
    encode_words,
    encode_board,
    board_to_string,
    find_all_moves_py,
)


# ============================================================
# Paths
# ============================================================

DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"
DEFAULT_REPLAY_ROOT = PROJECT_ROOT / "data" / "training" / "selfplay"


# ============================================================
# Model definition
# ============================================================

class PolicyMLP(nn.Module):
    """
    train_policy_torch.py と同じ構造。

    入力:
      feature vector

    出力:
      logit 1個
    """

    def __init__(
        self,
        input_size: int,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),

            nn.Linear(hidden3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Replay utilities
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_replay_files(
    replay_root: Path,
    league: str = "f2",
) -> List[Path]:
    if league == "all":
        dirs = [
            p for p in replay_root.iterdir()
            if p.is_dir() and p.name.startswith("f")
        ]
    else:
        league_name = league if league.startswith("f") else f"f{league}"
        dirs = [replay_root / league_name]

    files: List[Path] = []

    for d in sorted(dirs):
        if not d.exists():
            continue

        files.extend(sorted(d.glob("*.json")))

    return files


def replay_words(replay: Dict[str, Any]) -> List[str]:
    words = replay.get("words", [])

    return [
        w for w in words
        if isinstance(w, str)
        and 2 <= len(list(w)) <= MAX_WORD_LENGTH
    ]


def replay_board(replay: Dict[str, Any]) -> List[List[Any]]:
    board = replay.get("board")

    if board is None:
        raise ValueError("Replay has no board")

    return board


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


def build_codec_for_replay(
    words: List[str],
    board_raw: List[List[Any]],
):
    board_chars = collect_board_chars(board_raw)

    codec_source = list(words)

    if board_chars:
        codec_source.append("".join(board_chars))

    return build_codec(codec_source, wildcard_char="F")


# ============================================================
# Feature builder
# ============================================================

def feature_dim() -> int:
    # board 40
    # word chars 5
    # word length 1
    # word id 1
    # row 1
    # col 1
    # direction H 1
    # direction V 1
    # branch count 1
    # target f 1
    # depth 1
    return 40 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1


def make_feature_from_encoded_move(
    board: np.ndarray,
    move: np.ndarray,
    *,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    char_vocab_size: int,
    num_words: int,
    branch_count: int = 0,
    target_f: int = 0,
    depth: int = 0,
) -> np.ndarray:
    """
    board + encoded move -> feature vector

    move:
      [word_id, row, col, direction]
    """
    word_id = int(move[0])
    row = int(move[1])
    col = int(move[2])
    direction = int(move[3])

    board_flat = board.astype(np.float32).reshape(-1)
    board_flat = board_flat / max(1.0, float(char_vocab_size))

    word_chars = word_array[word_id].astype(np.float32)
    word_chars = word_chars / max(1.0, float(char_vocab_size))

    word_length = float(word_lengths[word_id]) / MAX_WORD_LENGTH

    word_id_norm = float(word_id) / max(1, num_words - 1)

    row_norm = float(row) / max(1, ROWS - 1)
    col_norm = float(col) / max(1, COLS - 1)

    dir_h = 1.0 if direction == DIR_H else 0.0
    dir_v = 1.0 if direction == DIR_V else 0.0

    branch_norm = np.log1p(branch_count) / np.log1p(100.0)
    target_f_norm = float(target_f) / 10.0
    depth_norm = float(depth) / 20.0

    extra = np.array(
        [
            word_length,
            word_id_norm,
            row_norm,
            col_norm,
            dir_h,
            dir_v,
            branch_norm,
            target_f_norm,
            depth_norm,
        ],
        dtype=np.float32,
    )

    feature = np.concatenate(
        [
            board_flat.astype(np.float32),
            word_chars.astype(np.float32),
            extra,
        ],
        axis=0,
    )

    expected = feature_dim()

    if feature.shape[0] != expected:
        raise ValueError(
            f"Invalid feature dim: {feature.shape[0]}, expected={expected}"
        )

    return feature.astype(np.float32)


def encoded_move_to_path(
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


def encoded_move_to_dict(
    move: np.ndarray,
    word_lengths: np.ndarray,
    words_text: List[str],
    score: Optional[float] = None,
    policy_score: Optional[float] = None,
) -> Dict[str, Any]:
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

    if score is not None:
        out["heuristicScore"] = float(score)

    if policy_score is not None:
        out["policyScore"] = float(policy_score)

    return out


# ============================================================
# Policy Inference Class
# ============================================================

class TorchPolicy:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: str = "auto",
    ) -> None:
        self.model_dir = Path(model_dir)

        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[PolicyMLP] = None
        self.checkpoint: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None

        self.input_size: int = feature_dim()
        self.char_vocab_size: int = 1
        self.num_words: int = 1
        self.word_array: Optional[np.ndarray] = None
        self.word_lengths: Optional[np.ndarray] = None

    def load(self) -> None:
        model_path = self.model_dir / "policy_model.pt"
        metadata_path = self.model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        # PyTorch 2.6+ では weights_only の既定が変わる可能性があるため明示。
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        args = checkpoint.get("args", {})

        self.input_size = int(checkpoint.get("input_size", feature_dim()))
        self.char_vocab_size = int(checkpoint.get("char_vocab_size", 1))
        self.num_words = int(checkpoint.get("num_words", 1))

        self.word_array = checkpoint.get("word_array")
        self.word_lengths = checkpoint.get("word_lengths")

        if isinstance(self.word_array, torch.Tensor):
            self.word_array = self.word_array.cpu().numpy()

        if isinstance(self.word_lengths, torch.Tensor):
            self.word_lengths = self.word_lengths.cpu().numpy()

        hidden1 = int(args.get("hidden1", 256))
        hidden2 = int(args.get("hidden2", 128))
        hidden3 = int(args.get("hidden3", 64))
        dropout = float(args.get("dropout", 0.15))

        model = PolicyMLP(
            input_size=self.input_size,
            hidden1=hidden1,
            hidden2=hidden2,
            hidden3=hidden3,
            dropout=dropout,
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.checkpoint = checkpoint
        self.metadata = metadata

    def is_loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def score_encoded_moves(
        self,
        board: np.ndarray,
        moves: np.ndarray,
        *,
        word_array: np.ndarray,
        word_lengths: np.ndarray,
        branch_count: Optional[int] = None,
        target_f: int = 0,
        depth: int = 0,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        encoded moves に policy score を付ける。

        moves:
          shape [N,4]

        return:
          scores shape [N]
        """
        if self.model is None:
            raise RuntimeError("Policy model is not loaded. Call load() first.")

        if moves.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        bcount = branch_count if branch_count is not None else int(moves.shape[0])

        features: List[np.ndarray] = []

        for i in range(moves.shape[0]):
            feat = make_feature_from_encoded_move(
                board,
                moves[i],
                word_array=word_array,
                word_lengths=word_lengths,
                char_vocab_size=self.char_vocab_size,
                num_words=word_array.shape[0],
                branch_count=bcount,
                target_f=target_f,
                depth=depth,
            )
            features.append(feat)

        x_np = np.stack(features, axis=0).astype(np.float32)

        scores: List[np.ndarray] = []

        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])

            x = torch.from_numpy(x_np[start:end]).to(self.device)

            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(-1)

            scores.append(probs.cpu().numpy())

        return np.concatenate(scores, axis=0).astype(np.float32)

    def order_encoded_moves(
        self,
        board: np.ndarray,
        moves: np.ndarray,
        heuristic_scores: Optional[np.ndarray],
        *,
        word_array: np.ndarray,
        word_lengths: np.ndarray,
        target_f: int = 0,
        depth: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        policy score 降順で moves を並べ替える。

        戻り値:
          ordered_moves
          ordered_policy_scores
        """
        policy_scores = self.score_encoded_moves(
            board,
            moves,
            word_array=word_array,
            word_lengths=word_lengths,
            branch_count=int(moves.shape[0]),
            target_f=target_f,
            depth=depth,
        )

        if heuristic_scores is None:
            heuristic_scores = np.zeros_like(policy_scores, dtype=np.float32)

        # policy score優先、同点なら heuristic score
        order = np.lexsort(
            (
                -heuristic_scores.astype(np.float32),
                -policy_scores.astype(np.float32),
            )
        )

        return moves[order], policy_scores[order]


# ============================================================
# Helper: replay -> encoded candidates
# ============================================================

def build_encoded_context_from_replay(
    replay: Dict[str, Any],
) -> Dict[str, Any]:
    words = replay_words(replay)

    if not words:
        raise ValueError("Replay has no words")

    board_raw = replay_board(replay)

    codec = build_codec_for_replay(words, board_raw)

    word_array, word_lengths, filtered_words = encode_words(
        words,
        codec,
        max_word_length=MAX_WORD_LENGTH,
    )

    board = encode_board(board_raw, codec)

    return {
        "board": board,
        "codec": codec,
        "word_array": word_array,
        "word_lengths": word_lengths,
        "words_text": filtered_words,
    }


def candidate_dict_to_encoded_move(
    candidate: Dict[str, Any],
    word_to_id: Dict[str, int],
) -> np.ndarray:
    word = candidate["word"]
    path = candidate["path"]

    wid = word_to_id[word]

    row = int(path[0][0])
    col = int(path[0][1])

    if len(path) >= 2:
        r0, c0 = int(path[0][0]), int(path[0][1])
        r1, c1 = int(path[1][0]), int(path[1][1])

        if r1 == r0 and c1 == c0 + 1:
            direction = DIR_H
        elif r1 == r0 + 1 and c1 == c0:
            direction = DIR_V
        else:
            raise ValueError(f"Invalid direction candidate: {candidate}")
    else:
        direction = DIR_H

    return np.array(
        [wid, row, col, direction],
        dtype=np.int16,
    )


def get_start_candidates_encoded(
    board: np.ndarray,
    word_array: np.ndarray,
    word_lengths: np.ndarray,
    codec: Any,
    words_text: List[str],
    max_paths_per_word: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    find_all_moves_py から encoded moves + heuristic scores を作る。

    Parameters
    ----------
    board:
        encoded board, shape=(8, 5)

    word_array:
        encoded words, shape=(num_words, 5)

    word_lengths:
        word lengths, shape=(num_words,)

    codec:
        Codec object. codec.wild_id を使う。

    words_text:
        word_id -> word string のリスト

    max_paths_per_word:
        1単語あたりの最大候補数

    Returns
    -------
    moves:
        np.ndarray, shape=(N, 4)
        move format: [word_id, row, col, direction]

    scores:
        np.ndarray, shape=(N,)
        heuristic scores from find_all_moves_py
    """
    candidates = find_all_moves_py(
        board,
        word_array,
        word_lengths,
        codec.wild_id,
        words_text,
        max_paths_per_word=max_paths_per_word,
    )

    word_to_id = {
        word: i for i, word in enumerate(words_text)
    }

    moves: List[np.ndarray] = []
    scores: List[float] = []

    for cand in candidates:
        encoded_move = candidate_dict_to_encoded_move(
            cand,
            word_to_id,
        )

        moves.append(encoded_move)
        scores.append(float(cand.get("score", 0.0)))

    if not moves:
        return (
            np.zeros((0, 4), dtype=np.int16),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(moves, axis=0).astype(np.int16),
        np.array(scores, dtype=np.float32),
    )

# ============================================================
# CLI demo
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch policy inference demo"
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
    )

    parser.add_argument(
        "--replay",
        default=None,
        help="Path to replay json",
    )

    parser.add_argument(
        "--league",
        default="f2",
        help="Used when --replay is omitted",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Replay index in league when --replay is omitted",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--max-paths-per-word",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.replay:
        replay_file = Path(args.replay).resolve()
    else:
        files = list_replay_files(
            DEFAULT_REPLAY_ROOT,
            league=args.league,
        )

        if not files:
            raise FileNotFoundError(
                f"No replay files found for league={args.league}"
            )

        index = max(0, min(args.index, len(files) - 1))
        replay_file = files[index]

    print("=== TORCH POLICY INFERENCE ===")
    print("Replay:", replay_file)
    print("Model dir:", args.model_dir)

    replay = load_json(replay_file)

    ctx = build_encoded_context_from_replay(replay)

    board = ctx["board"]
    codec = ctx["codec"]
    word_array = ctx["word_array"]
    word_lengths = ctx["word_lengths"]
    words_text = ctx["words_text"]

    print("\n=== BOARD ===")
    print(board_to_string(board, codec))

    moves, heuristic_scores = get_start_candidates_encoded(
        board,
        word_array,
        word_lengths,
        codec,
        words_text,
        max_paths_per_word=args.max_paths_per_word,
    )

    print("\nCandidate count:", moves.shape[0])

    if moves.shape[0] == 0:
        print("No candidates.")
        return

    target_f = int(replay.get("targetWildcards", 0))

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
        target_f=target_f,
        depth=0,
    )

    print("\n=== TOP MOVES BY POLICY ===")

    top_k = min(args.top_k, ordered_moves.shape[0])

    for i in range(top_k):
        move = ordered_moves[i]
        pscore = float(policy_scores[i])

        # heuristic scoreも表示したいので元のmoveと照合
        hscore = 0.0
        for j in range(moves.shape[0]):
            if np.array_equal(moves[j], move):
                hscore = float(heuristic_scores[j])
                break

        info = encoded_move_to_dict(
            move,
            word_lengths,
            words_text,
            score=hscore,
            policy_score=pscore,
        )

        print(
            f"{i + 1:02d}. "
            f"{info['word']} "
            f"{info['direction']} "
            f"({info['row']},{info['col']}) "
            f"path={info['path']} "
            f"policy={pscore:.4f} "
            f"heuristic={hscore:.2f}"
        )


if __name__ == "__main__":
    main()