# python/src/keshimasu_py/torch_policy.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


FEATURE_DIM = 54


def feature_dim() -> int:
    """
    Backward compatible feature dimension API.

    policy_inference.py imports feature_dim().
    """
    return int(FEATURE_DIM)


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_board_flat(
    boards: np.ndarray,
    *,
    char_vocab_size: int,
) -> np.ndarray:
    """
    Convert boards to normalized flat board features.

    boards:
      shape = (N, 8, 5), int ids

    return:
      shape = (N, 40), float32
    """
    boards = boards.astype(np.float32)

    denom = float(max(1, char_vocab_size))

    return boards.reshape((boards.shape[0], -1)) / denom


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
    """
    moves format:
      [word_id, row, col, direction]

    direction:
      H = 0
      V = 1
    """
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
    rows: int = 8,
    cols: int = 5,
    max_depth_norm: float = 120.0,
    max_branch_norm: float = 256.0,
    max_target_f_norm: float = 3.0,
) -> np.ndarray:
    """
    Build precomputed policy move feature matrix.

    Output:
      X shape = (N, 54)

    Feature layout:
      00-39: board flattened / char_vocab_size
      40   : word_id / (num_words - 1)
      41   : row / (rows - 1)
      42   : col / (cols - 1)
      43   : direction raw, H=0, V=1
      44   : is_horizontal
      45   : is_vertical
      46   : word_length / 5
      47   : target_f / 3
      48   : depth / 120
      49   : log1p(branch_count) / log1p(256)
      50   : start_index / (rows * cols - 1)
      51   : end_row / (rows - 1)
      52   : end_col / (cols - 1)
      53   : filled_ratio
    """
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
    is_horizontal = (directions == 0).astype(np.float32)
    is_vertical = (directions == 1).astype(np.float32)

    word_len_norm = word_len_for_moves / 5.0

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
        np.count_nonzero(boards, axis=(1, 2)).astype(np.float32) /
        float(rows * cols)
    )

    extra = np.stack(
        [
            word_id_norm,
            row_norm,
            col_norm,
            direction_norm,
            is_horizontal,
            is_vertical,
            word_len_norm,
            target_f_norm,
            depth_norm,
            branch_norm,
            start_index_norm,
            end_row_norm,
            end_col_norm,
            filled_ratio,
        ],
        axis=1,
    ).astype(np.float32)

    x = np.concatenate(
        [
            board_features.astype(np.float32),
            extra,
        ],
        axis=1,
    ).astype(np.float32)

    if x.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Feature dim mismatch: got {x.shape[1]}, expected {FEATURE_DIM}"
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
    """
    Build X.npy / Y.npy arrays from raw dataset arrays.
    """
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


def make_feature_for_encoded_moves(
    *,
    board: np.ndarray,
    moves: np.ndarray,
    word_lengths: np.ndarray,
    char_vocab_size: int,
    num_words: int,
    target_f: int = 0,
    depth: int = 0,
    branch_count: Optional[int] = None,
) -> np.ndarray:
    """
    Build feature rows for inference.

    One board + many encoded moves -> X matrix.
    """
    moves = np.asarray(moves, dtype=np.int16)

    if moves.ndim == 1:
        moves = moves.reshape(1, 4)

    if moves.shape[0] == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    if branch_count is None:
        branch_count = int(moves.shape[0])

    board = np.asarray(board, dtype=np.int16)

    boards = np.repeat(
        board[None, :, :],
        moves.shape[0],
        axis=0,
    )

    branch_counts = np.full(
        (moves.shape[0],),
        int(branch_count),
        dtype=np.int16,
    )

    target_fs = np.full(
        (moves.shape[0],),
        int(target_f),
        dtype=np.int16,
    )

    depths = np.full(
        (moves.shape[0],),
        int(depth),
        dtype=np.int16,
    )

    return build_feature_matrix(
        boards=boards,
        moves=moves,
        branch_counts=branch_counts,
        target_f=target_fs,
        depths=depths,
        word_lengths=word_lengths,
        char_vocab_size=char_vocab_size,
        num_words=num_words,
    )


def make_feature_from_arrays(
    board: np.ndarray,
    moves: np.ndarray | None = None,
    *,
    move: np.ndarray | None = None,
    word_array: np.ndarray | None = None,
    word_lengths: np.ndarray | None = None,
    char_vocab_size: int,
    num_words: int,
    target_f: int = 0,
    depth: int = 0,
    branch_count: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Backward compatible feature builder.

    policy_inference.py imports make_feature_from_arrays().
    Some callers pass word_array=..., so this wrapper accepts it
    even though the current feature calculation only needs word_lengths.

    Supported call styles:
      make_feature_from_arrays(board, move=move, ...)
      make_feature_from_arrays(board, moves=moves, ...)
      make_feature_from_arrays(board, moves, word_array=..., ...)
    """
    if word_lengths is None:
        raise ValueError("word_lengths must be provided.")

    if moves is None:
        if move is None:
            raise ValueError("Either moves or move must be provided.")

        moves_arr = np.asarray(move, dtype=np.int16)
    else:
        moves_arr = np.asarray(moves, dtype=np.int16)

    if moves_arr.ndim == 1:
        moves_arr = moves_arr.reshape(1, 4)

    return make_feature_for_encoded_moves(
        board=np.asarray(board, dtype=np.int16),
        moves=moves_arr,
        word_lengths=np.asarray(word_lengths, dtype=np.int16),
        char_vocab_size=int(char_vocab_size),
        num_words=int(num_words),
        target_f=int(target_f),
        depth=int(depth),
        branch_count=branch_count,
    )


class PolicyMLP(nn.Module):
    """
    Policy MLP model.

    Must match saved checkpoint architecture:
      input -> 256 -> 128 -> 64 -> 1
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
            nn.Dropout(dropout),

            nn.Linear(hidden3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PolicyMoveDataset(Dataset):
    """
    X.npy / Y.npy 専用の事前前処理済み Policy dataset。

    前提:
      build_numpy_dataset.py が以下を出力していること。

        X.npy: shape=(N, 54), dtype=float32
        Y.npy: shape=(N, 1), dtype=float32

    この Dataset では __getitem__ 内で特徴量計算を一切しない。
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        max_negative_ratio: int = 3,
        balance: bool = True,
        mmap: bool = False,
        seed: int = 42,
        **_ignored_kwargs: Any,
    ) -> None:
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        self.mmap = bool(mmap)
        self.balance = bool(balance)
        self.max_negative_ratio = int(max_negative_ratio)
        self.seed = int(seed)

        x_path = self.dataset_dir / "X.npy"
        y_path = self.dataset_dir / "Y.npy"
        metadata_path = self.dataset_dir / "metadata.json"

        if not x_path.exists():
            raise FileNotFoundError(
                f"X.npy not found: {x_path}. "
                "Run python/apps/build_numpy_dataset.py first."
            )

        if not y_path.exists():
            raise FileNotFoundError(
                f"Y.npy not found: {y_path}. "
                "Run python/apps/build_numpy_dataset.py first."
            )

        if self.mmap:
            self.X = np.load(x_path, mmap_mode="r")
            self.Y = np.load(y_path, mmap_mode="r")
        else:
            self.X = np.load(x_path).astype(np.float32, copy=False)
            self.Y = np.load(y_path).astype(np.float32, copy=False)

        if self.X.ndim != 2:
            raise ValueError(
                f"X.npy must be 2D, got shape={self.X.shape}"
            )

        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)

        if self.Y.ndim != 2:
            raise ValueError(
                f"Y.npy must be 2D, got shape={self.Y.shape}"
            )

        # 【修正1】タイポ (shaperaise) を正しい条件分岐と raise に修正
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                "X/Y row count mismatch: "
                f"X={self.X.shape[0]}, Y={self.Y.shape[0]}"
            )

        self.input_size = int(self.X.shape[1])
        self.input_dim = self.input_size

        self.metadata: Dict[str, Any] = {}

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.char_vocab_size = int(
            self.metadata.get(
                "charVocabSize",
                self.metadata.get(
                    "summary",
                    {},
                ).get("charVocabSize", 0),
            )
            or 0
        )

        self.num_words = int(
            self.metadata.get(
                "numWords",
                len(self.metadata.get("words", [])),
            )
            or 0
        )

        # 【修正2】エラーで要求されている word_array と word_lengths の読み込みを追加
        word_array_path = self.dataset_dir / "word_array.npy"
        if word_array_path.exists():
            self.word_array = np.load(word_array_path)
        else:
            self.word_array = None

        word_lengths_path = self.dataset_dir / "word_lengths.npy"
        if word_lengths_path.exists():
            self.word_lengths = np.load(word_lengths_path)
        else:
            self.word_lengths = None

        # 【修正3】labels 変数を self.labels に変更し、クラスの属性として保持
        self.labels = np.asarray(self.Y).reshape(-1).astype(np.int8)

        self.indices = self._build_indices(
            labels=self.labels,
            max_negative_ratio=self.max_negative_ratio,
            balance=self.balance,
            seed=self.seed,
        ).astype(np.int64, copy=False)

        selected_labels = self.labels[self.indices]

        self.positive_count = int(np.count_nonzero(selected_labels == 1))
        self.negative_count = int(np.count_nonzero(selected_labels == 0))

        # 既存 train_policy.py との互換用 alias
        self.positive_samples = self.positive_count
        self.negative_samples = self.negative_count
        self.pos_count = self.positive_count
        self.neg_count = self.negative_count

    def _build_indices(
        self,
        *,
        labels: np.ndarray,
        max_negative_ratio: int,
        balance: bool,
        seed: int,
    ) -> np.ndarray:
        labels = labels.astype(np.int8, copy=False)

        if not balance:
            return np.arange(labels.shape[0], dtype=np.int64)

        pos_idx = np.where(labels == 1)[0].astype(np.int64)
        neg_idx = np.where(labels == 0)[0].astype(np.int64)

        if pos_idx.shape[0] == 0:
            return np.arange(labels.shape[0], dtype=np.int64)

        max_neg = int(pos_idx.shape[0] * max(1, max_negative_ratio))

        rng = np.random.default_rng(seed)

        if neg_idx.shape[0] > max_neg:
            neg_idx = rng.choice(
                neg_idx,
                size=max_neg,
                replace=False,
            ).astype(np.int64)

        indices = np.concatenate(
            [
                pos_idx,
                neg_idx,
            ],
            axis=0,
        ).astype(np.int64)

        rng.shuffle(indices)

        return indices

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])

        x_np = np.asarray(
            self.X[real_idx],
            dtype=np.float32,
        )

        y_np = np.asarray(
            self.Y[real_idx],
            dtype=np.float32,
        )

        # mmap_mode="r" の場合、NumPy配列が read-only になることがある。
        if not x_np.flags.writeable:
            x_np = x_np.copy()

        if not y_np.flags.writeable:
            y_np = y_np.copy()

        return (
            torch.from_numpy(x_np),
            torch.from_numpy(y_np),
        )
    # 【修正4】最末尾に存在した、古い重複メソッド（__len__, __getitem__）は削除しました