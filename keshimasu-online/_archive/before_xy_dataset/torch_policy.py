# python/src/keshimasu_py/torch_policy.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import (
    ROWS,
    COLS,
    MAX_WORD_LENGTH,
    DIR_H,
    DIR_V,
)


class PolicyMoveDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        max_negative_ratio: int = 3,
        balance: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)

        self.boards = np.load(self.dataset_dir / "boards.npy")
        self.moves = np.load(self.dataset_dir / "moves.npy")
        self.labels = np.load(self.dataset_dir / "labels.npy")
        self.branch_counts = np.load(self.dataset_dir / "branch_counts.npy")
        self.target_f = np.load(self.dataset_dir / "target_f.npy")
        self.depths = np.load(self.dataset_dir / "depths.npy")

        self.word_array = np.load(self.dataset_dir / "word_array.npy")
        self.word_lengths = np.load(self.dataset_dir / "word_lengths.npy")

        import json

        with (self.dataset_dir / "metadata.json").open(
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        self.char_vocab_size = max(
            int(v)
            for v in self.metadata["charToId"].values()
        )

        self.num_words = self.word_array.shape[0]

        indices = np.arange(self.labels.shape[0])

        if balance:
            pos = indices[self.labels == 1]
            neg = indices[self.labels == 0]

            max_neg = min(len(neg), len(pos) * max_negative_ratio)

            sampled_neg = np.random.choice(
                neg,
                size=max_neg,
                replace=False,
            )

            indices = np.concatenate([pos, sampled_neg])
            np.random.shuffle(indices)

        self.indices = indices.astype(np.int64)
        self.input_size = feature_dim()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])

        x = make_feature_from_arrays(
            board=self.boards[idx],
            move=self.moves[idx],
            word_array=self.word_array,
            word_lengths=self.word_lengths,
            char_vocab_size=self.char_vocab_size,
            num_words=self.num_words,
            branch_count=int(self.branch_counts[idx]),
            target_f=int(self.target_f[idx]),
            depth=int(self.depths[idx]),
        )

        y = np.array([float(self.labels[idx])], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


class PolicyMLP(nn.Module):
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


def feature_dim() -> int:
    return 40 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1


def make_feature_from_arrays(
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

    branch_norm = math.log1p(branch_count) / math.log1p(100.0)
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

    return feature.astype(np.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )