# python/train_policy_torch.py
#
# PyTorch Policy Network Trainer
#
# 入力:
#   data/numpy_dataset/policy/
#     boards.npy          [N, 8, 5] int16
#     moves.npy           [N, 4] int16
#     labels.npy          [N] int8
#     branch_counts.npy   [N] int16
#     target_f.npy        [N] int16
#     depths.npy          [N] int16
#     word_array.npy      [W, 5] int16
#     word_lengths.npy    [W] int16
#     metadata.json
#
# 出力:
#   data/models/torch_policy/
#     policy_model.pt
#     metadata.json
#
# 実行:
#   python python/train_policy_torch.py
#
# オプション:
#   python python/train_policy_torch.py --epochs 20
#   python python/train_policy_torch.py --batch-size 256
#   python python/train_policy_torch.py --device cpu
#   python python/train_policy_torch.py --device cuda
#

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "numpy_dataset" / "policy"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"


# ============================================================
# Constants
# ============================================================

ROWS = 8
COLS = 5
BOARD_SIZE = ROWS * COLS
MAX_WORD_LENGTH = 5

DIR_H = 0
DIR_V = 1


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Dataset
# ============================================================

class PolicyMoveDataset(Dataset):
    """
    move-level dataset

    1 sample:
      board: [8,5]
      move : [word_id,row,col,direction]
      label: 0/1

    入力特徴:
      - board flattened normalized
      - word chars encoded normalized
      - word length
      - move row/col/direction
      - branch_count
      - target_f
      - depth
    """

    def __init__(
        self,
        dataset_dir: Path,
        max_negative_ratio: int = 3,
        balance: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir

        self.boards = np.load(dataset_dir / "boards.npy")
        self.moves = np.load(dataset_dir / "moves.npy")
        self.labels = np.load(dataset_dir / "labels.npy")
        self.branch_counts = np.load(dataset_dir / "branch_counts.npy")
        self.target_f = np.load(dataset_dir / "target_f.npy")
        self.depths = np.load(dataset_dir / "depths.npy")

        self.word_array = np.load(dataset_dir / "word_array.npy")
        self.word_lengths = np.load(dataset_dir / "word_lengths.npy")

        self.metadata = load_json(dataset_dir / "metadata.json")

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

        self.input_size = self._feature_dim()

    def __len__(self) -> int:
        return len(self.indices)

    def _feature_dim(self) -> int:
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

    def _make_features(self, idx: int) -> np.ndarray:
        board = self.boards[idx].astype(np.float32)
        move = self.moves[idx].astype(np.int64)

        word_id = int(move[0])
        row = int(move[1])
        col = int(move[2])
        direction = int(move[3])

        branch_count = int(self.branch_counts[idx])
        target_f = int(self.target_f[idx])
        depth = int(self.depths[idx])

        # board normalized
        board_flat = board.reshape(-1) / max(1.0, float(self.char_vocab_size))

        # word chars normalized
        word_chars = self.word_array[word_id].astype(np.float32)
        word_chars = word_chars / max(1.0, float(self.char_vocab_size))

        word_length = float(self.word_lengths[word_id]) / MAX_WORD_LENGTH

        word_id_norm = float(word_id) / max(1, self.num_words - 1)

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

        features = np.concatenate(
            [
                board_flat.astype(np.float32),
                word_chars.astype(np.float32),
                extra,
            ],
            axis=0,
        )

        return features.astype(np.float32)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])

        x = self._make_features(idx)
        y = np.array([float(self.labels[idx])], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
# Model
# ============================================================

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


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    tp = 0
    fp = 0
    fn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()

        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / max(1e-8, precision + recall)
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Training
# ============================================================

def train_policy(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    print("=== TRAIN TORCH POLICY ===")
    print("Dataset:", dataset_dir)
    print("Model dir:", model_dir)

    dataset = PolicyMoveDataset(
        dataset_dir=dataset_dir,
        max_negative_ratio=args.max_negative_ratio,
        balance=not args.no_balance,
    )

    print("Dataset size:", len(dataset))
    print("Input size:", dataset.input_size)
    print("Char vocab size:", dataset.char_vocab_size)
    print("Num words:", dataset.num_words)

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Device:", device)

    model = PolicyMLP(
        input_size=dataset.input_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        hidden3=args.hidden3,
        dropout=args.dropout,
    ).to(device)

    print("Trainable params:", count_parameters(model))

    # pos_weight は dataset balance 後の比率で計算
    labels_balanced = dataset.labels[dataset.indices]
    pos = max(1, int((labels_balanced == 1).sum()))
    neg = max(1, int((labels_balanced == 0).sum()))

    pos_weight_value = neg / pos

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    ensure_dir(model_dir)

    best_val_loss = float("inf")
    best_epoch = -1
    best_path = model_dir / "policy_model.pt"

    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = total_loss / max(1, total)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "trainLoss": train_loss,
            "valLoss": val_metrics["loss"],
            "valAcc": val_metrics["acc"],
            "valPrecision": val_metrics["precision"],
            "valRecall": val_metrics["recall"],
            "valF1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            "epoch={epoch:03d} "
            "train_loss={trainLoss:.4f} "
            "val_loss={valLoss:.4f} "
            "val_acc={valAcc:.4f} "
            "val_f1={valF1:.4f} "
            "precision={valPrecision:.4f} "
            "recall={valRecall:.4f} "
            "lr={lr:.6f}".format(**row)
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": dataset.input_size,
                    "char_vocab_size": dataset.char_vocab_size,
                    "num_words": dataset.num_words,
                    "word_array": dataset.word_array,
                    "word_lengths": dataset.word_lengths,
                    "metadata": dataset.metadata,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

            print(f"  saved best -> {best_path}")

        if args.early_stop > 0:
            if epoch - best_epoch >= args.early_stop:
                print(
                    f"Early stopping: no improvement for {args.early_stop} epochs."
                )
                break

    elapsed = time.time() - start_time

    final_metadata = {
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasetDir": str(dataset_dir),
        "modelPath": str(best_path),
        "bestEpoch": best_epoch,
        "bestValLoss": best_val_loss,
        "elapsedSec": elapsed,
        "inputSize": dataset.input_size,
        "charVocabSize": dataset.char_vocab_size,
        "numWords": dataset.num_words,
        "history": history,
        "args": vars(args),
    }

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            final_metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== TRAINING DONE ===")
    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)
    print("Model:", best_path)
    print("Metadata:", model_dir / "metadata.json")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyTorch policy network from NumPy dataset"
    )

    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--hidden1",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden2",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hidden3",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-negative-ratio",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--no-balance",
        action="store_true",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--early-stop",
        type=int,
        default=8,
        help="Stop if no val_loss improvement for N epochs. 0 disables.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train_policy(parse_args())