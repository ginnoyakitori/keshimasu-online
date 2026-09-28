# python/apps/train_policy.py

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from keshimasu_py.config import PROJECT_ROOT
from keshimasu_py.torch_policy import (
    PolicyMoveDataset,
    PolicyMLP,
    count_parameters,
)


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "numpy_dataset" / "policy"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

        print("WARNING: CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        print("WARNING: MPS is not available. Falling back to CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"

    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    tn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        logits = model(x).squeeze(-1)
        labels = y.squeeze(-1)

        loss = criterion(logits, labels)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        labels_long = labels.long()

        correct += (preds == labels_long).sum().item()

        tp += ((preds == 1) & (labels_long == 1)).sum().item()
        fp += ((preds == 1) & (labels_long == 0)).sum().item()
        fn += ((preds == 0) & (labels_long == 1)).sum().item()
        tn += ((preds == 0) & (labels_long == 0)).sum().item()

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    specificity = tn / max(1, tn + fp)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    model.train()

    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        logits = model(x).squeeze(-1)
        labels = y.squeeze(-1)

        loss = criterion(logits, labels)

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / max(1, total)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=== TRAIN TORCH POLICY ===")
    print("Dataset:", dataset_dir)
    print("Model dir:", model_dir)

    dataset_metadata = load_dataset_metadata(dataset_dir)

    if dataset_metadata:
        summary = dataset_metadata.get("summary", {})
        print("Loaded metadata.json")
        print("  moveExamples:", summary.get("moveExamples"))
        print("  policyExamples:", summary.get("policyExamples"))
        print("  rows:", summary.get("rows"))
        print("  cols:", summary.get("cols"))
        print("  maxWordLength:", summary.get("maxWordLength"))

    dataset = PolicyMoveDataset(
        dataset_dir=dataset_dir,
        max_negative_ratio=args.max_negative_ratio,
        balance=not args.no_balance,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your .npy files.")

    print("Dataset size:", len(dataset))
    print("Input size:", dataset.input_size)
    print("Char vocab size:", dataset.char_vocab_size)
    print("Num words:", dataset.num_words)

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("Train size became zero. Reduce --val-split.")

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
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    device = choose_device(args.device)
    print("Device:", device)

    input_size = dataset.input_size

    model = PolicyMLP(
        input_size=input_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        hidden3=args.hidden3,
        dropout=args.dropout,
    ).to(device)

    print("Trainable params:", count_parameters(model))

    labels_balanced = dataset.labels[dataset.indices]
    pos = max(1, int((labels_balanced == 1).sum()))
    neg = max(1, int((labels_balanced == 0).sum()))

    if args.no_pos_weight:
        pos_weight_value = 1.0
    else:
        pos_weight_value = neg / pos

    print("Positive samples:", pos)
    print("Negative samples:", neg)
    print("pos_weight:", pos_weight_value)

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
        patience=args.lr_patience,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_path = model_dir / "policy_model.pt"

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        scheduler.step(val_metrics["loss"])

        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "trainLoss": float(train_loss),
            "valLoss": float(val_metrics["loss"]),
            "valAcc": float(val_metrics["acc"]),
            "valPrecision": float(val_metrics["precision"]),
            "valRecall": float(val_metrics["recall"]),
            "valSpecificity": float(val_metrics["specificity"]),
            "valF1": float(val_metrics["f1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsedSec": float(elapsed),
        }

        history.append(row)

        print(
            "epoch={epoch:03d} "
            "time={elapsedSec:.1f}s "
            "train_loss={trainLoss:.4f} "
            "val_loss={valLoss:.4f} "
            "val_acc={valAcc:.4f} "
            "val_f1={valF1:.4f} "
            "precision={valPrecision:.4f} "
            "recall={valRecall:.4f} "
            "lr={lr:.6f}".format(**row)
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),

                # policy_inference.py / solver_policy.py 互換に必須
                "input_size": dataset.input_size,
                "char_vocab_size": dataset.char_vocab_size,
                "num_words": dataset.num_words,
                "word_array": dataset.word_array,
                "word_lengths": dataset.word_lengths,
                "metadata": dataset.metadata,

                # 再現・解析用
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "history": history,
                "pos_weight": float(pos_weight_value),
            }

            torch.save(
                checkpoint,
                best_path,
            )

            print(f"  saved best -> {best_path}")

        if args.early_stop > 0:
            if epoch - best_epoch >= args.early_stop:
                print(
                    f"Early stopping: no improvement for {args.early_stop} epochs."
                )
                break

    elapsed_total = time.time() - start_time

    final_metadata = {
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasetDir": str(dataset_dir),
        "modelPath": str(best_path),
        "bestEpoch": best_epoch,
        "bestValLoss": best_val_loss,
        "elapsedSec": elapsed_total,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PyTorch policy network from NumPy dataset"
    )

    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--hidden3", type=int, default=64)

    parser.add_argument("--max-negative-ratio", type=int, default=3)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--no-pos-weight", action="store_true")

    parser.add_argument("--val-split", type=float, default=0.15)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--early-stop",
        type=int,
        default=8,
        help="Stop if no val_loss improvement for N epochs. 0 disables.",
    )

    parser.add_argument("--lr-patience", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())