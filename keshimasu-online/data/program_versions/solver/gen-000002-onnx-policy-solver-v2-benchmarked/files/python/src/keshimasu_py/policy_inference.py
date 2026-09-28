# python/src/keshimasu_py/policy_inference.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .config import PROJECT_ROOT
from .torch_policy import (
    PolicyMLP,
    make_feature_from_arrays,
    feature_dim,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"


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

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        import json

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
        if self.model is None:
            raise RuntimeError("Policy model is not loaded. Call load() first.")

        if moves.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        bcount = branch_count if branch_count is not None else int(moves.shape[0])

        features = []

        for i in range(moves.shape[0]):
            feat = make_feature_from_arrays(
                board=board,
                move=moves[i],
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

        scores = []

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
        policy_weight: float = 0.75,
        heuristic_weight: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        heuristic_scores = heuristic_scores.astype(np.float32)

        h_min = float(np.min(heuristic_scores))
        h_max = float(np.max(heuristic_scores))

        if h_max > h_min:
            heuristic_norm = (heuristic_scores - h_min) / (h_max - h_min)
        else:
            heuristic_norm = np.zeros_like(heuristic_scores, dtype=np.float32)

        final_scores = (
            policy_scores.astype(np.float32) * float(policy_weight)
            + heuristic_norm.astype(np.float32) * float(heuristic_weight)
        )

        order = np.argsort(-final_scores)

        return moves[order], policy_scores[order]