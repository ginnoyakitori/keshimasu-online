# python/apps/export_policy_onnx.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
PY_SRC = ROOT / "python" / "src"

if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))


from keshimasu_py.config import PROJECT_ROOT  # noqa: E402
from keshimasu_py.torch_policy import PolicyMLP, feature_dim  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "torch_policy"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "numpy_dataset" / "policy"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def torch_load_checkpoint(path: Path, device: torch.device) -> Any:
    """
    PyTorch 2.6+ では weights_only の挙動が環境により変わる場合があるため、
    互換性優先で weights_only=False を試す。
    """
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    DataParallel 等で module. prefix が付いている場合に除去する。
    """
    out: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            out[key[len("module.") :]] = value
        else:
            out[key] = value

    return out


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    checkpoint 形式の揺れを吸収する。

    対応:
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - {"model": ...}
      - 直接 state_dict
    """
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "modelStateDict",
        ]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return strip_module_prefix(checkpoint[key])

        # 直接 state_dict の可能性
        if checkpoint:
            tensor_like = all(
                hasattr(v, "shape")
                for v in checkpoint.values()
            )

            if tensor_like:
                return strip_module_prefix(checkpoint)

    raise ValueError(
        "Could not extract model state_dict from checkpoint. "
        "Expected key: model_state_dict / state_dict / model."
    )


def extract_checkpoint_metadata(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in [
            "metadata",
            "meta",
            "model_metadata",
        ]:
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

    return {}


def infer_input_size(
    *,
    checkpoint: Any,
    metadata: Dict[str, Any],
    dataset_dir: Path,
) -> int:
    """
    input_size を複数候補から推定する。
    """
    if isinstance(checkpoint, dict):
        for key in [
            "input_size",
            "inputSize",
            "feature_dim",
            "featureDim",
        ]:
            if key in checkpoint:
                return int(checkpoint[key])

    summary = metadata.get("summary", {})

    feature_format = summary.get("featureFormat", {})

    for key in [
        "inputSize",
        "input_size",
        "featureDim",
        "feature_dim",
    ]:
        if key in feature_format:
            return int(feature_format[key])

    for key in [
        "inputSize",
        "input_size",
    ]:
        if key in metadata:
            return int(metadata[key])

    x_path = dataset_dir / "X.npy"

    if x_path.exists():
        x = np.load(x_path, mmap_mode="r")
        return int(x.shape[1])

    return int(feature_dim())


def infer_char_vocab_size(metadata: Dict[str, Any]) -> int:
    char_to_id = metadata.get("charToId", {})

    if isinstance(char_to_id, dict):
        return int(len(char_to_id) + 1)

    return 0


def infer_num_words(metadata: Dict[str, Any], dataset_dir: Path) -> int:
    words = metadata.get("words", [])

    if isinstance(words, list) and words:
        return int(len(words))

    word_array_path = dataset_dir / "word_array.npy"

    if word_array_path.exists():
        word_array = np.load(word_array_path, mmap_mode="r")
        return int(word_array.shape[0])

    return 0


def build_model(
    *,
    input_size: int,
    checkpoint: Any,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> PolicyMLP:
    """
    現在の PolicyMLP アーキテクチャでモデルを復元する。
    """
    hidden1 = 256
    hidden2 = 128
    hidden3 = 64
    dropout = 0.15

    if isinstance(checkpoint, dict):
        hidden1 = int(checkpoint.get("hidden1", hidden1))
        hidden2 = int(checkpoint.get("hidden2", hidden2))
        hidden3 = int(checkpoint.get("hidden3", hidden3))
        dropout = float(checkpoint.get("dropout", dropout))

        arch = checkpoint.get("architecture", {})

        if isinstance(arch, dict):
            hidden1 = int(arch.get("hidden1", hidden1))
            hidden2 = int(arch.get("hidden2", hidden2))
            hidden3 = int(arch.get("hidden3", hidden3))
            dropout = float(arch.get("dropout", dropout))

    model = PolicyMLP(
        input_size=input_size,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        dropout=dropout,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def export_onnx(
    *,
    model: torch.nn.Module,
    input_size: int,
    output_path: Path,
    device: torch.device,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(
        1,
        input_size,
        dtype=torch.float32,
        device=device,
    )

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={
            "x": {
                0: "batch",
            },
            "logits": {
                0: "batch",
            },
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def try_embed_onnx_metadata(
    *,
    onnx_path: Path,
    metadata_payload: Dict[str, Any],
) -> bool:
    """
    ONNX metadata_props に JSON 文字列を埋め込む。

    onnx パッケージが無い場合は False を返す。
    """
    try:
        import onnx  # type: ignore
    except Exception as e:
        print(
            "WARNING: Could not import onnx package. "
            "ONNX metadata embedding skipped."
        )
        print("Reason:", e)
        return False

    model = onnx.load(str(onnx_path))

    existing_keys = {
        prop.key
        for prop in model.metadata_props
    }

    def add_metadata(key: str, value: Any) -> None:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False)
        )

        if key in existing_keys:
            for prop in model.metadata_props:
                if prop.key == key:
                    prop.value = text
                    return

        prop = model.metadata_props.add()
        prop.key = key
        prop.value = text

    add_metadata("keshimasu_policy_metadata", metadata_payload)
    add_metadata("input_size", str(metadata_payload.get("inputSize", "")))
    add_metadata("feature_dim", str(metadata_payload.get("featureDim", "")))
    add_metadata("char_vocab_size", str(metadata_payload.get("charVocabSize", "")))
    add_metadata("num_words", str(metadata_payload.get("numWords", "")))
    add_metadata("move_format", "word_id,row,col,direction")
    add_metadata("direction", {"H": 0, "V": 1})

    onnx.save(model, str(onnx_path))

    return True


def verify_with_torch(
    *,
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.no_grad():
        x = torch.zeros(
            3,
            input_size,
            dtype=torch.float32,
            device=device,
        )

        logits = model(x)
        probs = torch.sigmoid(logits)

    return {
        "torchLogitsShape": list(logits.shape),
        "torchProbsShape": list(probs.shape),
        "torchProbsSample": [
            float(v)
            for v in probs.reshape(-1).detach().cpu().numpy().tolist()
        ],
    }


def try_verify_with_onnxruntime(
    *,
    onnx_path: Path,
    input_size: int,
) -> Dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        return {
            "onnxRuntimeAvailable": False,
            "reason": str(e),
        }

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    x = np.zeros(
        (3, input_size),
        dtype=np.float32,
    )

    outputs = session.run(
        None,
        {
            "x": x,
        },
    )

    logits = outputs[0]

    return {
        "onnxRuntimeAvailable": True,
        "onnxLogitsShape": list(logits.shape),
        "onnxLogitsSample": [
            float(v)
            for v in logits.reshape(-1).tolist()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained Torch policy model to ONNX."
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing policy_model.pt and metadata.json.",
    )

    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory containing X.npy / word_array.npy.",
    )

    parser.add_argument(
        "--checkpoint",
        default="policy_model.pt",
        help="Checkpoint file name or path.",
    )

    parser.add_argument(
        "--output",
        default="policy_model.onnx",
        help="ONNX output file name or path.",
    )

    parser.add_argument(
        "--metadata-output",
        default="onnx_metadata.json",
        help="Sidecar metadata json file name or path.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run basic torch and optional onnxruntime verification.",
    )

    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return base_dir / path


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    checkpoint_path = resolve_path(
        model_dir,
        args.checkpoint,
    ).resolve()

    onnx_path = resolve_path(
        model_dir,
        args.output,
    ).resolve()

    metadata_output_path = resolve_path(
        model_dir,
        args.metadata_output,
    ).resolve()

    device = torch.device(args.device)

    print("=== EXPORT POLICY ONNX ===")
    print("Project root:", PROJECT_ROOT)
    print("Model dir:", model_dir)
    print("Dataset dir:", dataset_dir)
    print("Checkpoint:", checkpoint_path)
    print("Output ONNX:", onnx_path)
    print("Metadata output:", metadata_output_path)
    print("Device:", device)
    print("Opset:", args.opset)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch_load_checkpoint(
        checkpoint_path,
        device,
    )

    state_dict = extract_state_dict(checkpoint)

    checkpoint_metadata = extract_checkpoint_metadata(checkpoint)
    sidecar_metadata = load_json(model_dir / "metadata.json")
    dataset_metadata = load_json(dataset_dir / "metadata.json")

    metadata: Dict[str, Any] = {}

    # 優先順位:
    # dataset metadata > model sidecar metadata > checkpoint metadata
    metadata.update(checkpoint_metadata)
    metadata.update(sidecar_metadata)
    metadata.update(dataset_metadata)

    input_size = infer_input_size(
        checkpoint=checkpoint,
        metadata=metadata,
        dataset_dir=dataset_dir,
    )

    char_vocab_size = infer_char_vocab_size(metadata)
    num_words = infer_num_words(
        metadata,
        dataset_dir,
    )

    print("Input size:", input_size)
    print("Feature dim:", feature_dim())
    print("Char vocab size:", char_vocab_size)
    print("Num words:", num_words)

    model = build_model(
        input_size=input_size,
        checkpoint=checkpoint,
        state_dict=state_dict,
        device=device,
    )

    export_onnx(
        model=model,
        input_size=input_size,
        output_path=onnx_path,
        device=device,
        opset=args.opset,
    )

    metadata_payload: Dict[str, Any] = {
        "format": "keshimasu_policy_onnx",
        "modelType": "PolicyMLP",
        "inputName": "x",
        "outputName": "logits",
        "inputSize": int(input_size),
        "featureDim": int(feature_dim()),
        "charVocabSize": int(char_vocab_size),
        "numWords": int(num_words),
        "moveFormat": [
            "word_id",
            "row",
            "col",
            "direction",
        ],
        "direction": {
            "H": 0,
            "V": 1,
        },
        "featureLayout": {
            "0_39": "board_flattened / char_vocab_size",
            "40": "word_id / (num_words - 1)",
            "41": "row / 7",
            "42": "col / 4",
            "43": "direction raw H=0 V=1",
            "44": "is_horizontal",
            "45": "is_vertical",
            "46": "word_length / 5",
            "47": "target_f / 3",
            "48": "depth / 120",
            "49": "log1p(branch_count) / log1p(256)",
            "50": "start_index / 39",
            "51": "end_row / 7",
            "52": "end_col / 4",
            "53": "filled_ratio",
        },
        "sourceCheckpoint": str(checkpoint_path),
        "sourceMetadata": str(model_dir / "metadata.json"),
        "sourceDatasetMetadata": str(dataset_dir / "metadata.json"),
    }

    if "words" in metadata:
        metadata_payload["words"] = metadata["words"]

    if "wordLengths" in metadata:
        metadata_payload["wordLengths"] = metadata["wordLengths"]

    if "charToId" in metadata:
        metadata_payload["charToId"] = metadata["charToId"]

    if "idToChar" in metadata:
        metadata_payload["idToChar"] = metadata["idToChar"]

    if "wildId" in metadata:
        metadata_payload["wildId"] = metadata["wildId"]

    if "emptyId" in metadata:
        metadata_payload["emptyId"] = metadata["emptyId"]

    embedded = try_embed_onnx_metadata(
        onnx_path=onnx_path,
        metadata_payload=metadata_payload,
    )

    metadata_payload["onnxMetadataEmbedded"] = bool(embedded)

    verification: Dict[str, Any] = {}

    if args.verify:
        verification["torch"] = verify_with_torch(
            model=model,
            input_size=input_size,
            device=device,
        )

        verification["onnxruntime"] = try_verify_with_onnxruntime(
            onnx_path=onnx_path,
            input_size=input_size,
        )

        metadata_payload["verification"] = verification

    save_json(
        metadata_output_path,
        metadata_payload,
    )

    print("\n=== EXPORT DONE ===")
    print("ONNX:", onnx_path)
    print("Metadata:", metadata_output_path)
    print("Metadata embedded:", embedded)

    if args.verify:
        print("\n=== VERIFY ===")
        print(
            json.dumps(
                verification,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()