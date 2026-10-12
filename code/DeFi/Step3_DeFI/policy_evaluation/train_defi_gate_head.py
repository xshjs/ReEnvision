import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import (
    GateHead,
    build_gate_input,
    encode_texts_t5,
    load_feature,
    load_labeled_rows,
    load_memory_arrays,
    load_rollout_rows,
    repair_label,
    retrieve_memory_topk,
    split_sequence_indices,
)


def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.1, 0.9, 17):
        pred = probs >= threshold
        tp = float(np.logical_and(pred, labels == 1).sum())
        fp = float(np.logical_and(pred, labels == 0).sum())
        fn = float(np.logical_and(~pred, labels == 1).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(f1), "precision": float(precision), "recall": float(recall)}
    return best


def build_dataset(
    rollout_rows: Dict[int, Dict[str, Any]],
    labeled_rows: List[Dict[str, Any]],
    feature_root: Path,
    memory_arrays: Dict[str, np.ndarray],
    reflection_embeddings: np.ndarray,
    topk: int,
) -> Dict[str, np.ndarray]:
    x_rows = []
    y_rows = []
    seq_rows = []
    meta_rows = []
    for idx, row in enumerate(labeled_rows):
        row_id = int(row["row_id"])
        rollout = rollout_rows[row_id]
        feat = load_feature(feature_root / rollout["feature_path"])
        current_feature = feat["obs_current_feature"].astype(np.float32)
        future_feature = feat["defi_future_feature"].astype(np.float32)
        memory_future, memory_delta, memory_row_ids, memory_scores = retrieve_memory_topk(
            str(row["task"]),
            future_feature,
            memory_arrays,
            topk,
            exclude_row_id=row_id,
        )
        gate_input = build_gate_input(
            current_feature,
            future_feature,
            memory_future,
            memory_delta,
            reflection_embeddings[idx],
            memory_scores,
        )
        x_rows.append(gate_input)
        y_rows.append(float(repair_label(row)))
        seq_rows.append(int(row["sequence_index"]))
        meta_rows.append(
            {
                "row_id": row_id,
                "task": str(row["task"]),
                "memory_row_ids": [int(item) for item in memory_row_ids.tolist()],
                "memory_scores": [float(item) for item in memory_scores.tolist()],
            }
        )
    return {
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "y": np.asarray(y_rows, dtype=np.float32),
        "sequence_index": np.asarray(seq_rows, dtype=np.int64),
        "meta": meta_rows,
    }


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, float]:
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.sigmoid(logits).cpu().numpy()
    pred = probs >= threshold
    acc = float((pred == (y > 0.5)).mean())
    tp = float(np.logical_and(pred, y > 0.5).sum())
    fp = float(np.logical_and(pred, y <= 0.5).sum())
    fn = float(np.logical_and(~pred, y > 0.5).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(pred.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeFi single-future trust/gate head.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--labeled-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t5-path", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rollout_rows = load_rollout_rows(args.rollout_jsonl)
    labeled_rows = load_labeled_rows(args.labeled_jsonl)
    memory_arrays = load_memory_arrays(args.memory_npz)
    reflection_texts = [str(row.get("reflection_text", "") or "uncertain") for row in labeled_rows]
    reflection_embeddings = encode_texts_t5(reflection_texts, args.t5_path, device, batch_size=32)
    dataset = build_dataset(rollout_rows, labeled_rows, args.feature_root, memory_arrays, reflection_embeddings, args.topk)

    train_idx, val_idx = split_sequence_indices(dataset["sequence_index"])
    x_train = dataset["x"][train_idx]
    y_train = dataset["y"][train_idx]
    x_val = dataset["x"][val_idx]
    y_val = dataset["y"][val_idx]

    model = GateHead(x_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_score = -1.0
    best_threshold = 0.5
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.from_numpy(x_val).to(device))).cpu().numpy()
        best = best_f1_threshold(val_probs, y_val)
        metrics = evaluate_model(model.cpu(), x_val, y_val, best["threshold"])
        model.to(device)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_threshold": best["threshold"],
            "val_f1": metrics["f1"],
            "val_acc": metrics["acc"],
        }
        history.append(record)
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_threshold = float(best["threshold"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "threshold": float(best_threshold),
        "topk": int(args.topk),
        "t5_path": str(args.t5_path),
    }
    torch.save(checkpoint, args.output_dir / "gate_head.pt")

    summary = {
        "train_num": int(len(train_idx)),
        "val_num": int(len(val_idx)),
        "positive_train": int((y_train > 0.5).sum()),
        "positive_val": int((y_val > 0.5).sum()),
        "best_threshold": float(best_threshold),
        "best_val": evaluate_model(model, x_val, y_val, best_threshold),
        "history_tail": history[-5:],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
