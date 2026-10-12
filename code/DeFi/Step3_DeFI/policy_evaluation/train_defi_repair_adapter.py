import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import (
    RepairAdapter,
    build_repair_input,
    encode_texts_t5,
    load_feature,
    load_labeled_rows,
    load_memory_arrays,
    load_rollout_rows,
    repair_target_mask,
    retrieve_memory_topk,
    split_sequence_indices,
)


def build_dataset(
    rollout_rows: Dict[int, Dict[str, Any]],
    labeled_rows: List[Dict[str, Any]],
    feature_root: Path,
    memory_arrays: Dict[str, np.ndarray],
    repair_targets_npz: Dict[str, np.ndarray],
    reflection_embeddings: np.ndarray,
    topk: int,
) -> Dict[str, np.ndarray]:
    row_ids = repair_targets_npz["row_ids"].astype(np.int64)
    repair_deltas = repair_targets_npz["repair_deltas"].astype(np.float32)
    row_to_idx = {int(row_id): idx for idx, row_id in enumerate(row_ids.tolist())}

    x_rows = []
    y_rows = []
    seq_rows = []
    meta_rows = []
    target_shape = None
    for idx, row in enumerate(labeled_rows):
        row_id = int(row["row_id"])
        repair_delta = repair_deltas[row_to_idx[row_id]]
        if not repair_target_mask(repair_delta, row):
            continue
        if target_shape is None:
            target_shape = list(repair_delta.shape)
        rollout = rollout_rows[row_id]
        feat = load_feature(feature_root / rollout["feature_path"])
        future_feature = feat["defi_future_feature"].astype(np.float32)
        memory_future, memory_delta, memory_row_ids, memory_scores = retrieve_memory_topk(
            str(row["task"]),
            future_feature,
            memory_arrays,
            topk,
            exclude_row_id=row_id,
        )
        x_rows.append(
            build_repair_input(
                future_feature,
                memory_future,
                memory_delta,
                reflection_embeddings[idx],
            )
        )
        y_rows.append(repair_delta.astype(np.float32).reshape(-1))
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
        "y": np.stack(y_rows, axis=0).astype(np.float32),
        "sequence_index": np.asarray(seq_rows, dtype=np.int64),
        "meta": meta_rows,
        "target_shape": target_shape or [],
    }


def mse_metrics(model: RepairAdapter, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy()
    mse = float(np.mean((pred - y) ** 2))
    corrected_norm = float(np.mean(np.linalg.norm(pred, axis=1)))
    target_norm = float(np.mean(np.linalg.norm(y, axis=1)))
    cosine = np.sum(pred * y, axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    return {
        "mse": mse,
        "delta_norm_pred": corrected_norm,
        "delta_norm_target": target_norm,
        "delta_cosine": float(np.mean(cosine)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeFi single-future repair adapter.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--labeled-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--repair-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t5-path", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
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
    repair_targets_npz = load_memory_arrays(args.repair_targets)
    reflection_texts = [str(row.get("reflection_text", "") or "uncertain") for row in labeled_rows]
    reflection_embeddings = encode_texts_t5(reflection_texts, args.t5_path, device, batch_size=32)
    dataset = build_dataset(
        rollout_rows,
        labeled_rows,
        args.feature_root,
        memory_arrays,
        repair_targets_npz,
        reflection_embeddings,
        args.topk,
    )

    train_idx, val_idx = split_sequence_indices(dataset["sequence_index"])
    x_train = dataset["x"][train_idx]
    y_train = dataset["y"][train_idx]
    x_val = dataset["x"][val_idx]
    y_val = dataset["y"][val_idx]

    model = RepairAdapter(x_train.shape[1], y_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_mse = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        metrics = mse_metrics(model.cpu(), x_val, y_val)
        model.to(device)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_mse": metrics["mse"],
            "val_delta_cosine": metrics["delta_cosine"],
        }
        history.append(record)
        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "future_dim": int(y_train.shape[1]),
        "future_shape": list(dataset["target_shape"]),
        "hidden_dim": int(args.hidden_dim),
        "topk": int(args.topk),
        "t5_path": str(args.t5_path),
    }
    torch.save(checkpoint, args.output_dir / "repair_adapter.pt")

    summary = {
        "train_num": int(len(train_idx)),
        "val_num": int(len(val_idx)),
        "best_val": mse_metrics(model, x_val, y_val),
        "history_tail": history[-5:],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
