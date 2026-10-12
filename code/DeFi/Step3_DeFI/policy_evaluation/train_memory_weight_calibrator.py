from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import (
    encode_texts_t5,
    load_feature,
    load_labeled_rows,
    load_memory_arrays,
    load_rollout_rows,
    split_sequence_indices,
)
from policy_evaluation.memory_weight_calibrator import MemoryWeightCalibrator, pad_weight_inputs


def mean_pool_feature(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)


def parse_qwen_weight_map(row: Dict[str, Any], memory_ids: List[str]) -> Dict[str, float]:
    raw = row.get("memory_weights")
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    if memory_ids:
        value = 1.0 / float(len(memory_ids))
        return {memory_id: value for memory_id in memory_ids}
    return {}


def build_examples(
    rollout_rows: Dict[int, Dict[str, Any]],
    labeled_rows: List[Dict[str, Any]],
    feature_root: Path,
    memory_npz: Dict[str, np.ndarray],
    repair_targets_npz: Dict[str, np.ndarray],
    topk: int,
) -> Dict[str, np.ndarray]:
    row_ids = memory_npz["row_ids"].astype(np.int64)
    tasks = [str(item) for item in memory_npz["tasks"].tolist()]
    future_bank = memory_npz["future_features"].astype(np.float32)
    delta_bank = memory_npz["repair_deltas"].astype(np.float32)
    row_to_bank = {int(row_id): idx for idx, row_id in enumerate(row_ids.tolist())}

    target_row_ids = repair_targets_npz["row_ids"].astype(np.int64)
    repair_target_deltas = repair_targets_npz["repair_deltas"].astype(np.float32)
    target_map = {int(row_id): repair_target_deltas[idx] for idx, row_id in enumerate(target_row_ids.tolist())}

    future_dim = future_bank.shape[-1]
    x_future = []
    x_memory = []
    x_qwen = []
    x_sims = []
    x_mask = []
    y_weights = []
    seq_rows = []
    meta_rows = []

    for row in labeled_rows:
        row_id = int(row["row_id"])
        rollout = rollout_rows.get(row_id)
        target_delta = target_map.get(row_id)
        if rollout is None or target_delta is None:
            continue
        feat = load_feature(feature_root / str(rollout["feature_path"]))
        future_feature = mean_pool_feature(feat["defi_future_feature"])
        task = str(row["task"])

        candidate_ids = [
            int(candidate_row_id)
            for candidate_row_id, candidate_task in zip(row_ids.tolist(), tasks)
            if candidate_task == task and int(candidate_row_id) != row_id
        ]
        if not candidate_ids:
            candidate_ids = [int(candidate_row_id) for candidate_row_id in row_ids.tolist() if int(candidate_row_id) != row_id]
        if not candidate_ids:
            continue

        scored = []
        for candidate_row_id in candidate_ids:
            bank_idx = row_to_bank[candidate_row_id]
            cand_future = mean_pool_feature(future_bank[bank_idx])
            sim = float(np.dot(future_feature, cand_future) / (np.linalg.norm(future_feature) * np.linalg.norm(cand_future) + 1e-12))
            scored.append((sim, candidate_row_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:topk]
        memory_ids = [f"KEY_{candidate_row_id}" for _, candidate_row_id in top]
        similarity_map = {f"KEY_{candidate_row_id}": sim for sim, candidate_row_id in top}
        qwen_weight_map = parse_qwen_weight_map(row, memory_ids)
        qwen_weights, sims, mask = pad_weight_inputs(qwen_weight_map, similarity_map, memory_ids, topk)

        mem_feats = np.zeros((topk, future_dim), dtype=np.float32)
        target_scores = np.zeros((topk,), dtype=np.float32)
        pooled_target = mean_pool_feature(target_delta)
        for idx, (_, candidate_row_id) in enumerate(top):
            bank_idx = row_to_bank[candidate_row_id]
            mem_feats[idx] = mean_pool_feature(delta_bank[bank_idx])
            target_scores[idx] = float(
                np.dot(pooled_target, mem_feats[idx]) / (np.linalg.norm(pooled_target) * np.linalg.norm(mem_feats[idx]) + 1e-12)
            )
        if float(mask.sum()) <= 0:
            continue
        target_scores = np.clip(target_scores, 1e-6, None)
        target_scores = target_scores / float(target_scores.sum())

        x_future.append(future_feature.astype(np.float32))
        x_memory.append(mem_feats.astype(np.float32))
        x_qwen.append(qwen_weights.numpy().astype(np.float32))
        x_sims.append(sims.numpy().astype(np.float32))
        x_mask.append(mask.numpy())
        y_weights.append(target_scores.astype(np.float32))
        seq_rows.append(int(row["sequence_index"]))
        meta_rows.append({"row_id": row_id, "task": task, "memory_ids": memory_ids})

    return {
        "future": np.stack(x_future, axis=0).astype(np.float32),
        "memory": np.stack(x_memory, axis=0).astype(np.float32),
        "qwen": np.stack(x_qwen, axis=0).astype(np.float32),
        "sims": np.stack(x_sims, axis=0).astype(np.float32),
        "mask": np.stack(x_mask, axis=0).astype(bool),
        "target": np.stack(y_weights, axis=0).astype(np.float32),
        "sequence_index": np.asarray(seq_rows, dtype=np.int64),
        "meta": meta_rows,
    }


def evaluate_model(
    model: MemoryWeightCalibrator,
    future: np.ndarray,
    memory: np.ndarray,
    qwen: np.ndarray,
    sims: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    with torch.no_grad():
        pred, _, _ = model(
            torch.from_numpy(future),
            torch.from_numpy(memory),
            torch.from_numpy(qwen),
            torch.from_numpy(sims),
            torch.from_numpy(mask),
        )
    pred_np = pred.cpu().numpy()
    mse = float(np.mean((pred_np - target) ** 2))
    kl = float(np.mean(np.sum(target * (np.log(target + 1e-12) - np.log(pred_np + 1e-12)), axis=1)))
    top1_acc = float(np.mean(np.argmax(pred_np, axis=1) == np.argmax(target, axis=1)))
    return {"mse": mse, "kl": kl, "top1_acc": top1_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory weight calibrator from Qwen weights + retrieval scores.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--labeled-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--repair-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rollout_rows = load_rollout_rows(args.rollout_jsonl)
    labeled_rows = load_labeled_rows(args.labeled_jsonl)
    memory_npz = load_memory_arrays(args.memory_npz)
    repair_targets_npz = load_memory_arrays(args.repair_targets)
    dataset = build_examples(rollout_rows, labeled_rows, args.feature_root, memory_npz, repair_targets_npz, args.topk)

    train_idx, val_idx = split_sequence_indices(dataset["sequence_index"])
    train = {k: dataset[k][train_idx] for k in ["future", "memory", "qwen", "sims", "mask", "target"]}
    val = {k: dataset[k][val_idx] for k in ["future", "memory", "qwen", "sims", "mask", "target"]}

    model = MemoryWeightCalibrator(
        future_dim=train["future"].shape[1],
        memory_dim=train["memory"].shape[2],
        max_memories=args.topk,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["future"]),
            torch.from_numpy(train["memory"]),
            torch.from_numpy(train["qwen"]),
            torch.from_numpy(train["sims"]),
            torch.from_numpy(train["mask"]),
            torch.from_numpy(train["target"]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_kl = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for future_b, memory_b, qwen_b, sims_b, mask_b, target_b in loader:
            future_b = future_b.to(device)
            memory_b = memory_b.to(device)
            qwen_b = qwen_b.to(device)
            sims_b = sims_b.to(device)
            mask_b = mask_b.to(device)
            target_b = target_b.to(device)
            pred, _, aux = model(future_b, memory_b, qwen_b, sims_b, mask_b)
            loss = F.kl_div(torch.log(pred + 1e-12), target_b, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        metrics = evaluate_model(model.cpu(), val["future"], val["memory"], val["qwen"], val["sims"], val["mask"], val["target"])
        model.to(device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
        if metrics["kl"] < best_kl:
            best_kl = metrics["kl"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    torch.save(
        {
            "model_state": model.state_dict(),
            "future_dim": int(train["future"].shape[1]),
            "memory_dim": int(train["memory"].shape[2]),
            "max_memories": int(args.topk),
            "hidden_dim": int(args.hidden_dim),
        },
        args.output_dir / "memory_weight_calibrator.pt",
    )
    summary = {
        "train_num": int(len(train_idx)),
        "val_num": int(len(val_idx)),
        "best_val": evaluate_model(model, val["future"], val["memory"], val["qwen"], val["sims"], val["mask"], val["target"]),
        "history_tail": history[-5:],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
