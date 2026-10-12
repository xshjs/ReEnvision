from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import (
    load_feature,
    load_labeled_rows,
    load_memory_arrays,
    load_rollout_rows,
    split_sequence_indices,
)
from policy_evaluation.memory_state_guided_adapter import MemoryStateGuidedAdapter
from policy_evaluation.memory_weight_calibrator import MemoryWeightCalibrator, pad_weight_inputs


def mean_pool_feature(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)


def token_feature(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(-1, arr.shape[-1]).astype(np.float32)


def build_examples(
    rollout_rows: Dict[int, Dict[str, Any]],
    labeled_rows: list[Dict[str, Any]],
    feature_root: Path,
    memory_npz: Dict[str, np.ndarray],
    repair_targets_npz: Dict[str, np.ndarray],
    calibrator_ckpt: Dict[str, Any],
    topk: int,
    device: torch.device,
    token_level: bool,
) -> Dict[str, np.ndarray]:
    row_ids = memory_npz["row_ids"].astype(np.int64)
    tasks = [str(item) for item in memory_npz["tasks"].tolist()]
    future_bank = memory_npz["future_features"].astype(np.float32)
    delta_bank = memory_npz["repair_deltas"].astype(np.float32)
    row_to_bank = {int(row_id): idx for idx, row_id in enumerate(row_ids.tolist())}

    target_row_ids = repair_targets_npz["row_ids"].astype(np.int64)
    repair_target_deltas = repair_targets_npz["repair_deltas"].astype(np.float32)
    target_map = {
        int(row_id): token_feature(repair_target_deltas[idx]) if token_level else mean_pool_feature(repair_target_deltas[idx])
        for idx, row_id in enumerate(target_row_ids.tolist())
    }

    calibrator = MemoryWeightCalibrator(
        future_dim=int(calibrator_ckpt["future_dim"]),
        memory_dim=int(calibrator_ckpt["memory_dim"]),
        max_memories=int(calibrator_ckpt["max_memories"]),
        hidden_dim=int(calibrator_ckpt["hidden_dim"]),
    )
    calibrator.load_state_dict(calibrator_ckpt["model_state"])
    calibrator = calibrator.to(device).eval()

    x_future = []
    x_memory_state = []
    y_delta = []
    seq_rows = []

    for row in labeled_rows:
        row_id = int(row["row_id"])
        rollout = rollout_rows.get(row_id)
        target_delta = target_map.get(row_id)
        if rollout is None or target_delta is None:
            continue
        feat = load_feature(feature_root / str(rollout["feature_path"]))
        future_tokens = token_feature(feat["defi_future_feature"]) if token_level else mean_pool_feature(feat["defi_future_feature"])
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
        qwen_weight_map = {memory_id: 1.0 / float(len(memory_ids)) for memory_id in memory_ids} if memory_ids else {}
        qwen_weights, sims, mask = pad_weight_inputs(qwen_weight_map, similarity_map, memory_ids, topk)

        memory_feats = np.zeros((topk, mean_pool_feature(delta_bank[0]).shape[0]), dtype=np.float32)
        for idx, (_, candidate_row_id) in enumerate(top):
            bank_idx = row_to_bank[candidate_row_id]
            memory_feats[idx] = mean_pool_feature(delta_bank[bank_idx])

        with torch.no_grad():
            _, memory_state, _ = calibrator(
                torch.from_numpy(future_feature).unsqueeze(0).to(device),
                torch.from_numpy(memory_feats).unsqueeze(0).to(device),
                qwen_weights.unsqueeze(0).to(device),
                sims.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
            )

        x_future.append(future_tokens.astype(np.float32))
        x_memory_state.append(memory_state.squeeze(0).cpu().numpy().astype(np.float32))
        y_delta.append(target_delta.astype(np.float32))
        seq_rows.append(int(row["sequence_index"]))

    return {
        "future": np.stack(x_future, axis=0).astype(np.float32),
        "memory_state": np.stack(x_memory_state, axis=0).astype(np.float32),
        "target_delta": np.stack(y_delta, axis=0).astype(np.float32),
        "sequence_index": np.asarray(seq_rows, dtype=np.int64),
    }


def evaluate_model(model: MemoryStateGuidedAdapter, future: np.ndarray, memory_state: np.ndarray, target_delta: np.ndarray) -> Dict[str, float]:
    with torch.no_grad():
        corrected, _, aux = model(torch.from_numpy(future), torch.from_numpy(memory_state))
    pred_delta = corrected.cpu().numpy() - future
    mse = float(np.mean((pred_delta - target_delta) ** 2))
    pred_flat = pred_delta.reshape(pred_delta.shape[0], -1)
    target_flat = target_delta.reshape(target_delta.shape[0], -1)
    cosine = np.sum(pred_flat * target_flat, axis=1) / (np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(target_flat, axis=1) + 1e-12)
    gate_mean = float(aux["gate"].mean().item())
    return {"mse": mse, "delta_cosine": float(np.mean(cosine)), "gate_mean": gate_mean}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory-state guided adapter on pooled or token-level future deltas.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--labeled-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--repair-targets", type=Path, required=True)
    parser.add_argument("--calibrator-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-level", action="store_true", help="Train on full [T,D] future tokens instead of mean-pooled [D] futures.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rollout_rows = load_rollout_rows(args.rollout_jsonl)
    labeled_rows = load_labeled_rows(args.labeled_jsonl)
    memory_npz = load_memory_arrays(args.memory_npz)
    repair_targets_npz = load_memory_arrays(args.repair_targets)
    calibrator_ckpt = torch.load(args.calibrator_ckpt, map_location="cpu")
    dataset = build_examples(
        rollout_rows,
        labeled_rows,
        args.feature_root,
        memory_npz,
        repair_targets_npz,
        calibrator_ckpt,
        args.topk,
        device,
        args.token_level,
    )

    train_idx, val_idx = split_sequence_indices(dataset["sequence_index"])
    train = {k: dataset[k][train_idx] for k in ["future", "memory_state", "target_delta"]}
    val = {k: dataset[k][val_idx] for k in ["future", "memory_state", "target_delta"]}

    model = MemoryStateGuidedAdapter(
        future_dim=train["future"].shape[-1],
        memory_state_dim=train["memory_state"].shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["future"]),
            torch.from_numpy(train["memory_state"]),
            torch.from_numpy(train["target_delta"]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_mse = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for future_b, memory_state_b, target_delta_b in loader:
            future_b = future_b.to(device)
            memory_state_b = memory_state_b.to(device)
            target_delta_b = target_delta_b.to(device)
            corrected, _, _ = model(future_b, memory_state_b)
            pred_delta = corrected - future_b
            loss = F.mse_loss(pred_delta, target_delta_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        metrics = evaluate_model(model.cpu(), val["future"], val["memory_state"], val["target_delta"])
        model.to(device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    torch.save(
        {
            "model_state": model.state_dict(),
            "future_dim": int(train["future"].shape[-1]),
            "future_token_shape": tuple(int(x) for x in train["future"].shape[1:]),
            "memory_state_dim": int(train["memory_state"].shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "token_level": bool(args.token_level),
        },
        args.output_dir / "memory_state_guided_adapter.pt",
    )
    summary = {
        "train_num": int(len(train_idx)),
        "val_num": int(len(val_idx)),
        "token_level": bool(args.token_level),
        "future_shape": [int(x) for x in train["future"].shape[1:]],
        "best_val": evaluate_model(model, val["future"], val["memory_state"], val["target_delta"]),
        "history_tail": history[-5:],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
