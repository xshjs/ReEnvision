from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FACTOR_ORDER = ["contact", "drawer_slider_progress", "goal_completion", "object_displacement", "object_identity", "none", "unknown"]


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_rows(path: Path, include_neutral: bool) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        improve = int(row.get("label_improve", 0))
        harm = int(row.get("label_harm", 0))
        if improve and not harm:
            row["_label"] = 1
        elif harm and not improve:
            row["_label"] = 0
        elif include_neutral:
            row["_label"] = 0
            row["_neutral"] = True
        else:
            continue
        rows.append(row)
    return rows


def build_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    tasks = sorted({str(row.get("task", "")) for row in rows})
    task_to_id = {task: idx for idx, task in enumerate(tasks)}
    feature_names = [
        "subtask_index",
        "base_to_target_cos",
        "proposal_to_target_cos",
        "gated_to_target_cos",
        "final_to_target_cos",
        "proposal_cos_gain",
        "gated_cos_gain",
        "final_cos_gain",
        "proposal_to_target_delta_cos",
        "gated_to_target_delta_cos",
        "final_to_target_delta_cos",
        "proposal_delta_norm_log",
        "gated_delta_norm_log",
        "final_delta_norm_log",
        "target_delta_norm_log",
        "counterfactual_gate",
        "counterfactual_memory_support",
        "counterfactual_alignment",
        "counterfactual_residual_norm_log",
        "counterfactual_shift_norm_log",
        "adapter_delta_norm_log",
        "adapter_gate_mean",
        "adapter_weight_mean",
        "memory_score_mean",
        "memory_score_max",
        "factor_mask_applied",
    ]
    feature_names += [f"factor_{factor}" for factor in FACTOR_ORDER]
    feature_names += [f"task_{task}" for task in tasks]
    x, y = [], []
    for row in rows:
        factor = str(row.get("failure_factor") or row.get("applied_factor_mask") or "unknown")
        factor_onehot = [1.0 if factor == name else 0.0 for name in FACTOR_ORDER]
        task_onehot = [0.0] * len(tasks)
        task = str(row.get("task", ""))
        if task in task_to_id:
            task_onehot[task_to_id[task]] = 1.0
        numeric = [
            safe_float(row.get("subtask_index"), -1.0),
            safe_float(row.get("base_to_target_cos")),
            safe_float(row.get("proposal_to_target_cos")),
            safe_float(row.get("gated_to_target_cos")),
            safe_float(row.get("final_to_target_cos")),
            safe_float(row.get("proposal_cos_gain")),
            safe_float(row.get("gated_cos_gain")),
            safe_float(row.get("final_cos_gain")),
            safe_float(row.get("proposal_to_target_delta_cos")),
            safe_float(row.get("gated_to_target_delta_cos")),
            safe_float(row.get("final_to_target_delta_cos")),
            math.log1p(max(safe_float(row.get("proposal_delta_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("gated_delta_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("final_delta_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("target_delta_norm")), 0.0)),
            safe_float(row.get("counterfactual_gate")),
            safe_float(row.get("counterfactual_memory_support")),
            safe_float(row.get("counterfactual_alignment")),
            math.log1p(max(safe_float(row.get("counterfactual_residual_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("counterfactual_shift_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("adapter_delta_norm")), 0.0)),
            safe_float(row.get("adapter_gate_mean")),
            safe_float(row.get("adapter_weight_mean")),
            safe_float(row.get("memory_score_mean")),
            safe_float(row.get("memory_score_max")),
            1.0 if bool(row.get("factor_mask_applied", False)) else 0.0,
        ]
        x.append(numeric + factor_onehot + task_onehot)
        y.append(int(row["_label"]))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), feature_names, tasks


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(rows)})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_seqs = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        seq = int(row.get("sequence_index", idx))
        (val_idx if seq in val_seqs else train_idx).append(idx)
    return np.asarray(train_idx), np.asarray(val_idx)


class VerifierMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = (p >= 0.5).astype(np.int64)
    yi = y.astype(np.int64)
    tp = int(((pred == 1) & (yi == 1)).sum())
    fp = int(((pred == 1) & (yi == 0)).sum())
    fn = int(((pred == 0) & (yi == 1)).sum())
    tn = int(((pred == 0) & (yi == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    pos = p[yi == 1]
    neg = p[yi == 0]
    auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())) if len(pos) and len(neg) else 0.0
    return {
        "acc": float((pred == yi).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "positive_rate": float(yi.mean()),
        "mean_prob": float(p.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def train_seed(x: np.ndarray, y: np.ndarray, rows: list[dict[str, Any]], args: argparse.Namespace, seed: int, feature_names: list[str], tasks: list[str]) -> dict[str, Any]:
    set_seed(seed)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(x[train_idx].std(axis=0, keepdims=True), 1e-6)
    x_train = (x[train_idx] - mean) / std
    x_val = (x[val_idx] - mean) / std
    y_train = y[train_idx]
    y_val = y[val_idx]
    device = torch.device(args.device)
    model = VerifierMLP(x.shape[1], args.hidden_dim).to(device)
    tx = torch.from_numpy(x_train).to(device)
    ty = torch.from_numpy(y_train).to(device)
    vx = torch.from_numpy(x_val).to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    best_state = None
    for step in range(args.steps):
        model.train()
        logits = model(tx)
        loss = F.binary_cross_entropy_with_logits(logits, ty, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(vx)).detach().cpu().numpy()
            m = metrics(y_val, prob)
            score = m["auc"] + m["f1"]
            if best is None or score > best["score"]:
                best = {"step": step + 1, "loss": float(loss.item()), "score": float(score), **m}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best is not None and best_state is not None
    ckpt_path = args.output_dir / f"counterfactual_verifier_seed{seed}.pt"
    torch.save(
        {
            "model_state": best_state,
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "feature_names": feature_names,
            "tasks": tasks,
            "hidden_dim": int(args.hidden_dim),
        },
        ckpt_path,
    )
    best["checkpoint_path"] = str(ckpt_path)
    best["seed"] = seed
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.dataset_jsonl, args.include_neutral)
    x, y, feature_names, tasks = build_features(rows)
    results = [train_seed(x, y, rows, args, seed, feature_names, tasks) for seed in args.seeds]
    best = max(results, key=lambda item: item["score"])
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_examples": len(rows),
        "num_positive": int(y.sum()),
        "num_negative": int(len(y) - y.sum()),
        "num_features": int(x.shape[1]),
        "tasks": tasks,
        "results": results,
        "best_checkpoint_path": best["checkpoint_path"],
        "best": best,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
