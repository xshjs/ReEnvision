from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "object_displacement_underestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
    "goal_progress_hallucinated": "goal_completion",
}

FACTOR_ORDER = ["contact", "drawer_slider_progress", "goal_completion", "object_displacement", "object_identity", "none", "unknown"]


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def infer_factor(row: dict[str, Any]) -> str:
    for key in ("failure_factor", "selected_failure_factor", "applied_factor_mask"):
        value = str(row.get(key, "") or "").strip()
        if value in FACTOR_ORDER:
            return value
    mismatch = str(row.get("mismatch_type", "") or "")
    return MISMATCH_TO_FACTOR.get(mismatch, "unknown")


def factor_vote_stats(row: dict[str, Any], memory_factor_by_id: dict[int, str]) -> tuple[list[float], float, float]:
    votes = defaultdict(float)
    ids = row.get("adapter_memory_ids") or []
    scores = row.get("adapter_memory_scores") or []
    for memory_id, score in zip(ids, scores):
        text = str(memory_id)
        if text.startswith("KEY_"):
            text = text[4:]
        try:
            row_id = int(text)
        except ValueError:
            continue
        factor = memory_factor_by_id.get(row_id, "unknown")
        votes[factor] += max(safe_float(score), 0.0)
    total = sum(votes.values())
    dist = [votes.get(factor, 0.0) / max(total, 1e-8) for factor in FACTOR_ORDER]
    entropy = -sum(p * math.log(max(p, 1e-8)) for p in dist)
    confidence = max(dist) if dist else 0.0
    return dist, float(entropy), float(confidence)


def build_memory_factor_map(memory_jsonl: Path | None) -> dict[int, str]:
    if memory_jsonl is None or not memory_jsonl.exists():
        return {}
    out = {}
    for idx, row in enumerate(iter_jsonl(memory_jsonl)):
        row_id = int(row.get("row_id", idx))
        mismatch = str(row.get("mismatch_type", "") or "")
        out[row_id] = MISMATCH_TO_FACTOR.get(mismatch, "unknown")
    return out


def build_features(rows: list[dict[str, Any]], memory_factor_by_id: dict[int, str], pre_rollout_only: bool) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    tasks = sorted({str(row.get("task", "")) for row in rows})
    task_to_id = {task: idx for idx, task in enumerate(tasks)}
    feature_names = ["subtask_index"]
    if not pre_rollout_only:
        feature_names.append("steps")
    feature_names += [
            "counterfactual_gate",
            "counterfactual_memory_support",
            "counterfactual_alignment",
            "counterfactual_residual_norm_log",
            "counterfactual_shift_norm_log",
            "base_future_norm_log",
            "proposal_future_norm_log",
    ]
    if not pre_rollout_only:
        feature_names += [
            "adapter_delta_norm_log",
            "adapter_gate_mean",
            "adapter_weight_mean",
        ]
    feature_names += [
            "factor_vote_entropy",
            "factor_vote_confidence",
    ]
    feature_names += [f"factor_vote_{factor}" for factor in FACTOR_ORDER]
    feature_names += [f"qwen_factor_{factor}" for factor in FACTOR_ORDER]
    feature_names += [f"task_{task}" for task in tasks]

    x = []
    y = []
    for row in rows:
        vote_dist, vote_entropy, vote_confidence = factor_vote_stats(row, memory_factor_by_id)
        factor = infer_factor(row)
        factor_onehot = [1.0 if factor == name else 0.0 for name in FACTOR_ORDER]
        task = str(row.get("task", ""))
        task_onehot = [0.0] * len(tasks)
        if task in task_to_id:
            task_onehot[task_to_id[task]] = 1.0
        numeric = [safe_float(row.get("subtask_index"), -1.0)]
        if not pre_rollout_only:
            numeric.append(safe_float(row.get("steps"), 0.0))
        numeric += [
            safe_float(row.get("counterfactual_gate")),
            safe_float(row.get("counterfactual_memory_support")),
            safe_float(row.get("counterfactual_alignment")),
            math.log1p(max(safe_float(row.get("counterfactual_residual_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("counterfactual_shift_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("base_future_norm")), 0.0)),
            math.log1p(max(safe_float(row.get("proposal_future_norm")), 0.0)),
        ]
        if not pre_rollout_only:
            numeric += [
                math.log1p(max(safe_float(row.get("adapter_delta_norm")), 0.0)),
                safe_float(row.get("adapter_gate_mean")),
                safe_float(row.get("adapter_weight_mean")),
            ]
        numeric += [vote_entropy, vote_confidence]
        x.append(numeric + vote_dist + factor_onehot + task_onehot)
        y.append(0 if bool(row.get("success", False)) else 1)
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
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def standardize(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    pred = (prob >= 0.5).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    order = np.argsort(-prob)
    precision_at = {}
    for frac in [0.01, 0.02, 0.05, 0.10, 0.20]:
        k = max(1, int(round(len(prob) * frac)))
        top = order[:k]
        precision_at[f"top_{int(frac * 100)}pct_fail_rate"] = float(y[top].mean())
        precision_at[f"top_{int(frac * 100)}pct_fail_count"] = int(y[top].sum())
        precision_at[f"top_{int(frac * 100)}pct_count"] = int(k)
    # Rank AUC without sklearn.
    pos = prob[y == 1]
    neg = prob[y == 0]
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    else:
        auc = 0.0
    return {
        "base_fail_rate": float(y.mean()) if len(y) else 0.0,
        "acc_at_0_5": float((pred == y).mean()) if len(y) else 0.0,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **precision_at,
    }


def train_one_seed(
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
    feature_names: list[str],
    tasks: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, seed)
    x_train, x_val, mean, std = standardize(x[train_idx], x[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    device = torch.device(args.device)
    model = RiskMLP(x.shape[1], args.hidden_dim).to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tx = torch.from_numpy(x_train).to(device)
    ty = torch.from_numpy(y_train).to(device)
    vx = torch.from_numpy(x_val).to(device)
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
            metrics = binary_metrics(y_val, prob)
            score = metrics["top_10pct_fail_rate"]
            if best is None or score > best["top_10pct_fail_rate"]:
                best = {**metrics, "step": int(step), "loss": float(loss.detach().cpu().item())}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(tx)).detach().cpu().numpy()
        val_prob = torch.sigmoid(model(vx)).detach().cpu().numpy()
    checkpoint_path = output_dir / f"risk_detector_seed{seed}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "feature_names": feature_names,
            "tasks": tasks,
            "hidden_dim": int(args.hidden_dim),
            "pre_rollout_only": bool(args.pre_rollout_only),
            "seed": int(seed),
        },
        checkpoint_path,
    )
    return {
        "seed": int(seed),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "train_fail_rate": float(y_train.mean()),
        "val_fail_rate": float(y_val.mean()),
        "train_metrics": binary_metrics(y_train, train_prob),
        "val_metrics": binary_metrics(y_val, val_prob),
        "best": best,
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout risk detector for gating future intervention.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--memory-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pre-rollout-only", action="store_true")
    args = parser.parse_args()

    rows = list(iter_jsonl(args.rollout_jsonl))
    memory_factor_by_id = build_memory_factor_map(args.memory_jsonl)
    x, y, feature_names, tasks = build_features(rows, memory_factor_by_id, bool(args.pre_rollout_only))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_one_seed(x, y, rows, args, seed, feature_names, tasks, args.output_dir) for seed in args.seeds]
    val_auc = [result["val_metrics"]["auc"] for result in results]
    top10 = [result["val_metrics"]["top_10pct_fail_rate"] for result in results]
    top5 = [result["val_metrics"]["top_5pct_fail_rate"] for result in results]
    summary = {
        "num_rows": int(len(rows)),
        "num_features": int(x.shape[1]),
        "fail_count": int(y.sum()),
        "success_count": int(len(y) - y.sum()),
        "base_fail_rate": float(y.mean()),
        "task_count": len(tasks),
        "pre_rollout_only": bool(args.pre_rollout_only),
        "feature_names": feature_names,
        "seeds": [int(seed) for seed in args.seeds],
        "val_auc_mean": float(np.mean(val_auc)),
        "val_auc_std": float(np.std(val_auc)),
        "val_top5_fail_rate_mean": float(np.mean(top5)),
        "val_top5_fail_rate_std": float(np.std(top5)),
        "val_top10_fail_rate_mean": float(np.mean(top10)),
        "val_top10_fail_rate_std": float(np.std(top10)),
        "runs": results,
    }
    best_result = max(results, key=lambda result: result["val_metrics"]["top_10pct_fail_rate"])
    summary["best_checkpoint_path"] = best_result["checkpoint_path"]
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
