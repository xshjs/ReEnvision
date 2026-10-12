"""Train a balanced subtask-level trigger for joint future editing.

The trigger predicts whether a pre-rollout future should be edited. It is trained
from saved rollout rows plus future traces and deliberately balances rare failure
rows against success rows to avoid the all-safe classifier.
"""

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


FACTORS = [
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
    "none",
    "unknown",
]


class TriggerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def resolve_trace(row: dict[str, Any], trace_dir: Path) -> Path | None:
    trace_path = row.get("trace_path") or row.get("target_proxy_trace_path") or row.get("collection_trace_path")
    if trace_path:
        path = Path(str(trace_path))
        if path.exists():
            return path
        candidate = trace_dir / path.name
        if candidate.exists():
            return candidate
    idx = row.get("_trace_index", row.get("row_index"))
    if idx is not None:
        candidate = trace_dir / f"future_trace_{int(idx):04d}.npz"
        if candidate.exists():
            return candidate
    return None


def future_stats(trace_path: Path, future_key: str) -> np.ndarray:
    data = np.load(trace_path)
    if future_key in data:
        arr = data[future_key]
    elif "base_future" in data:
        arr = data["base_future"]
    elif "original_base_future" in data:
        arr = data["original_base_future"]
    else:
        raise KeyError(f"missing future key {future_key} in {trace_path}")
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(-1)
    token_norm = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1) if arr.ndim >= 2 else np.asarray([np.linalg.norm(flat)])
    channel_mean_abs = np.mean(np.abs(arr.reshape(-1, arr.shape[-1])), axis=0) if arr.ndim >= 2 else np.abs(flat)
    qs = np.quantile(flat, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).astype(np.float32)
    return np.asarray(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.linalg.norm(flat)),
            float(np.mean(np.abs(flat))),
            float(np.max(np.abs(flat))),
            float(np.mean(token_norm)),
            float(np.std(token_norm)),
            float(np.max(token_norm)),
            float(np.mean(channel_mean_abs)),
            float(np.std(channel_mean_abs)),
            float(np.max(channel_mean_abs)),
            *[float(x) for x in qs],
        ],
        dtype=np.float32,
    )


def make_label(row: dict[str, Any], mode: str, min_gain: float) -> int:
    success = bool(row.get("success", False))
    if mode == "failure_only":
        return 0 if success else 1
    if not success:
        return 1
    if bool(row.get("label_harm", False)):
        return 0
    if safe_float(row.get("final_cos_gain"), 0.0) <= min_gain:
        return 0
    if bool(row.get("label_improve", False)) and safe_float(row.get("final_cos_gain"), 0.0) > min_gain:
        return 1
    return 0


def build_feature(row: dict[str, Any], stats: np.ndarray, task_to_idx: dict[str, int], online_only: bool = False) -> np.ndarray:
    task = str(row.get("task", "unknown"))
    factor = str(row.get("failure_factor") or row.get("applied_factor_mask") or "unknown")
    if factor not in FACTORS:
        factor = "unknown"
    task_onehot = np.zeros(len(task_to_idx), dtype=np.float32)
    task_onehot[task_to_idx.get(task, 0)] = 1.0
    factor_onehot = np.zeros(len(FACTORS), dtype=np.float32)
    factor_onehot[FACTORS.index(factor)] = 1.0
    pre_rollout_scalar_names = [
        "subtask_index",
        "memory_score_mean",
        "memory_score_max",
        "counterfactual_memory_support",
        "counterfactual_alignment",
        "counterfactual_gate",
        "counterfactual_residual_norm",
        "counterfactual_shift_norm",
    ]
    post_rollout_scalar_names = [
        "steps",
        "proposal_cos_gain",
        "final_cos_gain",
        "base_to_target_cos",
        "proposal_to_target_cos",
        "final_to_target_cos",
        "adapter_delta_norm",
        "adapter_gate_mean",
    ]
    scalar_names = pre_rollout_scalar_names if online_only else pre_rollout_scalar_names + post_rollout_scalar_names
    scalars = np.asarray([safe_float(row.get(name), 0.0) for name in scalar_names], dtype=np.float32)
    for idx, name in enumerate(scalar_names):
        if name in {"counterfactual_residual_norm", "counterfactual_shift_norm", "adapter_delta_norm"}:
            scalars[idx] = np.log1p(max(float(scalars[idx]), 0.0))
    return np.concatenate([stats, scalars, task_onehot, factor_onehot], axis=0).astype(np.float32)


def balanced_indices(rows: list[dict[str, Any]], labels: list[int], neg_per_pos: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg_by_task: dict[str, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        if y == 0:
            neg_by_task[str(rows[i].get("task", "unknown"))].append(i)
    neg_target = min(sum(len(v) for v in neg_by_task.values()), max(1, int(round(len(pos) * neg_per_pos))))
    neg = []
    tasks = list(neg_by_task)
    while len(neg) < neg_target and tasks:
        rng.shuffle(tasks)
        kept = []
        for task in tasks:
            bucket = neg_by_task[task]
            if bucket and len(neg) < neg_target:
                neg.append(bucket.pop(rng.randrange(len(bucket))))
            if bucket:
                kept.append(task)
        tasks = kept
    out = pos + neg
    rng.shuffle(out)
    return out


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    gold = labels.bool()
    tp = int((pred & gold).sum().item())
    fp = int((pred & ~gold).sum().item())
    fn = int((~pred & gold).sum().item())
    tn = int((~pred & ~gold).sum().item())
    return {
        "acc": float((pred == gold).float().mean().item()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_trigger_rate": fp / max(fp + tn, 1),
        "mean_prob": float(probs.mean().item()),
        "positive_rate": float(pred.float().mean().item()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--label-mode", choices=["failure_only", "failure_or_improve"], default="failure_only")
    parser.add_argument("--online-only-features", action="store_true")
    parser.add_argument("--min-improve-gain", type=float, default=0.0)
    parser.add_argument("--neg-per-pos", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.dataset_jsonl))
    for idx, row in enumerate(rows):
        row["_trace_index"] = idx
        row["row_index"] = idx
    trace_dir = Path(args.future_trace_dir)
    usable = []
    missing = 0
    for row in rows:
        path = resolve_trace(row, trace_dir)
        if path is None:
            missing += 1
            continue
        row["_resolved_trace_path"] = str(path)
        usable.append(row)
    tasks = sorted({str(row.get("task", "unknown")) for row in usable})
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    labels = [make_label(row, args.label_mode, args.min_improve_gain) for row in usable]
    selected = balanced_indices(usable, labels, args.neg_per_pos, args.seed)
    rng.shuffle(selected)
    split = int(0.8 * len(selected))
    train_idx = selected[:split]
    eval_idx = selected[split:]

    features = {}
    for i in selected:
        row = usable[i]
        stats = future_stats(Path(row["_resolved_trace_path"]), args.future_key)
        features[i] = build_feature(row, stats, task_to_idx, online_only=bool(args.online_only_features))
    x_all = np.stack([features[i] for i in selected], axis=0)
    mean = x_all.mean(axis=0).astype(np.float32)
    std = np.maximum(x_all.std(axis=0).astype(np.float32), 1e-6)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = TriggerMLP(x_all.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best = None
    best_score = -1.0
    for step in range(1, args.steps + 1):
        batch = [rng.choice(train_idx) for _ in range(args.batch_size)]
        xb = torch.from_numpy(((np.stack([features[i] for i in batch]) - mean) / std).astype(np.float32)).to(device)
        yb = torch.tensor([labels[i] for i in batch], dtype=torch.float32, device=device)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                xe = torch.from_numpy(((np.stack([features[i] for i in eval_idx]) - mean) / std).astype(np.float32)).to(device)
                ye = torch.tensor([labels[i] for i in eval_idx], dtype=torch.float32, device=device)
                eval_logits = model(xe)
                m = metrics_from_logits(eval_logits, ye)
                score = m["f1"] - 0.5 * m["false_trigger_rate"]
            rec = {"step": step, "loss": float(loss.item()), **m}
            history.append(rec)
            if score > best_score:
                best_score = score
                best = rec
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "feature_mean": mean,
                        "feature_std": std,
                        "task_to_idx": task_to_idx,
                        "factors": FACTORS,
                        "hidden_dim": args.hidden_dim,
                        "input_dim": int(x_all.shape[1]),
                        "future_key": args.future_key,
                        "label_mode": args.label_mode,
                        "online_only_features": bool(args.online_only_features),
                    },
                    out_dir / "joint_editor_trigger.pt",
                )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "num_rows": len(rows),
        "num_usable": len(usable),
        "missing_trace": missing,
        "num_selected": len(selected),
        "num_train": len(train_idx),
        "num_eval": len(eval_idx),
        "label_counts_all": dict(Counter(labels)),
        "label_counts_selected": dict(Counter(labels[i] for i in selected)),
        "task_counts_selected": dict(Counter(str(usable[i].get("task", "unknown")) for i in selected).most_common()),
        "best": best,
        "online_only_features": bool(args.online_only_features),
        "history": history,
        "output_ckpt": str(out_dir / "joint_editor_trigger.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
