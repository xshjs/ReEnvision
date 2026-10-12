#!/usr/bin/env python3
"""Train relation-aware probes on DeFI future latents and CALVIN strong GT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TARGETS = ["risk", "contact", "object_motion", "progress", "goal"]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        with np.load(str(path)) as data:
            if key not in data.files:
                return None
            x = np.asarray(data[key], dtype=np.float32)
    except Exception:
        return None
    return x if x.ndim == 2 else None


def task_hash(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % dim
        vec[idx] += 1.0
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def nested_values(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from nested_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from nested_values(value)
    else:
        yield obj


def max_abs_delta(obj: Any) -> float:
    vals = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() == "delta":
                try:
                    vals.append(abs(float(value)))
                except Exception:
                    pass
            else:
                vals.append(max_abs_delta(value))
    elif isinstance(obj, list):
        for value in obj:
            vals.append(max_abs_delta(value))
    return float(max(vals)) if vals else 0.0


def max_block_distance(row: dict[str, Any]) -> float:
    motion = row.get("block_displacement") or row.get("object_motion") or {}
    vals = []
    if isinstance(motion, dict):
        for obj in motion.values():
            if isinstance(obj, dict):
                try:
                    vals.append(float(obj.get("distance", 0.0)))
                except Exception:
                    pass
    return float(max(vals)) if vals else 0.0


def has_robot_block_contact(row: dict[str, Any]) -> bool:
    contact = row.get("contact_state") or {}
    text = json.dumps(contact, ensure_ascii=False).lower()
    return ("robot" in text) and ("block" in text)


def task_target_color(task: str) -> str | None:
    for color in ["red", "blue", "pink"]:
        if color in task:
            return color
    return None


def target_block_motion(row: dict[str, Any], task: str) -> float:
    color = task_target_color(task)
    motion = row.get("block_displacement") or row.get("object_motion") or {}
    if not isinstance(motion, dict):
        return max_block_distance(row)
    if color:
        key = f"block_{color}"
        if isinstance(motion.get(key), dict):
            try:
                return float(motion[key].get("distance", 0.0))
            except Exception:
                return 0.0
    return max_block_distance(row)


def target_block_lift(row: dict[str, Any], task: str) -> float:
    color = task_target_color(task)
    motion = row.get("block_displacement") or row.get("object_motion") or {}
    if not isinstance(motion, dict):
        return 0.0
    candidates = [f"block_{color}"] if color else list(motion)
    vals = []
    for key in candidates:
        obj = motion.get(key)
        if isinstance(obj, dict):
            try:
                vals.append(float(obj.get("delta_z", 0.0)))
            except Exception:
                pass
    return float(max(vals)) if vals else 0.0


def infer_relation_labels(row: dict[str, Any]) -> np.ndarray:
    task = str(row.get("task", "")).lower()
    success = bool(row.get("success", row.get("strong_gt_success", False)))
    risk = 0.0 if success else 1.0

    lift_tasks = any(k in task for k in ["lift", "stack", "unstack", "place"])
    push_tasks = "push" in task
    progress_tasks = any(k in task for k in ["drawer", "slider"])
    goal_tasks = any(k in task for k in ["led", "lightbulb"])
    object_tasks = any(k in task for k in ["block", "push", "lift", "stack", "place"])

    contact_ok = has_robot_block_contact(row)
    object_motion = target_block_motion(row, task)
    target_lift = target_block_lift(row, task)
    progress_delta = max_abs_delta(row.get("drawer_slider_progress_delta") or {})
    goal_ok = bool((row.get("goal_predicate") or {}).get("success", False))

    # These are mismatch labels: 1 means this relation appears wrong or insufficient.
    contact_bad = 1.0 if lift_tasks and not success and (not contact_ok or target_lift < 0.015) else 0.0
    object_bad = 1.0 if object_tasks and not success and object_motion < (0.03 if push_tasks else 0.015) else 0.0
    progress_bad = 1.0 if progress_tasks and not success and progress_delta < 0.02 else 0.0
    goal_bad = 1.0 if goal_tasks and not goal_ok else 0.0

    return np.asarray([risk, contact_bad, object_bad, progress_bad, goal_bad], dtype=np.float32)


def load_examples(path: Path, future_key: str, task_dim: int):
    examples = []
    skipped = Counter()
    for row in iter_jsonl(path):
        trace = row.get("trace_path") or row.get("collection_trace_path")
        if not trace or not Path(str(trace)).exists():
            skipped["missing_trace"] += 1
            continue
        future = load_future(trace, future_key)
        if future is None:
            skipped["bad_future"] += 1
            continue
        labels = infer_relation_labels(row)
        pooled = future.mean(axis=0).astype(np.float32)
        std = future.std(axis=0).astype(np.float32)
        feat = np.concatenate([pooled, std, task_hash(str(row.get("task", "")), task_dim)], axis=0).astype(np.float32)
        examples.append(
            {
                "feature": feat,
                "labels": labels,
                "task": str(row.get("task", "")),
                "success": bool(row.get("success", row.get("strong_gt_success", False))),
            }
        )
    if not examples:
        raise ValueError(f"no usable examples from {path}; skipped={dict(skipped)}")
    return examples, skipped


class RelationProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(TARGETS)),
        )

    def forward(self, x):
        return self.net(x)


def batchify(examples, batch_size):
    batch = random.sample(examples, min(batch_size, len(examples)))
    x = torch.from_numpy(np.stack([b["feature"] for b in batch], axis=0))
    y = torch.from_numpy(np.stack([b["labels"] for b in batch], axis=0))
    return x, y


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()
    x = torch.from_numpy(np.stack([b["feature"] for b in examples], axis=0)).to(device)
    y = torch.from_numpy(np.stack([b["labels"] for b in examples], axis=0)).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).float()
    out = {}
    for i, name in enumerate(TARGETS):
        yi = y[:, i]
        pi = pred[:, i]
        tp = int(((pi == 1) & (yi == 1)).sum().item())
        fp = int(((pi == 1) & (yi == 0)).sum().item())
        fn = int(((pi == 0) & (yi == 1)).sum().item())
        tn = int(((pi == 0) & (yi == 0)).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        out[name] = {
            "acc": float((pi == yi).float().mean().item()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "positive_rate": float(yi.mean().item()),
            "mean_prob": float(prob[:, i].mean().item()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-gt-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", type=str, default="base_future")
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples, skipped = load_examples(args.strong_gt_jsonl, args.future_key, args.task_dim)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio))
    val = examples[:n_val]
    train = examples[n_val:]

    in_dim = int(train[0]["feature"].shape[0])
    y_train = np.stack([b["labels"] for b in train], axis=0)
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weight = torch.from_numpy((neg / np.maximum(pos, 1.0)).astype(np.float32)).to(device)

    model = RelationProbe(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_score = -1.0
    best = None
    for step in range(1, args.steps + 1):
        x, y = batchify(train, args.batch_size)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            metrics = evaluate(model, val, device)
            score = float(np.mean([metrics[k]["f1"] for k in TARGETS]))
            rec = {"step": step, "loss": float(loss.item()), "mean_f1": score, **metrics}
            history.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            if score > best_score:
                best_score = score
                best = {
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "task_dim": args.task_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "future_key": args.future_key,
                    "targets": TARGETS,
                    "metrics": metrics,
                    "step": step,
                    "pos_weight": pos_weight.detach().cpu().numpy(),
                }

    ckpt_path = args.output_dir / "relation_future_probe.pt"
    torch.save(best, ckpt_path)
    summary = {
        "strong_gt_jsonl": str(args.strong_gt_jsonl),
        "num_examples": len(examples),
        "num_train": len(train),
        "num_val": len(val),
        "skipped": dict(skipped),
        "target_positive_counts_train": {name: float(pos[i]) for i, name in enumerate(TARGETS)},
        "task_counts": dict(Counter(b["task"] for b in examples).most_common()),
        "history": history,
        "best_checkpoint": str(ckpt_path),
        "best_step": int(best["step"]) if best else None,
        "best_metrics": best["metrics"] if best else {},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
