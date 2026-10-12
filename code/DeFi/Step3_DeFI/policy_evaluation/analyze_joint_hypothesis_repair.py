from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class JointHypothesisRepairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, future_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_head = nn.Linear(hidden_dim, future_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        pred_future = self.future_head(h)
        pred_action = self.action_head(h)
        return {
            "pred_future": pred_future,
            "pred_action": pred_action,
        }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def make_arrays(rows: list[dict[str, Any]]):
    tasks = sorted({str(row.get("task", "unknown")) for row in rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}
    states = np.stack([np.asarray(row["state_neg"], dtype=np.float32) for row in rows], axis=0)
    future_neg = np.stack([np.asarray(row["future_neg"], dtype=np.float32) for row in rows], axis=0)
    action_neg = np.stack([np.asarray(row["action_neg"], dtype=np.float32) for row in rows], axis=0)
    future_pos = np.stack([np.asarray(row["future_pos"], dtype=np.float32) for row in rows], axis=0)
    action_pos = np.stack([np.asarray(row["action_pos"], dtype=np.float32) for row in rows], axis=0)
    task_ids = np.asarray([task_to_id[str(row.get("task", "unknown"))] for row in rows], dtype=np.int64)
    task_onehot = np.eye(len(tasks), dtype=np.float32)[task_ids]
    stages = np.asarray([float(row.get("stage", 0)) for row in rows], dtype=np.float32).reshape(-1, 1)
    max_stage = max(float(stages.max()), 1.0)
    stages = stages / max_stage
    x = np.concatenate([states, future_neg, action_neg, task_onehot, stages], axis=1).astype(np.float32)
    return x, future_neg, action_neg, future_pos, action_pos


def summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline analysis for joint hypothesis repair.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--repair-ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows = load_rows(args.pairs_jsonl)
    x_np, future_neg_np, action_neg_np, future_pos_np, action_pos_np = make_arrays(rows)

    ckpt = torch.load(args.repair_ckpt, map_location="cpu")
    model = JointHypothesisRepairMLP(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        future_dim=int(ckpt["future_dim"]),
        action_dim=int(ckpt["action_dim"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    x = torch.from_numpy(x_np).to(device)
    with torch.no_grad():
        out = model(x)
        delta_future = out["pred_future"].cpu().numpy()
        delta_action = out["pred_action"].cpu().numpy()

    future_refined = future_neg_np + delta_future
    action_refined = action_neg_np + delta_action

    future_before_l2 = np.linalg.norm(future_neg_np - future_pos_np, axis=1)
    future_after_l2 = np.linalg.norm(future_refined - future_pos_np, axis=1)
    action_before_l2 = np.linalg.norm(action_neg_np - action_pos_np, axis=1)
    action_after_l2 = np.linalg.norm(action_refined - action_pos_np, axis=1)

    future_before_cos = F.cosine_similarity(
        torch.from_numpy(future_neg_np), torch.from_numpy(future_pos_np), dim=-1
    ).numpy()
    future_after_cos = F.cosine_similarity(
        torch.from_numpy(future_refined), torch.from_numpy(future_pos_np), dim=-1
    ).numpy()
    action_before_cos = F.cosine_similarity(
        torch.from_numpy(action_neg_np), torch.from_numpy(action_pos_np), dim=-1
    ).numpy()
    action_after_cos = F.cosine_similarity(
        torch.from_numpy(action_refined), torch.from_numpy(action_pos_np), dim=-1
    ).numpy()

    summary = {
        "num_rows": len(rows),
        "future_l2_before": summarize(future_before_l2.tolist()),
        "future_l2_after": summarize(future_after_l2.tolist()),
        "future_l2_improved_rate": float(np.mean((future_after_l2 < future_before_l2).astype(np.float32))),
        "action_l2_before": summarize(action_before_l2.tolist()),
        "action_l2_after": summarize(action_after_l2.tolist()),
        "action_l2_improved_rate": float(np.mean((action_after_l2 < action_before_l2).astype(np.float32))),
        "future_cos_before": summarize(future_before_cos.tolist()),
        "future_cos_after": summarize(future_after_cos.tolist()),
        "future_cos_improved_rate": float(np.mean((future_after_cos > future_before_cos).astype(np.float32))),
        "action_cos_before": summarize(action_before_cos.tolist()),
        "action_cos_after": summarize(action_after_cos.tolist()),
        "action_cos_improved_rate": float(np.mean((action_after_cos > action_before_cos).astype(np.float32))),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
