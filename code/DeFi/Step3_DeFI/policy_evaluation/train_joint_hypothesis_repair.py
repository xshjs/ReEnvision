from __future__ import annotations

import argparse
import json
import random
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


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted(
        {
            int(row.get("sequence_index_neg", -1))
            for row in rows
            if int(row.get("sequence_index_neg", -1)) >= 0
        }
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val_set = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        seq = int(row.get("sequence_index_neg", -1))
        (val_idx if seq in val_set else train_idx).append(idx)
    if not train_idx:
        train_idx = list(range(max(0, len(rows) - 1)))
    if not val_idx:
        val_idx = [len(rows) - 1]
    return train_idx, val_idx


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def make_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    states = np.stack([np.asarray(row["state_neg"], dtype=np.float32) for row in rows], axis=0)
    future_neg = np.stack([np.asarray(row["future_neg"], dtype=np.float32) for row in rows], axis=0)
    action_neg = np.stack([np.asarray(row["action_neg"], dtype=np.float32) for row in rows], axis=0)
    future_pos = np.stack([np.asarray(row["future_pos"], dtype=np.float32) for row in rows], axis=0)
    action_pos = np.stack([np.asarray(row["action_pos"], dtype=np.float32) for row in rows], axis=0)
    delta_future = future_pos - future_neg
    delta_action = action_pos - action_neg

    tasks = sorted({str(row.get("task", "unknown")) for row in rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}
    task_ids = np.asarray([task_to_id[str(row.get("task", "unknown"))] for row in rows], dtype=np.int64)
    task_onehot = np.eye(len(tasks), dtype=np.float32)[task_ids]
    stages = np.asarray([float(row.get("stage", 0)) for row in rows], dtype=np.float32).reshape(-1, 1)
    max_stage = max(float(stages.max()), 1.0)
    stages = stages / max_stage
    weights = np.asarray([float(row.get("return_gap", 1.0)) for row in rows], dtype=np.float32)
    weights = np.maximum(weights, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)

    x = np.concatenate([states, future_neg, action_neg, task_onehot, stages], axis=1).astype(np.float32)
    y = np.concatenate([delta_future, delta_action], axis=1).astype(np.float32)
    return {
        "x": x,
        "y": y,
        "weights": weights,
        "future_dim": np.asarray([future_neg.shape[1]], dtype=np.int64),
        "action_dim": np.asarray([action_neg.shape[1]], dtype=np.int64),
        "num_tasks": np.asarray([len(tasks)], dtype=np.int64),
        "task_names": np.asarray(tasks, dtype=object),
    }


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
            "pred": torch.cat([pred_future, pred_action], dim=-1),
        }


def evaluate(
    model: JointHypothesisRepairMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    future_dim: int,
):
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred = out["pred"]
        diff = pred - y
        l2 = torch.sqrt((diff**2).mean(dim=-1) + 1e-8)
        loss = (l2 * w).mean()
        pred_f, pred_a = pred[:, :future_dim], pred[:, future_dim:]
        tgt_f, tgt_a = y[:, :future_dim], y[:, future_dim:]
        future_loss = F.smooth_l1_loss(pred_f, tgt_f)
        action_loss = F.smooth_l1_loss(pred_a, tgt_a)
        future_cos = F.cosine_similarity(pred_f, tgt_f, dim=-1).mean()
        action_cos = F.cosine_similarity(pred_a, tgt_a, dim=-1).mean()
    return {
        "loss": float(loss.item()),
        "future_loss": float(future_loss.item()),
        "action_loss": float(action_loss.item()),
        "future_cos": float(future_cos.item()),
        "action_cos": float(action_cos.item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train repair operator R(h,s,p)->(delta_zf, delta_za).")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--future-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.pairs_jsonl)
    arrays = make_arrays(rows)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(arrays["x"]).to(device)
    y = torch.from_numpy(arrays["y"]).to(device)
    w = torch.from_numpy(arrays["weights"]).to(device)
    future_dim = int(arrays["future_dim"][0])
    action_dim = int(arrays["action_dim"][0])

    x_train, y_train, w_train = x[train_idx], y[train_idx], w[train_idx]
    x_val, y_val, w_val = x[val_idx], y[val_idx], w[val_idx]

    model = JointHypothesisRepairMLP(
        input_dim=x.shape[1],
        hidden_dim=int(args.hidden_dim),
        future_dim=int(future_dim),
        action_dim=int(action_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_ids = torch.randint(0, x_train.shape[0], (min(int(args.batch_size), x_train.shape[0]),), device=device)
        out = model(x_train[batch_ids])
        pred_future = out["pred_future"]
        pred_action = out["pred_action"]
        tgt_future = y_train[batch_ids][:, :future_dim]
        tgt_action = y_train[batch_ids][:, future_dim:]
        future_err = ((pred_future - tgt_future) ** 2).mean(dim=-1)
        action_err = ((pred_action - tgt_action) ** 2).mean(dim=-1)
        per_example = (
            float(args.future_loss_weight) * torch.sqrt(future_err + 1e-8)
            + float(args.action_loss_weight) * torch.sqrt(action_err + 1e-8)
        )
        loss = (per_example * w_train[batch_ids]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == args.steps:
            metrics = {"step": step, "train_loss": float(loss.item()), **evaluate(model, x_val, y_val, w_val, future_dim)}
            history.append(metrics)
            if best is None or metrics["loss"] < best["loss"]:
                best = metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": int(x.shape[1]),
                        "output_dim": int(y.shape[1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "future_loss_weight": float(args.future_loss_weight),
                        "action_loss_weight": float(args.action_loss_weight),
                        "future_dim": int(future_dim),
                        "action_dim": int(action_dim),
                        "num_tasks": int(arrays["num_tasks"][0]),
                        "task_names": arrays["task_names"].tolist(),
                    },
                    args.output_dir / "joint_hypothesis_repair.pt",
                )

    summary = {
        "num_rows": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "input_dim": int(x.shape[1]),
        "future_dim": int(future_dim),
        "action_dim": int(action_dim),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "joint_hypothesis_repair.pt"),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
