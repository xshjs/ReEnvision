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


def load_pairs(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row.get("sequence_index_neg", -1)) for row in rows if int(row.get("sequence_index_neg", -1)) >= 0})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row.get("sequence_index_neg", -1)) in val else train_idx).append(idx)
    return train_idx, val_idx


def make_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted({str(row.get("task", "unknown")) for row in rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}
    num_tasks = len(tasks)
    max_stage = max(float(row.get("stage", 0)) for row in rows)
    max_stage = max(max_stage, 1.0)

    x_neg, x_pos, weights = [], [], []
    for row in rows:
        task_onehot = np.zeros((num_tasks,), dtype=np.float32)
        task_onehot[task_to_id[str(row.get("task", "unknown"))]] = 1.0
        stage = np.asarray([float(row.get("stage", 0)) / max_stage], dtype=np.float32)
        neg = np.concatenate(
            [
                np.asarray(row["state_neg"], dtype=np.float32),
                np.asarray(row["future_neg"], dtype=np.float32),
                np.asarray(row["action_neg"], dtype=np.float32),
                task_onehot,
                stage,
            ],
            axis=0,
        ).astype(np.float32)
        pos = np.concatenate(
            [
                np.asarray(row["state_pos"], dtype=np.float32),
                np.asarray(row["future_pos"], dtype=np.float32),
                np.asarray(row["action_pos"], dtype=np.float32),
                task_onehot,
                stage,
            ],
            axis=0,
        ).astype(np.float32)
        x_neg.append(neg)
        x_pos.append(pos)
        weights.append(float(row.get("return_gap", 1.0)))
    w = np.asarray(weights, dtype=np.float32)
    w = np.maximum(w, 1.0)
    w = w / max(float(w.mean()), 1e-6)
    return {
        "x_neg": np.stack(x_neg, axis=0).astype(np.float32),
        "x_pos": np.stack(x_pos, axis=0).astype(np.float32),
        "weights": w,
        "task_names": tasks,
        "num_tasks": int(num_tasks),
        "max_stage": float(max_stage),
    }


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def tensorize(arrays: dict[str, np.ndarray], ids: list[int], device: torch.device):
    x_neg = torch.from_numpy(arrays["x_neg"][ids]).to(device)
    x_pos = torch.from_numpy(arrays["x_pos"][ids]).to(device)
    w = torch.from_numpy(arrays["weights"][ids]).to(device)
    return x_neg, x_pos, w


def evaluate(model: ValueMLP, x_neg: torch.Tensor, x_pos: torch.Tensor, w: torch.Tensor, margin: float):
    model.eval()
    with torch.no_grad():
        s_neg = model(x_neg)
        s_pos = model(x_pos)
        diff = s_pos - s_neg
        loss = (F.relu(margin - diff) * w).mean()
        acc = float((diff > 0).float().mean().item())
        mean_gap = float(diff.mean().item())
        median_gap = float(torch.median(diff).item())
    return {
        "loss": float(loss.item()),
        "pair_acc": acc,
        "mean_gap": mean_gap,
        "median_gap": median_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Train pairwise G(h) ranker for joint hypothesis.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pairs(args.pairs_jsonl)
    arrays = make_arrays(rows)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    x_neg_train, x_pos_train, w_train = tensorize(arrays, train_idx, device)
    x_neg_val, x_pos_val, w_val = tensorize(arrays, val_idx, device)

    model = ValueMLP(x_neg_train.shape[-1], int(args.hidden_dim), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_ids = torch.randint(0, x_neg_train.shape[0], (min(int(args.batch_size), x_neg_train.shape[0]),), device=device)
        s_neg = model(x_neg_train[batch_ids])
        s_pos = model(x_pos_train[batch_ids])
        diff = s_pos - s_neg
        loss = (F.relu(float(args.margin) - diff) * w_train[batch_ids]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == args.steps:
            metrics = {"step": step, "train_loss": float(loss.item()), **evaluate(model, x_neg_val, x_pos_val, w_val, float(args.margin))}
            history.append(metrics)
            if best is None or metrics["loss"] < best["loss"]:
                best = metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": int(x_neg_train.shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "margin": float(args.margin),
                        "task_names": list(arrays.get("task_names", [])),
                        "num_tasks": int(arrays.get("num_tasks", 0)),
                        "max_stage": float(arrays.get("max_stage", 1.0)),
                    },
                    args.output_dir / "joint_hypothesis_value_rank.pt",
                )

    summary = {
        "num_pairs": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "joint_hypothesis_value_rank.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
