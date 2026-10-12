from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row.get("sequence_index", -1)) for row in rows if int(row.get("sequence_index", -1)) >= 0})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row.get("sequence_index", -1)) in val else train_idx).append(idx)
    return train_idx, val_idx


def load_pair_rows(path: Path) -> list[dict[str, Any]]:
    pair_rows = list(iter_jsonl(path))
    if not pair_rows:
        raise ValueError(f"no rows in {path}")
    rows: list[dict[str, Any]] = []
    tasks = sorted({str(row.get("task", "unknown")) for row in pair_rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}
    num_tasks = len(tasks)
    max_stage = max(float(row.get("stage", 0)) for row in pair_rows)
    max_stage = max(max_stage, 1.0)
    for row in pair_rows:
        task_onehot = np.zeros((num_tasks,), dtype=np.float32)
        task_onehot[task_to_id[str(row.get("task", "unknown"))]] = 1.0
        stage = np.asarray([float(row.get("stage", 0)) / max_stage], dtype=np.float32)
        common = {
            "task": str(row.get("task", "unknown")),
            "sequence_index": int(row.get("sequence_index_neg", -1)),
            "stage": int(row.get("stage", 0)),
        }
        neg_feat = np.concatenate(
            [
                np.asarray(row["state_neg"], dtype=np.float32),
                np.asarray(row["future_neg"], dtype=np.float32),
                np.asarray(row["action_neg"], dtype=np.float32),
                task_onehot,
                stage,
            ],
            axis=0,
        ).astype(np.float32)
        pos_feat = np.concatenate(
            [
                np.asarray(row["state_pos"], dtype=np.float32),
                np.asarray(row["future_pos"], dtype=np.float32),
                np.asarray(row["action_pos"], dtype=np.float32),
                task_onehot,
                stage,
            ],
            axis=0,
        ).astype(np.float32)
        rows.append({**common, "x": neg_feat, "y": float(row.get("g_neg", 0.0)), "weight": float(row.get("return_gap", 1.0))})
        rows.append(
            {
                "task": str(row.get("task", "unknown")),
                "sequence_index": int(row.get("sequence_index_pos", row.get("sequence_index_neg", -1))),
                "stage": int(row.get("stage", 0)),
                "x": pos_feat,
                "y": float(row.get("g_pos", 0.0)),
                "weight": float(row.get("return_gap", 1.0)),
            }
        )
    return rows


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


def tensorize(rows: list[dict[str, Any]], ids: list[int], device: torch.device):
    x = torch.from_numpy(np.stack([rows[i]["x"] for i in ids], axis=0)).to(device)
    y = torch.tensor([rows[i]["y"] for i in ids], dtype=torch.float32, device=device)
    w = torch.tensor([rows[i]["weight"] for i in ids], dtype=torch.float32, device=device)
    w = w / max(float(w.mean().item()), 1e-6)
    return x, y, w


def evaluate(model: ValueMLP, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        mse = ((pred - y) ** 2).mean()
        mae = (pred - y).abs().mean()
        weighted = (((pred - y) ** 2) * w).mean()
        corr = float(np.corrcoef(pred.detach().cpu().numpy(), y.detach().cpu().numpy())[0, 1]) if len(y) > 1 else 0.0
    return {
        "loss": float(weighted.item()),
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "corr": float(0.0 if np.isnan(corr) else corr),
    }


def main():
    parser = argparse.ArgumentParser(description="Train G(h) value head for joint hypothesis.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pair_rows(args.pairs_jsonl)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    x_train, y_train, w_train = tensorize(rows, train_idx, device)
    x_val, y_val, w_val = tensorize(rows, val_idx, device)

    model = ValueMLP(x_train.shape[-1], int(args.hidden_dim), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_ids = torch.randint(0, x_train.shape[0], (min(int(args.batch_size), x_train.shape[0]),), device=device)
        pred = model(x_train[batch_ids])
        loss = ((((pred - y_train[batch_ids]) ** 2) * w_train[batch_ids])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == args.steps:
            metrics = {"step": step, "train_loss": float(loss.item()), **evaluate(model, x_val, y_val, w_val)}
            history.append(metrics)
            if best is None or metrics["loss"] < best["loss"]:
                best = metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": int(x_train.shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                    },
                    args.output_dir / "joint_hypothesis_value.pt",
                )

    summary = {
        "num_rows": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "joint_hypothesis_value.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
