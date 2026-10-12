import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class RiskDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_groups(meta: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([f'{json.loads(str(row))["sequence_index"]}:{json.loads(str(row))["subtask_index"]}' for row in meta])
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_groups = set(unique[:n_val].tolist())
    train_idx = np.asarray([idx for idx, group in enumerate(groups) if group not in val_groups], dtype=np.int64)
    val_idx = np.asarray([idx for idx, group in enumerate(groups) if group in val_groups], dtype=np.int64)
    return train_idx, val_idx, groups


def metrics(logits: np.ndarray, labels: np.ndarray, meta: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = probs >= 0.5
    rows = [json.loads(str(row)) for row in meta]
    top1_idx = np.asarray([idx for idx, row in enumerate(rows) if row["router_rank"] == 0], dtype=np.int64)
    return {
        "acc": float(np.mean(pred == labels)) if len(labels) else 0.0,
        "positive_rate": float(np.mean(labels)) if len(labels) else 0.0,
        "pred_positive_rate": float(np.mean(pred)) if len(labels) else 0.0,
        "top1_risk_mean": float(np.mean(probs[top1_idx])) if len(top1_idx) else 0.0,
        "top1_positive_rate": float(np.mean(labels[top1_idx])) if len(top1_idx) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conservative bad-future risk detector.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--risk-threshold-value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    value = data["y"].astype(np.float32)
    meta = data["meta"]
    y = (value < args.risk_threshold_value).astype(np.float32)
    train_idx, val_idx, _ = split_groups(meta, args.val_fraction, args.seed)

    model = RiskDetector(x.shape[1], args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])
    pos_weight = torch.tensor([(len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)], dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        logits = model(x_train)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.binary_cross_entropy_with_logits(model(x_val), y_val)
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_all = model(torch.from_numpy(x)).numpy()
        logits_train = model(x_train).numpy()
        logits_val = model(x_val).numpy()
    out = {
        "num_examples": int(len(x)),
        "input_dim": int(x.shape[1]),
        "risk_label": f"value < {args.risk_threshold_value}",
        "all": metrics(logits_all, y, meta),
        "train": metrics(logits_train, y[train_idx], meta[train_idx]),
        "val": metrics(logits_val, y[val_idx], meta[val_idx]),
    }
    torch.save(
        {"model_state": model.state_dict(), "input_dim": int(x.shape[1]), "hidden_dim": args.hidden_dim},
        args.output_dir / "reflection_risk_detector.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
