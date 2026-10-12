import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


BRANCHES = ["continue", "retry_demo", "recover"]


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def immediate_values(row: Dict[str, Any], ep_len: int) -> np.ndarray:
    by_name = {item["name"]: item for item in row["branches"]}
    values = []
    for name in BRANCHES:
        item = by_name.get(name)
        if item is None or not item["success"]:
            values.append(0.0)
        else:
            values.append(1.0 + (ep_len - float(item["steps"])) / ep_len)
    return np.asarray(values, dtype=np.float32)


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:], idx[:n_val]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a)
    rb = rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    flat_spearman = spearman(y_true.reshape(-1), y_pred.reshape(-1))
    per_branch = {name: spearman(y_true[:, i], y_pred[:, i]) for i, name in enumerate(BRANCHES)}
    per_state = [spearman(y_true[i], y_pred[i]) for i in range(len(y_true))]
    top1 = float(np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)))
    return {
        "spearman_flat": flat_spearman,
        "spearman_state_mean": float(np.mean(per_state)),
        "top1_branch_match": top1,
        **{f"spearman_{name}": value for name, value in per_branch.items()},
    }


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy()
    return {
        "mse": float(np.mean((pred - y) ** 2)),
        "mae": float(np.mean(np.abs(pred - y))),
        "r2": r2_score(y, pred),
        **rank_metrics(y, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train value predictor for continue/retry/recover branches.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ep-len", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.examples)
    x = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    y = np.stack([immediate_values(row, args.ep_len) for row in rows], axis=0)
    train_idx, val_idx = split_indices(len(rows), args.val_fraction, args.seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std

    x_train = torch.from_numpy(x_norm[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x_norm[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    model = ValueMLP(x.shape[1], args.hidden_dim, len(BRANCHES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        if val_loss <= best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "num_examples": len(rows),
        "input_dim": int(x.shape[1]),
        "branches": BRANCHES,
        "target": "immediate branch value: 0 if fail else 1 + (ep_len - steps)/ep_len",
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "train": evaluate(model, x_norm[train_idx], y[train_idx]),
        "val": evaluate(model, x_norm[val_idx], y[val_idx]),
        "all": evaluate(model, x_norm, y),
        "limitation": "Input feature currently contains state + continue/GFDM future; retry/recover branch futures were not saved in this dataset.",
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "branches": BRANCHES,
            "mean": mean.squeeze(0),
            "std": std.squeeze(0),
        },
        args.output_dir / "value_mlp.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
