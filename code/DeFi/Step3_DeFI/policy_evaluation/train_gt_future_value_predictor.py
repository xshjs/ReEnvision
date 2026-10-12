import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


BRANCHES = ["continue", "retry_demo", "recover"]
SCALARS = [
    "contradiction_score",
    "drift_score",
    "error_memory_0",
    "error_memory_1",
    "error_memory_2",
    "error_memory_3",
    "error_score",
    "l2_error",
]


class VectorValueMLP(nn.Module):
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


def values(row: Dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row["hypothesis_values"].get(branch, 0.0)) for branch in BRANCHES], dtype=np.float32)


def best_future(row: Dict[str, Any]) -> np.ndarray:
    y = values(row)
    for idx in np.argsort(-y):
        branch = BRANCHES[int(idx)]
        if branch in row["hypothesis_features"]:
            return np.asarray(row["hypothesis_features"][branch], dtype=np.float32)
    return np.asarray(row["hypothesis_features"]["continue"], dtype=np.float32)


def make_feature(row: Dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["state_feature"], dtype=np.float32)
    future = best_future(row)
    scalars = np.asarray([float(row["scalars"][name]) for name in SCALARS], dtype=np.float32)
    return np.concatenate([state, future, future - state, scalars], axis=0)


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:], idx[:n_val]


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
    if len(a) < 2:
        return 0.0
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if denom <= 1e-12 else 1.0 - float(np.sum((y - pred) ** 2)) / denom


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy()
    state_corr = [spearman(y[i], pred[i]) for i in range(len(y))]
    return {
        "mse": float(np.mean((pred - y) ** 2)),
        "mae": float(np.mean(np.abs(pred - y))),
        "r2": r2(y, pred),
        "spearman_flat": spearman(y.reshape(-1), pred.reshape(-1)),
        "state_spearman_mean": float(np.mean(state_corr)),
        "top1_branch_match": float(np.mean(np.argmax(y, axis=1) == np.argmax(pred, axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vector value predictor with GT-future proxy input.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.examples)
    x = np.stack([make_feature(row) for row in rows], axis=0).astype(np.float32)
    y = np.stack([values(row) for row in rows], axis=0).astype(np.float32)
    train_idx, val_idx = split_indices(len(rows), args.val_fraction, args.seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std

    model = VectorValueMLP(x.shape[1], args.hidden_dim, len(BRANCHES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    x_train = torch.from_numpy(x_norm[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x_norm[val_idx])
    y_val = torch.from_numpy(y[val_idx])

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
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "num_examples": len(rows),
        "input_dim": int(x.shape[1]),
        "branches": BRANCHES,
        "input": "state + GT-future proxy + delta + scalars",
        "gt_future_proxy": "future feature of the branch with max observed immediate value",
        "target": "[V_continue, V_retry_demo, V_recover]",
        "train": evaluate(model, x_norm[train_idx], y[train_idx]),
        "val": evaluate(model, x_norm[val_idx], y[val_idx]),
        "all": evaluate(model, x_norm, y),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "branches": BRANCHES,
            "scalars": SCALARS,
            "mean": mean.squeeze(0),
            "std": std.squeeze(0),
        },
        args.output_dir / "gt_future_value_mlp.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
