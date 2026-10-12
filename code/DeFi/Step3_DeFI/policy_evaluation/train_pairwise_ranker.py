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


class RankMLP(nn.Module):
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


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def feature(row: Dict[str, Any], branch: str, mode: str) -> np.ndarray:
    state = np.asarray(row["state_feature"], dtype=np.float32)
    if mode == "branch_future":
        future = np.asarray(row["hypothesis_features"][branch], dtype=np.float32)
    elif mode == "gt_future_proxy":
        vals = values(row)
        best = BRANCHES[int(np.argmax(vals))]
        future = np.asarray(row["hypothesis_features"][best], dtype=np.float32)
    else:
        raise ValueError(mode)
    scalars = np.asarray([float(row["scalars"][name]) for name in SCALARS], dtype=np.float32)
    branch_onehot = np.asarray([float(branch == name) for name in BRANCHES], dtype=np.float32)
    return np.concatenate([state, future, future - state, scalars, branch_onehot], axis=0)


def values(row: Dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row["hypothesis_values"].get(branch, 0.0)) for branch in BRANCHES], dtype=np.float32)


def build_arrays(rows: List[Dict[str, Any]], mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for group_id, row in enumerate(rows):
        vals = values(row)
        for idx, branch in enumerate(BRANCHES):
            xs.append(feature(row, branch, mode))
            ys.append(vals[idx])
            groups.append(group_id)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(groups, dtype=np.int64)


def split_groups(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.arange(n)
    rng.shuffle(groups)
    n_val = max(1, int(round(n * val_fraction)))
    return groups[n_val:], groups[:n_val]


def pair_indices(y: np.ndarray, groups: np.ndarray, group_subset: np.ndarray, min_margin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = set(group_subset.tolist())
    left, right, target = [], [], []
    for group in sorted(subset):
        idx = np.flatnonzero(groups == group)
        for a in idx:
            for b in idx:
                if a == b:
                    continue
                diff = float(y[a] - y[b])
                if abs(diff) < min_margin:
                    continue
                if diff > 0:
                    left.append(a); right.append(b); target.append(1.0)
                else:
                    left.append(b); right.append(a); target.append(1.0)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64), np.asarray(target, dtype=np.float32)


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
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, groups: np.ndarray, group_subset: np.ndarray) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        score = model(torch.from_numpy(x)).cpu().numpy()
    top1, corr = [], []
    for group in group_subset:
        idx = np.flatnonzero(groups == group)
        top1.append(int(np.argmax(y[idx]) == np.argmax(score[idx])))
        corr.append(spearman(y[idx], score[idx]))
    return {
        "top1_branch_match": float(np.mean(top1)),
        "state_spearman_mean": float(np.mean(corr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise branch ranker.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["branch_future", "gt_future_proxy"], default="gt_future_proxy")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.examples)
    x, y, groups = build_arrays(rows, args.mode)
    train_groups, val_groups = split_groups(len(rows), args.val_fraction, args.seed)
    mean = x[np.isin(groups, train_groups)].mean(axis=0, keepdims=True)
    std = x[np.isin(groups, train_groups)].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std

    left, right, _ = pair_indices(y, groups, train_groups, args.min_margin)
    if len(left) == 0:
        raise ValueError("No pairwise comparisons after min-margin filtering")
    x_tensor = torch.from_numpy(x_norm)
    left_t = torch.from_numpy(left)
    right_t = torch.from_numpy(right)
    model = RankMLP(x.shape[1], args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MarginRankingLoss(margin=0.1)
    target = torch.ones(len(left_t), dtype=torch.float32)

    best_state = None
    best_top1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(x_tensor)
        loss = criterion(scores[left_t], scores[right_t], target)
        loss.backward()
        optimizer.step()
        if epoch == args.epochs - 1 or (epoch + 1) % args.eval_every == 0:
            metrics = evaluate(model, x_norm, y, groups, val_groups)
            if metrics["top1_branch_match"] >= best_top1:
                best_top1 = metrics["top1_branch_match"]
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "num_decisions": len(rows),
        "num_train_pairs": int(len(left)),
        "input_dim": int(x.shape[1]),
        "mode": args.mode,
        "min_margin": args.min_margin,
        "train": evaluate(model, x_norm, y, groups, train_groups),
        "val": evaluate(model, x_norm, y, groups, val_groups),
        "all": evaluate(model, x_norm, y, groups, np.arange(len(rows))),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "mode": args.mode,
            "branches": BRANCHES,
            "scalars": SCALARS,
            "mean": mean.squeeze(0),
            "std": std.squeeze(0),
        },
        args.output_dir / "pairwise_ranker.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
