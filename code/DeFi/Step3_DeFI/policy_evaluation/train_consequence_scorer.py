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


class ConsequenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.success_head = nn.Linear(hidden_dim, 1)
        self.steps_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return (
            self.success_head(h).squeeze(-1),
            self.steps_head(h).squeeze(-1),
            self.value_head(h).squeeze(-1),
        )


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def branch_outcome(row: Dict[str, Any], branch: str, ep_len: int) -> Tuple[float, float, float]:
    by_name = {item["name"]: item for item in row["branches"]}
    item = by_name[branch]
    success = float(bool(item["success"]))
    steps_norm = float(item["steps"]) / float(ep_len)
    value = float(row["hypothesis_values"][branch])
    return success, steps_norm, value


def feature(row: Dict[str, Any], branch: str) -> np.ndarray:
    state = np.asarray(row["state_feature"], dtype=np.float32)
    future = np.asarray(row["hypothesis_features"][branch], dtype=np.float32)
    scalars = np.asarray([float(row["scalars"][name]) for name in SCALARS], dtype=np.float32)
    branch_onehot = np.asarray([float(branch == name) for name in BRANCHES], dtype=np.float32)
    return np.concatenate([state, future, future - state, scalars, branch_onehot], axis=0)


def build_arrays(rows: List[Dict[str, Any]], ep_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, success, steps, value, groups = [], [], [], [], []
    for group_id, row in enumerate(rows):
        for branch in BRANCHES:
            if branch not in row["hypothesis_features"] or branch not in row["hypothesis_values"]:
                continue
            y_success, y_steps, y_value = branch_outcome(row, branch, ep_len)
            xs.append(feature(row, branch))
            success.append(y_success)
            steps.append(y_steps)
            value.append(y_value)
            groups.append(group_id)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(success, dtype=np.float32),
        np.asarray(steps, dtype=np.float32),
        np.asarray(value, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
    )


def split_groups(num_groups: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.arange(num_groups)
    rng.shuffle(groups)
    n_val = max(1, int(round(num_groups * val_fraction)))
    return groups[n_val:], groups[:n_val]


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
    denom = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if denom <= 1e-12 else 1.0 - float(np.sum((y - pred) ** 2)) / denom


def evaluate(
    model: ConsequenceMLP,
    x: np.ndarray,
    y_success: np.ndarray,
    y_steps: np.ndarray,
    y_value: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        success_logit, pred_steps, pred_value = model(torch.from_numpy(x))
        pred_success = torch.sigmoid(success_logit).cpu().numpy()
        pred_steps_np = pred_steps.cpu().numpy()
        pred_value_np = pred_value.cpu().numpy()

    top1_value = []
    top1_success = []
    top1_success_steps = []
    state_corr = []
    for group in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == group)
        oracle = int(np.argmax(y_value[idx]))
        top1_value.append(int(oracle == np.argmax(pred_value_np[idx])))
        top1_success.append(int(oracle == np.argmax(pred_success[idx])))
        top1_success_steps.append(int(oracle == np.argmax(pred_success[idx] - pred_steps_np[idx])))
        state_corr.append(spearman(y_value[idx], pred_value_np[idx]))
    success_pred = (pred_success >= 0.5).astype(np.float32)
    return {
        "success_acc": float(np.mean(success_pred == y_success)),
        "success_brier": float(np.mean((pred_success - y_success) ** 2)),
        "steps_mae": float(np.mean(np.abs(pred_steps_np - y_steps))),
        "value_mse": float(np.mean((pred_value_np - y_value) ** 2)),
        "value_r2": r2(y_value, pred_value_np),
        "value_spearman": spearman(y_value, pred_value_np),
        "state_spearman_mean": float(np.mean(state_corr)) if state_corr else 0.0,
        "top1_branch_match": float(np.mean(top1_value)) if top1_value else 0.0,
        "top1_success_score": float(np.mean(top1_success)) if top1_success else 0.0,
        "top1_success_minus_steps_score": float(np.mean(top1_success_steps)) if top1_success_steps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local consequence scorer for hypothesis selection.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ep-len", type=int, default=120)
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
    x, y_success, y_steps, y_value, groups = build_arrays(rows, args.ep_len)
    train_groups, val_groups = split_groups(len(rows), args.val_fraction, args.seed)
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)

    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std

    model = ConsequenceMLP(x.shape[1], args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    x_train = torch.from_numpy(x_norm[train_mask])
    success_train = torch.from_numpy(y_success[train_mask])
    steps_train = torch.from_numpy(y_steps[train_mask])
    value_train = torch.from_numpy(y_value[train_mask])
    x_val = torch.from_numpy(x_norm[val_mask])
    value_val = torch.from_numpy(y_value[val_mask])

    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        success_logit, pred_steps, pred_value = model(x_train)
        loss = (
            bce(success_logit, success_train)
            + 0.5 * mse(pred_steps, steps_train)
            + mse(pred_value, value_train)
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, _, pred_value_val = model(x_val)
            val_loss = float(mse(pred_value_val, value_val).item())
        if val_loss <= best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "num_decisions": len(rows),
        "num_branch_examples": int(len(x)),
        "input_dim": int(x.shape[1]),
        "target": "consequence proxy: branch success, normalized steps, recoverability value",
        "train": evaluate(model, x_norm[train_mask], y_success[train_mask], y_steps[train_mask], y_value[train_mask], groups[train_mask]),
        "val": evaluate(model, x_norm[val_mask], y_success[val_mask], y_steps[val_mask], y_value[val_mask], groups[val_mask]),
        "all": evaluate(model, x_norm, y_success, y_steps, y_value, groups),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "mean": mean.squeeze(0),
            "std": std.squeeze(0),
            "branches": BRANCHES,
            "scalars": SCALARS,
        },
        args.output_dir / "consequence_scorer.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
