import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


class SkillProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            steps = item.get("steps") or []
            if not steps:
                continue
            pred = steps[0].get("visual", {}).get("pred", [])
            if not pred:
                continue
            rows.append(
                {
                    "trajectory_id": item["trajectory_id"],
                    "task": item["task"],
                    "success": bool(item["success"]),
                    "feature": np.asarray(pred, dtype=np.float32),
                }
            )
    return rows


def make_stratified_folds(labels: np.ndarray, num_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(num_folds)]
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            folds[offset % num_folds].append(int(index))
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = SkillProbe(x_train.shape[1], int(y_train.max()) + 1, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    rng = np.random.default_rng(args.seed)
    model.train()
    for _ in range(args.epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            loss = criterion(model(x_tensor[batch]), y_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu()
    probs = torch.softmax(logits, dim=1).numpy()
    return logits.numpy(), probs


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    top1 = Counter(row["future_skill_top1"] for row in rows)
    goal = Counter(row["task"] for row in rows)
    matches = sum(row["future_skill_top1"] == row["task"] for row in rows)
    return {
        "total": total,
        "goal_task_counts": dict(goal.most_common()),
        "future_skill_top1_counts": dict(top1.most_common()),
        "future_skill_top1_rates": {key: value / total for key, value in top1.most_common()} if total else {},
        "goal_equals_future_skill": matches,
        "goal_equals_future_skill_rate": matches / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Success vs predicted future skill from saved GFDM future features.")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.trajectories)
    skills = sorted({row["task"] for row in rows})
    skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
    idx_to_skill = {idx: skill for skill, idx in skill_to_idx.items()}
    x = np.stack([row["feature"] for row in rows]).astype(np.float32)
    y = np.asarray([skill_to_idx[row["task"]] for row in rows], dtype=np.int64)
    folds = make_stratified_folds(y, args.num_folds, args.seed)

    logits_all = np.zeros((len(rows), len(skills)), dtype=np.float32)
    probs_all = np.zeros_like(logits_all)
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.asarray([idx for idx in range(len(rows)) if idx not in set(test_idx.tolist())], dtype=np.int64)
        mean = x[train_idx].mean(axis=0, keepdims=True)
        std = x[train_idx].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        logits, probs = train_predict((x[train_idx] - mean) / std, y[train_idx], (x[test_idx] - mean) / std, args)
        logits_all[test_idx] = logits
        probs_all[test_idx] = probs
        print(f"fold {fold_idx + 1}/{len(folds)} done: n_test={len(test_idx)}")

    predictions = []
    for idx, row in enumerate(rows):
        order = np.argsort(-logits_all[idx])
        item = {
            "trajectory_id": row["trajectory_id"],
            "task": row["task"],
            "success": row["success"],
            "future_skill_top1": idx_to_skill[int(order[0])],
            "top1_prob": float(probs_all[idx, order[0]]),
            "top3": [idx_to_skill[int(item)] for item in order[:3]],
        }
        predictions.append(item)

    by_success = {
        "success": summarize_group([row for row in predictions if row["success"]]),
        "failure": summarize_group([row for row in predictions if not row["success"]]),
    }
    per_task_success = defaultdict(list)
    for row in predictions:
        per_task_success[(row["task"], "success" if row["success"] else "failure")].append(row)
    per_task = {
        f"{task}:{status}": summarize_group(items)
        for (task, status), items in sorted(per_task_success.items())
    }
    summary = {
        "source": str(args.trajectories),
        "n": len(rows),
        "skills": skills,
        "probe": {
            "type": "5-fold cross-validated MLP",
            "input": "saved step[0].visual.pred GFDM future feature",
            "num_folds": args.num_folds,
        },
        "by_success": by_success,
        "per_task_success": per_task,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
