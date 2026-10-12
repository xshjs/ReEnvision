import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class ReflectionCritic(nn.Module):
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


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:], idx[:n_val]


def rank_metrics(pred: np.ndarray, y: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    hits = []
    regrets = []
    for gid in sorted(set(group.tolist())):
        idx = np.where(group == gid)[0]
        if len(idx) < 2:
            continue
        chosen = idx[int(np.argmax(pred[idx]))]
        best = idx[int(np.argmax(y[idx]))]
        hits.append(float(chosen == best))
        regrets.append(float(y[best] - y[chosen]))
    return {
        "top1_match": float(np.mean(hits)) if hits else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "num_groups": int(len(hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reflection Critic for MoE top-k future reranking.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    meta = [json.loads(str(row)) for row in data["meta"]]
    groups = np.asarray([f'{m["sequence_index"]}:{m["subtask_index"]}' for m in meta])
    train_idx, val_idx = split_indices(len(x), args.val_fraction, args.seed)

    model = ReflectionCritic(x.shape[1], args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])
    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        pred = model(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(x_val) - y_val) ** 2)
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_all = model(torch.from_numpy(x)).numpy()
        pred_val = model(x_val).numpy()
    metrics = {
        "num_examples": int(len(x)),
        "input_dim": int(x.shape[1]),
        "target": "branch value: 0 if fail else 1 + remaining_step_fraction",
        "mse_all": float(np.mean((pred_all - y) ** 2)),
        "mse_val": float(np.mean((pred_val - y[val_idx]) ** 2)),
        "rank_all": rank_metrics(pred_all, y, groups),
        "rank_val": rank_metrics(pred_val, y[val_idx], groups[val_idx]),
    }
    torch.save(
        {"model_state": model.state_dict(), "input_dim": int(x.shape[1]), "hidden_dim": args.hidden_dim},
        args.output_dir / "reflection_critic.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
