import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def load_probe_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:], idx[:n_val]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def evaluate(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    sims = cosine(pred, target)
    l2 = np.linalg.norm(pred - target, axis=1)
    return {
        "cosine_mean": float(sims.mean()),
        "cosine_median": float(np.median(sims)),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train small V-JEPA2 current->future adapter.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_probe_features(args.features)
    x = data["train_start"].astype(np.float32)
    y = data["train_end"].astype(np.float32)
    train_idx, val_idx = split_indices(len(x), args.val_fraction, args.seed)
    model = Adapter(x.shape[1], args.hidden_dim)
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
        loss = 1.0 - torch.mean(torch.sum(pred * y_train, dim=-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = 1.0 - torch.mean(torch.sum(model(x_val) * y_val, dim=-1))
        if float(val_loss) <= best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_train = model(torch.from_numpy(x[train_idx])).numpy()
        pred_val = model(torch.from_numpy(x[val_idx])).numpy()
        pred_all = model(torch.from_numpy(x)).numpy()
    metrics = {
        "num_examples": int(len(x)),
        "input_dim": int(x.shape[1]),
        "target": "V-JEPA2 current feature -> V-JEPA2 GT future feature",
        "current_to_future_all": evaluate(x, y),
        "train": evaluate(pred_train, y[train_idx]),
        "val": evaluate(pred_val, y[val_idx]),
        "all": evaluate(pred_all, y),
    }
    torch.save(
        {"model_state": model.state_dict(), "input_dim": int(x.shape[1]), "hidden_dim": args.hidden_dim},
        args.output_dir / "vjepa2_future_adapter.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
