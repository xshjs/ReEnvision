import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


CLASSES = ["continue", "retry_demo", "recover"]


class SelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_examples(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    label_to_id = {name: idx for idx, name in enumerate(CLASSES)}
    rows = [row for row in rows if row["label"] in label_to_id]
    x = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    y = np.asarray([label_to_id[row["label"]] for row in rows], dtype=np.int64)
    return x, y, rows


def stratified_split(y: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def accuracy_and_confusion(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, List[List[int]]]:
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == y).float().mean().item()) if y.numel() else 0.0
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    for target, output in zip(y.cpu().tolist(), pred.cpu().tolist()):
        confusion[target, output] += 1
    return acc, confusion.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small selector classifier for continue/retry/recover.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, rows = load_examples(args.examples)
    if len(rows) < 6:
        raise ValueError(f"Need at least 6 examples, got {len(rows)}")
    train_idx, val_idx = stratified_split(y, args.val_fraction, args.seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std

    x_train = torch.from_numpy(x_norm[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x_norm[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    counts = Counter(y_train.tolist())
    class_weights = torch.tensor(
        [len(y_train) / max(1, len(CLASSES) * counts.get(i, 0)) for i in range(len(CLASSES))],
        dtype=torch.float32,
    )
    model = SelectorMLP(input_dim=x.shape[1], hidden_dim=args.hidden_dim, num_classes=len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val = -1.0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc, _ = accuracy_and_confusion(model(x_val), y_val)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc, train_confusion = accuracy_and_confusion(model(x_train), y_train)
        val_acc, val_confusion = accuracy_and_confusion(model(x_val), y_val)
        all_acc, all_confusion = accuracy_and_confusion(model(torch.from_numpy(x_norm)), torch.from_numpy(y))

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": args.hidden_dim,
        "classes": CLASSES,
        "mean": mean.squeeze(0),
        "std": std.squeeze(0),
    }
    torch.save(checkpoint, args.output_dir / "selector_mlp.pt")
    metrics = {
        "num_examples": len(rows),
        "input_dim": int(x.shape[1]),
        "class_distribution": {CLASSES[idx]: int(count) for idx, count in Counter(y.tolist()).items()},
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "all_accuracy": all_acc,
        "train_confusion": train_confusion,
        "val_confusion": val_confusion,
        "all_confusion": all_confusion,
        "confusion_rows": CLASSES,
        "confusion_cols": CLASSES,
        "feature_note": "mean-pooled GFDM current/future/delta features plus error memory, contradiction proxy, and drift score",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
