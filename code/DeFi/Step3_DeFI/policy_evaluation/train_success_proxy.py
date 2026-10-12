import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from policy_evaluation.defi_memory_models import feature_signature, hashed_text_features


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rows(paths: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    rows: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            rows.append((path, row))
    return rows


def resolve_feature_path(jsonl_path: Path, feature_path: str) -> Path:
    path = Path(feature_path)
    if path.is_absolute():
        return path
    return (jsonl_path.parent / path).resolve()


def read_feature(feature_path: Path) -> Dict[str, np.ndarray]:
    with np.load(feature_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def make_sample(row: Dict[str, Any], feat: Dict[str, np.ndarray]) -> np.ndarray:
    current = feat.get("obs_current_feature")
    if current is None:
        current = feat.get("current_feature")
    future = feat.get("defi_future_feature")
    if future is None:
        future = feat.get("future_feature")
    end = feat.get("end_obs_current_feature")
    if end is None:
        end = future if future is not None else current
    current_sig = feature_signature(current, dim=768)
    future_sig = feature_signature(future if future is not None else current, dim=768)
    end_sig = feature_signature(end if end is not None else future, dim=768)
    delta_sig = feature_signature(future_sig - current_sig, dim=768)
    task_sig = hashed_text_features([str(row.get("task", ""))], dim=64)[0]
    lang_sig = hashed_text_features([str(row.get("lang_goal", row.get("lang_text", "")))], dim=64)[0]
    meta = np.asarray(
        [
            float(row.get("steps", 0)) / float(max(int(row.get("ep_len", 25)), 1)),
            float(row.get("invalid_actions", 0)) / float(max(int(row.get("ep_len", 25)), 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([current_sig, future_sig, end_sig, delta_sig, task_sig, lang_sig, meta], axis=0).astype(np.float32)


def build_dataset(rows: List[Tuple[Path, Dict[str, Any]]], baseline_only: bool = True) -> Dict[str, np.ndarray]:
    x_rows = []
    y_rows = []
    task_rows = []
    meta_rows = []
    for jsonl_path, row in rows:
        if baseline_only and str(row.get("attempt_type", "baseline")) != "baseline":
            continue
        feature_path = row.get("feature_path")
        if not feature_path:
            continue
        resolved = resolve_feature_path(jsonl_path, str(feature_path))
        if not resolved.exists():
            continue
        feat = read_feature(resolved)
        x_rows.append(make_sample(row, feat))
        y_rows.append(1.0 if bool(row.get("success", False)) else 0.0)
        task_rows.append(str(row.get("task", "unknown")))
        meta_rows.append(
            {
                "task": str(row.get("task", "unknown")),
                "lang_goal": str(row.get("lang_goal", "")),
                "success": bool(row.get("success", False)),
                "attempt_type": str(row.get("attempt_type", "baseline")),
                "feature_path": str(resolved),
            }
        )
    if not x_rows:
        raise ValueError("No usable rows were found")
    return {
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "y": np.asarray(y_rows, dtype=np.float32),
        "tasks": np.asarray(task_rows),
        "meta": meta_rows,
    }


def split_by_task(tasks: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(tasks.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_tasks = set(unique[:n_val].tolist())
    train_idx = np.asarray([idx for idx, task in enumerate(tasks) if task not in val_tasks], dtype=np.int64)
    val_idx = np.asarray([idx for idx, task in enumerate(tasks) if task in val_tasks], dtype=np.int64)
    return train_idx, val_idx


def balance_train_indices(y: np.ndarray, idx: np.ndarray, seed: int) -> np.ndarray:
    if idx.size == 0:
        return idx
    train_y = y[idx]
    pos_idx = idx[train_y > 0.5]
    neg_idx = idx[train_y <= 0.5]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return idx
    n_keep = min(pos_idx.size, neg_idx.size)
    rng = np.random.default_rng(seed)
    pos_keep = rng.choice(pos_idx, size=n_keep, replace=False)
    neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)
    balanced = np.concatenate([pos_keep, neg_keep], axis=0)
    rng.shuffle(balanced)
    return balanced.astype(np.int64)


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.1, 0.9, 33):
        pred = probs >= threshold
        tp = float(np.logical_and(pred, labels > 0.5).sum())
        fp = float(np.logical_and(pred, labels <= 0.5).sum())
        fn = float(np.logical_and(~pred, labels > 0.5).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(f1), "precision": float(precision), "recall": float(recall)}
    return best


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, float]:
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = torch.sigmoid(logits).cpu().numpy()
    pred = probs >= threshold
    acc = float((pred == (y > 0.5)).mean())
    tp = float(np.logical_and(pred, y > 0.5).sum())
    fp = float(np.logical_and(pred, y <= 0.5).sum())
    fn = float(np.logical_and(~pred, y > 0.5).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)
    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(pred.mean()),
        "mean_prob": float(probs.mean()),
    }


class SuccessProxyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight success/fail proxy from rollout rows.")
    parser.add_argument("--rollout-jsonls", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-only", action="store_true", default=True)
    parser.add_argument("--include-repair-rows", action="store_true")
    parser.add_argument("--balance-train", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows = load_rows(args.rollout_jsonls)
    dataset = build_dataset(rows, baseline_only=not args.include_repair_rows)
    train_idx, val_idx = split_by_task(dataset["tasks"], args.val_fraction, args.seed)
    if args.balance_train:
        train_idx = balance_train_indices(dataset["y"], train_idx, args.seed)
    x_train = dataset["x"][train_idx]
    y_train = dataset["y"][train_idx]
    x_val = dataset["x"][val_idx]
    y_val = dataset["y"][val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    model = SuccessProxyMLP(x_train.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_score = -1.0
    best_thr = 0.5
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.from_numpy(x_val).to(device))).cpu().numpy()
        thr = best_threshold(val_probs, y_val)
        metrics = evaluate(model.cpu(), x_val, y_val, thr["threshold"])
        model.to(device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_f1": metrics["f1"],
                "val_acc": metrics["acc"],
                "val_threshold": thr["threshold"],
            }
        )
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_thr = float(thr["threshold"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "threshold": float(best_thr),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "baseline_only": bool(not args.include_repair_rows),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_layout": [
            "current_signature_768",
            "future_signature_768",
            "end_signature_768",
            "delta_signature_768",
            "task_hash_64",
            "lang_hash_64",
            "steps_norm",
            "invalid_actions_norm",
        ],
    }
    torch.save(checkpoint, args.output_dir / "success_proxy.pt")

    summary = {
        "num_examples": int(len(dataset["x"])),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "positive_rate": float(dataset["y"].mean()),
        "train_positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "best_threshold": float(best_thr),
        "best_val": evaluate(model, x_val, y_val, best_thr),
        "history_tail": history[-5:],
        "baseline_only": bool(not args.include_repair_rows),
        "balance_train": bool(args.balance_train),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
