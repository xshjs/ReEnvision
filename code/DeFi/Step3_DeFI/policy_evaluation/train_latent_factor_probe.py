from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "object_displacement_underestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
    "goal_progress_hallucinated": "goal_completion",
}


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pool_feature(future: np.ndarray, delta: np.ndarray, mode: str) -> np.ndarray:
    future_mean = future.mean(axis=1)
    future_std = future.std(axis=1)
    delta_mean = delta.mean(axis=1)
    delta_abs = np.abs(delta).mean(axis=1)
    delta_norm_token = np.linalg.norm(delta, axis=2)
    delta_token_stats = np.stack(
        [
            delta_norm_token.mean(axis=1),
            delta_norm_token.std(axis=1),
            delta_norm_token.max(axis=1),
        ],
        axis=1,
    )
    if mode == "future":
        return future_mean
    if mode == "delta":
        return delta_mean
    if mode == "abs_delta":
        return delta_abs
    if mode == "future_abs_delta":
        return np.concatenate([future_mean, delta_abs], axis=1)
    if mode == "future_stats_abs_delta":
        return np.concatenate([future_mean, future_std, delta_abs, delta_token_stats], axis=1)
    raise ValueError(f"Unsupported feature mode: {mode}")


def stratified_split(labels: list[int], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)
    train_idx = []
    val_idx = []
    for indices in by_label.values():
        indices = list(indices)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 1
        n_val = min(n_val, max(1, len(indices) - 1)) if len(indices) > 1 else 1
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def standardize_train_val(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


class Probe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, model_type: str, hidden_dim: int, dropout: float):
        super().__init__()
        if model_type == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif model_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    per_class = {}
    f1s = []
    matrix = {name: Counter() for name in label_names}
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[label_names[t]][label_names[p]] += 1
    for idx, name in enumerate(label_names):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1s.append(f1)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((y_true == idx).sum()),
        }
    return {
        "acc": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion": {gold: dict(counter) for gold, counter in matrix.items()},
    }


def train_once(
    x: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    model_type: str,
    seed: int,
    val_ratio: float,
    steps: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    device: str,
) -> dict[str, Any]:
    set_seed(seed)
    train_idx, val_idx = stratified_split(y.tolist(), val_ratio, seed)
    x_train, x_val, mean, std = standardize_train_val(x[train_idx], x[val_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = Probe(x.shape[1], len(label_names), model_type, hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tx = torch.from_numpy(x_train.astype(np.float32)).to(device)
    ty = torch.from_numpy(y_train.astype(np.int64)).to(device)
    vx = torch.from_numpy(x_val.astype(np.float32)).to(device)

    best = None
    best_state = None
    for step in range(steps):
        model.train()
        logits = model(tx)
        loss = F.cross_entropy(logits, ty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == steps - 1 or step % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(vx).argmax(dim=1).detach().cpu().numpy()
            m = metrics(y_val, pred, label_names)
            score = m["macro_f1"]
            if best is None or score > best["macro_f1"]:
                best = {**m, "step": int(step), "loss": float(loss.detach().cpu().item())}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(tx).argmax(dim=1).detach().cpu().numpy()
        val_pred = model(vx).argmax(dim=1).detach().cpu().numpy()
    return {
        "model_type": model_type,
        "seed": int(seed),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "feature_dim": int(x.shape[1]),
        "label_names": label_names,
        "train_metrics": metrics(y_train, train_pred, label_names),
        "val_metrics": metrics(y_val, val_pred, label_names),
        "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear/MLP probes for weak latent failure factors.")
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["future", "delta", "abs_delta", "future_abs_delta", "future_stats_abs_delta"], default="future_stats_abs_delta")
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    rows = list(iter_jsonl(args.memory_jsonl))
    arrays = np.load(args.memory_npz, allow_pickle=True)
    future = arrays["future_features"].astype(np.float32)
    delta = arrays["repair_deltas"].astype(np.float32)
    if len(rows) != future.shape[0] or len(rows) != delta.shape[0]:
        raise ValueError(f"Row/array mismatch: rows={len(rows)} future={future.shape} delta={delta.shape}")

    labels_raw = [MISMATCH_TO_FACTOR.get(str(row.get("mismatch_type", "")), "unknown") for row in rows]
    keep = [idx for idx, label in enumerate(labels_raw) if label != "unknown"]
    label_names = sorted(set(labels_raw[idx] for idx in keep))
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    x = pool_feature(future[keep], delta[keep], args.feature_mode).astype(np.float32)
    y = np.asarray([label_to_id[labels_raw[idx]] for idx in keep], dtype=np.int64)

    results = []
    for seed in args.seeds:
        results.append(
            train_once(
                x,
                y,
                label_names,
                args.model_type,
                int(seed),
                float(args.val_ratio),
                int(args.steps),
                float(args.lr),
                float(args.weight_decay),
                int(args.hidden_dim),
                float(args.dropout),
                str(args.device),
            )
        )

    val_acc = [r["val_metrics"]["acc"] for r in results]
    val_f1 = [r["val_metrics"]["macro_f1"] for r in results]
    summary = {
        "num_examples": int(len(y)),
        "feature_mode": args.feature_mode,
        "model_type": args.model_type,
        "label_counts": dict(Counter(labels_raw[idx] for idx in keep).most_common()),
        "seeds": [int(s) for s in args.seeds],
        "val_acc_mean": float(np.mean(val_acc)),
        "val_acc_std": float(np.std(val_acc)),
        "val_macro_f1_mean": float(np.mean(val_f1)),
        "val_macro_f1_std": float(np.std(val_f1)),
        "runs": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
