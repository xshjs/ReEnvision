from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        data = np.load(str(path))
    except Exception:
        return None
    if key not in data.files:
        return None
    future = np.asarray(data[key], dtype=np.float32)
    return future if future.ndim == 2 else None


def hashed_text_features(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / max(float(norm), 1e-6)


def summarize_future(future: np.ndarray) -> np.ndarray:
    mean = future.mean(axis=0)
    std = future.std(axis=0)
    first = future[0]
    last = future[-1]
    delta = last - first
    return np.concatenate([mean, std, delta], axis=0).astype(np.float32)


def load_dataset(path: Path, future_key: str, task_dim: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs, ys, meta = [], [], []
    for row in iter_jsonl(path):
        trace_path = row.get("trace_path")
        if not trace_path or not Path(str(trace_path)).exists():
            continue
        future = load_future(trace_path, future_key)
        if future is None:
            continue
        task = str(row.get("task", "unknown"))
        subtask_index = float(row.get("subtask_index", 0.0)) / 5.0
        future_sig = summarize_future(future)
        task_sig = hashed_text_features(task, task_dim)
        x = np.concatenate([future_sig, task_sig, np.asarray([subtask_index], dtype=np.float32)], axis=0)
        y = 1.0 if bool(row.get("success", False)) else 0.0
        xs.append(x)
        ys.append(y)
        meta.append(
            {
                "sequence_index": row.get("sequence_index"),
                "subtask_index": row.get("subtask_index"),
                "task": task,
                "success": bool(row.get("success", False)),
            }
        )
    if not xs:
        raise ValueError("no usable future rows")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), meta


def split_by_sequence(meta: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(meta)})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_seqs = set(seqs[:n_val])
    train, val = [], []
    for idx, row in enumerate(meta):
        seq = int(row.get("sequence_index", idx))
        (val if seq in val_seqs else train).append(idx)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class FutureSuccessVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def metrics(y: np.ndarray, prob_success: np.ndarray) -> dict[str, Any]:
    risk = 1.0 - prob_success
    y_fail = (y < 0.5).astype(np.int64)
    pred_fail = (risk >= 0.5).astype(np.int64)
    tp = int(((pred_fail == 1) & (y_fail == 1)).sum())
    fp = int(((pred_fail == 1) & (y_fail == 0)).sum())
    fn = int(((pred_fail == 0) & (y_fail == 1)).sum())
    tn = int(((pred_fail == 0) & (y_fail == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    pos = risk[y_fail == 1]
    neg = risk[y_fail == 0]
    auc = 0.0
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    order = np.argsort(-risk)
    top = {}
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(1, int(round(len(order) * frac)))
        idx = order[:k]
        top[f"top_{int(frac*100)}pct_fail_rate"] = float(y_fail[idx].mean())
        top[f"top_{int(frac*100)}pct_fail_count"] = int(y_fail[idx].sum())
        top[f"top_{int(frac*100)}pct_count"] = int(k)
    return {
        "fail_rate": float(y_fail.mean()),
        "acc_at_0_5": float((pred_fail == y_fail).mean()),
        "precision_fail_at_0_5": float(precision),
        "recall_fail_at_0_5": float(recall),
        "f1_fail_at_0_5": float(f1),
        "auc_fail_risk": float(auc),
        "mean_prob_success": float(prob_success.mean()),
        "mean_risk_success_rows": float(risk[y > 0.5].mean()) if (y > 0.5).any() else 0.0,
        "mean_risk_fail_rows": float(risk[y < 0.5].mean()) if (y < 0.5).any() else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a subtask-level future success verifier from base_future traces.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", choices=["base_future", "proposal_future", "gated_future", "final_future"], default="base_future")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    x, y, meta = load_dataset(args.dataset_jsonl, args.future_key, args.task_dim)
    train_idx, val_idx = split_by_sequence(meta, args.val_ratio, args.seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(x[train_idx].std(axis=0, keepdims=True), 1e-6)
    xz = (x - mean) / std

    model = FutureSuccessVerifier(x.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tx = torch.from_numpy(xz[train_idx]).to(device)
    ty = torch.from_numpy(y[train_idx]).to(device)
    vx = torch.from_numpy(xz[val_idx]).to(device)
    vy = y[val_idx]

    success_count = float(ty.sum().item())
    fail_count = float(len(ty) - success_count)
    success_weight = len(ty) / (2.0 * max(success_count, 1.0))
    fail_weight = len(ty) / (2.0 * max(fail_count, 1.0))
    history = []
    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, tx.shape[0], (max(1, args.batch_size),), device=device)
        xb = tx[idx]
        yb = ty[idx]
        logits = model(xb)
        weights = torch.where(
            yb > 0.5,
            torch.full_like(yb, float(success_weight)),
            torch.full_like(yb, float(fail_weight)),
        )
        loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * weights).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(vx)).detach().cpu().numpy()
            record = {"step": step, "loss": float(loss.detach().cpu()), **metrics(vy, prob)}
            history.append(record)
            score = record["auc_fail_risk"] + record["top_10pct_fail_rate"]
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(record), flush=True)

    assert best is not None and best_state is not None
    ckpt_path = args.output_dir / "future_success_verifier.pt"
    torch.save(
        {
            "model_state": best_state,
            "input_dim": int(x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "future_key": args.future_key,
            "task_dim": int(args.task_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
        ckpt_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_examples": int(len(y)),
        "success_count": int(y.sum()),
        "failure_count": int(len(y) - y.sum()),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "task_counts": dict(Counter(row["task"] for row in meta).most_common()),
        "best": best,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
