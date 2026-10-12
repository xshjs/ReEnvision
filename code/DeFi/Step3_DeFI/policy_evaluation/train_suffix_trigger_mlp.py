from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hashed_task_vec(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    digest = hashlib.sha256(str(task).encode("utf-8")).digest()
    for i, b in enumerate(digest):
        vec[(b + i * 17) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-6)


class SuffixTriggerMLP(nn.Module):
    """Conservative suffix trigger calibrated from rollout decisions."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.05):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


RUNTIME_SCALARS = [
    "residual_norm",
    "compat",
    "success_prob",
    "current_to_target_norm",
    "shift_norm_vs_slow",
    "committed_scalar",
    "learned_gate",
    "joint_gate",
    "joint_error",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def build_suffix_trigger_feature(
    *,
    task: str,
    subtask_index: int,
    residual_norm: float,
    compat: float,
    success_prob: float,
    current_to_target_norm: float = 0.0,
    shift_norm_vs_slow: float = 0.0,
    committed_scalar: float = 0.0,
    learned_gate: float = 0.0,
    joint_gate: float = 0.0,
    joint_error: float = 0.0,
    task_dim: int = 32,
) -> np.ndarray:
    scalars = np.asarray(
        [
            np.log1p(max(_safe_float(residual_norm), 0.0)),
            np.clip(_safe_float(compat), -1.0, 1.0),
            np.clip(_safe_float(success_prob), 0.0, 1.0),
            np.log1p(max(_safe_float(current_to_target_norm), 0.0)),
            np.log1p(max(_safe_float(shift_norm_vs_slow), 0.0)),
            np.tanh(_safe_float(committed_scalar)),
            np.clip(_safe_float(learned_gate), 0.0, 1.0),
            np.clip(_safe_float(joint_gate), 0.0, 1.0),
            np.log1p(max(_safe_float(joint_error), 0.0)),
            float(subtask_index) / 5.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([scalars, hashed_task_vec(task, task_dim)], axis=0).astype(np.float32)


def load_rows(paths: list[Path], task_dim: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                decision = row.get("dynamic_coupling_chunk_decision")
                if decision not in {"keep", "repair_suffix"}:
                    continue
                xs.append(
                    build_suffix_trigger_feature(
                        task=str(row.get("task", "unknown")),
                        subtask_index=int(row.get("subtask_index", 0) or 0),
                        residual_norm=_safe_float(row.get("dynamic_coupling_chunk_residual_norm")),
                        compat=_safe_float(row.get("dynamic_coupling_chunk_compat")),
                        success_prob=_safe_float(row.get("dynamic_coupling_chunk_success_prob")),
                        current_to_target_norm=_safe_float(row.get("dynamic_coupling_current_to_target_norm")),
                        shift_norm_vs_slow=_safe_float(row.get("dynamic_coupling_shift_norm_vs_slow")),
                        committed_scalar=_safe_float(row.get("dynamic_coupling_committed_scalar")),
                        learned_gate=_safe_float(row.get("joint_execution_learned_gate")),
                        joint_gate=_safe_float(row.get("joint_execution_gate")),
                        joint_error=_safe_float(row.get("joint_execution_error")),
                        task_dim=task_dim,
                    )
                )
                ys.append(1.0 if decision == "repair_suffix" else 0.0)
    if not xs:
        raise ValueError("no keep/repair_suffix rows found")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def balanced_indices(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.where(y > 0.5)[0]
    neg = np.where(y <= 0.5)[0]
    if len(pos) == 0 or len(neg) == 0:
        ids = np.arange(len(y))
        rng.shuffle(ids)
        return ids
    n = max(len(pos), len(neg))
    out = np.concatenate(
        [
            rng.choice(pos, size=n, replace=len(pos) < n),
            rng.choice(neg, size=n, replace=len(neg) < n),
        ]
    )
    rng.shuffle(out)
    return out


def eval_model(model: SuffixTriggerMLP, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        logits = model(xt)
        prob = logits.sigmoid()
        pred = prob >= 0.5
        truth = yt > 0.5
        tp = int((pred & truth).sum().item())
        fp = int((pred & ~truth).sum().item())
        fn = int((~pred & truth).sum().item())
        tn = int((~pred & ~truth).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        return {
            "loss": float(F.binary_cross_entropy_with_logits(logits, yt).item()),
            "acc": float((pred == truth).float().mean().item()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "trigger_rate": float(pred.float().mean().item()),
            "target_rate": float(truth.float().mean().item()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conservative suffix trigger from rollout decisions.")
    parser.add_argument("--rows-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y = load_rows(args.rows_jsonl, task_dim=args.task_dim)
    rng = np.random.default_rng(args.seed)
    seq = np.arange(len(y))
    rng.shuffle(seq)
    n_val = max(1, int(len(seq) * args.val_ratio))
    val_ids = seq[:n_val]
    train_pool = seq[n_val:]
    train_bal = train_pool[balanced_indices(y[train_pool], args.seed)]
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model = SuffixTriggerMLP(x.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best: dict[str, Any] | None = None
    best_state = None
    log_path = args.output_dir / "train_log.jsonl"

    with log_path.open("w") as log_f:
        for step in range(1, args.steps + 1):
            ids = np.random.choice(train_bal, size=args.batch_size, replace=len(train_bal) < args.batch_size)
            xt = torch.from_numpy(x[ids]).to(device)
            yt = torch.from_numpy(y[ids]).to(device)
            model.train()
            logits = model(xt)
            loss = F.binary_cross_entropy_with_logits(logits, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % 200 == 0 or step == args.steps:
                metrics = eval_model(model, x[val_ids], y[val_ids], device)
                metrics.update({"step": step, "train_loss": float(loss.item())})
                log_f.write(json.dumps(metrics) + "\n")
                log_f.flush()
                if best is None or metrics["f1"] > best["f1"] or (metrics["f1"] == best["f1"] and metrics["trigger_rate"] < best["trigger_rate"]):
                    best = dict(metrics)
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = args.output_dir / "suffix_trigger_mlp.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": int(x.shape[1]),
            "task_dim": int(args.task_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "feature_names": ["log_residual", "compat", "success_prob", "log_current_to_target", "log_shift", "committed", "learned_gate", "joint_gate", "log_joint_error", "subtask_norm", "task_hash"],
            "best": best,
        },
        ckpt_path,
    )
    summary = {
        "rows_jsonl": [str(p) for p in args.rows_jsonl],
        "num_rows": int(len(y)),
        "num_positive": int((y > 0.5).sum()),
        "num_negative": int((y <= 0.5).sum()),
        "num_train": int(len(train_bal)),
        "num_val": int(len(val_ids)),
        "steps": int(args.steps),
        "best": best,
        "output_ckpt": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
