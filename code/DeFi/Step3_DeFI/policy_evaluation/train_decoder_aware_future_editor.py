#!/usr/bin/env python3
"""Train a smoke decoder-aware future editor from action-sensitivity traces.

This does not backprop through the CALVIN decoder. Instead it uses the measured
action-sensitive repair direction as a teacher and a frozen Q critic as a
success proxy:

    Editor(F, task) -> dF
    loss = teacher_direction + Q_success + small_delta + preserve_success
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def read_collect_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[collect-row]" not in line:
                continue
            try:
                rows.append(json.loads(line.split("[collect-row]", 1)[1].strip()))
            except Exception:
                continue
    return rows


def hashed_task(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def future_summary_torch(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False), f[:, -1] - f[:, 0]], dim=-1)


def load_trace(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    path = row.get("trace") or row.get("trace_path") or row.get("collection_trace_path")
    if not path or not Path(path).exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    if "base_future" in data:
        base = np.asarray(data["base_future"], dtype=np.float32)
    elif "original_base_future" in data:
        base = np.asarray(data["original_base_future"], dtype=np.float32)
    else:
        return None
    if "edit_direction" in data:
        direction = np.asarray(data["edit_direction"], dtype=np.float32)
    elif "final_future" in data:
        direction = np.asarray(data["final_future"], dtype=np.float32) - base
    else:
        return None
    if base.ndim != 2 or direction.shape != base.shape:
        return None
    return base, direction


class QCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
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


class TokenEditor(nn.Module):
    def __init__(self, future_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.future_proj = nn.Linear(future_dim + 1, hidden_dim)
        self.task_proj = nn.Linear(task_dim + 1, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, future: torch.Tensor, task_cond: torch.Tensor) -> torch.Tensor:
        b, t, _ = future.shape
        pos = torch.linspace(0.0, 1.0, t, dtype=future.dtype, device=future.device)[None, :, None].expand(b, t, 1)
        h = self.future_proj(torch.cat([future, pos], dim=-1)) + self.task_proj(task_cond)[:, None, :]
        return self.net(h)


def build_q_features(future: torch.Tensor, task_cond: torch.Tensor, q_mean: torch.Tensor, q_std: torch.Tensor) -> torch.Tensor:
    # task_cond stores [task_hash, subtask_scalar]; Q was trained with the same layout.
    x = torch.cat([future_summary_torch(future), task_cond], dim=-1)
    return (x - q_mean) / q_std


def split_indices(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(r.get("seq", r.get("sequence_index", i))) for i, r in enumerate(rows)})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val_seqs = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train, val = [], []
    for i, r in enumerate(rows):
        (val if int(r.get("seq", r.get("sequence_index", i))) in val_seqs else train).append(i)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-log", required=True)
    parser.add_argument("--q-critic-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-action-rel", type=float, default=0.05)
    parser.add_argument("--max-teacher-norm", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-q", type=float, default=0.5)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument("--lambda-small", type=float, default=0.01)
    parser.add_argument("--lambda-preserve-success", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    raw_rows = read_collect_rows(Path(args.sensitivity_log))
    rows, bases, teachers, task_conds, labels, weights = [], [], [], [], [], []
    for r in raw_rows:
        loaded = load_trace(r)
        if loaded is None:
            continue
        base, direction = loaded
        rel = float(r.get("action_delta_rel_eps_1") or 0.0)
        norm = float(np.linalg.norm(direction.reshape(-1)))
        if norm <= 1e-8:
            weight = 0.0
        else:
            weight = max(rel - args.min_action_rel, 0.0) / max(args.min_action_rel, 1e-6)
            weight = min(weight, 3.0)
            if norm > args.max_teacher_norm:
                direction = direction * (args.max_teacher_norm / norm)
        sub = float(r.get("sub", r.get("subtask_index", 0.0))) / 5.0
        task_dim = 128
        rows.append(r)
        bases.append(base)
        teachers.append(direction)
        task_conds.append(np.concatenate([hashed_task(str(r.get("task", "unknown")), task_dim), np.asarray([sub], dtype=np.float32)]))
        labels.append(1.0 if bool(r.get("success")) else 0.0)
        weights.append(float(weight))
    if not rows:
        raise RuntimeError("no usable sensitivity rows")

    bases_np = np.stack(bases).astype(np.float32)
    teachers_np = np.stack(teachers).astype(np.float32)
    cond_np = np.stack(task_conds).astype(np.float32)
    y_np = np.asarray(labels, dtype=np.float32)
    w_np = np.asarray(weights, dtype=np.float32)
    train_idx, val_idx = split_indices(rows, args.val_ratio, args.seed)

    q_ckpt = torch.load(args.q_critic_ckpt, map_location="cpu")
    q = QCritic(int(q_ckpt["input_dim"]), int(q_ckpt.get("hidden_dim", 512)), float(q_ckpt.get("dropout", 0.05))).to(device)
    q.load_state_dict(q_ckpt["model_state"])
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)
    q_mean = torch.from_numpy(np.asarray(q_ckpt["mean"], dtype=np.float32)).to(device)
    q_std = torch.from_numpy(np.asarray(q_ckpt["std"], dtype=np.float32)).to(device)

    editor = TokenEditor(bases_np.shape[-1], cond_np.shape[-1] - 1, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(editor.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch(ids):
        return (
            torch.from_numpy(bases_np[ids]).to(device),
            torch.from_numpy(teachers_np[ids]).to(device),
            torch.from_numpy(cond_np[ids]).to(device),
            torch.from_numpy(y_np[ids]).to(device),
            torch.from_numpy(w_np[ids]).to(device),
        )

    history = []
    best = None
    best_state = None
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        ids = rng.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=len(train_idx) < args.batch_size)
        base, teacher, cond, label, weight = batch(ids)
        pred = editor(base, cond)
        final = base + pred
        q_prob = torch.sigmoid(q(build_q_features(final, cond, q_mean, q_std)))
        teacher_cos = F.cosine_similarity(pred.reshape(pred.shape[0], -1), teacher.reshape(teacher.shape[0], -1), dim=-1)
        teacher_mse = ((pred - teacher) ** 2).mean(dim=(1, 2))
        teacher_loss = ((1.0 - teacher_cos) * weight + 0.05 * teacher_mse * weight).mean()
        q_loss = F.binary_cross_entropy(q_prob, torch.ones_like(q_prob))
        small_loss = torch.norm(pred.reshape(pred.shape[0], -1), dim=-1).mean()
        preserve_loss = (torch.norm(pred.reshape(pred.shape[0], -1), dim=-1) * label).mean()
        loss = args.lambda_teacher * teacher_loss + args.lambda_q * q_loss + args.lambda_small * small_loss + args.lambda_preserve_success * preserve_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            with torch.no_grad():
                vb, vt, vc, vy, vw = batch(val_idx)
                vp = editor(vb, vc)
                vf = vb + vp
                vq_base = torch.sigmoid(q(build_q_features(vb, vc, q_mean, q_std)))
                vq_final = torch.sigmoid(q(build_q_features(vf, vc, q_mean, q_std)))
                vcos = F.cosine_similarity(vp.reshape(vp.shape[0], -1), vt.reshape(vt.shape[0], -1), dim=-1)
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "teacher_loss": float(teacher_loss.item()),
                    "q_loss": float(q_loss.item()),
                    "small_loss": float(small_loss.item()),
                    "preserve_loss": float(preserve_loss.item()),
                    "val_q_base": float(vq_base.mean().item()),
                    "val_q_final": float(vq_final.mean().item()),
                    "val_q_gain": float((vq_final - vq_base).mean().item()),
                    "val_direction_cos": float(vcos.mean().item()),
                    "val_weighted_direction_cos": float(((vcos * vw).sum() / torch.clamp(vw.sum(), min=1e-6)).item()),
                    "val_edit_norm": float(torch.norm(vp.reshape(vp.shape[0], -1), dim=-1).mean().item()),
                    "num_rows": len(rows),
                    "num_train": int(len(train_idx)),
                    "num_val": int(len(val_idx)),
                    "mean_teacher_weight": float(w_np.mean()),
                }
                history.append(rec)
                score = rec["val_q_gain"] + 0.1 * rec["val_weighted_direction_cos"] - 0.001 * rec["val_edit_norm"]
                if best is None or score > best["score"]:
                    best = dict(rec, score=float(score))
                    best_state = {k: v.detach().cpu() for k, v in editor.state_dict().items()}
                print(json.dumps(rec), flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "decoder_aware_future_editor.pt"
    torch.save(
        {
            "model_state": best_state,
            "future_dim": int(bases_np.shape[-1]),
            "task_cond_dim": int(cond_np.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "q_critic_ckpt": str(args.q_critic_ckpt),
            "best": best,
            "history": history,
        },
        ckpt_path,
    )
    with (out / "summary.json").open("w") as f:
        json.dump({"best": best, "checkpoint_path": str(ckpt_path), "history": history}, f, indent=2)
    print(json.dumps({"best": best, "checkpoint_path": str(ckpt_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
