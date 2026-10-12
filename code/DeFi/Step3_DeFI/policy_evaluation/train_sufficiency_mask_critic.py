#!/usr/bin/env python3
"""Train a smoke Q critic and sufficiency mask for future-channel intervention.

The Q critic learns P(success | future, task) from rollout labels. The mask net
then learns a sparse channel mask M such that applying the same success delta
only through M scores higher than complement/random branches under Q.
"""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_trace(row: dict[str, Any], trace_dir: Path, idx: int) -> Path | None:
    raw = row.get("collection_trace_path") or row.get("target_proxy_trace_path") or row.get("trace_path") or row.get("trace")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
        candidate = trace_dir / path.name
        if candidate.exists():
            return candidate
    candidate = trace_dir / f"future_trace_{idx:04d}.npz"
    return candidate if candidate.exists() else None


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = data[key]
            elif "base_future" in data:
                arr = data["base_future"]
            elif "original_base_future" in data:
                arr = data["original_base_future"]
            else:
                return None
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    return arr if arr.ndim == 2 else None


def hashed_task(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def future_summary(f: np.ndarray) -> np.ndarray:
    return np.concatenate([f.mean(axis=0), f.std(axis=0), f[-1] - f[0]], axis=0).astype(np.float32)


def future_summary_torch(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False), f[:, -1] - f[:, 0]], dim=-1)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return 0.0 if denom <= 1e-8 else float(np.dot(af, bf) / denom)


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


class MaskNet(nn.Module):
    def __init__(self, input_dim: int, future_dim: int, hidden_dim: int, temperature: float):
        super().__init__()
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x) / max(self.temperature, 1e-6))


def metrics_from_q(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    risk = 1.0 - prob
    fail = (y < 0.5).astype(np.int64)
    pred_fail = (risk >= 0.5).astype(np.int64)
    tp = int(((pred_fail == 1) & (fail == 1)).sum())
    fp = int(((pred_fail == 1) & (fail == 0)).sum())
    fn = int(((pred_fail == 0) & (fail == 1)).sum())
    tn = int(((pred_fail == 0) & (fail == 0)).sum())
    pos = risk[fail == 1]
    neg = risk[fail == 0]
    auc = 0.0
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    return {
        "acc": float((pred_fail == fail).mean()),
        "precision_fail": tp / max(tp + fp, 1),
        "recall_fail": tp / max(tp + fn, 1),
        "f1_fail": 2 * tp / max(2 * tp + fp + fn, 1),
        "auc_fail_risk": auc,
        "mean_prob_success": float(prob.mean()),
        "mean_risk_success": float(risk[y > 0.5].mean()) if (y > 0.5).any() else 0.0,
        "mean_risk_fail": float(risk[y < 0.5].mean()) if (y < 0.5).any() else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_dataset(args):
    raw_rows = read_jsonl(Path(args.dataset_jsonl))
    trace_dir = Path(args.future_trace_dir)
    rows, futures, xs, ys = [], [], [], []
    for idx, row in enumerate(raw_rows):
        trace = resolve_trace(row, trace_dir, idx)
        if trace is None:
            continue
        future = load_future(trace, args.future_key)
        if future is None:
            continue
        task = str(row.get("task", "unknown"))
        sub = float(row.get("subtask_index", 0.0)) / 5.0
        x = np.concatenate([future_summary(future), hashed_task(task, args.task_dim), np.asarray([sub], dtype=np.float32)])
        rows.append(dict(row, _trace_path=str(trace)))
        futures.append(future)
        xs.append(x.astype(np.float32))
        ys.append(1.0 if bool(row.get("success", False)) else 0.0)
    if not rows:
        raise RuntimeError("no usable rows")
    return rows, np.stack(futures).astype(np.float32), np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def split_indices(rows, val_ratio: float, seed: int):
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(rows)})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val_seqs = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train, val = [], []
    for idx, row in enumerate(rows):
        (val if int(row.get("sequence_index", idx)) in val_seqs else train).append(idx)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def nearest_success_targets(rows, futures, train_pool, same_task: bool, k: int):
    pooled = futures.mean(axis=1)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    tasks = [str(row.get("task", "unknown")) for row in rows]
    success = [int(i) for i in train_pool if bool(rows[int(i)].get("success", False))]
    out = {}
    for i, row in enumerate(rows):
        candidates = [j for j in success if j != i and (not same_task or tasks[j] == tasks[i])]
        if not candidates:
            candidates = [j for j in success if j != i]
        if not candidates:
            continue
        sims = pooled[i][None, :] @ pooled[candidates].T
        order = np.argsort(-sims.reshape(-1))[: max(1, min(k, len(candidates)))]
        ids = [candidates[int(o)] for o in order]
        weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
        if float(weights.sum()) <= 1e-8:
            weights[:] = 1.0 / max(len(weights), 1)
        else:
            weights /= weights.sum()
        out[i] = np.tensordot(weights, futures[ids], axes=(0, 0)).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--q-steps", type=int, default=1000)
    parser.add_argument("--mask-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--same-task-target", action="store_true")
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--mask-temperature", type=float, default=0.5)
    parser.add_argument("--target-density", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda-nec", type=float, default=1.0)
    parser.add_argument("--lambda-rand", type=float, default=1.0)
    parser.add_argument("--lambda-sparse", type=float, default=0.5)
    parser.add_argument("--lambda-edit", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows, futures, x, y = build_dataset(args)
    context_feats = []
    for row in rows:
        task = str(row.get("task", "unknown"))
        sub = float(row.get("subtask_index", 0.0)) / 5.0
        context_feats.append(np.concatenate([hashed_task(task, args.task_dim), np.asarray([sub], dtype=np.float32)]))
    context_feats = np.stack(context_feats).astype(np.float32)
    train_idx, val_idx = split_indices(rows, args.val_ratio, args.seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(x[train_idx].std(axis=0, keepdims=True), 1e-6)
    xz = ((x - mean) / std).astype(np.float32)
    tx = torch.from_numpy(xz[train_idx]).to(device)
    ty = torch.from_numpy(y[train_idx]).to(device)
    vx = torch.from_numpy(xz[val_idx]).to(device)
    vy = torch.from_numpy(y[val_idx]).to(device)

    q = QCritic(x.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-4)
    succ = float(ty.sum().item())
    fail = float(len(ty) - succ)
    w_s = len(ty) / (2.0 * max(succ, 1.0))
    w_f = len(ty) / (2.0 * max(fail, 1.0))
    q_history = []
    best_q, best_q_state = None, None
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.q_steps + 1):
        ids = torch.randint(0, tx.shape[0], (args.batch_size,), device=device)
        xb, yb = tx[ids], ty[ids]
        logits = q(xb)
        weights = torch.where(yb > 0.5, torch.full_like(yb, w_s), torch.full_like(yb, w_f))
        loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * weights).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, args.q_steps // 10) == 0:
            q.eval()
            with torch.no_grad():
                rec = {"step": step, "loss": float(loss.item()), **metrics_from_q(q(vx), vy)}
            q.train()
            q_history.append(rec)
            score = rec["auc_fail_risk"] + rec["recall_fail"] - 0.5 * rec["fp"] / max(rec["fp"] + rec["tn"], 1)
            if best_q is None or score > best_q["score"]:
                best_q = dict(rec, score=float(score))
                best_q_state = {k: v.detach().cpu() for k, v in q.state_dict().items()}
            print(json.dumps({"q": rec}), flush=True)
    assert best_q_state is not None
    q.load_state_dict(best_q_state)
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)

    targets = nearest_success_targets(rows, futures, train_idx, args.same_task_target, args.target_k)
    fail_train = [int(i) for i in train_idx if y[int(i)] < 0.5 and int(i) in targets]
    if not fail_train:
        raise RuntimeError("no train failures with success targets")
    mask = MaskNet(x.shape[1], futures.shape[-1], args.hidden_dim, args.mask_temperature).to(device)
    opt_m = torch.optim.AdamW(mask.parameters(), lr=args.lr, weight_decay=1e-4)
    mask_history = []
    best_mask, best_mask_state = None, None

    future_mean_t = torch.from_numpy(mean).to(device)
    future_std_t = torch.from_numpy(std).to(device)

    context_feats_t = torch.from_numpy(context_feats).to(device)

    def q_features(f_batch: torch.Tensor, row_ids: list[int]) -> torch.Tensor:
        row_index = torch.as_tensor(row_ids, dtype=torch.long, device=device)
        arr = torch.cat([future_summary_torch(f_batch), context_feats_t[row_index]], dim=-1)
        return (arr - future_mean_t) / future_std_t

    for step in range(1, args.mask_steps + 1):
        row_ids = [int(rng.choice(fail_train)) for _ in range(args.batch_size)]
        xb = torch.from_numpy(xz[row_ids]).to(device)
        f_base = torch.from_numpy(futures[row_ids]).to(device)
        f_tgt = torch.from_numpy(np.stack([targets[i] for i in row_ids]).astype(np.float32)).to(device)
        delta = f_tgt - f_base
        m = mask(xb)
        density = m.mean()
        if args.target_density > 0:
            k = max(1, int(round(m.shape[-1] * args.target_density)))
            topk_ids = torch.topk(m.detach(), k=min(k, m.shape[-1]), dim=-1).indices
            hard = torch.zeros_like(m).scatter_(1, topk_ids, 1.0)
            m_eff = hard.detach() - m.detach() + m
        else:
            m_eff = m
        m3 = m_eff.unsqueeze(1)
        rand = torch.zeros_like(m_eff)
        k = max(1, int(round(m_eff.shape[-1] * max(args.target_density, 1.0 / m_eff.shape[-1]))))
        for bi in range(rand.shape[0]):
            ridx = torch.randperm(rand.shape[-1], device=device)[:k]
            rand[bi, ridx] = 1.0
        rand3 = rand.unsqueeze(1)
        f_suff = f_base + m3 * delta
        f_comp = f_base + (1.0 - m3) * delta
        f_rand = f_base + rand3 * delta
        q_s = torch.sigmoid(q(q_features(f_suff, row_ids)))
        q_c = torch.sigmoid(q(q_features(f_comp, row_ids)))
        q_r = torch.sigmoid(q(q_features(f_rand, row_ids)))
        suff_loss = -torch.log(q_s.clamp_min(1e-6)).mean()
        nec_loss = F.relu(args.margin + q_c - q_s).mean()
        rand_loss = F.relu(args.margin + q_r - q_s).mean()
        edit_loss = ((m3 * delta) ** 2).mean()
        sparse_loss = (density - float(args.target_density)).abs()
        loss = suff_loss + args.lambda_nec * nec_loss + args.lambda_rand * rand_loss + args.lambda_sparse * sparse_loss + args.lambda_edit * edit_loss
        opt_m.zero_grad(set_to_none=True)
        loss.backward()
        opt_m.step()
        if step == 1 or step % max(1, args.mask_steps // 10) == 0:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "suff_loss": float(suff_loss.item()),
                "nec_loss": float(nec_loss.item()),
                "rand_loss": float(rand_loss.item()),
                "sparse_loss": float(sparse_loss.item()),
                "edit_loss": float(edit_loss.item()),
                "q_suff": float(q_s.mean().item()),
                "q_comp": float(q_c.mean().item()),
                "q_rand": float(q_r.mean().item()),
                "suff_minus_comp": float((q_s - q_c).mean().item()),
                "suff_minus_rand": float((q_s - q_r).mean().item()),
                "mask_mean": float(m.mean().item()),
                "mask_eff_density": float(m_eff.mean().item()),
            }
            mask_history.append(rec)
            score = rec["suff_minus_comp"] + rec["suff_minus_rand"] - 0.1 * abs(rec["mask_eff_density"] - args.target_density)
            if best_mask is None or score > best_mask["score"]:
                best_mask = dict(rec, score=float(score))
                best_mask_state = {k: v.detach().cpu() for k, v in mask.state_dict().items()}
            print(json.dumps({"mask": rec}), flush=True)

    if best_mask_state is None:
        best_mask_state = {k: v.detach().cpu() for k, v in mask.state_dict().items()}
        best_mask = {"skipped": True, "reason": "mask_steps<=0"}
        mask_history.append(best_mask)
    assert best_mask_state is not None
    q_path = out_dir / "q_critic.pt"
    mask_path = out_dir / "sufficiency_mask_net.pt"
    torch.save(
        {
            "model_state": best_q_state,
            "input_dim": int(x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "future_key": args.future_key,
            "task_dim": int(args.task_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
        q_path,
    )
    torch.save(
        {
            "model_state": best_mask_state,
            "input_dim": int(x.shape[1]),
            "future_dim": int(futures.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "temperature": float(args.mask_temperature),
            "target_density": float(args.target_density),
            "future_key": args.future_key,
            "task_dim": int(args.task_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        },
        mask_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "num_rows": int(len(rows)),
        "success_count": int(y.sum()),
        "failure_count": int(len(y) - y.sum()),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "fail_train_with_targets": int(len(fail_train)),
        "task_counts": dict(Counter(str(r.get("task", "unknown")) for r in rows).most_common()),
        "best_q": best_q,
        "best_mask": best_mask,
        "q_history": q_history,
        "mask_history": mask_history,
        "q_ckpt": str(q_path),
        "mask_ckpt": str(mask_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
