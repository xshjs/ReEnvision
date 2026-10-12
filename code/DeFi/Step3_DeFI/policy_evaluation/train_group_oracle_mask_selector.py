#!/usr/bin/env python3
"""Train a Q critic, enumerate oracle channel groups, then train a selector.

This is a more stable alternative to dense end-to-end mask learning:
1. Train Q(F, task) -> P(success) from rollout rows.
2. For each failed future, find same-task successful target futures.
3. Enumerate channel groups and label the group that maximizes Q after patching.
4. Train a group selector to predict that oracle-best group.
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
            for k in [key, "base_future", "original_base_future"]:
                if k in data:
                    arr = np.asarray(data[k], dtype=np.float32)
                    return arr if arr.ndim == 2 else None
    except Exception:
        return None
    return None


def hashed_task(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def future_summary(f: np.ndarray) -> np.ndarray:
    return np.concatenate([f.mean(axis=0), f.std(axis=0), f[-1] - f[0]], axis=0).astype(np.float32)


def build_feature(future: np.ndarray, row: dict[str, Any], task_dim: int) -> np.ndarray:
    sub = float(row.get("subtask_index", 0.0)) / 5.0
    task = str(row.get("task", "unknown"))
    return np.concatenate([future_summary(future), hashed_task(task, task_dim), np.asarray([sub], dtype=np.float32)]).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return 0.0 if denom <= 1e-8 else float(np.dot(af, bf) / denom)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics_q(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    risk = 1.0 - prob
    fail = (y < 0.5).astype(np.int64)
    pred = (risk >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (fail == 1)).sum())
    fp = int(((pred == 1) & (fail == 0)).sum())
    fn = int(((pred == 0) & (fail == 1)).sum())
    tn = int(((pred == 0) & (fail == 0)).sum())
    pos = risk[fail == 1]
    neg = risk[fail == 0]
    auc = 0.0
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    return {
        "acc": float((pred == fail).mean()),
        "precision_fail": tp / max(tp + fp, 1),
        "recall_fail": tp / max(tp + fn, 1),
        "f1_fail": 2 * tp / max(2 * tp + fp + fn, 1),
        "auc_fail_risk": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "mean_prob_success": float(prob.mean()),
    }


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(rows)})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_ids, val_ids = [], []
    for idx, row in enumerate(rows):
        (val_ids if int(row.get("sequence_index", idx)) in val else train_ids).append(idx)
    return np.asarray(train_ids, dtype=np.int64), np.asarray(val_ids, dtype=np.int64)


def nearest_success_targets(rows, futures, pool_ids, same_task: bool, k: int) -> dict[int, np.ndarray]:
    pooled = futures.mean(axis=1)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    tasks = [str(row.get("task", "unknown")) for row in rows]
    success = [int(i) for i in pool_ids if bool(rows[int(i)].get("success", False))]
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
        weights = weights / max(float(weights.sum()), 1e-8)
        out[i] = np.tensordot(weights, futures[ids], axes=(0, 0)).astype(np.float32)
    return out


def channel_groups(dim: int, group_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + group_size, dim)) for s in range(0, dim, group_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--same-task-target", action="store_true")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--q-steps", type=int, default=2000)
    parser.add_argument("--selector-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    raw_rows = read_jsonl(Path(args.dataset_jsonl))
    rows, futures, feats, labels = [], [], [], []
    trace_dir = Path(args.future_trace_dir)
    for idx, row in enumerate(raw_rows):
        trace = resolve_trace(row, trace_dir, idx)
        if trace is None:
            continue
        future = load_future(trace, args.future_key)
        if future is None:
            continue
        rows.append(dict(row, _trace_path=str(trace)))
        futures.append(future)
        feats.append(build_feature(future, row, args.task_dim))
        labels.append(1.0 if bool(row.get("success", False)) else 0.0)
    if not rows:
        raise RuntimeError("no usable rows")
    futures_np = np.stack(futures).astype(np.float32)
    feats_np = np.stack(feats).astype(np.float32)
    labels_np = np.asarray(labels, dtype=np.float32)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    mean = feats_np[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(feats_np[train_idx].std(axis=0, keepdims=True), 1e-6)
    feats_z = ((feats_np - mean) / std).astype(np.float32)

    q = MLP(feats_z.shape[1], 1, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-4)
    tx = torch.from_numpy(feats_z[train_idx]).to(device)
    ty = torch.from_numpy(labels_np[train_idx]).to(device)
    vx = torch.from_numpy(feats_z[val_idx]).to(device)
    vy = torch.from_numpy(labels_np[val_idx]).to(device)
    succ = float(ty.sum().item())
    fail = float(len(ty) - succ)
    w_s = len(ty) / (2.0 * max(succ, 1.0))
    w_f = len(ty) / (2.0 * max(fail, 1.0))
    best_q, best_q_state, q_history = None, None, []
    for step in range(1, args.q_steps + 1):
        ids = torch.randint(0, tx.shape[0], (args.batch_size,), device=device)
        xb, yb = tx[ids], ty[ids]
        logits = q(xb).squeeze(-1)
        weights = torch.where(yb > 0.5, torch.full_like(yb, w_s), torch.full_like(yb, w_f))
        loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * weights).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, args.q_steps // 10) == 0:
            q.eval()
            with torch.no_grad():
                rec = {"step": step, "loss": float(loss.item()), **metrics_q(q(vx).squeeze(-1), vy)}
            q.train()
            q_history.append(rec)
            false_trigger_rate = rec["fp"] / max(rec["fp"] + rec["tn"], 1)
            score = rec["auc_fail_risk"] + rec["f1_fail"] - 0.2 * false_trigger_rate
            if best_q is None or score > best_q["score"]:
                best_q = dict(rec, score=float(score))
                best_q_state = {k: v.detach().cpu() for k, v in q.state_dict().items()}
            print(json.dumps({"q": rec}), flush=True)
    q.load_state_dict(best_q_state)
    q.eval()

    targets = nearest_success_targets(rows, futures_np, train_idx, args.same_task_target, args.target_k)
    groups = channel_groups(futures_np.shape[-1], args.group_size)
    oracle_rows = []
    oracle_path = out_dir / "oracle_group_rows.jsonl"
    with oracle_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            if labels_np[idx] > 0.5 or idx not in targets:
                continue
            base = futures_np[idx]
            target = targets[idx]
            delta = target - base
            base_feat = ((build_feature(base, row, args.task_dim)[None, :] - mean) / std).astype(np.float32)
            with torch.no_grad():
                base_q = float(torch.sigmoid(q(torch.from_numpy(base_feat).to(device))).item())
            cand = []
            for gi, (start, end) in enumerate(groups):
                patched = base.copy()
                patched[:, start:end] += delta[:, start:end]
                feat = ((build_feature(patched, row, args.task_dim)[None, :] - mean) / std).astype(np.float32)
                with torch.no_grad():
                    qv = float(torch.sigmoid(q(torch.from_numpy(feat).to(device))).item())
                cand.append((qv - base_q, qv, gi, start, end))
            cand.sort(reverse=True)
            best_gain, best_score, best_gi, best_start, best_end = cand[0]
            rec = {
                "row_index": int(idx),
                "sequence_index": row.get("sequence_index"),
                "subtask_index": row.get("subtask_index"),
                "task": row.get("task"),
                "base_q": base_q,
                "best_q": best_score,
                "best_gain": best_gain,
                "best_group_index": int(best_gi),
                "best_group": [int(best_start), int(best_end)],
                "top_groups": [
                    {"gain": float(g), "q": float(qv), "group_index": int(gi), "group": [int(s), int(e)]}
                    for g, qv, gi, s, e in cand[:10]
                ],
            }
            oracle_rows.append(rec)
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if not oracle_rows:
        raise RuntimeError("no failure rows with oracle groups")

    oracle_ids = np.asarray([r["row_index"] for r in oracle_rows], dtype=np.int64)
    oracle_y = np.asarray([r["best_group_index"] for r in oracle_rows], dtype=np.int64)
    oracle_x = feats_z[oracle_ids]
    oracle_groups = np.asarray([str(rows[i].get("sequence_index", i)) for i in oracle_ids])
    unique = sorted(set(oracle_groups.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(unique)
    val_groups = set(unique[: max(1, int(round(len(unique) * args.val_ratio)))])
    sel_train = np.asarray([i for i, g in enumerate(oracle_groups) if g not in val_groups], dtype=np.int64)
    sel_val = np.asarray([i for i, g in enumerate(oracle_groups) if g in val_groups], dtype=np.int64)
    selector = MLP(oracle_x.shape[1], len(groups), args.hidden_dim, args.dropout).to(device)
    opt_s = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=1e-4)
    sx = torch.from_numpy(oracle_x).to(device)
    sy = torch.from_numpy(oracle_y).to(device)
    best_sel, best_sel_state, selector_history = None, None, []
    for step in range(1, args.selector_steps + 1):
        batch = torch.from_numpy(np.random.choice(sel_train, size=args.batch_size, replace=True)).to(device)
        logits = selector(sx[batch])
        loss = F.cross_entropy(logits, sy[batch])
        opt_s.zero_grad(set_to_none=True)
        loss.backward()
        opt_s.step()
        if step == 1 or step % max(1, args.selector_steps // 10) == 0:
            selector.eval()
            with torch.no_grad():
                logits = selector(sx[sel_val])
                pred = logits.argmax(dim=-1)
                top5 = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "top1_acc": float((pred == sy[sel_val]).float().mean().item()),
                    "top5_acc": float((top5 == sy[sel_val].unsqueeze(-1)).any(dim=-1).float().mean().item()),
                    "num_train_failures": int(len(sel_train)),
                    "num_val_failures": int(len(sel_val)),
                }
            selector.train()
            selector_history.append(rec)
            score = rec["top1_acc"] + 0.2 * rec["top5_acc"]
            if best_sel is None or score > best_sel["score"]:
                best_sel = dict(rec, score=float(score))
                best_sel_state = {k: v.detach().cpu() for k, v in selector.state_dict().items()}
            print(json.dumps({"selector": rec}), flush=True)

    q_path = out_dir / "q_critic.pt"
    sel_path = out_dir / "group_selector.pt"
    torch.save(
        {
            "model_state": best_q_state,
            "input_dim": int(feats_z.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "task_dim": int(args.task_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "future_key": args.future_key,
        },
        q_path,
    )
    torch.save(
        {
            "model_state": best_sel_state,
            "input_dim": int(feats_z.shape[1]),
            "num_groups": int(len(groups)),
            "group_size": int(args.group_size),
            "groups": groups,
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "task_dim": int(args.task_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "future_key": args.future_key,
        },
        sel_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "num_rows": int(len(rows)),
        "success_count": int(labels_np.sum()),
        "failure_count": int(len(labels_np) - labels_np.sum()),
        "group_size": int(args.group_size),
        "num_groups": int(len(groups)),
        "oracle_failure_count": int(len(oracle_rows)),
        "oracle_positive_gain_rate": float(np.mean([r["best_gain"] > 0 for r in oracle_rows])),
        "oracle_mean_best_gain": float(np.mean([r["best_gain"] for r in oracle_rows])),
        "best_group_counts": dict(Counter(str(r["best_group"]) for r in oracle_rows).most_common(20)),
        "task_counts": dict(Counter(str(r.get("task", "unknown")) for r in rows).most_common()),
        "failure_task_counts": dict(Counter(str(rows[i].get("task", "unknown")) for i in oracle_ids).most_common()),
        "best_q": best_q,
        "best_selector": best_sel,
        "q_history": q_history,
        "selector_history": selector_history,
        "q_ckpt": str(q_path),
        "selector_ckpt": str(sel_path),
        "oracle_rows": str(oracle_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
