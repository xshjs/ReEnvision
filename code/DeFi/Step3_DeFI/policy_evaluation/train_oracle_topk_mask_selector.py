#!/usr/bin/env python3
"""Explicit oracle top-k mask search and selector imitation.

For each failed future, greedily search sparse channel-group mask M* maximizing:
    S(M) + lambda_n * N(M) - lambda_s * C(M)

Then train a selector to imitate the multi-label oracle mask.
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
    return np.concatenate(
        [future_summary(future), hashed_task(str(row.get("task", "unknown")), task_dim), np.asarray([sub], dtype=np.float32)]
    ).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
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


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(rows)})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    tr, va = [], []
    for idx, row in enumerate(rows):
        (va if int(row.get("sequence_index", idx)) in val else tr).append(idx)
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)


def q_metrics(logits, labels):
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    fail = (y < 0.5).astype(np.int64)
    pred = ((1.0 - prob) >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (fail == 1)).sum())
    fp = int(((pred == 1) & (fail == 0)).sum())
    fn = int(((pred == 0) & (fail == 1)).sum())
    tn = int(((pred == 0) & (fail == 0)).sum())
    pos, neg = (1.0 - prob)[fail == 1], (1.0 - prob)[fail == 0]
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
    }


def nearest_success_targets(rows, futures, pool_ids, same_task: bool, same_subtask: bool, step_tolerance: int, k: int):
    pooled = futures.mean(axis=1)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    tasks = [str(r.get("task", "unknown")) for r in rows]
    subtasks = [int(r.get("subtask_index", -1)) for r in rows]
    steps = [int(r.get("steps", -1)) for r in rows]
    success = [int(i) for i in pool_ids if bool(rows[int(i)].get("success", False))]
    out = {}
    for i in range(len(rows)):
        base = [j for j in success if j != i]
        stages = [
            [
                j
                for j in base
                if (not same_task or tasks[j] == tasks[i])
                and (not same_subtask or subtasks[j] == subtasks[i])
                and (step_tolerance <= 0 or abs(steps[j] - steps[i]) <= step_tolerance)
            ],
            [j for j in base if (not same_task or tasks[j] == tasks[i]) and (not same_subtask or subtasks[j] == subtasks[i])],
            [j for j in base if (not same_task or tasks[j] == tasks[i])],
            base,
        ]
        candidates = next((x for x in stages if x), [])
        if not candidates:
            continue
        sims = pooled[i][None, :] @ pooled[candidates].T
        order = np.argsort(-sims.reshape(-1))[: max(1, min(k, len(candidates)))]
        ids = [candidates[int(o)] for o in order]
        weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        out[i] = np.tensordot(weights, futures[ids], axes=(0, 0)).astype(np.float32)
    return out


def channel_groups(dim: int, group_size: int):
    return [(s, min(s + group_size, dim)) for s in range(0, dim, group_size)]


def mask_from_ids(ids, groups, dim):
    m = np.zeros((dim,), dtype=np.float32)
    for gi in ids:
        s, e = groups[int(gi)]
        m[s:e] = 1.0
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--oracle-topk", type=int, default=4)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--same-task-target", action="store_true")
    parser.add_argument("--same-subtask-target", action="store_true")
    parser.add_argument("--target-step-tolerance", type=int, default=160)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--q-steps", type=int, default=3000)
    parser.add_argument("--selector-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lambda-necessity", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    raw = read_jsonl(Path(args.dataset_jsonl))
    rows, futures, feats, labels = [], [], [], []
    for idx, row in enumerate(raw):
        trace = resolve_trace(row, Path(args.future_trace_dir), idx)
        if trace is None:
            continue
        fut = load_future(trace, args.future_key)
        if fut is None:
            continue
        rows.append(row)
        futures.append(fut)
        feats.append(build_feature(fut, row, args.task_dim))
        labels.append(1.0 if bool(row.get("success", False)) else 0.0)
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
    w_s = len(ty) / (2 * max(succ, 1.0))
    w_f = len(ty) / (2 * max(fail, 1.0))
    best_q, best_q_state, q_hist = None, None, []
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
                rec = {"step": step, "loss": float(loss.item()), **q_metrics(q(vx).squeeze(-1), vy)}
            q.train()
            q_hist.append(rec)
            ftr = rec["fp"] / max(rec["fp"] + rec["tn"], 1)
            score = rec["auc_fail_risk"] + rec["f1_fail"] - 0.2 * ftr
            if best_q is None or score > best_q["score"]:
                best_q = dict(rec, score=float(score))
                best_q_state = {k: v.detach().cpu() for k, v in q.state_dict().items()}
            print(json.dumps({"q": rec}), flush=True)
    q.load_state_dict(best_q_state)
    q.eval()

    targets = nearest_success_targets(
        rows,
        futures_np,
        train_idx,
        args.same_task_target,
        args.same_subtask_target,
        args.target_step_tolerance,
        args.target_k,
    )
    groups = channel_groups(futures_np.shape[-1], args.group_size)
    failure_ids = [int(i) for i in range(len(rows)) if labels_np[i] < 0.5 and i in targets]
    oracle_rows = []
    for idx in failure_ids:
        row = rows[idx]
        base = futures_np[idx]
        delta = targets[idx] - base
        full = base + delta
        base_q = float(torch.sigmoid(q(torch.from_numpy(((build_feature(base, row, args.task_dim)[None, :] - mean) / std).astype(np.float32)).to(device))).item())
        full_q = float(torch.sigmoid(q(torch.from_numpy(((build_feature(full, row, args.task_dim)[None, :] - mean) / std).astype(np.float32)).to(device))).item())
        denom = max(abs(full_q - base_q), 1e-6)
        selected: list[int] = []
        remaining = set(range(len(groups)))
        trace = []
        for _ in range(max(1, int(args.oracle_topk))):
            best = None
            for gi in list(remaining):
                cand_ids = selected + [gi]
                m = mask_from_ids(cand_ids, groups, futures_np.shape[-1])
                fm = base + m[None, :] * delta
                fcomp = full - m[None, :] * delta
                fm_q = float(torch.sigmoid(q(torch.from_numpy(((build_feature(fm, row, args.task_dim)[None, :] - mean) / std).astype(np.float32)).to(device))).item())
                fc_q = float(torch.sigmoid(q(torch.from_numpy(((build_feature(fcomp, row, args.task_dim)[None, :] - mean) / std).astype(np.float32)).to(device))).item())
                suff = (fm_q - base_q) / denom
                nec = full_q - fc_q
                cost = len(cand_ids) / max(len(groups), 1)
                score = suff + float(args.lambda_necessity) * nec - float(args.lambda_cost) * cost
                item = (score, suff, nec, cost, fm_q, fc_q, gi)
                if best is None or item > best:
                    best = item
            assert best is not None
            selected.append(int(best[-1]))
            remaining.remove(int(best[-1]))
            trace.append(
                {
                    "score": float(best[0]),
                    "sufficiency": float(best[1]),
                    "necessity": float(best[2]),
                    "cost": float(best[3]),
                    "q_suff": float(best[4]),
                    "q_comp": float(best[5]),
                    "group_index": int(best[6]),
                    "groups_selected": [int(x) for x in selected],
                }
            )
        oracle_rows.append(
            {
                "row_index": int(idx),
                "sequence_index": row.get("sequence_index"),
                "subtask_index": row.get("subtask_index"),
                "task": row.get("task"),
                "base_q": float(base_q),
                "full_q": float(full_q),
                "oracle_group_indices": [int(x) for x in selected],
                "oracle_groups": [[int(groups[i][0]), int(groups[i][1])] for i in selected],
                "oracle_trace": trace,
                "final_score": float(trace[-1]["score"]),
                "final_sufficiency": float(trace[-1]["sufficiency"]),
                "final_necessity": float(trace[-1]["necessity"]),
            }
        )
    oracle_path = out / "oracle_topk_mask_rows.jsonl"
    with oracle_path.open("w", encoding="utf-8") as handle:
        for row in oracle_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not oracle_rows:
        raise RuntimeError("no oracle rows")

    oracle_ids = np.asarray([r["row_index"] for r in oracle_rows], dtype=np.int64)
    oracle_x = feats_z[oracle_ids]
    oracle_y = np.zeros((len(oracle_rows), len(groups)), dtype=np.float32)
    for i, row in enumerate(oracle_rows):
        oracle_y[i, row["oracle_group_indices"]] = 1.0
    seq_groups = np.asarray([str(rows[i].get("sequence_index", i)) for i in oracle_ids])
    unique = sorted(set(seq_groups.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(unique)
    val_seqs = set(unique[: max(1, int(round(len(unique) * args.val_ratio)))])
    tr = np.asarray([i for i, g in enumerate(seq_groups) if g not in val_seqs], dtype=np.int64)
    va = np.asarray([i for i, g in enumerate(seq_groups) if g in val_seqs], dtype=np.int64)
    selector = MLP(oracle_x.shape[1], len(groups), args.hidden_dim, args.dropout).to(device)
    opt_s = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=1e-4)
    sx = torch.from_numpy(oracle_x).to(device)
    sy = torch.from_numpy(oracle_y).to(device)
    pos_weight = torch.full((len(groups),), max((len(groups) - args.oracle_topk) / max(args.oracle_topk, 1), 1.0), device=device)
    best_sel, best_sel_state, sel_hist = None, None, []
    for step in range(1, args.selector_steps + 1):
        batch = torch.from_numpy(np.random.choice(tr, size=args.batch_size, replace=True)).to(device)
        logits = selector(sx[batch])
        loss = F.binary_cross_entropy_with_logits(logits, sy[batch], pos_weight=pos_weight)
        opt_s.zero_grad(set_to_none=True)
        loss.backward()
        opt_s.step()
        if step == 1 or step % max(1, args.selector_steps // 10) == 0:
            selector.eval()
            with torch.no_grad():
                logits = selector(sx[va])
                topk = torch.topk(logits, k=max(1, int(args.oracle_topk)), dim=-1).indices
                pred = torch.zeros_like(sy[va])
                pred.scatter_(1, topk, 1.0)
                inter = (pred * sy[va]).sum(dim=-1)
                prec = inter / max(int(args.oracle_topk), 1)
                rec = inter / sy[va].sum(dim=-1).clamp_min(1.0)
                exact = (pred == sy[va]).all(dim=-1).float()
                log = {
                    "step": step,
                    "loss": float(loss.item()),
                    "topk_precision": float(prec.mean().item()),
                    "topk_recall": float(rec.mean().item()),
                    "exact_match": float(exact.mean().item()),
                    "num_train_failures": int(len(tr)),
                    "num_val_failures": int(len(va)),
                }
            selector.train()
            sel_hist.append(log)
            score = log["topk_recall"] + log["topk_precision"] + 0.2 * log["exact_match"]
            if best_sel is None or score > best_sel["score"]:
                best_sel = dict(log, score=float(score))
                best_sel_state = {k: v.detach().cpu() for k, v in selector.state_dict().items()}
            print(json.dumps({"selector": log}), flush=True)

    q_path = out / "q_critic.pt"
    selector_path = out / "group_selector.pt"
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
            "topk_groups": int(args.oracle_topk),
            "training_objective": "oracle_topk_sufficiency_necessity_imitation",
        },
        selector_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "num_rows": int(len(rows)),
        "success_count": int(labels_np.sum()),
        "failure_count": int(len(labels_np) - labels_np.sum()),
        "oracle_rows": int(len(oracle_rows)),
        "oracle_topk": int(args.oracle_topk),
        "oracle_final_sufficiency_mean": float(np.mean([r["final_sufficiency"] for r in oracle_rows])),
        "oracle_final_necessity_mean": float(np.mean([r["final_necessity"] for r in oracle_rows])),
        "oracle_group_counts": dict(Counter(g for r in oracle_rows for g in r["oracle_group_indices"]).most_common(30)),
        "failure_task_counts": dict(Counter(str(rows[i].get("task", "unknown")) for i in oracle_ids).most_common()),
        "best_q": best_q,
        "best_selector": best_sel,
        "q_history": q_hist,
        "selector_history": sel_hist,
        "q_ckpt": str(q_path),
        "selector_ckpt": str(selector_path),
        "oracle_mask_rows": str(oracle_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
