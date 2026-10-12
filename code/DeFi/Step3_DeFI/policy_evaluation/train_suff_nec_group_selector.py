#!/usr/bin/env python3
"""Train group selector with explicit sufficiency/necessity objectives.

The selector is optimized with a frozen Q critic:
  suff: Q(F + M*dF) should be high
  nec:  Q(F + (1-M)*dF) should be lower than suff
  rand: Q(F + Mrand*dF) should be lower than suff

This exports group_selector.pt in the same format consumed by
calvin_evaluate_with_memory_reflection.py.
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


def future_summary_torch(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False), f[:, -1] - f[:, 0]], dim=-1)


def build_feature(future: np.ndarray, row: dict[str, Any], task_dim: int) -> np.ndarray:
    sub = float(row.get("subtask_index", 0.0)) / 5.0
    task = str(row.get("task", "unknown"))
    return np.concatenate([future_summary(future), hashed_task(task, task_dim), np.asarray([sub], dtype=np.float32)]).astype(np.float32)


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


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(rows)})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_ids, val_ids = [], []
    for idx, row in enumerate(rows):
        (val_ids if int(row.get("sequence_index", idx)) in val else train_ids).append(idx)
    return np.asarray(train_ids, dtype=np.int64), np.asarray(val_ids, dtype=np.int64)


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
    pos, neg = risk[fail == 1], risk[fail == 0]
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


def nearest_success_targets(
    rows,
    futures,
    pool_ids,
    same_task: bool,
    same_subtask: bool,
    step_tolerance: int,
    min_similarity: float,
    strict_target: bool,
    k: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    pooled = futures.mean(axis=1)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    tasks = [str(row.get("task", "unknown")) for row in rows]
    subtasks = [int(row.get("subtask_index", -1)) for row in rows]
    steps = [int(row.get("steps", -1)) for row in rows]
    success = [int(i) for i in pool_ids if bool(rows[int(i)].get("success", False))]
    out, meta = {}, {}
    for i, row in enumerate(rows):
        stages = []
        base = [j for j in success if j != i]
        stages.append(
            (
                "same_task_subtask_step",
                [
                    j
                    for j in base
                    if (not same_task or tasks[j] == tasks[i])
                    and (not same_subtask or subtasks[j] == subtasks[i])
                    and (step_tolerance <= 0 or (steps[j] >= 0 and steps[i] >= 0 and abs(steps[j] - steps[i]) <= step_tolerance))
                ],
            )
        )
        stages.append(
            (
                "same_task_subtask",
                [
                    j
                    for j in base
                    if (not same_task or tasks[j] == tasks[i])
                    and (not same_subtask or subtasks[j] == subtasks[i])
                ],
            )
        )
        stages.append(("same_task", [j for j in base if (not same_task or tasks[j] == tasks[i])]))
        stages.append(("any_success", base))
        candidates, match_level = [], "none"
        for level, ids in stages:
            if ids:
                candidates, match_level = ids, level
                break
        if not candidates:
            continue
        sims = pooled[i][None, :] @ pooled[candidates].T
        order = np.argsort(-sims.reshape(-1))[: max(1, min(k, len(candidates)))]
        ids = [candidates[int(o)] for o in order]
        top_sim = float(sims.reshape(-1)[order[0]])
        if strict_target and match_level != "same_task_subtask_step":
            continue
        if min_similarity > -1.0 and top_sim < min_similarity:
            continue
        weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        out[i] = np.tensordot(weights, futures[ids], axes=(0, 0)).astype(np.float32)
        meta[i] = {
            "match_level": match_level,
            "top_similarity": top_sim,
            "candidate_count": int(len(candidates)),
            "target_ids": [int(x) for x in ids],
            "target_tasks": [tasks[x] for x in ids],
            "target_subtasks": [int(subtasks[x]) for x in ids],
            "target_steps": [int(steps[x]) for x in ids],
        }
    return out, meta


def channel_groups(dim: int, group_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + group_size, dim)) for s in range(0, dim, group_size)]


def group_probs_to_channel_mask(probs: torch.Tensor, groups: list[tuple[int, int]], dim: int) -> torch.Tensor:
    mask = torch.zeros((probs.shape[0], dim), device=probs.device, dtype=probs.dtype)
    for gi, (start, end) in enumerate(groups):
        mask[:, start:end] = probs[:, gi : gi + 1]
    return mask


def topk_group_mask(logits: torch.Tensor, groups: list[tuple[int, int]], dim: int, k: int) -> torch.Tensor:
    k = max(1, min(int(k), logits.shape[-1]))
    ids = torch.topk(logits, k=k, dim=-1).indices
    mask = torch.zeros((logits.shape[0], dim), device=logits.device, dtype=torch.float32)
    for gi, (start, end) in enumerate(groups):
        rows = (ids == gi).any(dim=-1)
        if bool(rows.any()):
            mask[rows, start:end] = 1.0
    return mask


def one_hot_group_mask(group_ids: torch.Tensor, groups: list[tuple[int, int]], dim: int) -> torch.Tensor:
    mask = torch.zeros((group_ids.shape[0], dim), device=group_ids.device, dtype=torch.float32)
    for gi, (start, end) in enumerate(groups):
        rows = group_ids == gi
        if bool(rows.any()):
            mask[rows, start:end] = 1.0
    return mask


def groups_to_channel_mask(group_ids: torch.Tensor, groups: list[tuple[int, int]], dim: int) -> torch.Tensor:
    mask = torch.zeros((group_ids.shape[0], dim), device=group_ids.device, dtype=torch.float32)
    for gi, (start, end) in enumerate(groups):
        rows = (group_ids == gi).any(dim=-1)
        if bool(rows.any()):
            mask[rows, start:end] = 1.0
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--same-task-target", action="store_true")
    parser.add_argument("--same-subtask-target", action="store_true")
    parser.add_argument("--target-step-tolerance", type=int, default=160)
    parser.add_argument("--target-min-similarity", type=float, default=-1.0)
    parser.add_argument("--strict-target-match", action="store_true")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--q-steps", type=int, default=2000)
    parser.add_argument("--selector-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--topk-groups", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda-nec", type=float, default=1.0)
    parser.add_argument("--lambda-rand", type=float, default=1.0)
    parser.add_argument("--lambda-full", type=float, default=1.0)
    parser.add_argument("--lambda-entropy", type=float, default=0.01)
    parser.add_argument("--lambda-edit", type=float, default=0.01)
    parser.add_argument("--lambda-oracle", type=float, default=1.0)
    parser.add_argument("--lambda-anti-comp", type=float, default=1.0)
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
    rows, futures, feats, labels, contexts = [], [], [], [], []
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
        sub = float(row.get("subtask_index", 0.0)) / 5.0
        contexts.append(np.concatenate([hashed_task(str(row.get("task", "unknown")), args.task_dim), np.asarray([sub], dtype=np.float32)]))
    futures_np = np.stack(futures).astype(np.float32)
    feats_np = np.stack(feats).astype(np.float32)
    labels_np = np.asarray(labels, dtype=np.float32)
    contexts_np = np.stack(contexts).astype(np.float32)
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
    succ, fail = float(ty.sum().item()), float(len(ty) - ty.sum().item())
    w_s, w_f = len(ty) / (2 * max(succ, 1.0)), len(ty) / (2 * max(fail, 1.0))
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
            ftr = rec["fp"] / max(rec["fp"] + rec["tn"], 1)
            score = rec["auc_fail_risk"] + rec["f1_fail"] - 0.2 * ftr
            if best_q is None or score > best_q["score"]:
                best_q = dict(rec, score=float(score))
                best_q_state = {k: v.detach().cpu() for k, v in q.state_dict().items()}
            print(json.dumps({"q": rec}), flush=True)
    q.load_state_dict(best_q_state)
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)

    targets, target_meta = nearest_success_targets(
        rows,
        futures_np,
        train_idx,
        args.same_task_target,
        args.same_subtask_target,
        int(args.target_step_tolerance),
        float(args.target_min_similarity),
        bool(args.strict_target_match),
        args.target_k,
    )
    q_path = out_dir / "q_critic.pt"
    if int(args.selector_steps) <= 0:
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
        summary = {
            "dataset_jsonl": str(args.dataset_jsonl),
            "future_trace_dir": str(args.future_trace_dir),
            "num_rows": int(len(rows)),
            "success_count": int(labels_np.sum()),
            "failure_count": int(len(labels_np) - labels_np.sum()),
            "best_q": best_q,
            "q_history": q_history,
            "q_ckpt": str(q_path),
            "selector_skipped": True,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return
    fail_train = [int(i) for i in train_idx if labels_np[int(i)] < 0.5 and int(i) in targets]
    fail_val = [int(i) for i in val_idx if labels_np[int(i)] < 0.5 and int(i) in targets]
    if not fail_train:
        raise RuntimeError("no train failures with targets")
    groups = channel_groups(futures_np.shape[-1], args.group_size)
    selector = MLP(feats_z.shape[1], len(groups), args.hidden_dim, args.dropout).to(device)
    opt_s = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=1e-4)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    context_t = torch.from_numpy(contexts_np).to(device)
    rng = np.random.default_rng(args.seed)

    def q_features(f_batch: torch.Tensor, row_ids: list[int]) -> torch.Tensor:
        rid = torch.as_tensor(row_ids, dtype=torch.long, device=device)
        raw = torch.cat([future_summary_torch(f_batch), context_t[rid]], dim=-1)
        return (raw - mean_t) / std_t

    def eval_selector(row_ids: list[int]) -> dict[str, float]:
        if not row_ids:
            return {}
        with torch.no_grad():
            xb = torch.from_numpy(feats_z[row_ids]).to(device)
            f_base = torch.from_numpy(futures_np[row_ids]).to(device)
            f_tgt = torch.from_numpy(np.stack([targets[i] for i in row_ids]).astype(np.float32)).to(device)
            delta = f_tgt - f_base
            logits = selector(xb)
            m = topk_group_mask(logits, groups, futures_np.shape[-1], args.topk_groups).unsqueeze(1)
            rand_score = torch.randn_like(logits)
            mr = topk_group_mask(rand_score, groups, futures_np.shape[-1], args.topk_groups).unsqueeze(1)
            full = f_base + delta
            fs = f_base + m * delta
            fc = full - m * delta
            fr = f_base + mr * delta
            q0 = torch.sigmoid(q(q_features(f_base, row_ids)))
            qf = torch.sigmoid(q(q_features(full, row_ids)))
            qs = torch.sigmoid(q(q_features(fs, row_ids)))
            qc = torch.sigmoid(q(q_features(fc, row_ids)))
            qr = torch.sigmoid(q(q_features(fr, row_ids)))
            denom = (qf - q0).abs().clamp_min(1e-6)
            suff_ratio = ((qs - q0) / denom).clamp(-10.0, 10.0)
            nec_drop = qf - qc
            return {
                "q_base": float(q0.mean().item()),
                "q_full": float(qf.mean().item()),
                "q_suff": float(qs.mean().item()),
                "q_comp": float(qc.mean().item()),
                "q_rand": float(qr.mean().item()),
                "suff_ratio": float(suff_ratio.mean().item()),
                "necessity_drop": float(nec_drop.mean().item()),
                "suff_vs_full_gap": float((qf - qs).mean().item()),
                "suff_minus_comp": float((qs - qc).mean().item()),
                "suff_minus_rand": float((qs - qr).mean().item()),
            }

    best_sel, best_sel_state, history = None, None, []
    for step in range(1, args.selector_steps + 1):
        row_ids = [int(rng.choice(fail_train)) for _ in range(args.batch_size)]
        xb = torch.from_numpy(feats_z[row_ids]).to(device)
        f_base = torch.from_numpy(futures_np[row_ids]).to(device)
        f_tgt = torch.from_numpy(np.stack([targets[i] for i in row_ids]).astype(np.float32)).to(device)
        delta = f_tgt - f_base
        logits = selector(xb)
        probs = torch.softmax(logits / max(args.temperature, 1e-6), dim=-1)
        if int(args.topk_groups) > 1:
            hard = topk_group_mask(logits, groups, futures_np.shape[-1], args.topk_groups)
            soft = group_probs_to_channel_mask(probs, groups, futures_np.shape[-1])
            m = (hard.detach() - soft.detach() + soft).unsqueeze(1)
        else:
            m = group_probs_to_channel_mask(probs, groups, futures_np.shape[-1]).unsqueeze(1)
        rand_score = torch.randn_like(logits)
        mr = topk_group_mask(rand_score, groups, futures_np.shape[-1], args.topk_groups).unsqueeze(1)
        full = f_base + delta
        fs = f_base + m * delta
        fc = full - m * delta
        fr = f_base + mr * delta
        q0 = torch.sigmoid(q(q_features(f_base, row_ids))).squeeze(-1)
        qf = torch.sigmoid(q(q_features(full, row_ids))).squeeze(-1)
        qs = torch.sigmoid(q(q_features(fs, row_ids))).squeeze(-1)
        qc = torch.sigmoid(q(q_features(fc, row_ids))).squeeze(-1)
        qr = torch.sigmoid(q(q_features(fr, row_ids))).squeeze(-1)

        # Q-oracle supervision: score every single group by whether it is
        # sufficient and necessary relative to the same full edit delta.
        oracle_scores = []
        for gi in range(len(groups)):
            gm = one_hot_group_mask(
                torch.full((len(row_ids),), gi, dtype=torch.long, device=device),
                groups,
                futures_np.shape[-1],
            ).unsqueeze(1)
            f_g = f_base + gm * delta
            f_not_g = full - gm * delta
            q_g = torch.sigmoid(q(q_features(f_g, row_ids))).squeeze(-1)
            q_not_g = torch.sigmoid(q(q_features(f_not_g, row_ids))).squeeze(-1)
            suff_gain = q_g - q0
            nec_gain = qf - q_not_g
            oracle_scores.append(suff_gain + args.lambda_nec * nec_gain)
        oracle_scores_t = torch.stack(oracle_scores, dim=-1)
        oracle_ids = torch.topk(
            oracle_scores_t,
            k=max(1, min(int(args.topk_groups), oracle_scores_t.shape[-1])),
            dim=-1,
        ).indices
        oracle_target = torch.zeros_like(logits)
        oracle_target.scatter_(1, oracle_ids, 1.0)

        denom = (qf - q0).abs().clamp_min(1e-6)
        suff_ratio = ((qs - q0) / denom).clamp(-10.0, 10.0)
        nec_drop = qf - qc
        full_loss = F.relu(args.margin + q0 - qf).mean()
        suff_loss = F.relu(1.0 - suff_ratio).mean()
        nec_loss = F.relu(args.margin - nec_drop).mean()
        anti_comp_loss = F.relu(args.margin + qc - qs).mean()
        rand_loss = F.relu(args.margin + qr - qs).mean()
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        edit_loss = ((m * delta) ** 2).mean()
        oracle_loss = F.binary_cross_entropy_with_logits(logits, oracle_target)
        loss = (
            args.lambda_full * full_loss
            + suff_loss
            + args.lambda_nec * nec_loss
            + args.lambda_anti_comp * anti_comp_loss
            + args.lambda_rand * rand_loss
            + args.lambda_oracle * oracle_loss
            - args.lambda_entropy * entropy
            + args.lambda_edit * edit_loss
        )
        opt_s.zero_grad(set_to_none=True)
        loss.backward()
        opt_s.step()
        if step == 1 or step % max(1, args.selector_steps // 10) == 0:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "full_loss": float(full_loss.item()),
                "suff_loss": float(suff_loss.item()),
                "nec_loss": float(nec_loss.item()),
                "anti_comp_loss": float(anti_comp_loss.item()),
                "rand_loss": float(rand_loss.item()),
                "oracle_loss": float(oracle_loss.item()),
                "entropy": float(entropy.item()),
                "edit_loss": float(edit_loss.item()),
                "q_base_train": float(q0.mean().item()),
                "q_full_train": float(qf.mean().item()),
                "q_suff_train": float(qs.mean().item()),
                "q_comp_train": float(qc.mean().item()),
                "q_rand_train": float(qr.mean().item()),
                "suff_ratio_train": float(suff_ratio.mean().item()),
                "necessity_drop_train": float(nec_drop.mean().item()),
                "oracle_overlap_train": float(((torch.sigmoid(logits) > 0.5).float() * oracle_target).sum(dim=-1).div(oracle_target.sum(dim=-1).clamp_min(1.0)).mean().item()),
                **{f"val_{k}": v for k, v in eval_selector(fail_val).items()},
                "num_train_failures": int(len(fail_train)),
                "num_val_failures": int(len(fail_val)),
            }
            history.append(rec)
            score = -abs(1.0 - rec.get("val_suff_ratio", 0.0)) + rec.get("val_necessity_drop", 0.0) + rec.get("val_suff_minus_rand", 0.0)
            if best_sel is None or score > best_sel["score"]:
                best_sel = dict(rec, score=float(score))
                best_sel_state = {k: v.detach().cpu() for k, v in selector.state_dict().items()}
            print(json.dumps({"selector": rec}), flush=True)

    selector_path = out_dir / "group_selector.pt"
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
            "training_objective": "sufficiency_necessity_random",
            "topk_groups": int(args.topk_groups),
            "uses_q_oracle_supervision": True,
        },
        selector_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "num_rows": int(len(rows)),
        "success_count": int(labels_np.sum()),
        "failure_count": int(len(labels_np) - labels_np.sum()),
        "group_size": int(args.group_size),
        "num_groups": int(len(groups)),
        "train_failures": int(len(fail_train)),
        "val_failures": int(len(fail_val)),
        "failure_task_counts": dict(Counter(str(rows[i].get("task", "unknown")) for i in fail_train + fail_val).most_common()),
        "target_match_counts": dict(Counter(target_meta[i]["match_level"] for i in fail_train + fail_val if i in target_meta).most_common()),
        "target_similarity_mean": float(np.mean([target_meta[i]["top_similarity"] for i in fail_train + fail_val if i in target_meta])) if any(i in target_meta for i in fail_train + fail_val) else 0.0,
        "target_similarity_min": float(np.min([target_meta[i]["top_similarity"] for i in fail_train + fail_val if i in target_meta])) if any(i in target_meta for i in fail_train + fail_val) else 0.0,
        "best_q": best_q,
        "best_selector": best_sel,
        "q_history": q_history,
        "selector_history": history,
        "q_ckpt": str(q_path),
        "selector_ckpt": str(selector_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
