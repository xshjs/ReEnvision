#!/usr/bin/env python3
"""Train a masked future editor from existing rollout future traces.

Unlike direct neighbor patching, this learns:
    delta_edit = Editor(F, task, stage, M)
    F' = F + M * delta_edit

The nearest success-failure delta is only a weak teacher hint. The main losses
use a frozen Q critic to enforce full-relative sufficiency, necessity, random
control, small edits, and preserving unmasked regions.
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


def future_summary_np(f: np.ndarray) -> np.ndarray:
    return np.concatenate([f.mean(axis=0), f.std(axis=0), f[-1] - f[0]], axis=0).astype(np.float32)


def future_summary_torch(f: torch.Tensor) -> torch.Tensor:
    return torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False), f[:, -1] - f[:, 0]], dim=-1)


def build_feature(future: np.ndarray, row: dict[str, Any], task_dim: int) -> np.ndarray:
    sub = float(row.get("subtask_index", 0.0)) / 5.0
    return np.concatenate(
        [future_summary_np(future), hashed_task(str(row.get("task", "unknown")), task_dim), np.asarray([sub], dtype=np.float32)]
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


class TokenEditor(nn.Module):
    def __init__(self, future_dim: int, cond_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.future_proj = nn.Linear(future_dim * 2 + 1, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, future: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # future/mask: [B,T,D], cond: [B,C]
        t = future.shape[1]
        pos = torch.linspace(0.0, 1.0, t, device=future.device, dtype=future.dtype)[None, :, None].expand(future.shape[0], t, 1)
        h = self.future_proj(torch.cat([future, mask, pos], dim=-1)) + self.cond_proj(cond)[:, None, :]
        return self.net(h)


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(row.get("sequence_index", idx)) for idx, row in enumerate(rows)})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train, val_ids = [], []
    for idx, row in enumerate(rows):
        (val_ids if int(row.get("sequence_index", idx)) in val else train).append(idx)
    return np.asarray(train, dtype=np.int64), np.asarray(val_ids, dtype=np.int64)


def nearest_success_targets(rows, futures, pool_ids, same_subtask: bool, step_tolerance: int, k: int):
    pooled = futures.mean(axis=1)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    tasks = [str(r.get("task", "unknown")) for r in rows]
    subtasks = [int(r.get("subtask_index", -1)) for r in rows]
    steps = [int(r.get("steps", -1)) for r in rows]
    success = [int(i) for i in pool_ids if bool(rows[int(i)].get("success", False))]
    out = {}
    for i, row in enumerate(rows):
        base = [j for j in success if j != i]
        stages = [
            [
                j
                for j in base
                if tasks[j] == tasks[i]
                and (not same_subtask or subtasks[j] == subtasks[i])
                and (step_tolerance <= 0 or abs(steps[j] - steps[i]) <= step_tolerance)
            ],
            [j for j in base if tasks[j] == tasks[i] and (not same_subtask or subtasks[j] == subtasks[i])],
            [j for j in base if tasks[j] == tasks[i]],
            base,
        ]
        candidates = next((x for x in stages if x), [])
        if not candidates:
            continue
        sims = pooled[i][None, :] @ pooled[candidates].T
        order = np.argsort(-sims.reshape(-1))[: max(1, min(k, len(candidates)))]
        ids = [candidates[int(o)] for o in order]
        w = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
        w = w / max(float(w.sum()), 1e-8)
        out[i] = np.tensordot(w, futures[ids], axes=(0, 0)).astype(np.float32)
    return out


def channel_groups(dim: int, group_size: int):
    return [(s, min(s + group_size, dim)) for s in range(0, dim, group_size)]


def topk_mask_from_selector(selector, x, groups, dim, topk):
    logits = selector(x)
    ids = torch.topk(logits, k=max(1, min(int(topk), logits.shape[-1])), dim=-1).indices
    mask = torch.zeros((x.shape[0], dim), device=x.device, dtype=torch.float32)
    for gi, (s, e) in enumerate(groups):
        rows = (ids == gi).any(dim=-1)
        if bool(rows.any()):
            mask[rows, s:e] = 1.0
    return mask, logits


def load_q(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    q = MLP(int(ckpt["input_dim"]), 1, int(ckpt.get("hidden_dim", 512)), float(ckpt.get("dropout", 0.1))).to(device)
    q.load_state_dict(ckpt["model_state"], strict=True)
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)
    return q, ckpt


def load_selector(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    selector = MLP(int(ckpt["input_dim"]), int(ckpt["num_groups"]), int(ckpt.get("hidden_dim", 512)), float(ckpt.get("dropout", 0.1))).to(device)
    selector.load_state_dict(ckpt["model_state"], strict=True)
    selector.eval()
    for p in selector.parameters():
        p.requires_grad_(False)
    return selector, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q-critic-ckpt", required=True)
    parser.add_argument("--q-critic-ckpts", nargs="*", default=[])
    parser.add_argument("--q-ensemble-mode", choices=["mean", "min"], default="mean")
    parser.add_argument("--selector-ckpt", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--topk-groups", type=int, default=4)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--same-subtask-target", action="store_true")
    parser.add_argument("--target-step-tolerance", type=int, default=160)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda-teacher", type=float, default=0.2)
    parser.add_argument("--lambda-nec", type=float, default=1.0)
    parser.add_argument("--lambda-rand", type=float, default=1.0)
    parser.add_argument("--lambda-small", type=float, default=0.02)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows_raw = read_jsonl(Path(args.dataset_jsonl))
    rows, futures, feats, labels, contexts = [], [], [], [], []
    trace_dir = Path(args.future_trace_dir)
    for idx, row in enumerate(rows_raw):
        trace = resolve_trace(row, trace_dir, idx)
        if trace is None:
            continue
        fut = load_future(trace, args.future_key)
        if fut is None:
            continue
        rows.append(row)
        futures.append(fut)
        feats.append(build_feature(fut, row, args.task_dim))
        labels.append(1.0 if bool(row.get("success", False)) else 0.0)
        sub = float(row.get("subtask_index", 0.0)) / 5.0
        contexts.append(np.concatenate([hashed_task(str(row.get("task", "unknown")), args.task_dim), np.asarray([sub], dtype=np.float32)]))
    futures_np = np.stack(futures).astype(np.float32)
    feats_np = np.stack(feats).astype(np.float32)
    labels_np = np.asarray(labels, dtype=np.float32)
    contexts_np = np.stack(contexts).astype(np.float32)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    q_paths = [str(args.q_critic_ckpt)] + [str(p) for p in args.q_critic_ckpts]
    q_bundles = [load_q(Path(p), device) for p in q_paths]
    q, q_ckpt = q_bundles[0]
    selector, sel_ckpt = load_selector(Path(args.selector_ckpt), device)
    groups = [tuple(map(int, g)) for g in sel_ckpt["groups"]]
    targets = nearest_success_targets(rows, futures_np, train_idx, args.same_subtask_target, args.target_step_tolerance, args.target_k)
    fail_train = [int(i) for i in train_idx if labels_np[int(i)] < 0.5 and i in targets]
    fail_val = [int(i) for i in val_idx if labels_np[int(i)] < 0.5 and i in targets]
    if not fail_train:
        raise RuntimeError("no train failures with matched targets")

    feats_z = ((feats_np - np.asarray(q_ckpt["mean"], dtype=np.float32)) / np.maximum(np.asarray(q_ckpt["std"], dtype=np.float32), 1e-6)).astype(np.float32)
    context_t = torch.from_numpy(contexts_np).to(device)

    def q_features_for_ckpt(f_batch, row_ids, ckpt):
        rid = torch.as_tensor(row_ids, dtype=torch.long, device=device)
        task_dim = int(ckpt.get("task_dim", args.task_dim))
        if task_dim == args.task_dim:
            ctx = context_t[rid]
        else:
            ctx_np = []
            for row_id in row_ids:
                sub = float(rows[row_id].get("subtask_index", 0.0)) / 5.0
                ctx_np.append(np.concatenate([hashed_task(str(rows[row_id].get("task", "unknown")), task_dim), np.asarray([sub], dtype=np.float32)]))
            ctx = torch.from_numpy(np.stack(ctx_np).astype(np.float32)).to(device)
        mean = torch.from_numpy(np.asarray(ckpt["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(ckpt["std"], dtype=np.float32)).to(device)
        raw = torch.cat([future_summary_torch(f_batch), ctx], dim=-1)
        return (raw - mean) / std

    def q_score(f_batch, row_ids):
        scores = []
        for model, ckpt in q_bundles:
            scores.append(torch.sigmoid(model(q_features_for_ckpt(f_batch, row_ids, ckpt))).squeeze(-1))
        stacked = torch.stack(scores, dim=0)
        if str(args.q_ensemble_mode) == "min":
            return stacked.min(dim=0).values
        return stacked.mean(dim=0)

    editor = TokenEditor(futures_np.shape[-1], contexts_np.shape[-1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(editor.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history, best, best_state = [], None, None

    def eval_rows(row_ids):
        if not row_ids:
            return {}
        with torch.no_grad():
            xb = torch.from_numpy(feats_z[row_ids]).to(device)
            fb = torch.from_numpy(futures_np[row_ids]).to(device)
            ft = torch.from_numpy(np.stack([targets[i] for i in row_ids]).astype(np.float32)).to(device)
            ctx = context_t[torch.as_tensor(row_ids, device=device)]
            teacher = ft - fb
            m2, _ = topk_mask_from_selector(selector, xb, groups, futures_np.shape[-1], args.topk_groups)
            m = m2.unsqueeze(1)
            de = editor(fb, m.expand_as(fb), ctx) * m
            full = fb + teacher
            fs = fb + de
            fc = full - de
            rand_score = torch.randn((len(row_ids), len(groups)), device=device)
            mr = topk_mask_from_selector(lambda x: rand_score, xb, groups, futures_np.shape[-1], args.topk_groups)[0].unsqueeze(1)
            fr = fb + mr * de
            q0 = q_score(fb, row_ids)
            qf = q_score(full, row_ids)
            qs = q_score(fs, row_ids)
            qc = q_score(fc, row_ids)
            qr = q_score(fr, row_ids)
            suff_ratio = ((qs - q0) / (qf - q0).abs().clamp_min(1e-6)).clamp(-10, 10)
            return {
                "q_base": float(q0.mean().item()),
                "q_full": float(qf.mean().item()),
                "q_suff": float(qs.mean().item()),
                "q_comp": float(qc.mean().item()),
                "q_rand": float(qr.mean().item()),
                "suff_ratio": float(suff_ratio.mean().item()),
                "necessity_drop": float((qf - qc).mean().item()),
                "suff_minus_rand": float((qs - qr).mean().item()),
                "edit_norm": float(torch.norm(de.reshape(de.shape[0], -1), dim=-1).mean().item()),
            }

    for step in range(1, args.steps + 1):
        row_ids = [int(rng.choice(fail_train)) for _ in range(args.batch_size)]
        xb = torch.from_numpy(feats_z[row_ids]).to(device)
        fb = torch.from_numpy(futures_np[row_ids]).to(device)
        ft = torch.from_numpy(np.stack([targets[i] for i in row_ids]).astype(np.float32)).to(device)
        ctx = context_t[torch.as_tensor(row_ids, device=device)]
        teacher = ft - fb
        m2, _ = topk_mask_from_selector(selector, xb, groups, futures_np.shape[-1], args.topk_groups)
        m = m2.unsqueeze(1)
        de_raw = editor(fb, m.expand_as(fb), ctx)
        de = de_raw * m
        full = fb + teacher
        fs = fb + de
        fc = full - de
        rand_score = torch.randn((args.batch_size, len(groups)), device=device)
        mr = topk_mask_from_selector(lambda x: rand_score, xb, groups, futures_np.shape[-1], args.topk_groups)[0].unsqueeze(1)
        fr = fb + mr * de
        q0 = q_score(fb, row_ids)
        qf = q_score(full, row_ids)
        qs = q_score(fs, row_ids)
        qc = q_score(fc, row_ids)
        qr = q_score(fr, row_ids)
        suff_ratio = ((qs - q0) / (qf - q0).abs().clamp_min(1e-6)).clamp(-10, 10)
        teacher_loss = F.mse_loss(de, m * teacher)
        suff_loss = F.relu(1.0 - suff_ratio).mean()
        nec_loss = F.relu(args.margin - (qf - qc)).mean()
        rand_loss = F.relu(args.margin + qr - qs).mean()
        small_loss = (de ** 2).mean()
        preserve_loss = (((1.0 - m) * de_raw) ** 2).mean()
        loss = suff_loss + args.lambda_nec * nec_loss + args.lambda_rand * rand_loss + args.lambda_teacher * teacher_loss + args.lambda_small * small_loss + args.lambda_preserve * preserve_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, args.steps // 10) == 0:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "teacher_loss": float(teacher_loss.item()),
                "suff_loss": float(suff_loss.item()),
                "nec_loss": float(nec_loss.item()),
                "rand_loss": float(rand_loss.item()),
                "small_loss": float(small_loss.item()),
                "preserve_loss": float(preserve_loss.item()),
                "q_suff_train": float(qs.mean().item()),
                "q_comp_train": float(qc.mean().item()),
                "q_rand_train": float(qr.mean().item()),
                "suff_ratio_train": float(suff_ratio.mean().item()),
                **{f"val_{k}": v for k, v in eval_rows(fail_val).items()},
                "num_train_failures": int(len(fail_train)),
                "num_val_failures": int(len(fail_val)),
            }
            history.append(rec)
            score = -abs(1.0 - rec.get("val_suff_ratio", 0.0)) + rec.get("val_necessity_drop", 0.0) + rec.get("val_suff_minus_rand", 0.0)
            if best is None or score > best["score"]:
                best = dict(rec, score=float(score))
                best_state = {k: v.detach().cpu() for k, v in editor.state_dict().items()}
            print(json.dumps({"editor": rec}), flush=True)

    ckpt_path = out / "masked_future_editor.pt"
    torch.save(
        {
            "model_state": best_state,
            "future_dim": int(futures_np.shape[-1]),
            "cond_dim": int(contexts_np.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "topk_groups": int(args.topk_groups),
            "groups": groups,
            "selector_ckpt": str(args.selector_ckpt),
            "q_critic_ckpt": str(args.q_critic_ckpt),
            "q_critic_ckpts": q_paths,
            "q_ensemble_mode": str(args.q_ensemble_mode),
        },
        ckpt_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "num_rows": int(len(rows)),
        "success_count": int(labels_np.sum()),
        "failure_count": int(len(labels_np) - labels_np.sum()),
        "train_failures": int(len(fail_train)),
        "val_failures": int(len(fail_val)),
        "best": best,
        "history": history,
        "editor_ckpt": str(ckpt_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
