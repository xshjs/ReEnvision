#!/usr/bin/env python3
"""Evaluate a masked future editor with multiple frozen Q critics.

This checks whether the editor only exploits its training Q or transfers to
independently trained Q critics.
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
        t = future.shape[1]
        pos = torch.linspace(0.0, 1.0, t, device=future.device, dtype=future.dtype)[None, :, None].expand(future.shape[0], t, 1)
        h = self.future_proj(torch.cat([future, mask, pos], dim=-1)) + self.cond_proj(cond)[:, None, :]
        return self.net(h)


def build_feature_np(future: np.ndarray, row: dict[str, Any], task_dim: int) -> np.ndarray:
    sub = float(row.get("subtask_index", 0.0)) / 5.0
    return np.concatenate([future_summary_np(future), hashed_task(str(row.get("task", "unknown")), task_dim), np.asarray([sub], dtype=np.float32)])


def load_q(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = MLP(int(ckpt["input_dim"]), 1, int(ckpt.get("hidden_dim", 512)), float(ckpt.get("dropout", 0.1))).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return {"model": model, "ckpt": ckpt, "path": str(path)}


def load_selector(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = MLP(int(ckpt["input_dim"]), int(ckpt["num_groups"]), int(ckpt.get("hidden_dim", 512)), float(ckpt.get("dropout", 0.1))).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def load_editor(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = TokenEditor(int(ckpt["future_dim"]), int(ckpt["cond_dim"]), int(ckpt.get("hidden_dim", 512)), float(ckpt.get("dropout", 0.05))).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def nearest_success_targets(rows, futures, pool_ids, same_subtask: bool, step_tolerance: int, k: int):
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


def topk_mask(selector, x, groups, dim, topk):
    logits = selector(x)
    ids = torch.topk(logits, k=max(1, min(int(topk), logits.shape[-1])), dim=-1).indices
    mask = torch.zeros((x.shape[0], dim), device=x.device, dtype=torch.float32)
    for gi, (s, e) in enumerate(groups):
        rows = (ids == gi).any(dim=-1)
        if bool(rows.any()):
            mask[rows, s:e] = 1.0
    return mask


def summarize(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selector-ckpt", required=True)
    parser.add_argument("--editor-ckpt", required=True)
    parser.add_argument("--q-critic-ckpts", nargs="+", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--topk-groups", type=int, default=4)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--same-subtask-target", action="store_true")
    parser.add_argument("--target-step-tolerance", type=int, default=160)
    parser.add_argument("--limit-failures", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows_raw = read_jsonl(Path(args.dataset_jsonl))
    rows, futures = [], []
    for idx, row in enumerate(rows_raw):
        trace = resolve_trace(row, Path(args.future_trace_dir), idx)
        if trace is None:
            continue
        fut = load_future(trace, args.future_key)
        if fut is None:
            continue
        rows.append(row)
        futures.append(fut)
    futures_np = np.stack(futures).astype(np.float32)
    labels = np.asarray([1.0 if bool(r.get("success", False)) else 0.0 for r in rows], dtype=np.float32)
    pool_ids = np.arange(len(rows), dtype=np.int64)
    targets = nearest_success_targets(rows, futures_np, pool_ids, args.same_subtask_target, args.target_step_tolerance, args.target_k)
    fail_ids = [i for i in range(len(rows)) if labels[i] < 0.5 and i in targets]
    if args.limit_failures > 0:
        fail_ids = fail_ids[: args.limit_failures]
    selector, sel_ckpt = load_selector(Path(args.selector_ckpt), device)
    editor, _ = load_editor(Path(args.editor_ckpt), device)
    q_list = [load_q(Path(p), device) for p in args.q_critic_ckpts]
    groups = [tuple(map(int, g)) for g in sel_ckpt["groups"]]
    task_dim = int(sel_ckpt.get("task_dim", 128))

    contexts = []
    selector_feats = []
    for i in fail_ids:
        row = rows[i]
        sub = float(row.get("subtask_index", 0.0)) / 5.0
        ctx = np.concatenate([hashed_task(str(row.get("task", "unknown")), task_dim), np.asarray([sub], dtype=np.float32)])
        contexts.append(ctx)
        selector_feats.append(build_feature_np(futures_np[i], row, task_dim))
    x_mean = np.asarray(sel_ckpt["mean"], dtype=np.float32)
    x_std = np.maximum(np.asarray(sel_ckpt["std"], dtype=np.float32), 1e-6)
    x = torch.from_numpy(((np.stack(selector_feats).astype(np.float32) - x_mean) / x_std).astype(np.float32)).to(device)
    ctx = torch.from_numpy(np.stack(contexts).astype(np.float32)).to(device)
    fb = torch.from_numpy(futures_np[fail_ids]).to(device)
    ft = torch.from_numpy(np.stack([targets[i] for i in fail_ids]).astype(np.float32)).to(device)
    teacher = ft - fb
    m = topk_mask(selector, x, groups, futures_np.shape[-1], args.topk_groups).unsqueeze(1)
    de = editor(fb, m.expand_as(fb), ctx) * m
    full = fb + teacher
    suff = fb + de
    comp = full - de
    rand_score = torch.randn((len(fail_ids), len(groups)), device=device)
    rand_mask = topk_mask(lambda _: rand_score, x, groups, futures_np.shape[-1], args.topk_groups).unsqueeze(1)
    rand = fb + rand_mask * de

    rows_out = []
    summary = {"num_failures": int(len(fail_ids)), "q": {}}
    for qi, qobj in enumerate(q_list):
        q = qobj["model"]
        ck = qobj["ckpt"]
        mean = torch.from_numpy(np.asarray(ck["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(ck["std"], dtype=np.float32)).to(device)
        q_task_dim = int(ck.get("task_dim", task_dim))
        q_contexts = []
        for i in fail_ids:
            sub = float(rows[i].get("subtask_index", 0.0)) / 5.0
            q_contexts.append(np.concatenate([hashed_task(str(rows[i].get("task", "unknown")), q_task_dim), np.asarray([sub], dtype=np.float32)]))
        qctx = torch.from_numpy(np.stack(q_contexts).astype(np.float32)).to(device)

        def score(fut):
            raw = torch.cat([future_summary_torch(fut), qctx], dim=-1)
            return torch.sigmoid(q((raw - mean) / std)).squeeze(-1).detach().cpu().numpy()

        vals = {name: score(fut) for name, fut in {"base": fb, "full": full, "suff": suff, "comp": comp, "rand": rand}.items()}
        denom = np.maximum(np.abs(vals["full"] - vals["base"]), 1e-6)
        suff_ratio = np.clip((vals["suff"] - vals["base"]) / denom, -10, 10)
        necessity_drop = vals["full"] - vals["comp"]
        suff_minus_rand = vals["suff"] - vals["rand"]
        rec = {
            "path": qobj["path"],
            "base": summarize(vals["base"]),
            "full": summarize(vals["full"]),
            "suff": summarize(vals["suff"]),
            "comp": summarize(vals["comp"]),
            "rand": summarize(vals["rand"]),
            "suff_ratio": summarize(suff_ratio),
            "necessity_drop": summarize(necessity_drop),
            "suff_minus_rand": summarize(suff_minus_rand),
            "positive_suff_vs_rand_rate": float((suff_minus_rand > 0).mean()),
            "positive_necessity_rate": float((necessity_drop > 0).mean()),
        }
        summary["q"][f"q{qi}"] = rec
        for n, row_id in enumerate(fail_ids):
            rows_out.append(
                {
                    "q_index": qi,
                    "row_index": int(row_id),
                    "task": rows[row_id].get("task"),
                    "subtask_index": rows[row_id].get("subtask_index"),
                    "base": float(vals["base"][n]),
                    "full": float(vals["full"][n]),
                    "suff": float(vals["suff"][n]),
                    "comp": float(vals["comp"][n]),
                    "rand": float(vals["rand"][n]),
                    "suff_ratio": float(suff_ratio[n]),
                    "necessity_drop": float(necessity_drop[n]),
                    "suff_minus_rand": float(suff_minus_rand[n]),
                }
            )
    with (out / "ensemble_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
