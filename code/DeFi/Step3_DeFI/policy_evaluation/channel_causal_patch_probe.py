#!/usr/bin/env python3
"""Offline channel patching probe for future-latent failure localization.

For each failed future, find nearby successful futures, patch channel groups from
the success center into the failed future, and rank groups by proxy improvement.
This is a cheap pre-rollout causal-probing approximation; rollout confirmation
is still needed for a strict causal claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(af, bf) / denom)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def nearest(query: np.ndarray, bank: np.ndarray, candidates: list[int], k: int) -> tuple[list[int], np.ndarray]:
    if not candidates:
        return [], np.zeros((0,), dtype=np.float32)
    sims = query[None, :] @ bank[candidates].T
    order = np.argsort(-sims.reshape(-1))[: max(1, min(k, len(candidates)))]
    ids = [candidates[int(i)] for i in order]
    weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones(len(ids), dtype=np.float32) / max(len(ids), 1)
    else:
        weights /= weights.sum()
    return ids, weights


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def channel_groups(dim: int, group_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + group_size, dim)) for start in range(0, dim, group_size)]


def patch_group(base: np.ndarray, donor: np.ndarray, start: int, end: int) -> np.ndarray:
    out = base.copy()
    out[:, start:end] = donor[:, start:end]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--future-trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--k-success", type=int, default=5)
    parser.add_argument("--k-failure", type=int, default=5)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--top-groups", type=int, default=8)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_raw = read_jsonl(Path(args.dataset_jsonl))
    trace_dir = Path(args.future_trace_dir)

    rows = []
    futures = []
    for idx, row in enumerate(rows_raw):
        path = resolve_trace(row, trace_dir, idx)
        if path is None:
            continue
        future = load_future(path, args.future_key)
        if future is None:
            continue
        row = dict(row)
        row["_trace_path"] = str(path)
        row["_row_index"] = idx
        rows.append(row)
        futures.append(future)

    if not rows:
        raise RuntimeError("no usable rows")

    futures_np = np.stack(futures, axis=0)
    pooled = normalize(futures_np.mean(axis=1).astype(np.float32))
    tasks = [str(row.get("task", "unknown")) for row in rows]
    success_ids = [i for i, row in enumerate(rows) if bool(row.get("success"))]
    failure_ids = [i for i, row in enumerate(rows) if not bool(row.get("success"))]
    if args.max_failures > 0:
        failure_ids = failure_ids[: args.max_failures]
    groups = channel_groups(futures_np.shape[-1], args.group_size)

    out_rows = []
    group_gains: dict[str, list[float]] = defaultdict(list)
    group_margin_gains: dict[str, list[float]] = defaultdict(list)
    best_group_counts = Counter()
    skipped = 0

    for idx in failure_ids:
        same_success = [j for j in success_ids if (not args.same_task_only or tasks[j] == tasks[idx])]
        same_failure = [j for j in failure_ids if j != idx and (not args.same_task_only or tasks[j] == tasks[idx])]
        if not same_success:
            same_success = success_ids
        if not same_failure:
            same_failure = [j for j in failure_ids if j != idx]
        succ_ids, succ_w = nearest(pooled[idx], pooled, same_success, args.k_success)
        fail_ids, fail_w = nearest(pooled[idx], pooled, same_failure, args.k_failure)
        if not succ_ids or not fail_ids:
            skipped += 1
            continue
        succ_center = np.tensordot(succ_w, futures_np[succ_ids], axes=(0, 0)).astype(np.float32)
        fail_center = np.tensordot(fail_w, futures_np[fail_ids], axes=(0, 0)).astype(np.float32)
        base = futures_np[idx]
        base_success_cos = cosine(base, succ_center)
        base_failure_cos = cosine(base, fail_center)

        candidates = []
        for start, end in groups:
            patched = patch_group(base, succ_center, start, end)
            success_gain = cosine(patched, succ_center) - base_success_cos
            failure_repulsion = base_failure_cos - cosine(patched, fail_center)
            margin_gain = success_gain + failure_repulsion
            key = f"{start}:{end}"
            group_gains[key].append(float(success_gain))
            group_margin_gains[key].append(float(margin_gain))
            candidates.append((margin_gain, success_gain, failure_repulsion, start, end))
        candidates.sort(reverse=True, key=lambda x: x[0])
        best = candidates[0]
        best_group_counts[f"{best[3]}:{best[4]}"] += 1
        out_rows.append(
            {
                "row_index": int(rows[idx]["_row_index"]),
                "sequence_index": rows[idx].get("sequence_index"),
                "subtask_index": rows[idx].get("subtask_index"),
                "task": tasks[idx],
                "trace_path": rows[idx]["_trace_path"],
                "base_success_cos": float(base_success_cos),
                "base_failure_cos": float(base_failure_cos),
                "best_group": f"{best[3]}:{best[4]}",
                "best_margin_gain": float(best[0]),
                "best_success_gain": float(best[1]),
                "best_failure_repulsion": float(best[2]),
                "top_groups": [
                    {
                        "channel_start": int(start),
                        "channel_end": int(end),
                        "margin_gain": float(margin_gain),
                        "success_gain": float(success_gain),
                        "failure_repulsion": float(failure_repulsion),
                    }
                    for margin_gain, success_gain, failure_repulsion, start, end in candidates[: args.top_groups]
                ],
            }
        )

    with (out_dir / "channel_patch_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_group_summary = []
    for key, vals in group_margin_gains.items():
        top_group_summary.append(
            {
                "group": key,
                "count_best": int(best_group_counts.get(key, 0)),
                "margin_gain": summarize(vals),
                "success_gain": summarize(group_gains[key]),
            }
        )
    top_group_summary.sort(key=lambda x: (x["count_best"], x["margin_gain"]["mean"]), reverse=True)
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_trace_dir": str(args.future_trace_dir),
        "future_key": args.future_key,
        "num_rows": len(rows),
        "num_success": len(success_ids),
        "num_failure": len(failure_ids),
        "num_probed": len(out_rows),
        "skipped": int(skipped),
        "same_task_only": bool(args.same_task_only),
        "group_size": int(args.group_size),
        "positive_best_margin_rate": float(np.mean([r["best_margin_gain"] > 0 for r in out_rows])) if out_rows else 0.0,
        "best_margin_gain": summarize([r["best_margin_gain"] for r in out_rows]),
        "best_success_gain": summarize([r["best_success_gain"] for r in out_rows]),
        "best_failure_repulsion": summarize([r["best_failure_repulsion"] for r in out_rows]),
        "top_group_summary": top_group_summary[:20],
        "output_rows": str(out_dir / "channel_patch_rows.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
