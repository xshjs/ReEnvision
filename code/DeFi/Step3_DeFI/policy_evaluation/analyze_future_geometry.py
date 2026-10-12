from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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
    x = np.asarray(data[key], dtype=np.float32)
    return x if x.ndim == 2 else None


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def auc_score(y_fail: np.ndarray, risk: np.ndarray) -> float:
    pos = risk[y_fail > 0.5]
    neg = risk[y_fail <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def topk_stats(y_fail: np.ndarray, risk: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-risk)
    out = {}
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(1, int(round(len(order) * frac)))
        idx = order[:k]
        out[f"top_{int(frac * 100)}pct_fail_rate"] = float(y_fail[idx].mean())
        out[f"top_{int(frac * 100)}pct_fail_count"] = int(y_fail[idx].sum())
        out[f"top_{int(frac * 100)}pct_count"] = int(k)
    return out


def nearest_center(features: np.ndarray, candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = normalize(features) @ normalize(candidates).T
    kk = max(1, min(k, candidates.shape[0]))
    order = np.argsort(-sims, axis=1)[:, :kk]
    top_sims = np.take_along_axis(sims, order, axis=1)
    weights = np.maximum(top_sims, 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    centers = np.einsum("bk,bkd->bd", weights, candidates[order])
    return centers.astype(np.float32), top_sims.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze future latent geometry: success/failure manifolds and recovery directions.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", choices=["base_future", "proposal_future", "gated_future", "final_future"], default="base_future")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--same-task", action="store_true")
    parser.add_argument("--gate", type=float, default=0.005)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    futures: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for row in iter_jsonl(args.dataset_jsonl):
        trace_path = row.get("trace_path")
        if not trace_path or not Path(str(trace_path)).exists():
            continue
        future = load_future(trace_path, args.future_key)
        if future is None:
            continue
        rows.append(row)
        futures.append(future)
        pooled.append(future.mean(axis=0).astype(np.float32))
    x = np.stack(pooled, axis=0).astype(np.float32)
    y_success = np.asarray([1.0 if bool(row.get("success", False)) else 0.0 for row in rows], dtype=np.float32)
    y_fail = 1.0 - y_success
    tasks = np.asarray([str(row.get("task", "unknown")) for row in rows])
    success_idx = np.where(y_success > 0.5)[0]
    failure_idx = np.where(y_success <= 0.5)[0]
    if len(success_idx) == 0 or len(failure_idx) == 0:
        raise ValueError("need both success and failure rows")

    risk = np.zeros(len(rows), dtype=np.float32)
    geometry_rows = []
    success_gain = []
    failure_repulsion = []
    margin_gain = []
    same_task_success_hits = []
    same_task_failure_hits = []
    by_task = defaultdict(list)

    for idx, row in enumerate(rows):
        task = tasks[idx]
        succ_candidates = success_idx[tasks[success_idx] == task] if args.same_task else success_idx
        fail_candidates = failure_idx[tasks[failure_idx] == task] if args.same_task else failure_idx
        if len(succ_candidates) == 0:
            succ_candidates = success_idx
        if len(fail_candidates) == 0:
            fail_candidates = failure_idx
        cur = x[idx : idx + 1]
        succ_center, succ_sims = nearest_center(cur, x[succ_candidates], args.k)
        fail_center, fail_sims = nearest_center(cur, x[fail_candidates], args.k)
        sim_success = float((normalize(cur) * normalize(succ_center)).sum())
        sim_failure = float((normalize(cur) * normalize(fail_center)).sum())
        risk[idx] = sim_failure - sim_success
        delta = succ_center[0] - fail_center[0]
        edited = cur[0] + float(args.gate) * delta
        base_succ = float((normalize(cur)[0] * normalize(succ_center)[0]).sum())
        edit_succ = float((normalize(edited[None, :])[0] * normalize(succ_center)[0]).sum())
        base_fail = float((normalize(cur)[0] * normalize(fail_center)[0]).sum())
        edit_fail = float((normalize(edited[None, :])[0] * normalize(fail_center)[0]).sum())
        sg = edit_succ - base_succ
        fr = base_fail - edit_fail
        mg = (edit_succ - edit_fail) - (base_succ - base_fail)
        success_gain.append(sg)
        failure_repulsion.append(fr)
        margin_gain.append(mg)
        succ_all_sims = (normalize(cur) @ normalize(x[succ_candidates]).T).reshape(-1)
        fail_all_sims = (normalize(cur) @ normalize(x[fail_candidates]).T).reshape(-1)
        succ_top = succ_candidates[np.argsort(-succ_all_sims)[: max(1, min(args.k, len(succ_candidates)))]]
        fail_top = fail_candidates[np.argsort(-fail_all_sims)[: max(1, min(args.k, len(fail_candidates)))]]
        same_task_success_hits.append(float(np.mean(tasks[succ_top] == task)))
        same_task_failure_hits.append(float(np.mean(tasks[fail_top] == task)))
        by_task[task].append(idx)
        if y_fail[idx] > 0.5:
            geometry_rows.append(
                {
                    "row_index": int(idx),
                    "sequence_index": row.get("sequence_index"),
                    "subtask_index": row.get("subtask_index"),
                    "task": task,
                    "success": bool(row.get("success", False)),
                    "risk_margin": float(risk[idx]),
                    "sim_success_center": sim_success,
                    "sim_failure_center": sim_failure,
                    "success_cos_gain_after_delta": float(sg),
                    "failure_repulsion_gain_after_delta": float(fr),
                    "margin_gain_after_delta": float(mg),
                    "nearest_success_sim_mean": float(np.mean(succ_sims)),
                    "nearest_failure_sim_mean": float(np.mean(fail_sims)),
                }
            )

    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_key": args.future_key,
        "num_rows": int(len(rows)),
        "num_success": int(len(success_idx)),
        "num_failure": int(len(failure_idx)),
        "k": int(args.k),
        "same_task": bool(args.same_task),
        "gate": float(args.gate),
        "base_fail_rate": float(y_fail.mean()),
        "risk_auc_fail": auc_score(y_fail, risk),
        **topk_stats(y_fail, risk),
        "success_gain_after_delta": summarize(success_gain),
        "failure_repulsion_after_delta": summarize(failure_repulsion),
        "margin_gain_after_delta": summarize(margin_gain),
        "positive_success_gain_rate": float(np.mean(np.asarray(success_gain) > 0.0)),
        "positive_failure_repulsion_rate": float(np.mean(np.asarray(failure_repulsion) > 0.0)),
        "positive_margin_gain_rate": float(np.mean(np.asarray(margin_gain) > 0.0)),
        "same_task_success_hit_rate": float(np.mean(same_task_success_hits)),
        "same_task_failure_hit_rate": float(np.mean(same_task_failure_hits)),
        "task_counts": dict(Counter(tasks.tolist()).most_common()),
        "failure_task_counts": dict(Counter(tasks[failure_idx].tolist()).most_common()),
    }
    (args.output_dir / "future_geometry_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "future_geometry_failure_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in geometry_rows)
    )
    np.savez_compressed(
        args.output_dir / "future_geometry_arrays.npz",
        pooled_features=x,
        success=y_success,
        risk_margin=risk,
        tasks=tasks.astype(object),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
