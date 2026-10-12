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


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return a @ b.T


def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        data = np.load(str(path))
    except Exception:
        return None
    if key not in data.files:
        return None
    future = np.asarray(data[key], dtype=np.float32)
    if future.ndim != 2:
        return None
    return future


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether failure-to-success future deltas form stable repulsion directions.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", choices=["base_future", "proposal_future", "gated_future", "final_future"], default="base_future")
    parser.add_argument("--max-success", type=int, default=0)
    parser.add_argument("--topk-success", type=int, default=1)
    parser.add_argument("--pooled-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    success_features = []
    failure_features = []
    success_futures = []
    failure_futures = []

    for row in iter_jsonl(args.dataset_jsonl):
        path = row.get("trace_path")
        if not path or not Path(str(path)).exists():
            continue
        future = load_future(path, args.future_key)
        if future is None:
            continue
        pooled = future.mean(axis=0).astype(np.float32)
        if bool(row.get("success", False)):
            if args.max_success > 0 and len(success_rows) >= args.max_success:
                continue
            success_rows.append(row)
            success_features.append(pooled)
            if not args.pooled_only:
                success_futures.append(future)
        else:
            failure_rows.append(row)
            failure_features.append(pooled)
            if not args.pooled_only:
                failure_futures.append(future)

    if not success_rows or not failure_rows:
        raise ValueError(f"need both success and failure rows, got success={len(success_rows)} failure={len(failure_rows)}")

    success_x = np.stack(success_features, axis=0)
    failure_x = np.stack(failure_features, axis=0)
    sims = cosine_matrix(failure_x, success_x)
    topk = max(1, min(int(args.topk_success), success_x.shape[0]))
    nn_indices = np.argpartition(sims, -topk, axis=1)[:, -topk:]

    deltas = []
    delta_norms = []
    nearest_success_sims = []
    same_task_hits = []
    rows_out = []
    for fail_idx, row in enumerate(failure_rows):
        ranked = sorted(nn_indices[fail_idx].tolist(), key=lambda idx: float(sims[fail_idx, idx]), reverse=True)
        if args.pooled_only:
            success_future = np.mean([success_x[idx] for idx in ranked], axis=0).astype(np.float32)
            delta = (success_future - failure_x[fail_idx]).reshape(-1).astype(np.float32)
        else:
            success_future = np.mean([success_futures[idx] for idx in ranked], axis=0).astype(np.float32)
            delta = (success_future - failure_futures[fail_idx]).reshape(-1).astype(np.float32)
        deltas.append(delta)
        delta_norms.append(float(np.linalg.norm(delta)))
        nearest_success_sims.append(float(np.mean([sims[fail_idx, idx] for idx in ranked])))
        same_task_hits.append(float(any(str(success_rows[idx].get("task")) == str(row.get("task")) for idx in ranked)))
        rows_out.append(
            {
                "failure_index": int(fail_idx),
                "sequence_index": row.get("sequence_index"),
                "subtask_index": row.get("subtask_index"),
                "task": row.get("task"),
                "failure_factor": row.get("failure_factor"),
                "nearest_success_indices": [int(x) for x in ranked],
                "nearest_success_tasks": [success_rows[idx].get("task") for idx in ranked],
                "nearest_success_sim": float(np.mean([sims[fail_idx, idx] for idx in ranked])),
                "same_task_hit": bool(same_task_hits[-1]),
                "delta_norm": delta_norms[-1],
            }
        )

    delta_x = np.stack(deltas, axis=0)
    delta_unit = delta_x / np.maximum(np.linalg.norm(delta_x, axis=1, keepdims=True), 1e-8)
    mean_dir = delta_unit.mean(axis=0)
    mean_dir_norm = float(np.linalg.norm(mean_dir))
    mean_dir_unit = mean_dir / max(mean_dir_norm, 1e-8)
    alignment_to_mean = (delta_unit @ mean_dir_unit).astype(np.float32)

    # Pairwise consistency can be expensive; sample deterministically if needed.
    rng = np.random.default_rng(0)
    pairs = []
    max_pairs = 20000
    n = delta_unit.shape[0]
    if n * (n - 1) // 2 <= max_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(float(np.dot(delta_unit[i], delta_unit[j])))
    else:
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            pairs.append(float(np.dot(delta_unit[i], delta_unit[j])))

    by_task = defaultdict(list)
    by_factor = defaultdict(list)
    for idx, row in enumerate(failure_rows):
        by_task[str(row.get("task"))].append(float(alignment_to_mean[idx]))
        by_factor[str(row.get("failure_factor"))].append(float(alignment_to_mean[idx]))

    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_key": args.future_key,
        "num_success": int(len(success_rows)),
        "num_failure": int(len(failure_rows)),
        "topk_success": int(topk),
        "pooled_only": bool(args.pooled_only),
        "failure_task_counts": dict(Counter(str(row.get("task")) for row in failure_rows).most_common()),
        "failure_factor_counts": dict(Counter(str(row.get("failure_factor")) for row in failure_rows).most_common()),
        "nearest_success_sim": summarize(nearest_success_sims),
        "same_task_hit_rate": float(np.mean(same_task_hits)),
        "delta_norm": summarize(delta_norms),
        "mean_direction_norm": mean_dir_norm,
        "alignment_to_mean_direction": summarize([float(x) for x in alignment_to_mean]),
        "pairwise_delta_cosine": summarize(pairs),
        "alignment_by_task_top": {
            task: summarize(vals)
            for task, vals in sorted(by_task.items(), key=lambda item: len(item[1]), reverse=True)[:20]
        },
        "alignment_by_factor": {factor: summarize(vals) for factor, vals in by_factor.items()},
        "output_rows": str(args.output_dir / "failure_to_success_neighbors.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "failure_to_success_neighbors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out)
    )
    np.savez_compressed(
        args.output_dir / "failure_repulsion_directions.npz",
        mean_direction=mean_dir.astype(np.float32),
        alignment_to_mean=alignment_to_mean.astype(np.float32),
        delta_norms=np.asarray(delta_norms, dtype=np.float32),
        nearest_success_sims=np.asarray(nearest_success_sims, dtype=np.float32),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
