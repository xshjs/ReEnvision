from __future__ import annotations

import argparse
import json
import re
from collections import Counter
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


def infer_action_trace_path(row: dict[str, Any], action_trace_dir: Path | None) -> str | None:
    explicit = row.get("action_trace_path")
    if explicit and Path(str(explicit)).exists():
        return str(explicit)
    if action_trace_dir is None:
        return None
    trace = row.get("trace_path") or row.get("trace")
    if not trace:
        return None
    match = re.search(r"future_trace_(\d+)\.npz$", str(trace))
    if not match:
        return None
    path = action_trace_dir / f"action_trace_{match.group(1)}.npz"
    return str(path) if path.exists() else None


def load_actions(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        data = np.load(str(path))
    except Exception:
        return None
    if "actions" not in data.files:
        return None
    arr = np.asarray(data["actions"], dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr


def action_summary_feature(actions: np.ndarray | None, action_dim: int = 7) -> np.ndarray:
    """Fixed-size action descriptor for failure->success repair pairs."""
    out_dim = 3 + 4 * action_dim
    if actions is None or actions.size == 0:
        return np.zeros((out_dim,), dtype=np.float32)
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((out_dim,), dtype=np.float32)
    if arr.shape[1] < action_dim:
        pad = np.zeros((arr.shape[0], action_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > action_dim:
        arr = arr[:, :action_dim]
    n = float(arr.shape[0])
    step_l2 = np.linalg.norm(arr, axis=1) if arr.shape[0] else np.zeros((0,), dtype=np.float32)
    total_l2 = float(np.linalg.norm(arr.reshape(-1)))
    return np.concatenate(
        [
            np.asarray(
                [
                    np.log1p(n) / np.log1p(720.0),
                    float(step_l2.mean()) if step_l2.size else 0.0,
                    np.log1p(total_l2) / np.log1p(720.0 * np.sqrt(float(action_dim))),
                ],
                dtype=np.float32,
            ),
            arr.mean(axis=0).astype(np.float32),
            arr.std(axis=0).astype(np.float32),
            arr[0].astype(np.float32),
            arr[-1].astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local failure-manifold immune repulsion memory.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", choices=["base_future", "proposal_future", "gated_future", "final_future"], default="base_future")
    parser.add_argument("--topk-success", type=int, default=1)
    parser.add_argument("--same-task-success", action="store_true")
    parser.add_argument("--action-trace-dir", type=Path, default=None)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-retrieval-weight", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    futures = []
    pooled = []
    action_paths: list[str | None] = []
    action_features = []
    action_lengths = []
    for row in iter_jsonl(args.dataset_jsonl):
        path = row.get("trace_path")
        if not path or not Path(str(path)).exists():
            continue
        future = load_future(path, args.future_key)
        if future is None:
            continue
        action_path = infer_action_trace_path(row, args.action_trace_dir)
        actions = load_actions(action_path)
        rows.append(row)
        futures.append(future)
        pooled.append(future.mean(axis=0).astype(np.float32))
        action_paths.append(action_path)
        action_features.append(action_summary_feature(actions, action_dim=args.action_dim))
        action_lengths.append(int(actions.shape[0]) if actions is not None and actions.ndim == 2 else 0)
    x = np.stack(pooled, axis=0)
    x_norm = normalize(x)
    action_x = np.stack(action_features, axis=0).astype(np.float32)
    action_x_norm = normalize(action_x)
    success_indices = [idx for idx, row in enumerate(rows) if bool(row.get("success", False))]
    failure_indices = [idx for idx, row in enumerate(rows) if not bool(row.get("success", False))]
    if not success_indices or not failure_indices:
        raise ValueError("need success and failure rows")

    failure_features = []
    success_features = []
    attract_deltas = []
    failure_token_features = []
    success_token_features = []
    token_attract_deltas = []
    failure_action_features = []
    success_action_features = []
    action_attract_deltas = []
    failure_action_lengths = []
    success_action_lengths = []
    tasks = []
    failure_factors = []
    neighbor_rows = []
    topk = max(1, int(args.topk_success))
    for global_i in failure_indices:
        task = str(rows[global_i].get("task"))
        candidates = [idx for idx in success_indices if (not args.same_task_success or str(rows[idx].get("task")) == task)]
        if not candidates:
            candidates = success_indices
        future_sims = (x_norm[global_i][None, :] @ x_norm[candidates].T).reshape(-1)
        action_sims = (action_x_norm[global_i][None, :] @ action_x_norm[candidates].T).reshape(-1)
        action_weight = float(max(0.0, args.action_retrieval_weight))
        if action_weight > 0.0:
            has_action = np.asarray(
                [bool(action_paths[global_i]) and bool(action_paths[idx]) for idx in candidates],
                dtype=np.float32,
            )
            # If an action trace is missing, fall back to future-only for that candidate.
            combined_sims = future_sims + action_weight * action_sims * has_action
        else:
            combined_sims = future_sims
        order = np.argsort(-combined_sims)[: min(topk, len(candidates))]
        selected = [candidates[int(i)] for i in order]
        weights = np.maximum(combined_sims[order], 0.0).astype(np.float32)
        if float(weights.sum()) <= 1e-8:
            weights = np.ones(len(selected), dtype=np.float32) / float(len(selected))
        else:
            weights = weights / weights.sum()
        success_center = np.tensordot(weights, x[selected], axes=(0, 0)).astype(np.float32)
        success_token_center = np.tensordot(weights, np.stack([futures[idx] for idx in selected], axis=0), axes=(0, 0)).astype(np.float32)
        success_action_center = np.tensordot(weights, action_x[selected], axes=(0, 0)).astype(np.float32)
        success_action_length = float(np.dot(weights, np.asarray([action_lengths[idx] for idx in selected], dtype=np.float32)))
        failure_feature = x[global_i].astype(np.float32)
        failure_token_feature = futures[global_i].astype(np.float32)
        failure_action_feature = action_x[global_i].astype(np.float32)
        failure_features.append(failure_feature)
        success_features.append(success_center)
        attract_deltas.append((success_center - failure_feature).astype(np.float32))
        failure_token_features.append(failure_token_feature)
        success_token_features.append(success_token_center)
        token_attract_deltas.append((success_token_center - failure_token_feature).astype(np.float32))
        failure_action_features.append(failure_action_feature)
        success_action_features.append(success_action_center)
        action_attract_deltas.append((success_action_center - failure_action_feature).astype(np.float32))
        failure_action_lengths.append(int(action_lengths[global_i]))
        success_action_lengths.append(float(success_action_length))
        tasks.append(task)
        failure_factors.append(str(rows[global_i].get("failure_factor") or "unknown"))
        neighbor_rows.append(
            {
                "memory_index": len(failure_features) - 1,
                "row_index": int(global_i),
                "sequence_index": rows[global_i].get("sequence_index"),
                "subtask_index": rows[global_i].get("subtask_index"),
                "task": task,
                "failure_factor": failure_factors[-1],
                "failure_trace_path": rows[global_i].get("trace_path"),
                "failure_action_trace_path": action_paths[global_i],
                "failure_action_length": int(action_lengths[global_i]),
                "nearest_success_rows": [int(x) for x in selected],
                "nearest_success_tasks": [rows[idx].get("task") for idx in selected],
                "nearest_success_trace_paths": [rows[idx].get("trace_path") for idx in selected],
                "nearest_success_action_trace_paths": [action_paths[idx] for idx in selected],
                "nearest_success_action_lengths": [int(action_lengths[idx]) for idx in selected],
                "nearest_success_similarity": float(np.mean(combined_sims[order])),
                "nearest_success_future_similarity": float(np.mean(future_sims[order])),
                "nearest_success_action_similarity": float(np.mean(action_sims[order])),
                "action_retrieval_weight": float(action_weight),
            }
        )

    np.savez_compressed(
        args.output_dir / "immune_repulsion_memory.npz",
        failure_features=np.stack(failure_features, axis=0).astype(np.float32),
        success_features=np.stack(success_features, axis=0).astype(np.float32),
        attract_deltas=np.stack(attract_deltas, axis=0).astype(np.float32),
        failure_token_features=np.stack(failure_token_features, axis=0).astype(np.float32),
        success_token_features=np.stack(success_token_features, axis=0).astype(np.float32),
        token_attract_deltas=np.stack(token_attract_deltas, axis=0).astype(np.float32),
        failure_action_features=np.stack(failure_action_features, axis=0).astype(np.float32),
        success_action_features=np.stack(success_action_features, axis=0).astype(np.float32),
        action_attract_deltas=np.stack(action_attract_deltas, axis=0).astype(np.float32),
        failure_action_lengths=np.asarray(failure_action_lengths, dtype=np.float32),
        success_action_lengths=np.asarray(success_action_lengths, dtype=np.float32),
        failure_action_trace_paths=np.asarray([action_paths[idx] for idx in failure_indices], dtype=object),
        tasks=np.asarray(tasks, dtype=object),
        failure_factors=np.asarray(failure_factors, dtype=object),
    )
    (args.output_dir / "immune_repulsion_memory.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in neighbor_rows)
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_key": args.future_key,
        "num_rows": int(len(rows)),
        "num_success": int(len(success_indices)),
        "num_failure": int(len(failure_indices)),
        "topk_success": int(topk),
        "same_task_success": bool(args.same_task_success),
        "action_trace_dir": str(args.action_trace_dir) if args.action_trace_dir else None,
        "action_retrieval_weight": float(args.action_retrieval_weight),
        "num_rows_with_action_trace": int(sum(1 for p in action_paths if p)),
        "num_failures_with_action_trace": int(sum(1 for idx in failure_indices if action_paths[idx])),
        "action_feature_dim": int(action_x.shape[1]),
        "task_counts": dict(Counter(tasks).most_common()),
        "failure_factor_counts": dict(Counter(failure_factors).most_common()),
        "output_npz": str(args.output_dir / "immune_repulsion_memory.npz"),
        "output_jsonl": str(args.output_dir / "immune_repulsion_memory.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
