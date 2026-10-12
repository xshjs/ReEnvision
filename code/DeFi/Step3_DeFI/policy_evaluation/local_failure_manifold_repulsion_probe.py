from __future__ import annotations

import argparse
import json
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
    if x.ndim != 2:
        return None
    return x


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local failure-manifold repulsion using nearest failure-to-success deltas.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", choices=["base_future", "proposal_future", "gated_future", "final_future"], default="base_future")
    parser.add_argument("--failure-neighbor-jsonl", type=Path, required=True)
    parser.add_argument("--k-failure", type=int, default=5)
    parser.add_argument("--gate", type=float, default=0.05)
    parser.add_argument("--alpha-repel", type=float, default=1.0)
    parser.add_argument("--beta-attract", type=float, default=1.0)
    parser.add_argument("--decomposed-delta", action="store_true")
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    futures = []
    pooled = []
    for row in iter_jsonl(args.dataset_jsonl):
        path = row.get("trace_path")
        if not path or not Path(str(path)).exists():
            continue
        future = load_future(path, args.future_key)
        if future is None:
            continue
        rows.append(row)
        futures.append(future)
        pooled.append(future.mean(axis=0).astype(np.float32))
    x = np.stack(pooled, axis=0)
    x_norm = normalize(x)

    failure_indices = [idx for idx, row in enumerate(rows) if not bool(row.get("success", False))]
    if args.max_failures > 0:
        failure_indices = failure_indices[: int(args.max_failures)]
    success_indices = [idx for idx, row in enumerate(rows) if bool(row.get("success", False))]
    if not failure_indices or not success_indices:
        raise ValueError("need success and failure rows")

    # Reconstruct failure -> nearest success deltas in pooled space.
    success_x = x[success_indices]
    success_norm = x_norm[success_indices]
    failure_x = x[failure_indices]
    failure_norm = x_norm[failure_indices]
    fs_sims = failure_norm @ success_norm.T
    nearest_success_local = np.argmax(fs_sims, axis=1)
    failure_delta = success_x[nearest_success_local] - failure_x
    failure_to_global = {global_idx: local_idx for local_idx, global_idx in enumerate(failure_indices)}

    test_rows = []
    gain_success = []
    repel_failure = []
    risk_margin_change = []
    selected_same_task_rate = []

    with (args.output_dir / "local_repulsion_rows.jsonl").open("w") as handle:
        for local_i, global_i in enumerate(failure_indices):
            task = str(rows[global_i].get("task"))
            candidate_locals = []
            for local_j, global_j in enumerate(failure_indices):
                if local_i == local_j:
                    continue
                if args.same_task_only and str(rows[global_j].get("task")) != task:
                    continue
                candidate_locals.append(local_j)
            if not candidate_locals:
                candidate_locals = [j for j in range(len(failure_indices)) if j != local_i]
            sims = failure_norm[local_i][None, :] @ failure_norm[candidate_locals].T
            order = np.argsort(-sims.reshape(-1))[: max(1, min(args.k_failure, len(candidate_locals)))]
            neighbor_locals = [candidate_locals[int(idx)] for idx in order]
            weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
            if float(weights.sum()) <= 1e-8:
                weights = np.ones(len(neighbor_locals), dtype=np.float32) / float(len(neighbor_locals))
            else:
                weights = weights / weights.sum()
            if args.decomposed_delta:
                failure_center = np.tensordot(weights, failure_x[neighbor_locals], axes=(0, 0)).astype(np.float32)
                success_center = success_x[int(nearest_success_local[local_i])]
                repel = x[global_i] - failure_center
                attract = success_center - x[global_i]
                delta = float(args.alpha_repel) * repel + float(args.beta_attract) * attract
            else:
                delta = np.tensordot(weights, failure_delta[neighbor_locals], axes=(0, 0)).astype(np.float32)
            edited = x[global_i] + float(args.gate) * delta
            success_target = success_x[nearest_success_local[local_i]]
            nearest_failure = failure_x[neighbor_locals[0]]
            base_success_cos = cosine(x[global_i], success_target)
            edit_success_cos = cosine(edited, success_target)
            base_failure_cos = cosine(x[global_i], nearest_failure)
            edit_failure_cos = cosine(edited, nearest_failure)
            sgain = edit_success_cos - base_success_cos
            frepel = base_failure_cos - edit_failure_cos
            margin_delta = (edit_success_cos - edit_failure_cos) - (base_success_cos - base_failure_cos)
            gain_success.append(float(sgain))
            repel_failure.append(float(frepel))
            risk_margin_change.append(float(margin_delta))
            selected_same_task_rate.append(float(any(str(rows[failure_indices[j]].get("task")) == task for j in neighbor_locals)))
            out = {
                "row_index": int(global_i),
                "failure_local_index": int(local_i),
                "task": task,
                "sequence_index": rows[global_i].get("sequence_index"),
                "subtask_index": rows[global_i].get("subtask_index"),
                "neighbor_failure_tasks": [rows[failure_indices[j]].get("task") for j in neighbor_locals],
                "nearest_success_task": rows[success_indices[int(nearest_success_local[local_i])]].get("task"),
                "base_success_cos": float(base_success_cos),
                "edit_success_cos": float(edit_success_cos),
                "success_cos_gain": float(sgain),
                "base_failure_cos": float(base_failure_cos),
                "edit_failure_cos": float(edit_failure_cos),
                "failure_repulsion_gain": float(frepel),
                "risk_margin_change": float(margin_delta),
                "decomposed_delta": bool(args.decomposed_delta),
                "alpha_repel": float(args.alpha_repel),
                "beta_attract": float(args.beta_attract),
            }
            test_rows.append(out)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_key": args.future_key,
        "num_rows": int(len(rows)),
        "num_failure": int(len(failure_indices)),
        "num_success": int(len(success_indices)),
        "k_failure": int(args.k_failure),
        "gate": float(args.gate),
        "same_task_only": bool(args.same_task_only),
        "decomposed_delta": bool(args.decomposed_delta),
        "alpha_repel": float(args.alpha_repel),
        "beta_attract": float(args.beta_attract),
        "success_cos_gain": summarize(gain_success),
        "failure_repulsion_gain": summarize(repel_failure),
        "risk_margin_change": summarize(risk_margin_change),
        "positive_success_gain_rate": float(np.mean(np.asarray(gain_success) > 0)),
        "positive_failure_repulsion_rate": float(np.mean(np.asarray(repel_failure) > 0)),
        "positive_margin_change_rate": float(np.mean(np.asarray(risk_margin_change) > 0)),
        "selected_same_task_rate": float(np.mean(selected_same_task_rate)),
        "output_rows": str(args.output_dir / "local_repulsion_rows.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
