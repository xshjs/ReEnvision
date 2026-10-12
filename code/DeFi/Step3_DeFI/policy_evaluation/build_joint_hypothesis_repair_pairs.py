from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_memory(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        memory = {k: data[k] for k in data.files}
    required = [
        "base_summary_exec",
        "target_summary_exec",
        "action_chunk",
        "success",
        "task",
        "sequence_index",
        "subtask_index",
    ]
    missing = [k for k in required if k not in memory]
    if missing:
        raise ValueError(f"memory npz missing keys: {missing}")
    return memory


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _state_feature(memory: dict[str, np.ndarray], idx: int) -> np.ndarray:
    if "state_start" in memory:
        arr = np.asarray(memory["state_start"][idx], dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr
    if "base_summary_exec" in memory:
        return np.asarray(memory["base_summary_exec"][idx], dtype=np.float32).reshape(-1)
    return np.asarray(memory["base_summary"][idx], dtype=np.float32).reshape(-1)


def _future_latent(memory: dict[str, np.ndarray], idx: int) -> np.ndarray:
    return np.asarray(memory["target_summary_exec"][idx], dtype=np.float32).reshape(-1)


def _action_latent(memory: dict[str, np.ndarray], idx: int) -> np.ndarray:
    return np.asarray(memory["action_chunk"][idx], dtype=np.float32).reshape(-1)


def _proxy_return(success: bool, subtask_index: int, horizon: int) -> float:
    if not success:
        return 0.0
    return float(max(horizon - subtask_index, 0))


def build_pairs(
    memory: dict[str, np.ndarray],
    horizon: int,
    same_stage_only: bool,
    min_return_gap: float,
    top_k_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(memory["success"])):
        by_task[str(memory["task"][idx])].append(idx)

    state_cache = [_state_feature(memory, i) for i in range(len(memory["success"]))]
    future_cache = [_future_latent(memory, i) for i in range(len(memory["success"]))]
    action_cache = [_action_latent(memory, i) for i in range(len(memory["success"]))]

    pairs: list[dict[str, Any]] = []
    neg_count = 0
    task_counter: Counter[str] = Counter()
    gap_counter: list[float] = []

    for task, task_indices in by_task.items():
        positives_by_stage: dict[int, list[int]] = defaultdict(list)
        positives_all: list[int] = []
        negatives: list[int] = []
        for idx in task_indices:
            success = bool(memory["success"][idx] > 0.5)
            stage = _to_int(memory["subtask_index"][idx])
            if success:
                positives_by_stage[stage].append(idx)
                positives_all.append(idx)
            else:
                negatives.append(idx)
        for neg_idx in negatives:
            neg_count += 1
            stage = _to_int(memory["subtask_index"][neg_idx])
            candidate_ids = positives_by_stage.get(stage, []) if same_stage_only else positives_all
            if not candidate_ids:
                continue
            neg_state = state_cache[neg_idx]
            dists: list[tuple[float, int]] = []
            for pos_idx in candidate_ids:
                pos_state = state_cache[pos_idx]
                if pos_state.shape != neg_state.shape:
                    continue
                dist = float(np.linalg.norm(neg_state - pos_state))
                dists.append((dist, pos_idx))
            if not dists:
                continue
            dists.sort(key=lambda x: x[0])
            shortlist = dists[: max(1, top_k_state)]
            best_pos_idx = None
            best_gap = None
            best_dist = None
            neg_return = _proxy_return(False, stage, horizon)
            for dist, pos_idx in shortlist:
                pos_stage = _to_int(memory["subtask_index"][pos_idx])
                pos_return = _proxy_return(True, pos_stage, horizon)
                gap = pos_return - neg_return
                if gap < min_return_gap:
                    continue
                if best_pos_idx is None or gap > best_gap or (gap == best_gap and dist < best_dist):
                    best_pos_idx = pos_idx
                    best_gap = gap
                    best_dist = dist
            if best_pos_idx is None:
                continue
            pos_stage = _to_int(memory["subtask_index"][best_pos_idx])
            pos_return = _proxy_return(True, pos_stage, horizon)
            pair = {
                "task": task,
                "stage": stage,
                "sequence_index_neg": _to_int(memory["sequence_index"][neg_idx]),
                "sequence_index_pos": _to_int(memory["sequence_index"][best_pos_idx]),
                "memory_index_neg": neg_idx,
                "memory_index_pos": best_pos_idx,
                "state_neg": neg_state.astype(np.float32).tolist(),
                "future_neg": future_cache[neg_idx].astype(np.float32).tolist(),
                "action_neg": action_cache[neg_idx].astype(np.float32).tolist(),
                "state_pos": state_cache[best_pos_idx].astype(np.float32).tolist(),
                "future_pos": future_cache[best_pos_idx].astype(np.float32).tolist(),
                "action_pos": action_cache[best_pos_idx].astype(np.float32).tolist(),
                "g_neg": float(neg_return),
                "g_pos": float(pos_return),
                "return_gap": float(pos_return - neg_return),
                "state_distance": float(best_dist),
                "neg_factor": str(memory["factor"][neg_idx]) if "factor" in memory else "",
                "pos_factor": str(memory["factor"][best_pos_idx]) if "factor" in memory else "",
                "neg_outcome": str(memory["outcome"][neg_idx]) if "outcome" in memory else "",
                "pos_outcome": str(memory["outcome"][best_pos_idx]) if "outcome" in memory else "",
            }
            pairs.append(pair)
            task_counter[task] += 1
            gap_counter.append(float(pos_return - neg_return))

    summary = {
        "num_memory_rows": int(len(memory["success"])),
        "num_negative_rows": int(neg_count),
        "num_pairs": int(len(pairs)),
        "horizon": int(horizon),
        "same_stage_only": bool(same_stage_only),
        "min_return_gap": float(min_return_gap),
        "top_k_state": int(top_k_state),
        "task_counts": dict(task_counter.most_common()),
        "return_gap_mean": float(np.mean(gap_counter)) if gap_counter else 0.0,
        "return_gap_p50": float(np.median(gap_counter)) if gap_counter else 0.0,
    }
    return pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build h-/h+/return-gap pairs for joint hypothesis repair.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--same-stage-only", action="store_true")
    parser.add_argument("--min-return-gap", type=float, default=1.0)
    parser.add_argument("--top-k-state", type=int, default=8)
    args = parser.parse_args()

    memory = load_memory(args.memory_npz)
    pairs, summary = build_pairs(
        memory=memory,
        horizon=int(args.horizon),
        same_stage_only=bool(args.same_stage_only),
        min_return_gap=float(args.min_return_gap),
        top_k_state=int(args.top_k_state),
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for row in pairs:
            handle.write(json.dumps(row) + "\n")
    summary_path = args.output_jsonl.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_jsonl": str(args.output_jsonl), **summary}, indent=2))


if __name__ == "__main__":
    main()
