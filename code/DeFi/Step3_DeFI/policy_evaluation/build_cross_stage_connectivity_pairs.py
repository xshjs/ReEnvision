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
        "state_start",
        "target_summary_exec",
        "action_chunk",
        "task",
        "sequence_index",
        "subtask_index",
        "success",
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


def _task_index(memory: dict[str, np.ndarray]) -> dict[tuple[int, int], list[int]]:
    by_seq_stage: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(len(memory["success"])):
        seq = _to_int(memory["sequence_index"][idx])
        stage = _to_int(memory["subtask_index"][idx])
        by_seq_stage[(seq, stage)].append(idx)
    return by_seq_stage


def _flatten(memory: dict[str, np.ndarray], key: str, idx: int) -> np.ndarray:
    return np.asarray(memory[key][idx], dtype=np.float32).reshape(-1)


def _find_positive_next(
    memory: dict[str, np.ndarray],
    by_seq_stage: dict[tuple[int, int], list[int]],
    idx: int,
) -> int | None:
    seq = _to_int(memory["sequence_index"][idx])
    stage = _to_int(memory["subtask_index"][idx])
    next_candidates = by_seq_stage.get((seq, stage + 1), [])
    for cand in next_candidates:
        if bool(memory["success"][cand] > 0.5):
            return cand
    return None


def _find_negative_next(
    memory: dict[str, np.ndarray],
    next_pos_idx: int,
    all_next_rows: list[int],
) -> int | None:
    pos_task = str(memory["task"][next_pos_idx])
    pos_state = _flatten(memory, "state_start", next_pos_idx)
    best_idx = None
    best_dist = None
    for cand in all_next_rows:
        if cand == next_pos_idx:
            continue
        if str(memory["task"][cand]) != pos_task:
            continue
        cand_state = _flatten(memory, "state_start", cand)
        if cand_state.shape != pos_state.shape:
            continue
        dist = float(np.linalg.norm(cand_state - pos_state))
        if best_idx is None or dist < best_dist:
            best_idx = cand
            best_dist = dist
    if best_idx is not None:
        return best_idx
    for cand in all_next_rows:
        if cand != next_pos_idx:
            return cand
    return None


def build_pairs(memory: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_seq_stage = _task_index(memory)
    current_rows = [
        idx
        for idx in range(len(memory["success"]))
        if bool(memory["success"][idx] > 0.5)
    ]
    next_rows_by_stage: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(memory["success"])):
        next_rows_by_stage[_to_int(memory["subtask_index"][idx])].append(idx)

    pairs: list[dict[str, Any]] = []
    task_counter: Counter[str] = Counter()
    gap_counter: list[float] = []

    for idx in current_rows:
        next_pos_idx = _find_positive_next(memory, by_seq_stage, idx)
        if next_pos_idx is None:
            continue
        next_stage = _to_int(memory["subtask_index"][idx]) + 1
        next_neg_idx = _find_negative_next(memory, next_pos_idx, next_rows_by_stage.get(next_stage, []))
        if next_neg_idx is None:
            continue

        cur_state = _flatten(memory, "state_start", idx)
        cur_future = _flatten(memory, "target_summary_exec", idx)
        cur_action = _flatten(memory, "action_chunk", idx)

        next_pos_state = _flatten(memory, "state_start", next_pos_idx)
        next_pos_future = _flatten(memory, "target_summary_exec", next_pos_idx)
        next_pos_action = _flatten(memory, "action_chunk", next_pos_idx)

        next_neg_state = _flatten(memory, "state_start", next_neg_idx)
        next_neg_future = _flatten(memory, "target_summary_exec", next_neg_idx)
        next_neg_action = _flatten(memory, "action_chunk", next_neg_idx)

        pos_dist = float(np.linalg.norm(next_pos_state - cur_state))
        neg_dist = float(np.linalg.norm(next_neg_state - cur_state))

        row = {
            "task": str(memory["task"][idx]),
            "stage": _to_int(memory["subtask_index"][idx]),
            "sequence_index": _to_int(memory["sequence_index"][idx]),
            "current_index": int(idx),
            "next_pos_index": int(next_pos_idx),
            "next_neg_index": int(next_neg_idx),
            "current_state": cur_state.astype(np.float32).tolist(),
            "current_future": cur_future.astype(np.float32).tolist(),
            "current_action": cur_action.astype(np.float32).tolist(),
            "next_state_pos": next_pos_state.astype(np.float32).tolist(),
            "next_future_pos": next_pos_future.astype(np.float32).tolist(),
            "next_action_pos": next_pos_action.astype(np.float32).tolist(),
            "next_state_neg": next_neg_state.astype(np.float32).tolist(),
            "next_future_neg": next_neg_future.astype(np.float32).tolist(),
            "next_action_neg": next_neg_action.astype(np.float32).tolist(),
            "next_task_pos": str(memory["task"][next_pos_idx]),
            "next_task_neg": str(memory["task"][next_neg_idx]),
            "next_success_pos": float(memory["success"][next_pos_idx]),
            "next_success_neg": float(memory["success"][next_neg_idx]),
            "pos_state_distance": pos_dist,
            "neg_state_distance": neg_dist,
            "distance_margin": float(neg_dist - pos_dist),
        }
        pairs.append(row)
        task_counter[str(memory["task"][idx])] += 1
        gap_counter.append(float(neg_dist - pos_dist))

    summary = {
        "num_memory_rows": int(len(memory["success"])),
        "num_pairs": int(len(pairs)),
        "task_counts": dict(task_counter.most_common()),
        "distance_margin_mean": float(np.mean(gap_counter)) if gap_counter else 0.0,
        "distance_margin_p50": float(np.median(gap_counter)) if gap_counter else 0.0,
    }
    return pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-stage manifold connectivity pairs.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    memory = load_memory(args.memory_npz)
    pairs, summary = build_pairs(memory)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row) + "\n")
    args.output_jsonl.with_name("summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output_jsonl": str(args.output_jsonl), **summary}, indent=2))


if __name__ == "__main__":
    main()
