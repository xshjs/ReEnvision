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
        "state_transition",
        "base_summary_exec",
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


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _group_key(memory: dict[str, np.ndarray], idx: int) -> tuple[str, int, int, str]:
    trace_path = ""
    if "action_trace_path" in memory:
        trace_path = _to_str(memory["action_trace_path"][idx])
    elif "trace_path" in memory:
        trace_path = _to_str(memory["trace_path"][idx])
    return (
        _to_str(memory["task"][idx]),
        _to_int(memory["sequence_index"][idx], -1),
        _to_int(memory["subtask_index"][idx], -1),
        trace_path,
    )


def _order_value(memory: dict[str, np.ndarray], idx: int) -> tuple[int, int]:
    chunk_start = _to_int(memory["chunk_start"][idx], idx) if "chunk_start" in memory else idx
    return (chunk_start, idx)

def _unit_direction(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def build_rows(
    memory: dict[str, np.ndarray],
    history_len: int,
    target_mode: str,
    min_progress_norm: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[tuple[str, int, int, str], list[int]] = defaultdict(list)
    for idx in range(len(memory["success"])):
        by_group[_group_key(memory, idx)].append(idx)

    rows: list[dict[str, Any]] = []
    task_counter: Counter[str] = Counter()
    hist_counter: Counter[int] = Counter()
    progress_norms: list[float] = []
    skipped_small_progress = 0

    for (task, seq_idx, stage_idx, trace_path), ids in by_group.items():
        ids = sorted(ids, key=lambda i: _order_value(memory, i))
        if len(ids) <= 1:
            continue
        group_target_state = np.asarray(memory["state_end"][ids[-1]], dtype=np.float32).reshape(-1) if "state_end" in memory else None
        for pos in range(1, len(ids)):
            cur_idx = ids[pos]
            prev_idx = ids[pos - 1]
            hist_ids = ids[max(0, pos - history_len) : pos]
            if not hist_ids:
                continue

            state_t = np.asarray(memory["state_start"][cur_idx], dtype=np.float32).reshape(-1)
            base_t = np.asarray(memory["base_summary_exec"][cur_idx], dtype=np.float32).reshape(-1)
            target_t = np.asarray(memory["target_summary_exec"][cur_idx], dtype=np.float32).reshape(-1)
            base_prev = np.asarray(memory["base_summary_exec"][prev_idx], dtype=np.float32).reshape(-1)
            state_transition = np.asarray(memory["state_transition"][cur_idx], dtype=np.float32).reshape(-1)
            observed_future_effect = (base_t - base_prev).astype(np.float32)
            residual_to_target = (target_t - base_t).astype(np.float32)
            residual_before = (target_t - base_prev).astype(np.float32)
            # First decomposition pass:
            # observed_effect: what has already materialized between prev and current future summary
            # pending_effect: what is still missing relative to the current target future
            # residual_before: pre-update gap, useful for debugging and future multi-term training
            pending_effect = residual_to_target.astype(np.float32)

            if target_mode == "state_transition":
                future_progress = state_transition.astype(np.float32)
                progress_norm = float(np.linalg.norm(future_progress))
            elif target_mode == "residual_reduction":
                if group_target_state is None:
                    raise ValueError("target_mode=residual_reduction requires memory key `state_end`")
                residual_dir = _unit_direction(group_target_state - state_t)
                scalar_progress = float(np.dot(state_transition, residual_dir))
                future_progress = np.asarray([scalar_progress], dtype=np.float32)
                progress_norm = float(np.linalg.norm(future_progress))
            elif target_mode == "future_residual_reduction":
                prev_residual = target_t - base_prev
                cur_residual = target_t - base_t
                future_progress = (prev_residual - cur_residual).astype(np.float32)
                progress_norm = float(np.linalg.norm(future_progress))
            else:
                raise ValueError(f"unsupported target_mode: {target_mode}")
            if progress_norm <= float(min_progress_norm):
                skipped_small_progress += 1
                continue

            history_actions = np.zeros((history_len,) + tuple(np.asarray(memory["action_chunk"][cur_idx]).shape), dtype=np.float32)
            history_mask = np.zeros((history_len,), dtype=np.float32)
            offset = history_len - len(hist_ids)
            for j, hist_idx in enumerate(hist_ids):
                history_actions[offset + j] = np.asarray(memory["action_chunk"][hist_idx], dtype=np.float32)
                history_mask[offset + j] = 1.0

            row = {
                "task": task,
                "sequence_index": int(seq_idx),
                "subtask_index": int(stage_idx),
                "trace_path": trace_path,
                "current_index": int(cur_idx),
                "prev_index": int(prev_idx),
                "history_indices": [int(x) for x in hist_ids],
                "state_t": state_t.astype(np.float32).tolist(),
                "base_t": base_t.astype(np.float32).tolist(),
                "base_prev_t": base_prev.astype(np.float32).tolist(),
                "target_t": target_t.astype(np.float32).tolist(),
                "future_progress": future_progress.astype(np.float32).tolist(),
                "observed_effect": observed_future_effect.astype(np.float32).tolist(),
                "pending_effect": pending_effect.astype(np.float32).tolist(),
                "residual_to_target": residual_to_target.astype(np.float32).tolist(),
                "residual_before": residual_before.astype(np.float32).tolist(),
                "state_transition": state_transition.astype(np.float32).tolist(),
                "history_actions": history_actions.astype(np.float32).tolist(),
                "history_mask": history_mask.astype(np.float32).tolist(),
                "history_count": int(len(hist_ids)),
                "success": bool(memory["success"][cur_idx] > 0.5),
            }
            rows.append(row)
            task_counter[task] += 1
            hist_counter[len(hist_ids)] += 1
            progress_norms.append(float(np.linalg.norm(future_progress)))

    summary = {
        "num_memory_rows": int(len(memory["success"])),
        "num_groups": int(len(by_group)),
        "num_rows": int(len(rows)),
        "history_len": int(history_len),
        "target_mode": str(target_mode),
        "decomposition_fields": [
            "observed_effect",
            "pending_effect",
            "residual_to_target",
            "residual_before",
        ],
        "min_progress_norm": float(min_progress_norm),
        "skipped_small_progress": int(skipped_small_progress),
        "task_counts": dict(task_counter.most_common()),
        "history_count_histogram": {str(k): int(v) for k, v in sorted(hist_counter.items())},
        "future_progress_norm_mean": float(np.mean(progress_norms)) if progress_norms else 0.0,
        "future_progress_norm_p50": float(np.median(progress_norms)) if progress_norms else 0.0,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset for dynamic coupling operator Γ_t.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--target-mode", choices=["state_transition", "residual_reduction", "future_residual_reduction"], default="state_transition")
    parser.add_argument("--min-progress-norm", type=float, default=0.0)
    args = parser.parse_args()

    memory = load_memory(args.memory_npz)
    rows, summary = build_rows(memory, int(args.history_len), str(args.target_mode), float(args.min_progress_norm))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    args.output_jsonl.with_name("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_jsonl": str(args.output_jsonl), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
