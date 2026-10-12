from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def future_trace_from_row(row: dict[str, Any]) -> Path | None:
    for key in ("trace_path", "collection_trace_path", "target_proxy_trace_path"):
        raw = str(row.get(key, "") or "")
        if raw and Path(raw).exists():
            return Path(raw)
    return None


def action_trace_from_future(future_trace: Path, action_trace_dirs: list[Path]) -> Path | None:
    action_name = future_trace.name.replace("future_trace_", "action_trace_")
    for trace_dir in [future_trace.parent.parent / "action_traces", *action_trace_dirs]:
        candidate = trace_dir / action_name
        if candidate.exists():
            return candidate
    return None


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "target_proxy_future" and "final_future" in data:
                arr = np.asarray(data["final_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def summarize_future(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[-1] - arr[0]], axis=0).astype(np.float32)


def summarize_future_exec(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def chunk_starts(length: int, chunk_len: int, chunks_per_trace: int) -> list[int]:
    if length <= chunk_len:
        return [0]
    max_start = length - chunk_len
    if chunks_per_trace <= 1:
        return [max_start]
    anchors = [0, max_start // 2, max_start]
    starts = sorted({min(max_start, max(0, a)) for a in anchors[:chunks_per_trace]})
    return starts[:chunks_per_trace]


def pad_chunk(actions: np.ndarray, chunk_len: int) -> np.ndarray:
    out = np.zeros((chunk_len, actions.shape[-1]), dtype=np.float32)
    take = min(chunk_len, actions.shape[0])
    out[:take] = actions[:take]
    return out


def outcome_from_row(success: bool, remaining_steps: int, horizon: int) -> str:
    if success:
        return "success_within_horizon" if remaining_steps <= horizon else "success_after_horizon"
    return "failure_within_horizon" if remaining_steps <= horizon else "failure_after_horizon"


def factor_from_row(row: dict[str, Any]) -> str:
    if bool(row.get("success", False)):
        return "none"
    for key in ("factor_label", "wrong_component", "failure_factor", "selected_failure_factor", "applied_factor_mask"):
        value = str(row.get(key, "") or "")
        if value:
            return value
    return "unknown"


def row_vec(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        return None
    try:
        return np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal future-action pair memory from factorized rows.")
    parser.add_argument("--dataset-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--action-trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--base-future-key", default="base_future")
    parser.add_argument("--target-future-key", default="final_future")
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--chunks-per-trace", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    items = []
    task_counts = Counter()
    outcome_counts = Counter()
    factor_counts = Counter()
    written = 0
    skipped = 0

    for dataset in args.dataset_jsonl:
        for row in iter_jsonl(dataset):
            future_trace = future_trace_from_row(row)
            if future_trace is None:
                skipped += 1
                continue
            action_trace = action_trace_from_future(future_trace, list(args.action_trace_dir))
            if action_trace is None:
                skipped += 1
                continue
            success = bool(row.get("success", False))
            if args.success_only and not success:
                continue
            base = load_future(future_trace, args.base_future_key)
            target = load_future(future_trace, args.target_future_key)
            if base is None or target is None:
                skipped += 1
                continue
            try:
                with np.load(action_trace) as data:
                    actions = np.asarray(data["actions"], dtype=np.float32)
            except Exception:
                skipped += 1
                continue
            if actions.ndim != 2 or actions.shape[0] <= 0:
                skipped += 1
                continue
            starts = chunk_starts(actions.shape[0], args.chunk_len, args.chunks_per_trace)
            base_summary = summarize_future(base)
            target_summary = summarize_future(target)
            target_delta = (target_summary - base_summary).astype(np.float32)
            base_summary_exec = summarize_future_exec(base)
            target_summary_exec = summarize_future_exec(target)
            target_delta_exec = (target_summary_exec - base_summary_exec).astype(np.float32)
            task = str(row.get("task", "unknown"))
            factor = factor_from_row(row)
            robot_start = row_vec(row, "strong_gt_robot_obs_start")
            robot_end = row_vec(row, "strong_gt_robot_obs_end")
            scene_start = row_vec(row, "strong_gt_scene_obs_start")
            scene_end = row_vec(row, "strong_gt_scene_obs_end")
            state_start = None
            state_end = None
            state_transition = None
            if robot_start is not None and robot_end is not None and scene_start is not None and scene_end is not None:
                state_start = np.concatenate([robot_start, scene_start], axis=0).astype(np.float32)
                state_end = np.concatenate([robot_end, scene_end], axis=0).astype(np.float32)
                state_transition = (state_end - state_start).astype(np.float32)
            for start in starts:
                remaining_steps = max(actions.shape[0] - start, 1)
                outcome = outcome_from_row(success, remaining_steps, args.horizon)
                items.append(
                    {
                        "task": task,
                        "success": success,
                        "factor": factor,
                        "outcome": outcome,
                        "sequence_index": int(row.get("sequence_index", written) or 0),
                        "subtask_index": int(row.get("subtask_index", 0) or 0),
                        "trace_path": str(future_trace),
                        "action_trace_path": str(action_trace),
                        "chunk_start": int(start),
                        "base_summary": base_summary,
                        "target_summary": target_summary,
                        "target_delta": target_delta,
                        "base_summary_exec": base_summary_exec,
                        "target_summary_exec": target_summary_exec,
                        "target_delta_exec": target_delta_exec,
                        "action_chunk": pad_chunk(actions[start : start + args.chunk_len], args.chunk_len),
                        "state_start": state_start,
                        "state_end": state_end,
                        "state_transition": state_transition,
                    }
                )
            task_counts[task] += 1
            factor_counts[factor] += 1
            written += 1
            if args.max_rows > 0 and written >= args.max_rows:
                break
        if args.max_rows > 0 and written >= args.max_rows:
            break

    if not items:
        raise ValueError("no memory items built")

    for item in items:
        outcome_counts[item["outcome"]] += 1

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    has_state = any(item["state_start"] is not None and item["state_transition"] is not None for item in items)
    save_kwargs: dict[str, Any] = dict(
        base_summary=np.stack([item["base_summary"] for item in items]).astype(np.float32),
        target_summary=np.stack([item["target_summary"] for item in items]).astype(np.float32),
        target_delta=np.stack([item["target_delta"] for item in items]).astype(np.float32),
        base_summary_exec=np.stack([item["base_summary_exec"] for item in items]).astype(np.float32),
        target_summary_exec=np.stack([item["target_summary_exec"] for item in items]).astype(np.float32),
        target_delta_exec=np.stack([item["target_delta_exec"] for item in items]).astype(np.float32),
        action_chunk=np.stack([item["action_chunk"] for item in items]).astype(np.float32),
        success=np.asarray([1.0 if item["success"] else 0.0 for item in items], dtype=np.float32),
        task=np.asarray([item["task"] for item in items], dtype=object),
        factor=np.asarray([item["factor"] for item in items], dtype=object),
        outcome=np.asarray([item["outcome"] for item in items], dtype=object),
        sequence_index=np.asarray([item["sequence_index"] for item in items], dtype=np.int32),
        subtask_index=np.asarray([item["subtask_index"] for item in items], dtype=np.int32),
        trace_path=np.asarray([item["trace_path"] for item in items], dtype=object),
        action_trace_path=np.asarray([item["action_trace_path"] for item in items], dtype=object),
        chunk_start=np.asarray([item["chunk_start"] for item in items], dtype=np.int32),
    )
    if has_state:
        first_state_dim = next(len(item["state_start"]) for item in items if item["state_start"] is not None)
        zero = np.zeros((first_state_dim,), dtype=np.float32)
        save_kwargs["state_start"] = np.stack([(item["state_start"] if item["state_start"] is not None else zero) for item in items]).astype(np.float32)
        save_kwargs["state_end"] = np.stack([(item["state_end"] if item["state_end"] is not None else zero) for item in items]).astype(np.float32)
        save_kwargs["state_transition"] = np.stack([(item["state_transition"] if item["state_transition"] is not None else zero) for item in items]).astype(np.float32)
        save_kwargs["has_state"] = np.asarray(
            [1.0 if item["state_start"] is not None and item["state_transition"] is not None else 0.0 for item in items],
            dtype=np.float32,
        )
    np.savez_compressed(args.output_npz, **save_kwargs)

    summary = {
        "dataset_jsonl": [str(p) for p in args.dataset_jsonl],
        "action_trace_dirs": [str(p) for p in args.action_trace_dir],
        "output_npz": str(args.output_npz),
        "num_items": len(items),
        "source_rows_written": written,
        "skipped": skipped,
        "success_only": bool(args.success_only),
        "has_state": bool(has_state),
        "task_counts": dict(task_counts.most_common()),
        "factor_counts": dict(factor_counts.most_common()),
        "outcome_counts": dict(outcome_counts.most_common()),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
