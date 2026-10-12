from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from policy_evaluation.train_joint_future_action_energy import hashed_task
except ModuleNotFoundError:
    from train_joint_future_action_energy import hashed_task


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _resolve_feature_path(rows_path: Path, feature_path: str) -> Path:
    path = Path(feature_path)
    return path if path.is_absolute() else rows_path.parent / path


def _pool_summary(arr: np.ndarray, out_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == out_dim:
        return arr.astype(np.float32)
    if arr.size < out_dim:
        out = np.zeros((out_dim,), dtype=np.float32)
        out[: arr.size] = arr
        return out
    trim = (arr.size // out_dim) * out_dim
    return arr[:trim].reshape(out_dim, trim // out_dim).mean(axis=1).astype(np.float32)


def _coerce_state(arr: np.ndarray, state_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    out = np.zeros((state_dim,), dtype=np.float32)
    n = min(state_dim, arr.size)
    out[:n] = arr[:n]
    return out


def _coerce_chunk(arr: np.ndarray, chunk_len: int, action_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.zeros((chunk_len, action_dim), dtype=np.float32)
    if arr.size == 0:
        return out
    arr = arr.reshape(-1, action_dim)
    n = min(chunk_len, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _segments(mask: np.ndarray) -> int:
    count = 0
    active = False
    for value in mask.astype(bool).reshape(-1):
        if value and not active:
            count += 1
            active = True
        elif not value:
            active = False
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DirectionalSensitivityTCN memory from finite-difference teacher rows.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=39)
    parser.add_argument("--summary-dim", type=int, default=1024)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--target-delta-clip", type=float, default=0.003)
    parser.add_argument("--min-delta-norm", type=float, default=0.0)
    parser.add_argument("--affected-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    if args.summary_dim % 2 != 0:
        raise ValueError("--summary-dim must be even for pooled z_f/z_a concatenation")
    rows = _read_jsonl(args.rows_jsonl)
    half_dim = args.summary_dim // 2

    out: dict[str, list[Any]] = {
        "state_start": [],
        "state_end": [],
        "base_summary_exec": [],
        "target_summary_exec": [],
        "target_delta_exec": [],
        "action_chunk": [],
        "action_delta_target": [],
        "task_vec": [],
        "success": [],
        "task": [],
        "lang_text": [],
        "sequence_index": [],
        "subtask_index": [],
        "cut_idx": [],
        "remaining_steps": [],
        "full_chunk_len": [],
        "row_id": [],
        "executed_action": [],
        "mode_label": [],
        "teacher_step_scores": [],
        "source_feature_path": [],
    }
    skipped = 0
    delta_norms: list[float] = []
    affected_fracs: list[float] = []
    segment_counts: list[int] = []

    for row in rows:
        feature = _resolve_feature_path(args.rows_jsonl, str(row.get("feature_path", "")))
        if not feature.exists():
            skipped += 1
            continue
        with np.load(feature, allow_pickle=True) as data:
            if "teacher_suffix_delta_action" not in data or "teacher_suffix_action" not in data:
                skipped += 1
                continue
            action = _coerce_chunk(data["teacher_suffix_action"], args.chunk_len, args.action_dim)
            delta = _coerce_chunk(data["teacher_suffix_delta_action"], args.chunk_len, args.action_dim)
            if args.target_delta_clip > 0:
                delta = np.clip(delta, -args.target_delta_clip, args.target_delta_clip).astype(np.float32)
            delta_norm = float(np.linalg.norm(delta.reshape(-1), ord=2))
            if delta_norm < args.min_delta_norm:
                skipped += 1
                continue

            h_future = _pool_summary(data.get("h_future_before", np.zeros((0,), dtype=np.float32)), half_dim)
            h_action = _pool_summary(data.get("h_action_before", np.zeros((0,), dtype=np.float32)), half_dim)
            base_h = np.concatenate([h_future, h_action], axis=0).astype(np.float32)
            delta_h = _pool_summary(data.get("delta_h", np.zeros((0,), dtype=np.float32)), args.summary_dim)
            target_h = (base_h + delta_h).astype(np.float32)
            state_start = _coerce_state(data.get("state_before", np.zeros((0,), dtype=np.float32)), args.state_dim)
            state_end = _coerce_state(data.get("state_after", np.zeros((0,), dtype=np.float32)), args.state_dim)
            executed_action = np.asarray(data.get("executed_action", action[:1]), dtype=np.float32).reshape(-1, args.action_dim)[0]

            scores = np.linalg.norm(delta, axis=-1).astype(np.float32)
            if "teacher_step_scores" in data and np.asarray(data["teacher_step_scores"]).size:
                raw = np.asarray(data["teacher_step_scores"], dtype=np.float32).reshape(-1)
                n = min(args.chunk_len, raw.size)
                scores[:n] = raw[:n]

            arrays = [state_start, state_end, executed_action, base_h, target_h, delta_h, action, delta, scores]
            if not all(np.isfinite(x).all() for x in arrays):
                skipped += 1
                continue

            task = str(row.get("task", "unknown"))
            subtask_index = int(row.get("subtask_index", 0) or 0)
            affected = scores > float(args.affected_threshold)
            affected_fracs.append(float(affected.mean()))
            segment_counts.append(_segments(affected))
            delta_norms.append(delta_norm)

            out["state_start"].append(state_start)
            out["state_end"].append(state_end)
            out["base_summary_exec"].append(base_h)
            out["target_summary_exec"].append(target_h)
            out["target_delta_exec"].append(delta_h)
            out["action_chunk"].append(action)
            out["action_delta_target"].append(delta)
            out["task_vec"].append(hashed_task(task, args.task_dim).astype(np.float32))
            out["success"].append(float(row.get("success", 1.0) if row.get("success", None) is not None else 1.0))
            out["task"].append(task)
            out["lang_text"].append(str(row.get("lang_text", task)))
            out["sequence_index"].append(int(row.get("sequence_index", len(out["row_id"]))))
            out["subtask_index"].append(subtask_index)
            out["cut_idx"].append(int(row.get("step", 0) or 0))
            out["remaining_steps"].append(int(max(0, args.chunk_len - 1)))
            out["full_chunk_len"].append(int(args.chunk_len))
            out["row_id"].append(int(row.get("row_id", len(out["row_id"]))))
            out["executed_action"].append(executed_action)
            out["mode_label"].append(int(1 if affected.any() else 0))
            out["teacher_step_scores"].append(scores)
            out["source_feature_path"].append(str(feature))

            if args.max_rows > 0 and len(out["row_id"]) >= args.max_rows:
                break

    if not out["row_id"]:
        raise ValueError("no valid teacher rows found")

    arrays_out: dict[str, np.ndarray] = {}
    numeric_keys = {
        "state_start",
        "state_end",
        "base_summary_exec",
        "target_summary_exec",
        "target_delta_exec",
        "action_chunk",
        "action_delta_target",
        "task_vec",
        "success",
        "sequence_index",
        "subtask_index",
        "cut_idx",
        "remaining_steps",
        "full_chunk_len",
        "row_id",
        "executed_action",
        "mode_label",
        "teacher_step_scores",
    }
    for key, values in out.items():
        if key in numeric_keys:
            arrays_out[key] = np.asarray(values)
        else:
            arrays_out[key] = np.asarray(values, dtype=object)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays_out)

    delta_arr = np.asarray(delta_norms, dtype=np.float32)
    summary = {
        "rows_jsonl": str(args.rows_jsonl),
        "output_npz": str(args.output_npz),
        "num_rows": int(len(out["row_id"])),
        "num_skipped": int(skipped),
        "state_dim": int(args.state_dim),
        "summary_dim": int(args.summary_dim),
        "chunk_len": int(args.chunk_len),
        "action_dim": int(args.action_dim),
        "target_delta_clip": float(args.target_delta_clip),
        "affected_threshold": float(args.affected_threshold),
        "mode_counts": {str(k): int(v) for k, v in zip(*np.unique(arrays_out["mode_label"], return_counts=True))},
        "delta_norm_min_p50_p90_p99_max": [
            float(np.min(delta_arr)),
            float(np.quantile(delta_arr, 0.50)),
            float(np.quantile(delta_arr, 0.90)),
            float(np.quantile(delta_arr, 0.99)),
            float(np.max(delta_arr)),
        ],
        "affected_fraction_mean": float(np.mean(affected_fracs)),
        "affected_segments_mean": float(np.mean(segment_counts)),
    }
    summary_path = args.summary_json or args.output_npz.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
