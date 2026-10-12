from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def task_hash_vec(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    for i, b in enumerate(digest):
        vec[(int(b) + i * 13) % dim] += 1.0
    return vec / max(float(np.linalg.norm(vec)), 1e-6)


def pool_flat(x: np.ndarray, out_dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == out_dim:
        return arr
    if arr.size < out_dim:
        out = np.zeros(out_dim, dtype=np.float32)
        out[: arr.size] = arr
        return out
    trimmed = arr[: (arr.size // out_dim) * out_dim]
    return trimmed.reshape(out_dim, -1).mean(axis=1).astype(np.float32)


def pool_joint_belief(z_future: np.ndarray, z_action: np.ndarray, out_dim: int) -> np.ndarray:
    if np.asarray(z_action).size == 0:
        raise ValueError("missing z_a/action-intent latent; collect with --joint-pair-action-generator-ckpt")
    future_dim = max(1, out_dim // 2)
    action_dim = out_dim - future_dim
    return np.concatenate(
        [
            pool_flat(z_future, future_dim),
            pool_flat(z_action, action_dim),
        ],
        axis=0,
    ).astype(np.float32)


def load_success_map(log_dir: Path) -> dict[tuple[int, int, str], float]:
    out: dict[tuple[int, int, str], float] = {}
    path = log_dir / "memory_rollout_rows.jsonl"
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row.get("sequence_index", -1)), int(row.get("subtask_index", -1)), str(row.get("task", "")))
            out[key] = 1.0 if bool(row.get("success", False)) else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact T/D memory from collected joint_belief_transition rows.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-dim", type=int, default=1024)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--keep-delta-norm", type=float, default=0.02)
    parser.add_argument("--replan-delta-norm", type=float, default=8.5)
    args = parser.parse_args()

    rows_path = args.log_dir / "joint_belief_transition_rows.jsonl"
    feature_root = args.log_dir
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)

    success_map = load_success_map(args.log_dir)
    lines = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_rows and len(lines) > args.max_rows:
        rng = np.random.default_rng(args.sample_seed)
        ids = np.sort(rng.choice(np.arange(len(lines)), size=int(args.max_rows), replace=False))
        lines = [lines[int(i)] for i in ids]
    state_start, state_end, base, target, action, action_delta, executed_action = [], [], [], [], [], [], []
    success, task_vec, task, lang_text = [], [], [], []
    sequence_index, subtask_index, cut_idx, remaining_steps, full_chunk_len = [], [], [], [], []
    mode_label, mode_source = [], []
    skipped = 0
    for row_id, line in enumerate(lines):
            row = json.loads(line)
            path = feature_root / str(row["feature_path"])
            try:
                data = np.load(path, allow_pickle=True)
                s0 = np.asarray(data["state_before"], dtype=np.float32).reshape(-1)
                s1 = np.asarray(data["state_after"], dtype=np.float32).reshape(-1)
                at = np.asarray(data["action"], dtype=np.float32).reshape(-1)
                hf0 = pool_joint_belief(data["h_future_before"], data["h_action_before"], args.summary_dim)
                hf1 = pool_joint_belief(data["h_future_after"], data["h_action_after"], args.summary_dim)
                ar0 = np.asarray(data["action_remain_before"], dtype=np.float32).reshape(-1, 7)
                ar1 = np.asarray(data["action_remain_after"], dtype=np.float32).reshape(-1, 7)
                if ar0.shape[0] < 10:
                    pad = np.zeros((10 - ar0.shape[0], 7), dtype=np.float32)
                    ar0 = np.concatenate([ar0, pad], axis=0)
                if ar1.shape[0] < 10:
                    pad = np.zeros((10 - ar1.shape[0], 7), dtype=np.float32)
                    ar1 = np.concatenate([ar1, pad], axis=0)
                ar0 = ar0[:10]
                ar1 = ar1[:10]
                arrays = [s0, s1, at, hf0, hf1, ar0, ar1]
                if not all(np.isfinite(x).all() for x in arrays):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            seq = int(row.get("sequence_index", -1))
            sub = int(row.get("subtask_index", 0) or 0)
            task_name = str(row.get("task", "unknown"))
            succ = success_map.get((seq, sub, task_name), 0.0)
            state_start.append(s0)
            state_end.append(s1)
            base.append(hf0)
            target.append(hf1)
            action.append(ar0)
            delta_a = ar1 - ar0
            action_delta.append(delta_a)
            executed_action.append(at)
            success.append(succ)
            task_vec.append(task_hash_vec(task_name, args.task_dim))
            task.append(task_name)
            lang_text.append(str(row.get("lang_text", "")))
            sequence_index.append(seq)
            subtask_index.append(sub)
            cut_idx.append(int(row.get("step", 0) or 0) % 10)
            remaining_steps.append(10 - (int(row.get("step", 0) or 0) % 10))
            full_chunk_len.append(10)
            decision = str(
                row.get("dynamic_coupling_chunk_decision")
                or row.get("joint_belief_transition_predicted_mode")
                or ""
            ).lower()
            if decision in {"keep", "0"}:
                mode_label.append(0)
                mode_source.append("runtime_decision")
            elif decision in {"repair_suffix", "patch", "1"}:
                mode_label.append(1)
                mode_source.append("runtime_decision")
            elif decision in {"replan", "2"}:
                mode_label.append(2)
                mode_source.append("runtime_decision")
            else:
                # Baseline rollout has no oracle suffix label. This proxy says:
                # no suffix change -> keep; moderate change -> patch; large change -> replan.
                delta_norm = float(np.linalg.norm(delta_a.reshape(-1)))
                if delta_norm <= float(args.keep_delta_norm):
                    mode_label.append(0)
                elif delta_norm >= float(args.replan_delta_norm):
                    mode_label.append(2)
                else:
                    mode_label.append(1)
                mode_source.append("delta_norm_proxy")

    if not state_start:
        raise ValueError("no valid joint belief rows")
    out = {
        "state_start": np.stack(state_start).astype(np.float32),
        "state_end": np.stack(state_end).astype(np.float32),
        "base_summary_exec": np.stack(base).astype(np.float32),
        "target_summary_exec": np.stack(target).astype(np.float32),
        "target_delta_exec": (np.stack(target) - np.stack(base)).astype(np.float32),
        "action_chunk": np.stack(action).astype(np.float32),
        "action_delta_target": np.stack(action_delta).astype(np.float32),
        "executed_action": np.stack(executed_action).astype(np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "task_vec": np.stack(task_vec).astype(np.float32),
        "task": np.asarray(task, dtype=object),
        "lang_text": np.asarray(lang_text, dtype=object),
        "sequence_index": np.asarray(sequence_index, dtype=np.int32),
        "subtask_index": np.asarray(subtask_index, dtype=np.int32),
        "cut_idx": np.asarray(cut_idx, dtype=np.int32),
        "remaining_steps": np.asarray(remaining_steps, dtype=np.int32),
        "full_chunk_len": np.asarray(full_chunk_len, dtype=np.int32),
        "mode_label": np.asarray(mode_label, dtype=np.int64),
        "mode_source": np.asarray(mode_source, dtype=object),
        "row_id": np.arange(len(state_start), dtype=np.int32),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    summary = {
        "log_dir": str(args.log_dir),
        "output_npz": str(args.output_npz),
        "num_rows": int(len(state_start)),
        "skipped": int(skipped),
        "summary_dim": int(args.summary_dim),
        "state_dim": int(out["state_start"].shape[1]),
        "action_shape": list(out["action_chunk"].shape[1:]),
        "num_success": int((out["success"] > 0.5).sum()),
        "num_failure": int((out["success"] <= 0.5).sum()),
        "mode_counts": {str(k): int(v) for k, v in zip(*np.unique(out["mode_label"], return_counts=True))},
        "mode_source_counts": {str(k): int(v) for k, v in zip(*np.unique(out["mode_source"], return_counts=True))},
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
