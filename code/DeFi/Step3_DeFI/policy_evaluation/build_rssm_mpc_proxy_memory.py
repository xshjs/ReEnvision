from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = [
    "state_start",
    "state_end",
    "base_summary_exec",
    "target_summary_exec",
    "action_chunk",
    "task_vec",
    "sequence_index",
    "subtask_index",
    "cut_idx",
    "remaining_steps",
    "full_chunk_len",
]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def finite_rows(mem: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    n = len(mem[keys[0]])
    mask = np.ones(n, dtype=bool)
    for key in keys:
        arr = np.asarray(mem[key])
        if arr.dtype == object:
            continue
        mask &= np.isfinite(arr.reshape(n, -1)).all(axis=1)
    return mask


def build_index(mem: dict[str, np.ndarray]) -> dict[tuple[int, int], list[int]]:
    seq = np.asarray(mem["sequence_index"]).reshape(-1).astype(int)
    sub = np.asarray(mem["subtask_index"]).reshape(-1).astype(int)
    cut = np.asarray(mem["cut_idx"]).reshape(-1).astype(int)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for idx, key in enumerate(zip(seq, sub)):
        groups.setdefault(key, []).append((int(cut[idx]), idx))
    return {key: [idx for _, idx in sorted(items)] for key, items in groups.items()}


def pad_or_trim(seq: np.ndarray, length: int, tail_value: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[0] >= length:
        return seq[:length].astype(np.float32)
    pad_shape = (length - seq.shape[0],) + tuple(seq.shape[1:])
    pad = np.broadcast_to(np.asarray(tail_value, dtype=np.float32), pad_shape).copy()
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


def smoothness_cost(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[1] < 2:
        return np.zeros((actions.shape[0],), dtype=np.float32)
    diff = np.diff(actions, axis=1)
    return np.mean(diff * diff, axis=(1, 2)).astype(np.float32)


def velocity_cost(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    return np.mean(actions * actions, axis=(1, 2)).astype(np.float32)


def make_mode_from_delta(delta_norm: np.ndarray, keep: float, replan: float) -> np.ndarray:
    mode = np.ones(delta_norm.shape[0], dtype=np.int64)
    mode[delta_norm <= keep] = 0
    mode[delta_norm >= replan] = 2
    return mode


def balanced_indices(mode: np.ndarray, max_per_mode: int, seed: int) -> np.ndarray:
    if max_per_mode <= 0:
        return np.arange(mode.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for value in sorted(int(x) for x in np.unique(mode)):
        ids = np.where(mode == value)[0]
        if ids.size > max_per_mode:
            ids = rng.choice(ids, size=max_per_mode, replace=False)
        keep.append(np.asarray(ids, dtype=np.int64))
    out = np.concatenate(keep) if keep else np.empty((0,), dtype=np.int64)
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RSSM + MPC-proxy memory from existing rollout/factored memory."
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--expected-alpha", type=float, default=0.6)
    parser.add_argument("--target-delta-clip", type=float, default=0.05)
    parser.add_argument("--keep-delta-norm", type=float, default=0.02)
    parser.add_argument("--replan-delta-norm", type=float, default=0.25)
    parser.add_argument("--max-per-mode", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mem = load_npz(args.input_npz)
    missing = [key for key in REQUIRED_KEYS if key not in mem]
    if missing:
        raise ValueError(f"input memory missing keys: {missing}")

    n = len(mem["state_start"])
    good = finite_rows(
        mem,
        [
            "state_start",
            "state_end",
            "base_summary_exec",
            "target_summary_exec",
            "action_chunk",
            "task_vec",
        ],
    )
    groups = build_index(mem)

    h0_all = np.asarray(mem["base_summary_exec"], dtype=np.float32)
    h1_all = np.asarray(mem["target_summary_exec"], dtype=np.float32)
    action_all = np.asarray(mem["action_chunk"], dtype=np.float32)

    state_start = []
    state_end = []
    h_t = []
    h_post = []
    h_delta = []
    h_expected_traj = []
    h_posterior_traj = []
    action_chunk = []
    action_remain = []
    repair_target = []
    repair_delta = []
    task_vec = []
    success = []
    mode = []
    sequence_index = []
    subtask_index = []
    cut_idx = []
    remaining_steps = []
    full_chunk_len = []
    task = []
    lang_text = []
    row_id = []

    alpha = float(np.clip(args.expected_alpha, 0.0, 1.0))
    horizon = max(1, int(args.horizon))
    for group_indices in groups.values():
        pos = {int(np.asarray(mem["cut_idx"])[idx]): idx for idx in group_indices}
        for idx in group_indices:
            if idx < 0 or idx >= n or not good[idx]:
                continue
            c = int(np.asarray(mem["cut_idx"])[idx])
            future_ids = [pos.get(c + k) for k in range(horizon)]
            valid_future = [j for j in future_ids if j is not None and good[j]]

            h0 = h0_all[idx]
            h1 = h1_all[idx]
            local_delta_h = h1 - h0
            exp_seq = []
            post_seq = []
            cur_exp = h0.copy()
            for k in range(horizon):
                # Proxy RSSM prior: continue the observed local belief trend.
                cur_exp = cur_exp + alpha * local_delta_h
                exp_seq.append(cur_exp.copy())
                if future_ids[k] is not None and good[future_ids[k]]:
                    post_seq.append(h1_all[future_ids[k]].copy())
                else:
                    post_seq.append(h1.copy())

            a0 = action_all[idx].copy()
            if "action_delta_target" in mem:
                delta = np.asarray(mem["action_delta_target"], dtype=np.float32)[idx].copy()
            else:
                delta = np.zeros_like(a0, dtype=np.float32)
                if len(valid_future) > 1:
                    delta = action_all[valid_future[min(1, len(valid_future) - 1)]] - a0
            if float(args.target_delta_clip) > 0:
                delta = np.clip(delta, -float(args.target_delta_clip), float(args.target_delta_clip))
            repaired = a0 + delta
            delta_norm = float(np.linalg.norm(delta.reshape(-1), ord=2))

            if "mode_label" in mem:
                mode_i = int(np.asarray(mem["mode_label"]).reshape(-1)[idx])
            else:
                mode_i = int(make_mode_from_delta(np.asarray([delta_norm]), args.keep_delta_norm, args.replan_delta_norm)[0])

            state_start.append(np.asarray(mem["state_start"], dtype=np.float32)[idx])
            state_end.append(np.asarray(mem["state_end"], dtype=np.float32)[idx])
            h_t.append(h0)
            h_post.append(h1)
            h_delta.append(local_delta_h)
            h_expected_traj.append(np.stack(exp_seq).astype(np.float32))
            h_posterior_traj.append(np.stack(post_seq).astype(np.float32))
            action_chunk.append(a0)
            action_remain.append(a0)
            repair_target.append(repaired)
            repair_delta.append(delta)
            task_vec.append(np.asarray(mem["task_vec"], dtype=np.float32)[idx])
            success.append(float(np.asarray(mem["success"], dtype=np.float32).reshape(-1)[idx]) if "success" in mem else 0.0)
            mode.append(mode_i)
            sequence_index.append(int(np.asarray(mem["sequence_index"]).reshape(-1)[idx]))
            subtask_index.append(int(np.asarray(mem["subtask_index"]).reshape(-1)[idx]))
            cut_idx.append(c)
            remaining_steps.append(int(np.asarray(mem["remaining_steps"]).reshape(-1)[idx]))
            full_chunk_len.append(int(np.asarray(mem["full_chunk_len"]).reshape(-1)[idx]))
            task.append(str(np.asarray(mem["task"], dtype=object).reshape(-1)[idx]) if "task" in mem else "")
            lang_text.append(str(np.asarray(mem["lang_text"], dtype=object).reshape(-1)[idx]) if "lang_text" in mem else "")
            row_id.append(int(np.asarray(mem["row_id"]).reshape(-1)[idx]) if "row_id" in mem else idx)

    if not h_t:
        raise ValueError("no valid rows after filtering")

    out = {
        "state_start": np.stack(state_start).astype(np.float32),
        "state_end": np.stack(state_end).astype(np.float32),
        "base_summary_exec": np.stack(h_t).astype(np.float32),
        "target_summary_exec": np.stack(h_post).astype(np.float32),
        "target_delta_exec": np.stack(h_delta).astype(np.float32),
        "expected_h_traj": np.stack(h_expected_traj).astype(np.float32),
        "posterior_h_traj": np.stack(h_posterior_traj).astype(np.float32),
        "action_chunk": np.stack(action_chunk).astype(np.float32),
        "action_remain": np.stack(action_remain).astype(np.float32),
        "repair_action_target": np.stack(repair_target).astype(np.float32),
        "action_delta_target": np.stack(repair_delta).astype(np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "task_vec": np.stack(task_vec).astype(np.float32),
        "mode_label": np.asarray(mode, dtype=np.int64),
        "sequence_index": np.asarray(sequence_index, dtype=np.int32),
        "subtask_index": np.asarray(subtask_index, dtype=np.int32),
        "cut_idx": np.asarray(cut_idx, dtype=np.int32),
        "remaining_steps": np.asarray(remaining_steps, dtype=np.int32),
        "full_chunk_len": np.asarray(full_chunk_len, dtype=np.int32),
        "task": np.asarray(task, dtype=object),
        "lang_text": np.asarray(lang_text, dtype=object),
        "row_id": np.asarray(row_id, dtype=np.int32),
    }
    out["repair_delta_norm"] = np.linalg.norm(out["action_delta_target"].reshape(len(h_t), -1), axis=1).astype(np.float32)
    out["belief_revision_norm"] = np.linalg.norm(out["target_delta_exec"].reshape(len(h_t), -1), axis=1).astype(np.float32)
    out["mpc_smooth_cost"] = smoothness_cost(out["repair_action_target"])
    out["mpc_velocity_cost"] = velocity_cost(out["repair_action_target"])
    out["mpc_nominal_deviation_cost"] = np.mean(
        (out["repair_action_target"] - out["action_remain"]) ** 2,
        axis=(1, 2),
    ).astype(np.float32)
    out["mpc_proxy_cost"] = (
        out["mpc_nominal_deviation_cost"]
        + 0.1 * out["mpc_smooth_cost"]
        + 0.01 * out["mpc_velocity_cost"]
    ).astype(np.float32)

    select = balanced_indices(out["mode_label"], int(args.max_per_mode), int(args.seed))
    if int(args.max_rows) > 0 and select.size > int(args.max_rows):
        rng = np.random.default_rng(int(args.seed))
        select = rng.choice(select, size=int(args.max_rows), replace=False)
    if select.size != len(h_t):
        out = {key: value[select] for key, value in out.items()}

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    mode_vals, mode_counts = np.unique(out["mode_label"], return_counts=True)
    summary = {
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "num_rows": int(out["base_summary_exec"].shape[0]),
        "horizon": horizon,
        "summary_dim": int(out["base_summary_exec"].shape[1]),
        "state_dim": int(out["state_start"].shape[1]),
        "action_shape": list(out["action_chunk"].shape[1:]),
        "expected_alpha": alpha,
        "target_delta_clip": float(args.target_delta_clip),
        "mode_counts": {str(k): int(v) for k, v in zip(mode_vals, mode_counts)},
        "success_counts": {
            "success": int((out["success"] > 0.5).sum()),
            "failure": int((out["success"] <= 0.5).sum()),
        },
        "repair_delta_norm": {
            "min": float(np.min(out["repair_delta_norm"])),
            "p50": float(np.percentile(out["repair_delta_norm"], 50)),
            "p90": float(np.percentile(out["repair_delta_norm"], 90)),
            "max": float(np.max(out["repair_delta_norm"])),
        },
        "mpc_proxy_cost": {
            "min": float(np.min(out["mpc_proxy_cost"])),
            "p50": float(np.percentile(out["mpc_proxy_cost"], 50)),
            "p90": float(np.percentile(out["mpc_proxy_cost"], 90)),
            "max": float(np.max(out["mpc_proxy_cost"])),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
