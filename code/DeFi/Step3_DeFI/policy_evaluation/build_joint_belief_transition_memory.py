from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consecutive-step proxy memory for h_{t+1}=T(h_t,o_t,a_t,o_{t+1}).")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mem = load_npz(args.input_npz)
    required = [
        "state_start",
        "base_summary_exec",
        "target_summary_exec",
        "action_chunk",
        "success",
        "task_vec",
        "sequence_index",
        "subtask_index",
        "cut_idx",
        "remaining_steps",
        "full_chunk_len",
    ]
    missing = [k for k in required if k not in mem]
    if missing:
        raise ValueError(f"input memory missing keys: {missing}")

    seq = np.asarray(mem["sequence_index"]).reshape(-1)
    sub = np.asarray(mem["subtask_index"]).reshape(-1)
    cut = np.asarray(mem["cut_idx"]).reshape(-1)
    pairs: list[tuple[int, int]] = []
    groups: dict[tuple[int, int], dict[int, int]] = {}
    for i, key in enumerate(zip(seq.astype(int), sub.astype(int))):
        groups.setdefault(key, {})[int(cut[i])] = i
    for cut_to_idx in groups.values():
        for c, i in cut_to_idx.items():
            j = cut_to_idx.get(c + 1)
            if j is not None:
                pairs.append((i, j))
    if not pairs:
        raise ValueError("no consecutive cut_idx pairs found")

    pairs_arr = np.asarray(pairs, dtype=np.int64)
    success = np.asarray(mem["success"], dtype=np.float32).reshape(-1)
    if args.balance:
        rng = np.random.default_rng(args.seed)
        pair_success = success[pairs_arr[:, 0]] > 0.5
        pos = np.where(pair_success)[0]
        neg = np.where(~pair_success)[0]
        if len(pos) and len(neg):
            n = min(len(pos), len(neg))
            keep = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
            rng.shuffle(keep)
            pairs_arr = pairs_arr[keep]

    i = pairs_arr[:, 0]
    j = pairs_arr[:, 1]
    base = np.asarray(mem["base_summary_exec"], dtype=np.float32)
    target = np.asarray(mem["target_summary_exec"], dtype=np.float32)
    action = np.asarray(mem["action_chunk"], dtype=np.float32)
    out = {
        "state_start": np.asarray(mem["state_start"], dtype=np.float32)[i],
        # Use the next step's state as o_{t+1}; this is the closest available compact observation.
        "state_end": np.asarray(mem["state_start"], dtype=np.float32)[j],
        "base_summary_exec": base[i],
        "target_summary_exec": target[i],
        "target_delta_exec": base[j] - base[i],
        "action_chunk": action[i],
        "action_delta_target": action[j] - action[i],
        "success": success[i],
        "task_vec": np.asarray(mem["task_vec"], dtype=np.float32)[i],
        "sequence_index": seq[i].astype(np.int32),
        "subtask_index": sub[i].astype(np.int32),
        "cut_idx": cut[i].astype(np.int32),
        "remaining_steps": np.asarray(mem["remaining_steps"], dtype=np.int32).reshape(-1)[i],
        "full_chunk_len": np.asarray(mem["full_chunk_len"], dtype=np.int32).reshape(-1)[i],
    }
    for key in ["task", "lang_text", "row_id"]:
        if key in mem:
            out[key] = mem[key][i]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    summary = {
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "num_pairs": int(len(i)),
        "num_success": int((out["success"] > 0.5).sum()),
        "num_failure": int((out["success"] <= 0.5).sum()),
        "summary_dim": int(out["base_summary_exec"].shape[1]),
        "state_dim": int(out["state_start"].shape[1]),
        "action_shape": list(out["action_chunk"].shape[1:]),
        "has_action_delta_target": True,
        "balanced": bool(args.balance),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
