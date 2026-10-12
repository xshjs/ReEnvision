from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def take(memory: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[ids] for key, value in memory.items()}


def ensure_executed_action(memory: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "executed_action" not in memory:
        memory = dict(memory)
        memory["executed_action"] = np.asarray(memory["action_chunk"], dtype=np.float32)[:, 0]
    return memory


def concat_memories(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted(set.intersection(*(set(part.keys()) for part in parts)))
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance keep/patch/replan rows for factored T+D training.")
    parser.add_argument("--keep-npz", type=Path, required=True, help="Clean baseline-shadow memory; mostly keep rows.")
    parser.add_argument("--delta-npz", type=Path, required=True, help="Rows with non-zero action_delta_target.")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--num-keep", type=int, default=900)
    parser.add_argument("--num-patch", type=int, default=450)
    parser.add_argument("--num-replan", type=int, default=450)
    parser.add_argument("--patch-max-delta-norm", type=float, default=8.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    keep = ensure_executed_action(load_npz(args.keep_npz))
    delta = ensure_executed_action(load_npz(args.delta_npz))

    keep_ids = np.arange(keep["action_chunk"].shape[0])
    delta_norm = np.linalg.norm(delta["action_delta_target"].reshape(delta["action_delta_target"].shape[0], -1), axis=1)
    patch_ids = np.where(delta_norm <= float(args.patch_max_delta_norm))[0]
    replan_ids = np.where(delta_norm > float(args.patch_max_delta_norm))[0]
    if patch_ids.size == 0 or replan_ids.size == 0:
        order = np.argsort(delta_norm)
        split = max(1, min(order.size - 1, order.size // 2))
        patch_ids = order[:split]
        replan_ids = order[split:]

    keep_sel = rng.choice(keep_ids, size=int(args.num_keep), replace=keep_ids.size < int(args.num_keep))
    patch_sel = rng.choice(patch_ids, size=int(args.num_patch), replace=patch_ids.size < int(args.num_patch))
    replan_sel = rng.choice(replan_ids, size=int(args.num_replan), replace=replan_ids.size < int(args.num_replan))

    keep_part = take(keep, keep_sel)
    patch_part = take(delta, patch_sel)
    replan_part = take(delta, replan_sel)
    out = concat_memories([keep_part, patch_part, replan_part])
    n = out["action_chunk"].shape[0]
    mode = np.concatenate(
        [
            np.zeros((keep_part["action_chunk"].shape[0],), dtype=np.int64),
            np.ones((patch_part["action_chunk"].shape[0],), dtype=np.int64),
            np.full((replan_part["action_chunk"].shape[0],), 2, dtype=np.int64),
        ],
        axis=0,
    )
    perm = rng.permutation(n)
    out = {key: value[perm] for key, value in out.items()}
    out["mode_label"] = mode[perm]
    out["source_label"] = np.asarray(["keep"] * keep_part["action_chunk"].shape[0] + ["patch"] * patch_part["action_chunk"].shape[0] + ["replan"] * replan_part["action_chunk"].shape[0], dtype=object)[perm]
    out["row_id"] = np.arange(n, dtype=np.int32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    summary = {
        "keep_npz": str(args.keep_npz),
        "delta_npz": str(args.delta_npz),
        "output_npz": str(args.output_npz),
        "num_rows": int(n),
        "num_keep": int((out["mode_label"] == 0).sum()),
        "num_patch": int((out["mode_label"] == 1).sum()),
        "num_replan": int((out["mode_label"] == 2).sum()),
        "patch_max_delta_norm": float(args.patch_max_delta_norm),
        "delta_norm_patch_pool": [float(np.min(delta_norm[patch_ids])), float(np.max(delta_norm[patch_ids]))],
        "delta_norm_replan_pool": [float(np.min(delta_norm[replan_ids])), float(np.max(delta_norm[replan_ids]))],
        "seed": int(args.seed),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
