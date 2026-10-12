import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _load_rows(rows_jsonl: Path):
    rows = []
    with rows_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_npz(path: Path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _flatten_state(arr):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return arr


def _match_success_row(row, success_rows):
    candidates = [r for r in success_rows if r["task"] == row["task"]]
    if not candidates:
        candidates = success_rows
    if not candidates:
        return None
    step = int(row.get("step", 0))
    return min(candidates, key=lambda r: abs(int(r.get("step", 0)) - step))


def _pad_or_crop(arr, shape):
    arr = np.asarray(arr, dtype=np.float32)
    out = np.zeros(shape, dtype=np.float32)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[slices] = arr[slices]
    return out


def _task_vec(text, dim=64):
    vec = np.zeros((dim,), dtype=np.float32)
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=16).digest()
    for i, byte in enumerate(digest):
        vec[i % dim] += (float(byte) / 127.5) - 1.0
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--success-only-keep", action="store_true", default=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--delta-clip", type=float, default=0.0)
    parser.add_argument("--balance-task-mode", action="store_true")
    parser.add_argument("--max-per-task-mode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = _load_rows(args.rows_jsonl)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    success_rows = [r for r in rows if bool(r.get("success", False))]
    if not success_rows:
        raise ValueError("No successful suffix rows found; cannot build D targets.")

    examples = []
    for row in rows:
        path = Path(row["feature_path"])
        if not path.exists():
            continue
        data = _load_npz(path)
        target_data = data
        mode_label = int(row.get("mode_label", 0))
        if not bool(row.get("success", False)):
            matched = _match_success_row(row, success_rows)
            if matched is None:
                continue
            target_data = _load_npz(Path(matched["feature_path"]))
            mode_label = 1
        else:
            mode_label = 0

        action_remain = np.asarray(data["action_remain_before"], dtype=np.float32)
        target_action = _pad_or_crop(target_data["action_remain_before"], action_remain.shape)
        action_delta = target_action - action_remain
        if mode_label == 0:
            action_delta = np.zeros_like(action_delta)
            target_action = action_remain.copy()
        elif args.delta_clip > 0:
            action_delta = np.clip(action_delta, -float(args.delta_clip), float(args.delta_clip))
            target_action = action_remain + action_delta

        examples.append(
            {
                "state_before": _flatten_state(data["state_before"]),
                "state_after": _flatten_state(data["state_after"]),
                "executed_action": np.asarray(data["executed_action"], dtype=np.float32).reshape(-1),
                "h_future_before": np.asarray(data["h_future_before"], dtype=np.float32).reshape(-1),
                "h_action_before": np.asarray(data["h_action_before"], dtype=np.float32).reshape(-1),
                "h_future_after": np.asarray(data["h_future_after"], dtype=np.float32).reshape(-1),
                "h_action_after": np.asarray(data["h_action_after"], dtype=np.float32).reshape(-1),
                "action_remain_before": action_remain,
                "action_remain_after": target_action,
                "action_delta_target": action_delta,
                "mode_label": np.asarray(mode_label, dtype=np.int64),
                "success": np.asarray(float(row.get("success", False)), dtype=np.float32),
                "sequence_index": np.asarray(int(row.get("episode", 0)), dtype=np.int32),
                "subtask_index": np.asarray(0, dtype=np.int32),
                "cut_idx": np.asarray(int(row.get("step", 0)), dtype=np.int32),
                "remaining_steps": np.asarray(int(max(0, action_remain.shape[0])), dtype=np.int32),
                "full_chunk_len": np.asarray(int(action_remain.shape[0]), dtype=np.int32),
                "task_vec": _task_vec(row.get("task", "")),
                "task": str(row.get("task", "")),
                "lang_goal": str(row.get("lang_goal", "")),
                "source_feature_path": str(path),
            }
        )

    if not examples:
        raise ValueError("No valid examples built.")

    if args.balance_task_mode or args.max_per_task_mode > 0:
        rng = np.random.default_rng(args.seed)
        buckets = {}
        for idx, ex in enumerate(examples):
            buckets.setdefault((ex["task"], int(ex["mode_label"])), []).append(idx)
        selected = []
        for idxs in buckets.values():
            if not idxs:
                continue
            if args.max_per_task_mode > 0:
                target = min(len(idxs), int(args.max_per_task_mode))
            else:
                target = min(len(bucket) for bucket in buckets.values() if bucket)
            selected.extend(rng.choice(np.asarray(idxs), size=target, replace=False).tolist())
        rng.shuffle(selected)
        examples = [examples[i] for i in selected]

    keys = [
        "state_before",
        "state_after",
        "executed_action",
        "h_future_before",
        "h_action_before",
        "h_future_after",
        "h_action_after",
        "action_remain_before",
        "action_remain_after",
        "action_delta_target",
        "mode_label",
        "success",
        "sequence_index",
        "subtask_index",
        "cut_idx",
        "remaining_steps",
        "full_chunk_len",
        "task_vec",
    ]
    out = {k: np.stack([ex[k] for ex in examples], axis=0) for k in keys}
    out["state_start"] = out["state_before"]
    out["state_end"] = out["state_after"]
    out["base_summary_exec"] = np.concatenate([out["h_future_before"], out["h_action_before"]], axis=-1)
    out["target_summary_exec"] = np.concatenate([out["h_future_after"], out["h_action_after"]], axis=-1)
    out["action_chunk"] = out["action_remain_before"]
    out["task"] = np.asarray([ex["task"] for ex in examples])
    out["lang_goal"] = np.asarray([ex["lang_goal"] for ex in examples])
    out["source_feature_path"] = np.asarray([ex["source_feature_path"] for ex in examples])

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    mode_vals, mode_counts = np.unique(out["mode_label"], return_counts=True)
    delta_norm = np.linalg.norm(out["action_delta_target"].reshape(out["action_delta_target"].shape[0], -1), axis=1)
    summary = {
        "rows_jsonl": str(args.rows_jsonl),
        "output_npz": str(args.output_npz),
        "num_examples": int(out["mode_label"].shape[0]),
        "num_success_rows": int(len(success_rows)),
        "mode_counts": {str(int(k)): int(v) for k, v in zip(mode_vals, mode_counts)},
        "delta_norm": {
            "min": float(np.min(delta_norm)),
            "p50": float(np.percentile(delta_norm, 50)),
            "p90": float(np.percentile(delta_norm, 90)),
            "p99": float(np.percentile(delta_norm, 99)),
            "max": float(np.max(delta_norm)),
        },
        "shapes": {k: list(v.shape) for k, v in out.items() if hasattr(v, "shape")},
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
