from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_task_hash(task: str, task_dim: int) -> np.ndarray:
    vec = np.zeros((int(task_dim),), dtype=np.float32)
    key = hashlib.md5(str(task).encode("utf-8")).hexdigest()
    vec[int(key[:8], 16) % int(task_dim)] = 1.0
    return vec


def average_pool_1d(x: np.ndarray, out_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("non-finite summary")
    if x.size == out_dim:
        return x.astype(np.float32)
    if x.size < out_dim:
        out = np.zeros((out_dim,), dtype=np.float32)
        out[: x.size] = x
        return out
    trim = (x.size // out_dim) * out_dim
    if trim <= 0:
        raise ValueError(f"cannot pool summary size={x.size} to out_dim={out_dim}")
    return x[:trim].reshape(out_dim, trim // out_dim).mean(axis=1).astype(np.float32)


def resolve_feature_path(feature_root: Path, feature_path: str) -> Path:
    path = Path(feature_path)
    return path if path.is_absolute() else feature_root / path


def choose_balanced_indices(success: np.ndarray, mode: str, seed: int, max_rows: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(success > 0.5)
    neg = np.flatnonzero(success <= 0.5)
    if mode == "none":
        ids = np.arange(success.shape[0])
    elif mode == "downsample":
        n = min(pos.size, neg.size)
        if n == 0:
            raise ValueError(f"cannot balance; pos={pos.size}, neg={neg.size}")
        ids = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    elif mode == "oversample":
        n = max(pos.size, neg.size)
        if n == 0:
            raise ValueError("no rows to balance")
        ids = np.concatenate([
            rng.choice(pos, n, replace=pos.size < n),
            rng.choice(neg, n, replace=neg.size < n),
        ])
    else:
        raise ValueError(f"unknown balance mode: {mode}")
    rng.shuffle(ids)
    if max_rows is not None and max_rows > 0 and ids.size > max_rows:
        ids = ids[:max_rows]
    return ids.astype(np.int64)


def preselect_rows(rows: list[dict[str, Any]], balance_mode: str, seed: int, max_rows: int | None) -> list[dict[str, Any]]:
    success = np.asarray([float(bool(row.get("success", False))) for row in rows], dtype=np.float32)
    ids = choose_balanced_indices(success, mode=balance_mode, seed=seed, max_rows=max_rows)
    return [rows[int(i)] for i in ids.tolist()]


def load_phi_rows(
    rows_jsonl: Path,
    feature_root: Path,
    summary_dim: int,
    task_dim: int,
    balance_mode: str,
    seed: int,
    max_rows: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_rows = list(iter_jsonl(rows_jsonl))
    rows = preselect_rows(all_rows, balance_mode=balance_mode, seed=seed, max_rows=max_rows)
    if not rows:
        raise ValueError(f"no rows in {rows_jsonl}")

    state_start = []
    state_end = []
    base_summary = []
    target_summary = []
    action_chunk = []
    success = []
    task_vec = []
    task = []
    lang_text = []
    sequence_index = []
    subtask_index = []
    cut_idx = []
    remaining_steps = []
    full_chunk_len = []
    row_id = []
    skipped = []

    for row in rows:
        try:
            path = resolve_feature_path(feature_root, str(row["feature_path"]))
            with np.load(path, allow_pickle=False) as data:
                chunk = np.asarray(data["action_chunk"], dtype=np.float32)
                s0 = np.asarray(data["state_start"], dtype=np.float32).reshape(-1)
                s1 = np.asarray(data["state_end"], dtype=np.float32).reshape(-1)
                base = average_pool_1d(data["base_summary_exec"], summary_dim)
                target = average_pool_1d(data["target_summary_exec"], summary_dim)
            if chunk.ndim != 2 or not np.isfinite(chunk).all() or not np.isfinite(s0).all() or not np.isfinite(s1).all():
                raise ValueError("non-finite or invalid feature array")
        except Exception as exc:
            skipped.append({"row_id": row.get("row_id"), "feature_path": row.get("feature_path"), "error": str(exc)})
            continue

        state_start.append(s0)
        state_end.append(s1)
        base_summary.append(base)
        target_summary.append(target)
        action_chunk.append(chunk)
        ok = bool(row.get("success", False))
        success.append(float(ok))
        task_name = str(row.get("task", "unknown"))
        task_vec.append(stable_task_hash(task_name, task_dim))
        task.append(task_name)
        lang_text.append(str(row.get("lang_text", task_name)))
        sequence_index.append(int(row.get("sequence_index", -1)))
        subtask_index.append(int(row.get("subtask_index", 0)))
        cut_idx.append(int(row.get("cut_idx", 0)))
        remaining_steps.append(int(row.get("remaining_steps", chunk.shape[0])))
        full_chunk_len.append(int(row.get("full_chunk_len", chunk.shape[0])))
        row_id.append(int(row.get("row_id", len(row_id))))

    if not action_chunk:
        raise ValueError("no usable phi rows after filtering")

    base_arr = np.stack(base_summary).astype(np.float32)
    target_arr = np.stack(target_summary).astype(np.float32)
    arrays = {
        "state_start": np.stack(state_start).astype(np.float32),
        "state_end": np.stack(state_end).astype(np.float32),
        "state_transition": (np.stack(state_end) - np.stack(state_start)).astype(np.float32),
        "base_summary_exec": base_arr,
        "target_summary_exec": target_arr,
        "target_delta_exec": (target_arr - base_arr).astype(np.float32),
        "action_chunk": np.stack(action_chunk).astype(np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "task_vec": np.stack(task_vec).astype(np.float32),
        "task": np.asarray(task, dtype=object),
        "lang_text": np.asarray(lang_text, dtype=object),
        "sequence_index": np.asarray(sequence_index, dtype=np.int32),
        "subtask_index": np.asarray(subtask_index, dtype=np.int32),
        "cut_idx": np.asarray(cut_idx, dtype=np.int32),
        "remaining_steps": np.asarray(remaining_steps, dtype=np.int32),
        "full_chunk_len": np.asarray(full_chunk_len, dtype=np.int32),
        "row_id": np.asarray(row_id, dtype=np.int32),
    }
    summary = {
        "rows_jsonl": str(rows_jsonl),
        "feature_root": str(feature_root),
        "raw_rows": len(all_rows),
        "preselected_rows": len(rows),
        "usable_rows": int(arrays["success"].shape[0]),
        "skipped_rows": len(skipped),
        "num_success": int((arrays["success"] > 0.5).sum()),
        "num_failure": int((arrays["success"] <= 0.5).sum()),
        "summary_dim": int(summary_dim),
        "state_dim": int(arrays["state_start"].shape[-1]),
        "action_shape": list(arrays["action_chunk"].shape[1:]),
        "skipped_examples": skipped[:10],
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced, pooled phi/chunk-effect memory from collected CALVIN rows.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-dim", type=int, default=1024)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--balance-mode", choices=["downsample", "oversample", "none"], default="downsample")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arrays, summary = load_phi_rows(
        args.rows_jsonl,
        args.feature_root,
        args.summary_dim,
        args.task_dim,
        balance_mode=args.balance_mode,
        seed=args.seed,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )
    summary.update(
        {
            "balance_mode": args.balance_mode,
            "balanced_rows": int(arrays["success"].shape[0]),
            "balanced_success": int((arrays["success"] > 0.5).sum()),
            "balanced_failure": int((arrays["success"] <= 0.5).sum()),
            "output_npz": str(args.output_npz),
        }
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
