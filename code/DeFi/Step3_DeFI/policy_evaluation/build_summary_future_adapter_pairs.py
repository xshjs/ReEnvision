from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def future_summary_exec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def iter_npz(root: Path):
    for path in sorted(root.rglob("*.npz")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lightweight JSONL pairs for summary-conditioned future-token adapter training."
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--base-key", default="h_future_before")
    parser.add_argument("--target-key", default="h_future_after")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-delta-norm", type=float, default=0.0)
    parser.add_argument("--max-delta-norm", type=float, default=0.0)
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    num_seen = 0
    num_written = 0
    num_missing = 0
    num_nonfinite = 0
    num_shape_mismatch = 0
    num_delta_filtered = 0
    delta_norms: list[float] = []
    shape = None

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for path in iter_npz(args.trace_root):
            num_seen += 1
            try:
                with np.load(path, allow_pickle=True) as data:
                    if args.base_key not in data.files or args.target_key not in data.files:
                        num_missing += 1
                        continue
                    base = np.asarray(data[args.base_key], dtype=np.float32)
                    target = np.asarray(data[args.target_key], dtype=np.float32)
                    if base.shape != target.shape or base.ndim != 2:
                        num_shape_mismatch += 1
                        continue
                    if not np.isfinite(base).all() or not np.isfinite(target).all():
                        num_nonfinite += 1
                        continue
                    delta_norm = float(np.linalg.norm((target - base).reshape(-1), ord=2))
                    if float(args.min_delta_norm) > 0.0 and delta_norm < float(args.min_delta_norm):
                        num_delta_filtered += 1
                        continue
                    if float(args.max_delta_norm) > 0.0 and delta_norm > float(args.max_delta_norm):
                        num_delta_filtered += 1
                        continue
                    source_summary = future_summary_exec(base)
                    target_summary = future_summary_exec(target)
                    row = {
                        "base_trace_path": str(path),
                        "target_trace_path": str(path),
                        "base_future_key": args.base_key,
                        "target_future_key": args.target_key,
                        "source_summary": source_summary.tolist(),
                        "target_summary": target_summary.tolist(),
                        "sequence_index": int(num_written),
                        "task": "runtime_joint_belief",
                        "delta_norm": delta_norm,
                    }
            except Exception:
                num_nonfinite += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_written += 1
            delta_norms.append(delta_norm)
            shape = base.shape
            if int(args.max_rows) > 0 and num_written >= int(args.max_rows):
                break

    q = {}
    if delta_norms:
        arr = np.asarray(delta_norms, dtype=np.float32)
        q = {
            "min": float(np.min(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }
    summary = {
        "trace_root": str(args.trace_root),
        "output_jsonl": str(args.output_jsonl),
        "base_key": args.base_key,
        "target_key": args.target_key,
        "num_seen": int(num_seen),
        "num_written": int(num_written),
        "num_missing": int(num_missing),
        "num_nonfinite_or_read_error": int(num_nonfinite),
        "num_shape_mismatch": int(num_shape_mismatch),
        "num_delta_filtered": int(num_delta_filtered),
        "future_shape": None if shape is None else list(shape),
        "delta_norm": q,
    }
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
