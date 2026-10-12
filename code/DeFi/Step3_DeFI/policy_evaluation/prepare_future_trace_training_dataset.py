#!/usr/bin/env python3
"""Build a training index from CALVIN rollout rows and saved future traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_path(row: dict[str, Any], trace_dir: Path | None) -> Path | None:
    raw = row.get("collection_trace_path") or row.get("target_proxy_trace_path") or row.get("trace")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    if trace_dir is None:
        return None
    idx = row.get("trace_index")
    if idx is not None:
        candidate = trace_dir / f"future_trace_{int(idx):04d}.npz"
        if candidate.exists():
            return candidate
    return None


def _validate_trace(path: Path) -> tuple[bool, tuple[int, ...] | None, tuple[int, ...] | None]:
    try:
        with np.load(path) as data:
            if "original_base_future" not in data or "final_future" not in data:
                return False, None, None
            return True, tuple(data["original_base_future"].shape), tuple(data["final_future"].shape)
    except Exception:
        return False, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="future_trace_training_dataset")
    parser.add_argument("--write-npz", action="store_true", help="Also pack all F_base/F_final arrays into one NPZ.")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    rollout_jsonl = Path(args.rollout_jsonl)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(rollout_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    output_jsonl = output_dir / f"{args.output_name}.jsonl"
    output_npz = output_dir / f"{args.output_name}.npz"

    examples: list[dict[str, Any]] = []
    f_base: list[np.ndarray] = []
    f_final: list[np.ndarray] = []
    missing_trace = 0
    invalid_trace = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(rows):
            path = _trace_path(row, trace_dir)
            if path is None:
                missing_trace += 1
                continue
            valid, base_shape, final_shape = _validate_trace(path)
            if not valid:
                invalid_trace += 1
                continue

            success = bool(row.get("success"))
            example = {
                "idx": len(examples),
                "source_row_idx": row_idx,
                "sequence_index": row.get("sequence_index", row.get("seq")),
                "subtask_index": row.get("subtask_index", row.get("sub")),
                "task": row.get("task"),
                "lang_text": row.get("lang_text"),
                "success": success,
                "label_success": int(success),
                "steps": row.get("steps"),
                "trace_path": str(path),
                "f_base_key": "original_base_future",
                "f_final_key": "final_future",
                "f_base_shape": base_shape,
                "f_final_shape": final_shape,
                "energy": _as_float(row.get("future_energy_before", row.get("energy"))),
                "energy_after": _as_float(row.get("future_energy_after", row.get("energy_after"))),
                "energy_delta": _as_float(row.get("future_energy_delta")),
                "energy_reason": row.get("future_energy_reason", row.get("energy_reason")),
                "energy_factor": row.get("future_energy_selected_factor", row.get("energy_factor")),
                "energy_applied": bool(row.get("future_energy_applied", False)),
                "energy_shift_norm": _as_float(row.get("future_energy_shift_norm", row.get("immune_shift"))),
                "future_energy_threshold": _as_float(row.get("future_energy_threshold")),
                "future_energy_step_size": _as_float(row.get("future_energy_step_size")),
                "future_energy_mask_mode": row.get("future_energy_mask_mode"),
                "future_energy_mask_source": row.get("future_energy_mask_source"),
                "immune_only": bool(row.get("immune_only", False)),
                "immune_applied": bool(row.get("immune_applied", False)),
                "immune_reason": row.get("immune_reason"),
            }
            examples.append(example)
            out.write(json.dumps(example, ensure_ascii=False) + "\n")

            if args.write_npz:
                with np.load(path) as data:
                    f_base.append(data["original_base_future"].astype(np.float32))
                    f_final.append(data["final_future"].astype(np.float32))

    if args.write_npz and examples:
        np.savez_compressed(
            output_npz,
            F_base=np.stack(f_base),
            F_final=np.stack(f_final),
            success=np.asarray([ex["label_success"] for ex in examples], dtype=np.int64),
            task=np.asarray([ex["task"] or "" for ex in examples]),
            energy=np.asarray([np.nan if ex["energy"] is None else ex["energy"] for ex in examples], dtype=np.float32),
            factor=np.asarray([ex["energy_factor"] or "" for ex in examples]),
        )

    summary = {
        "rollout_jsonl": str(rollout_jsonl),
        "trace_dir": str(trace_dir) if trace_dir else None,
        "output_jsonl": str(output_jsonl),
        "output_npz": str(output_npz) if args.write_npz else None,
        "num_rollout_rows": len(rows),
        "num_examples": len(examples),
        "missing_trace": missing_trace,
        "invalid_trace": invalid_trace,
        "success_count": sum(ex["label_success"] for ex in examples),
        "failure_count": sum(1 - ex["label_success"] for ex in examples),
        "energy_applied_count": sum(int(ex["energy_applied"]) for ex in examples),
    }
    summary_path = output_dir / f"{args.output_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
