from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sparse localized-repair schedule from existing sensitivity teacher rows."
    )
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-affected-fraction", type=float, default=0.05)
    parser.add_argument("--max-affected-fraction", type=float, default=0.50)
    parser.add_argument("--min-delta-norm", type=float, default=0.0)
    parser.add_argument("--max-delta-norm", type=float, default=1.0)
    parser.add_argument("--max-per-subtask", type=int, default=2)
    parser.add_argument("--max-per-sequence", type=int, default=20)
    parser.add_argument("--prefer-high-delta", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.rows_jsonl)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        fraction = _as_float(
            row,
            "suffix_mask_ablation_affected_fraction",
            "directional_sensitivity_tcn_affected_fraction",
            default=0.0,
        )
        delta_norm = _as_float(
            row,
            "suffix_mask_ablation_delta_norm",
            "sensitivity_teacher_suffix_delta_norm",
            "teacher_suffix_delta_norm",
            default=0.0,
        )
        if fraction < args.min_affected_fraction or fraction > args.max_affected_fraction:
            continue
        if delta_norm < args.min_delta_norm or delta_norm > args.max_delta_norm:
            continue
        candidates.append(
            {
                "sequence_index": int(row.get("sequence_index", -1)),
                "subtask_index": int(row.get("subtask_index", -1)),
                "step": int(row.get("step", -1)),
                "task": str(row.get("task", "")),
                "affected_fraction": float(fraction),
                "delta_norm": float(delta_norm),
                "feature_path": str(row.get("feature_path", "")),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["sequence_index"],
            item["subtask_index"],
            -item["delta_norm"] if args.prefer_high_delta else item["step"],
        )
    )

    per_subtask: dict[tuple[int, int], int] = defaultdict(int)
    per_sequence: dict[int, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for item in candidates:
        key = (item["sequence_index"], item["subtask_index"], item["step"])
        if key in seen:
            continue
        if args.max_per_subtask > 0 and per_subtask[key[:2]] >= args.max_per_subtask:
            continue
        if args.max_per_sequence > 0 and per_sequence[item["sequence_index"]] >= args.max_per_sequence:
            continue
        seen.add(key)
        per_subtask[key[:2]] += 1
        per_sequence[item["sequence_index"]] += 1
        selected.append(item)

    payload = {
        "source_rows_jsonl": str(args.rows_jsonl),
        "num_rows": len(rows),
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "min_affected_fraction": float(args.min_affected_fraction),
        "max_affected_fraction": float(args.max_affected_fraction),
        "min_delta_norm": float(args.min_delta_norm),
        "max_delta_norm": float(args.max_delta_norm),
        "max_per_subtask": int(args.max_per_subtask),
        "max_per_sequence": int(args.max_per_sequence),
        "cases": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
