#!/usr/bin/env python3
"""Summarize CALVIN future-trace collection quality while rollout is running."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _chain_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seq = row.get("sequence_index", row.get("seq"))
        if seq is not None:
            per_seq[int(seq)].append(row)

    lengths: list[int] = []
    for seq_rows in per_seq.values():
        seq_rows = sorted(seq_rows, key=lambda r: int(r.get("subtask_index", r.get("sub", 0))))
        count = 0
        for row in seq_rows:
            if bool(row.get("success")):
                count += 1
            else:
                break
        lengths.append(count)

    chain = {}
    for k in range(1, 6):
        chain[f"{k}/5"] = (sum(int(x >= k) for x in lengths) / len(lengths)) if lengths else 0.0
    return {
        "num_sequences": len(lengths),
        "avg_successful_sequence_length": mean(lengths) if lengths else 0.0,
        "chain_success": chain,
        "length_histogram": dict(sorted(Counter(lengths).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--energy-threshold", type=float)
    args = parser.parse_args()

    path = Path(args.rollout_jsonl)
    rows = _read_jsonl(path)

    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "success": 0,
        "failure": 0,
        "edited": 0,
        "edited_success": 0,
        "edited_failure": 0,
        "high_energy": 0,
        "high_energy_success": 0,
        "high_energy_failure": 0,
        "energy_values": [],
        "failure_energy_values": [],
    })
    reasons = Counter()
    factors = Counter()
    trace_saved = 0

    for row in rows:
        task = row.get("task") or "unknown"
        success = bool(row.get("success"))
        edited = bool(row.get("future_energy_applied", False))
        energy = _float(row, "future_energy_before", "energy")
        threshold = args.energy_threshold
        if threshold is None:
            threshold = _float(row, "future_energy_threshold")
        high_energy = energy is not None and threshold is not None and energy >= threshold

        stat = by_task[task]
        stat["total"] += 1
        stat["success"] += int(success)
        stat["failure"] += int(not success)
        stat["edited"] += int(edited)
        stat["edited_success"] += int(edited and success)
        stat["edited_failure"] += int(edited and not success)
        stat["high_energy"] += int(high_energy)
        stat["high_energy_success"] += int(high_energy and success)
        stat["high_energy_failure"] += int(high_energy and not success)
        if energy is not None:
            stat["energy_values"].append(energy)
            if not success:
                stat["failure_energy_values"].append(energy)

        reasons[row.get("future_energy_reason", row.get("energy_reason", "missing"))] += 1
        factors[row.get("future_energy_selected_factor", row.get("energy_factor", "none"))] += 1
        trace_saved += int(bool(row.get("collection_trace_saved")) or bool(row.get("collection_trace_path")))

    task_summary = {}
    for task, stat in by_task.items():
        total = stat["total"]
        edited = stat["edited"]
        high = stat["high_energy"]
        task_summary[task] = {
            "total": total,
            "success": stat["success"],
            "failure": stat["failure"],
            "success_rate": stat["success"] / total if total else 0.0,
            "edited": edited,
            "edited_success": stat["edited_success"],
            "edited_failure": stat["edited_failure"],
            "edited_failure_rate": stat["edited_failure"] / edited if edited else 0.0,
            "high_energy": high,
            "high_energy_success": stat["high_energy_success"],
            "high_energy_failure": stat["high_energy_failure"],
            "high_energy_failure_rate": stat["high_energy_failure"] / high if high else 0.0,
            "mean_energy": mean(stat["energy_values"]) if stat["energy_values"] else None,
            "mean_failure_energy": mean(stat["failure_energy_values"]) if stat["failure_energy_values"] else None,
        }

    failures = [r for r in rows if not bool(r.get("success"))]
    edited_rows = [r for r in rows if bool(r.get("future_energy_applied", False))]
    high_energy_rows = [
        r for r in rows
        if (_float(r, "future_energy_before", "energy") is not None)
        and ((_float(r, "future_energy_threshold") if args.energy_threshold is None else args.energy_threshold) is not None)
        and (_float(r, "future_energy_before", "energy") >= (_float(r, "future_energy_threshold") if args.energy_threshold is None else args.energy_threshold))
    ]

    report = {
        "rollout_jsonl": str(path),
        "num_rows": len(rows),
        "success_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(rows) if rows else 0.0,
        "trace_saved_count": trace_saved,
        "energy_reason_counts": dict(reasons),
        "energy_factor_counts": dict(factors),
        "edited_count": len(edited_rows),
        "edited_success_count": sum(int(bool(r.get("success"))) for r in edited_rows),
        "edited_failure_count": sum(int(not bool(r.get("success"))) for r in edited_rows),
        "high_energy_count": len(high_energy_rows),
        "high_energy_success_count": sum(int(bool(r.get("success"))) for r in high_energy_rows),
        "high_energy_failure_count": sum(int(not bool(r.get("success"))) for r in high_energy_rows),
        "high_energy_failure_precision": (
            sum(int(not bool(r.get("success"))) for r in high_energy_rows) / len(high_energy_rows)
            if high_energy_rows else 0.0
        ),
        "failure_task_counts": dict(Counter(r.get("task") or "unknown" for r in failures).most_common()),
        "task_summary": dict(sorted(task_summary.items(), key=lambda kv: (-kv[1]["failure"], -kv[1]["total"], kv[0]))),
        "chain": _chain_stats(rows),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
