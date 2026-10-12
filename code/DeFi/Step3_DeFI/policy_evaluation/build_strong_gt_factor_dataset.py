from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def target_block(task: str) -> str | None:
    task = task.lower()
    for color in ("red", "blue", "pink"):
        if f"_{color}_block" in task or f"{color} block" in task:
            return f"block_{color}"
    return None


def contact_count(row: dict[str, Any], name: str, endpoint: str = "end_contact_sets") -> int:
    state = row.get("contact_state") or {}
    groups = state.get(endpoint) or {}
    vals = groups.get(name) or []
    return len(vals) if isinstance(vals, list) else 0


def target_motion(row: dict[str, Any]) -> dict[str, float]:
    block = target_block(str(row.get("task", "")))
    motions = row.get("block_displacement") or row.get("object_motion") or {}
    if block and isinstance(motions.get(block), dict):
        return motions[block]
    vals = [v for v in motions.values() if isinstance(v, dict)] if isinstance(motions, dict) else []
    if not vals:
        return {}
    return max(vals, key=lambda v: abs(float(v.get("distance", 0.0) or 0.0)))


def slider_drawer_delta(row: dict[str, Any]) -> float:
    delta = row.get("drawer_slider_progress_delta") or {}
    best = 0.0
    for group in ("doors", "buttons", "switches", "lights"):
        items = delta.get(group) or {}
        if not isinstance(items, dict):
            continue
        for vals in items.values():
            if not isinstance(vals, dict):
                continue
            for field in vals.values():
                if isinstance(field, dict) and "delta" in field:
                    best = max(best, abs(float(field.get("delta", 0.0) or 0.0)))
    return best


def pseudo_label(row: dict[str, Any]) -> tuple[str, str]:
    if bool(row.get("success", False)):
        return "none", "successful rollout; preserve future"

    task = str(row.get("task", "")).lower()
    motion = target_motion(row)
    dx = float(motion.get("delta_x", 0.0) or 0.0)
    dz = float(motion.get("delta_z", 0.0) or 0.0)
    dist = float(motion.get("distance", 0.0) or 0.0)
    robot_block = contact_count(row, "robot_block")
    block_drawer = contact_count(row, "block_drawer")
    block_block = contact_count(row, "block_block")
    progress = slider_drawer_delta(row)

    if "push" in task and "right" in task:
        if dx < 0.04:
            return "object_displacement", f"target block moved right insufficiently: dx={dx:.3f}"
        return "goal_completion", "object moved but goal predicate still failed"
    if "push" in task and "left" in task:
        if dx > -0.04:
            return "object_displacement", f"target block moved left insufficiently: dx={dx:.3f}"
        return "goal_completion", "object moved but goal predicate still failed"
    if "push_into_drawer" in task or "place_in_drawer" in task or "place_in_slider" in task:
        if block_drawer == 0 and progress < 0.03:
            return "drawer_slider_progress", "object did not reach drawer/slider contact or progress"
        return "goal_completion", "drawer/slider-related state changed but predicate failed"
    if "stack_block" in task or "unstack_block" in task:
        if block_block == 0:
            return "contact", "required block-block contact was not established"
        return "goal_completion", "contact exists but stack predicate failed"
    if "lift" in task:
        if robot_block == 0:
            return "contact", "robot-object contact was missing at rollout end"
        if dz < 0.03 and dist < 0.04:
            return "object_displacement", f"object lift/displacement insufficient: dz={dz:.3f}, dist={dist:.3f}"
        return "goal_completion", "object moved/contacted but goal predicate failed"
    if "drawer" in task or "slider" in task:
        if progress < 0.03:
            return "drawer_slider_progress", f"drawer/slider progress insufficient: delta={progress:.3f}"
        return "goal_completion", "progress changed but goal predicate failed"
    if "led" in task or "lightbulb" in task:
        return "goal_completion", "button/light logical goal was not completed"
    return "goal_completion", "failure observed but low-level mechanism is ambiguous"


def convert(input_path: Path, output_path: Path, trace_dir: Path | None, require_trace: bool) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    tasks = Counter()
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(input_path):
            trace = str(row.get("collection_trace_path") or row.get("trace_path") or "")
            if trace_dir is not None and trace:
                trace = str(trace_dir / Path(trace).name)
            if require_trace and (not trace or not Path(trace).exists()):
                skipped += 1
                continue
            factor, mechanism = pseudo_label(row)
            new_row = dict(row)
            new_row["trace_path"] = trace
            new_row["collection_trace_path"] = trace
            new_row["factor_label"] = factor
            new_row["wrong_component"] = factor
            new_row["pseudo_failure_mechanism"] = mechanism
            new_row["label_source"] = "strong_gt_rules_v1"
            new_row["sequence_index"] = int(row.get("sequence_index", row.get("seq", written)) or 0)
            new_row["subtask_index"] = int(row.get("subtask_index", row.get("sub", 0)) or 0)
            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            counts[factor] += 1
            tasks[str(row.get("task", "unknown"))] += 1
            written += 1
    return {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "trace_dir_override": str(trace_dir) if trace_dir else None,
        "written": written,
        "skipped_missing_trace": skipped,
        "factor_counts": dict(counts.most_common()),
        "task_counts": dict(tasks.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build factor-labeled future dataset from strong GT rollout rows.")
    parser.add_argument("--strong-gt-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--no-require-trace", action="store_true")
    args = parser.parse_args()

    summary = convert(args.strong_gt_jsonl, args.output_jsonl, args.trace_dir, not args.no_require_trace)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
