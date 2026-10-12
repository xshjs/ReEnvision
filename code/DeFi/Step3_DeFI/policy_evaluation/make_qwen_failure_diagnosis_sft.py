from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SYSTEM_PROMPT = (
    "You are a robot pre-rollout future-diagnosis module. Given a CALVIN task, "
    "predicted future evidence, and nearest success/failure memory summaries, diagnose why the predicted "
    "future may fail before executing the rollout. Return only one compact JSON object with keys: "
    '{"failure_mechanism": string, "wrong_component": string, "time_window": [number, number], '
    '"object": string, "edit_intent": string}. '
    "wrong_component must be one of contact, object_displacement, object_identity, "
    "drawer_slider_progress, goal_completion, none."
)


COLOR_WORDS = {"red", "blue", "pink", "green", "yellow", "black", "white", "orange", "purple"}


COMPONENT_RULES = [
    ("object_displacement", re.compile(r"push|move|place")),
    ("drawer_slider_progress", re.compile(r"drawer|slider|slide")),
    ("contact", re.compile(r"lift|stack|unstack|grasp|block")),
    ("goal_completion", re.compile(r"led|lightbulb|turn_on|turn_off")),
]


TEMPLATES = {
    "contact": {
        "failure_mechanism": "the predicted future may rely on weak or unstable contact before the object state changes",
        "time_window": [0.20, 0.55],
        "edit_intent": "stabilize contact before continuing the manipulation",
    },
    "object_displacement": {
        "failure_mechanism": "the predicted future may overestimate object displacement or move it in a misaligned direction",
        "time_window": [0.30, 0.75],
        "edit_intent": "align the object displacement with the commanded direction and distance",
    },
    "object_identity": {
        "failure_mechanism": "the predicted future may track or manipulate the wrong object",
        "time_window": [0.10, 0.45],
        "edit_intent": "focus the future on the instructed target object",
    },
    "drawer_slider_progress": {
        "failure_mechanism": "the predicted future may hallucinate drawer or slider progress before verified physical motion",
        "time_window": [0.35, 0.85],
        "edit_intent": "make articulated progress depend on verified pulling or sliding motion",
    },
    "goal_completion": {
        "failure_mechanism": "the predicted future may mark task completion before the physical goal state is verified",
        "time_window": [0.60, 1.00],
        "edit_intent": "delay goal completion until the physical state verifies success",
    },
    "none": {
        "failure_mechanism": "no failure mechanism is indicated by the evidence",
        "time_window": [0.00, 0.00],
        "edit_intent": "preserve the predicted future without latent intervention",
    },
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def object_from_task(task: str) -> str:
    words = task.replace("_", " ").split()
    color = next((word for word in words if word in COLOR_WORDS), "")
    if "block" in words:
        base = "block"
    elif "drawer" in words:
        base = "drawer"
    elif "slider" in words:
        base = "slider"
    elif "lightbulb" in words:
        base = "lightbulb"
    elif "led" in words:
        base = "led"
    else:
        base = words[-1] if words else "target object"
    return f"{color} {base}".strip()


def component_from_task(task: str) -> str:
    for component, pattern in COMPONENT_RULES:
        if pattern.search(task):
            return component
    return "goal_completion"


def future_summary(trace_path: str, base_key: str) -> dict[str, Any]:
    try:
        with np.load(trace_path) as data:
            arr = np.asarray(data[base_key], dtype=np.float32)
    except Exception:
        return {}
    if arr.ndim != 2:
        return {}
    token_energy = np.linalg.norm(arr, axis=1)
    channel_energy = np.mean(np.abs(arr), axis=0)
    top_time = np.argsort(-token_energy)[:5].tolist()
    top_channel = np.argsort(-channel_energy)[:8].tolist()
    return {
        "future_shape": list(arr.shape),
        "future_norm": round(float(np.linalg.norm(arr)), 4),
        "token_energy_mean": round(float(token_energy.mean()), 4),
        "token_energy_max": round(float(token_energy.max()), 4),
        "top_time_tokens": [int(x) for x in top_time],
        "top_channels": [int(x) for x in top_channel],
    }


def build_memory_index(rows: list[dict[str, Any]]):
    by_task_success = defaultdict(list)
    for row in rows:
        by_task_success[(str(row.get("task", "")), bool(row.get("success")))].append(row)
    return by_task_success


def compact_memory(rows: list[dict[str, Any]], limit: int = 3):
    out = []
    for row in rows[:limit]:
        out.append(
            {
                "task": row.get("task"),
                "success": bool(row.get("success")),
                "steps": row.get("steps"),
                "energy": row.get("energy"),
                "energy_factor": row.get("energy_factor"),
            }
        )
    return out


def diagnosis_for(row: dict[str, Any], include_success: bool) -> dict[str, Any]:
    task = str(row.get("task", "unknown"))
    if include_success and bool(row.get("success", False)):
        component = "none"
    else:
        component = str(row.get("energy_factor") or component_from_task(task))
    if component not in TEMPLATES:
        component = component_from_task(task)
    template = TEMPLATES[component]
    return {
        "failure_mechanism": template["failure_mechanism"],
        "wrong_component": component,
        "time_window": [float(template["time_window"][0]), float(template["time_window"][1])],
        "object": "none" if component == "none" else object_from_task(task),
        "edit_intent": template["edit_intent"],
    }


def make_example(row: dict[str, Any], memory_index, args: argparse.Namespace) -> dict[str, Any]:
    task = str(row.get("task", "unknown"))
    base_key = str(row.get("f_base_key", "original_base_future"))
    trace_path = str(row.get("trace_path", ""))
    success_mem = memory_index.get((task, True), [])
    fail_mem = memory_index.get((task, False), [])
    payload = {
        "task": task,
        "language": row.get("lang_text", ""),
        "sequence_context": {
            "sequence_index": int(row.get("sequence_index", -1)),
            "subtask_index": int(row.get("subtask_index", -1)),
        },
        "future_risk_evidence": {
            "energy": row.get("energy"),
            "energy_after": row.get("energy_after"),
            "energy_delta": row.get("energy_delta"),
            "energy_factor": row.get("energy_factor"),
            "energy_reason": row.get("energy_reason"),
        },
        "base_future_summary": future_summary(trace_path, base_key),
        "nearest_failure_memory": compact_memory(fail_mem, args.memory_topk),
        "nearest_success_memory": compact_memory(success_mem, args.memory_topk),
        "instruction": "Before rollout, diagnose where and why the predicted future may fail, and state the intended counterfactual edit. Do not assume the rollout outcome is known.",
    }
    if not bool(args.pre_rollout_input):
        payload["rollout_outcome"] = {
            "success": bool(row.get("success", False)),
            "steps": int(row.get("steps", 0) or 0),
            "sequence_index": int(row.get("sequence_index", -1)),
            "subtask_index": int(row.get("subtask_index", -1)),
        }
    target = diagnosis_for(row, include_success=bool(args.include_success))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "source": "calvin_future_diagnosis_pseudo",
            "pre_rollout_input": bool(args.pre_rollout_input),
            "idx": row.get("idx"),
            "sequence_index": row.get("sequence_index"),
            "subtask_index": row.get("subtask_index"),
            "task": task,
            "success": bool(row.get("success", False)),
            "wrong_component": target["wrong_component"],
            "trace_path": trace_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen failure-diagnosis SFT data from CALVIN future traces.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--include-success", action="store_true")
    parser.add_argument("--max-success-per-task", type=int, default=8)
    parser.add_argument("--memory-topk", type=int, default=3)
    parser.add_argument("--pre-rollout-input", action="store_true")
    parser.add_argument("--balance-components", action="store_true")
    parser.add_argument("--target-per-failure-component", type=int, default=40)
    parser.add_argument("--target-none", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.dataset_jsonl))
    memory_index = build_memory_index(rows)
    success_seen = Counter()
    examples = []
    for row in rows:
        success = bool(row.get("success", False))
        task = str(row.get("task", "unknown"))
        if args.failures_only and success:
            continue
        if success and not args.include_success:
            continue
        if success:
            if success_seen[task] >= args.max_success_per_task:
                continue
            success_seen[task] += 1
        examples.append(make_example(row, memory_index, args))

    if args.balance_components:
        rng = random.Random(args.seed)
        by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ex in examples:
            by_component[str(ex["metadata"]["wrong_component"])].append(ex)
        balanced = []
        for component, group in sorted(by_component.items()):
            if not group:
                continue
            target = int(args.target_none) if component == "none" else int(args.target_per_failure_component)
            if target <= 0:
                continue
            if len(group) >= target:
                balanced.extend(rng.sample(group, target))
            else:
                balanced.extend(group)
                balanced.extend(rng.choices(group, k=target - len(group)))
        rng.shuffle(balanced)
        examples = balanced

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")

    component_counts = Counter(ex["metadata"]["wrong_component"] for ex in examples)
    task_counts = Counter(ex["metadata"]["task"] for ex in examples)
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "num_input_rows": len(rows),
        "num_examples": len(examples),
        "failures_only": bool(args.failures_only),
        "include_success": bool(args.include_success),
        "pre_rollout_input": bool(args.pre_rollout_input),
        "component_counts": dict(component_counts),
        "top_tasks": dict(task_counts.most_common(30)),
    }
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
