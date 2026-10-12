from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a robot world-model intervention programmer. "
    "Given the current CALVIN task, rollout evidence, and retrieved memory evidence, "
    "produce a structured counterfactual intervention program for editing the predicted future latent. "
    "Return only one compact JSON object with keys: "
    '{"failure_factor": string, "object": string, "time_window": [number, number], '
    '"current_relation": string, "target_relation": string, "gamma": number}. '
    "failure_factor must be one of contact, object_displacement, object_identity, "
    "drawer_slider_progress, goal_completion, none."
)


FACTOR_PROGRAMS = {
    "contact": {
        "time_window": [0.20, 0.55],
        "current_relation": "contact is weak missing or unstable",
        "target_relation": "stable contact is maintained until the object state changes",
        "gamma": 0.20,
    },
    "object_displacement": {
        "time_window": [0.35, 0.75],
        "current_relation": "object displacement is overestimated or misaligned",
        "target_relation": "object displacement matches the commanded direction and distance",
        "gamma": 0.18,
    },
    "object_identity": {
        "time_window": [0.10, 0.45],
        "current_relation": "future may track or manipulate the wrong object",
        "target_relation": "future tracks the instructed target object and ignores distractors",
        "gamma": 0.16,
    },
    "drawer_slider_progress": {
        "time_window": [0.35, 0.85],
        "current_relation": "articulated object progress is hallucinated before motion is verified",
        "target_relation": "drawer or slider progress is verified by physical state change",
        "gamma": 0.18,
    },
    "goal_completion": {
        "time_window": [0.60, 1.00],
        "current_relation": "task completion is predicted before the physical state verifies it",
        "target_relation": "task completion is accepted only after verified physical progress",
        "gamma": 0.10,
    },
    "none": {
        "time_window": [0.00, 0.00],
        "current_relation": "future is consistent with observed progress",
        "target_relation": "preserve the current predicted future",
        "gamma": 0.00,
    },
}


COLOR_WORDS = {"red", "blue", "pink", "green", "yellow", "black", "white", "orange", "purple"}


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def safe_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


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


def program_for(task: str, factor: str, confidence: float | None = None) -> dict[str, Any]:
    factor = factor if factor in FACTOR_PROGRAMS else "goal_completion"
    template = FACTOR_PROGRAMS[factor]
    gamma = float(template["gamma"])
    if confidence is not None and factor != "none":
        gamma = max(0.05, min(0.25, gamma * (0.75 + 0.5 * float(confidence))))
    return {
        "failure_factor": factor,
        "object": "none" if factor == "none" else object_from_task(task),
        "time_window": [float(template["time_window"][0]), float(template["time_window"][1])],
        "current_relation": template["current_relation"],
        "target_relation": template["target_relation"],
        "gamma": round(gamma, 4),
    }


def user_payload_from_factor_sft(row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    user = safe_json(row["messages"][1]["content"])
    task = str(row.get("metadata", {}).get("task") or user.get("task", "unknown"))
    return {
        "task": task,
        "executed_outcome": user.get("executed_outcome", {}),
        "state_semantics": user.get("state_semantics", {}),
        "retrieved_memory_evidence": user.get("retrieved_memory_evidence", {}),
        "diagnosis": {
            "mismatch_type": target.get("mismatch_type", row.get("metadata", {}).get("mismatch_type", "unknown")),
            "failure_factor": target.get("failure_factor", row.get("metadata", {}).get("failure_factor", "unknown")),
            "correction_direction": target.get("correction_direction", ""),
            "intervention": target.get("intervention", ""),
        },
        "instruction": "Generate the minimal structured latent intervention program.",
    }


def sample_from_factor_sft(row: dict[str, Any]) -> dict[str, Any]:
    target = safe_json(row["messages"][-1]["content"])
    task = str(row.get("metadata", {}).get("task", "unknown"))
    factor = str(target.get("failure_factor", row.get("metadata", {}).get("failure_factor", "unknown")))
    payload = user_payload_from_factor_sft(row, target)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            {"role": "assistant", "content": json.dumps(program_for(task, factor), ensure_ascii=False)},
        ],
        "metadata": {
            "source": "factor_sft_program",
            "task": task,
            "failure_factor": factor,
            "mismatch_type": str(target.get("mismatch_type", row.get("metadata", {}).get("mismatch_type", "unknown"))),
        },
    }


def sample_from_pseudo_rollout(row: dict[str, Any]) -> dict[str, Any]:
    task = str(row.get("task", "unknown"))
    factor = str(row.get("pseudo_failure_factor", "unknown"))
    confidence = float(row.get("pseudo_confidence", 0.0) or 0.0)
    payload = {
        "task": task,
        "executed_outcome": {
            "success": bool(row.get("success", False)),
            "steps": int(row.get("steps", 0) or 0),
        },
        "retrieved_memory_evidence": {
            "factor_votes": row.get("factor_votes", {}),
            "mismatch_votes": row.get("mismatch_votes", {}),
            "top_memory": row.get("adapter_memory_evidence", [])[:4],
            "pseudo_confidence": confidence,
        },
        "diagnosis": {
            "mismatch_type": str(row.get("pseudo_mismatch_type", "unknown")),
            "failure_factor": factor,
        },
        "instruction": "Generate the minimal structured latent intervention program.",
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            {"role": "assistant", "content": json.dumps(program_for(task, factor, confidence), ensure_ascii=False)},
        ],
        "metadata": {
            "source": "retrieval_pseudo_program",
            "sequence_index": int(row.get("sequence_index", -1)),
            "subtask_index": int(row.get("subtask_index", -1)),
            "task": task,
            "failure_factor": factor,
            "mismatch_type": str(row.get("pseudo_mismatch_type", "unknown")),
            "pseudo_confidence": confidence,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen intervention-program SFT data.")
    parser.add_argument("--factor-sft-jsonl", type=Path, required=True)
    parser.add_argument("--pseudo-rollout-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pseudo-per-factor", type=int, default=160)
    parser.add_argument("--min-pseudo-confidence", type=float, default=0.34)
    parser.add_argument("--success-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = [sample_from_factor_sft(row) for row in iter_jsonl(args.factor_sft_jsonl)]

    pseudo_by_factor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    success_pseudo = []
    for row in iter_jsonl(args.pseudo_rollout_jsonl):
        sample = sample_from_pseudo_rollout(row)
        if float(sample["metadata"].get("pseudo_confidence", 0.0)) < args.min_pseudo_confidence:
            continue
        if bool(row.get("success", False)):
            success_pseudo.append(sample)
        else:
            pseudo_by_factor[str(sample["metadata"]["failure_factor"])].append(sample)

    for factor, rows in sorted(pseudo_by_factor.items()):
        rng.shuffle(rows)
        samples.extend(rows[: args.max_pseudo_per_factor])

    rng.shuffle(success_pseudo)
    max_success = int(round(len(samples) * args.success_ratio))
    samples.extend(success_pseudo[:max_success])
    rng.shuffle(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "written": len(samples),
        "output": str(args.output),
        "source_counts": Counter(str(row["metadata"]["source"]) for row in samples),
        "factor_counts": Counter(str(row["metadata"]["failure_factor"]) for row in samples),
        "mismatch_counts": Counter(str(row["metadata"].get("mismatch_type", "")) for row in samples),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
