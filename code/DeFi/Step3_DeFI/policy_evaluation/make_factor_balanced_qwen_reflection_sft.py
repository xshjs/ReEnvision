from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a memory-grounded physical-future diagnosis module for robot manipulation. "
    "Given the current task, rollout outcome evidence, and retrieved memory evidence, "
    "diagnose which latent factor in the predicted future is unreliable. "
    "Return only one compact JSON object with keys: "
    '{"trust": "high|medium|low", "mismatch_type": string, "failure_factor": string, '
    '"hypothetical_failure": string, "counterfactual_future": string, '
    '"correction_direction": string, "intervention": string, "recoverability": string}.'
)


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "object_displacement_underestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
    "goal_progress_hallucinated": "goal_completion",
}

FACTOR_TEXT = {
    "contact": {
        "hypothetical_failure": "the robot fails to establish or maintain stable contact",
        "counterfactual_future": "the gripper or tool keeps stable contact until the object state changes",
        "correction_direction": "increase contact stability",
        "intervention": "edit contact latent channels toward stable grasp or pushing contact",
    },
    "object_displacement": {
        "hypothetical_failure": "the object moves less or differently than the predicted future",
        "counterfactual_future": "the object displacement matches the required target direction and distance",
        "correction_direction": "reduce over-optimistic displacement and align object motion",
        "intervention": "edit object-motion latent channels toward measured displacement evidence",
    },
    "object_identity": {
        "hypothetical_failure": "the future tracks or manipulates the wrong object",
        "counterfactual_future": "the future follows the instructed object identity and ignores distractors",
        "correction_direction": "reselect the instructed object",
        "intervention": "edit object-identity latent channels toward the target object",
    },
    "drawer_slider_progress": {
        "hypothetical_failure": "the drawer or slider progress is hallucinated before it actually moves",
        "counterfactual_future": "the drawer or slider reaches the required open or closed state",
        "correction_direction": "verify articulated-object progress",
        "intervention": "edit articulated-progress latent channels using drawer or slider state evidence",
    },
    "goal_completion": {
        "hypothetical_failure": "the predicted future marks the goal as complete before the physical state verifies it",
        "counterfactual_future": "the future preserves progress only after the task state is physically verified",
        "correction_direction": "preserve verified progress and avoid premature completion",
        "intervention": "edit task-progress latent channels toward conservative completion evidence",
    },
    "none": {
        "hypothetical_failure": "none",
        "counterfactual_future": "preserve the current future because the task is progressing correctly",
        "correction_direction": "preserve progress",
        "intervention": "do not edit latent factors",
    },
}


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def nested(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_memory_row(row: dict[str, Any]) -> dict[str, Any]:
    mismatch = str(row.get("mismatch_type") or nested(row, "diagnostic_semantics", "mismatch_type", default="unknown"))
    factor = MISMATCH_TO_FACTOR.get(mismatch, "goal_completion")
    text = FACTOR_TEXT[factor]
    success = bool(nested(row, "outcome", "success", default=row.get("success", False)))
    steps = int(nested(row, "outcome", "steps", default=row.get("steps", 160)) or 160)
    ep_len = int(nested(row, "outcome", "ep_len", default=row.get("ep_len", 160)) or 160)
    retrieved_score = row.get("retrieved_success_score") or nested(row, "diagnostic_semantics", "retrieved_success_score", default=None)
    state_semantics = row.get("state_semantics") or {}
    return {
        "source": "memory_failure",
        "row_id": row.get("row_id"),
        "task": str(row.get("task", "unknown")),
        "success": success,
        "steps": steps,
        "ep_len": ep_len,
        "state_semantics": state_semantics,
        "retrieved_success_row_id": row.get("retrieved_success_row_id") or nested(row, "diagnostic_semantics", "retrieved_success_row_id", default=None),
        "retrieved_success_score": retrieved_score,
        "target": {
            "trust": "low",
            "mismatch_type": mismatch,
            "failure_factor": factor,
            "hypothetical_failure": text["hypothetical_failure"],
            "counterfactual_future": text["counterfactual_future"],
            "correction_direction": text["correction_direction"],
            "intervention": text["intervention"],
            "recoverability": "retry_needed" if mismatch != "wrong_object_identity" else "uncertain",
        },
    }


def normalize_success_row(row: dict[str, Any]) -> dict[str, Any]:
    text = FACTOR_TEXT["none"]
    return {
        "source": "rollout_success",
        "row_id": f"{row.get('sequence_index', 'x')}_{row.get('subtask_index', 'x')}",
        "task": str(row.get("task", "unknown")),
        "success": True,
        "steps": int(row.get("steps", 0) or 0),
        "ep_len": int(row.get("ep_len", 160) or 160),
        "state_semantics": row.get("state_semantics") or {},
        "retrieved_success_row_id": None,
        "retrieved_success_score": None,
        "target": {
            "trust": "high",
            "mismatch_type": "none",
            "failure_factor": "none",
            "hypothetical_failure": text["hypothetical_failure"],
            "counterfactual_future": text["counterfactual_future"],
            "correction_direction": text["correction_direction"],
            "intervention": text["intervention"],
            "recoverability": "not_needed",
        },
    }


def build_user_prompt(item: dict[str, Any]) -> str:
    payload = {
        "task": item["task"],
        "executed_outcome": {
            "success": item["success"],
            "steps": item["steps"],
            "ep_len": item["ep_len"],
        },
        "state_semantics": item.get("state_semantics", {}),
        "retrieved_memory_evidence": {
            "retrieved_success_row_id": item.get("retrieved_success_row_id"),
            "retrieved_success_score": item.get("retrieved_success_score"),
        },
        "instruction": (
            "Diagnose the unreliable physical-future latent factor. "
            "Use none/preserve progress only when the current outcome is successful."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_sample(handle, item: dict[str, Any]) -> None:
    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)},
            {"role": "assistant", "content": json.dumps(item["target"], ensure_ascii=False)},
        ],
        "metadata": {
            "source": item["source"],
            "row_id": item["row_id"],
            "task": item["task"],
            "failure_factor": item["target"]["failure_factor"],
            "mismatch_type": item["target"]["mismatch_type"],
        },
    }
    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build factor-balanced Qwen reflection SFT data from memory failures and rollout successes.")
    parser.add_argument("--memory-jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--success-rollout-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-factor", type=int, default=72)
    parser.add_argument("--success-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_factor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.memory_jsonl:
        for row in iter_jsonl(path):
            item = normalize_memory_row(row)
            by_factor[item["target"]["failure_factor"]].append(item)

    selected: list[dict[str, Any]] = []
    for factor, rows in sorted(by_factor.items()):
        rng.shuffle(rows)
        selected.extend(rows[: args.max_per_factor])

    success_rows = [normalize_success_row(row) for row in iter_jsonl(args.success_rollout_jsonl) if bool(row.get("success"))]
    rng.shuffle(success_rows)
    n_success = int(round(len(selected) * float(args.success_ratio)))
    selected.extend(success_rows[:n_success])
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for item in selected:
            write_sample(handle, item)

    summary = {
        "written": len(selected),
        "output": str(args.output),
        "factor_counts": Counter(item["target"]["failure_factor"] for item in selected),
        "mismatch_counts": Counter(item["target"]["mismatch_type"] for item in selected),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
