import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import load_feature, load_labeled_rows, load_rollout_rows
from policy_evaluation.oracle_hypothesis_rollout import template_postexec_reflection_decision


def infer_mismatch_type(task: str, success: bool, steps: int, ep_len: int) -> str:
    if success and steps < max(ep_len // 2, 1):
        return "none"
    if "drawer" in task and not success:
        return "drawer_progress_hallucinated"
    if task.startswith("push_") and not success:
        return "object_displacement_overestimated"
    if task.startswith("lift_") and not success:
        return "contact_not_realized"
    if task.startswith("stack_") and not success:
        return "alignment_or_identity_error"
    if task.startswith("place_") and not success:
        return "placement_alignment_failed"
    if not success:
        return "future_outcome_mismatch"
    return "slow_success"


def mean_pool(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim <= 1:
        return arr.reshape(-1).astype(np.float32)
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)


def nearest_success_target(
    task: str,
    current_feature: np.ndarray,
    success_bank: Dict[str, List[Dict[str, Any]]],
) -> tuple[Optional[np.ndarray], Optional[int], float]:
    candidates = success_bank.get(task, [])
    if not candidates:
        return None, None, -1.0
    best_idx = None
    best_score = -float("inf")
    best_target = None
    for item in candidates:
        score = cosine(current_feature, item["obs_current_feature"])
        if score > best_score:
            best_score = score
            best_idx = int(item["row_id"])
            best_target = item["defi_future_feature"]
    return best_target, best_idx, float(best_score)


def infer_state_semantics(task: str, success: bool) -> Dict[str, Any]:
    if "drawer" in task:
        return {
            "completed_actions": [task] if success else [],
            "object_state_summary": "drawer=verified_open" if success else "drawer=unverified",
            "relation_summary": "drawer_handle_contact_verified" if success else "drawer_handle_contact_unverified",
            "precondition_status": "drawer_access_required",
        }
    if task.startswith("push_"):
        return {
            "completed_actions": [task] if success else [],
            "object_state_summary": "object_displaced" if success else "object_not_displaced",
            "relation_summary": "contact_established" if success else "contact_missing",
            "precondition_status": "contact_required",
        }
    if task.startswith("lift_"):
        return {
            "completed_actions": [task] if success else [],
            "object_state_summary": "object_lifted" if success else "object_not_lifted",
            "relation_summary": "grasp_verified" if success else "grasp_unverified",
            "precondition_status": "stable_grasp_required",
        }
    if task.startswith("place_"):
        return {
            "completed_actions": [task] if success else [],
            "object_state_summary": "placement_verified" if success else "placement_unverified",
            "relation_summary": "alignment_verified" if success else "alignment_unverified",
            "precondition_status": "grasp_then_alignment_required",
        }
    if task.startswith("stack_"):
        return {
            "completed_actions": [task] if success else [],
            "object_state_summary": "stack_verified" if success else "stack_unverified",
            "relation_summary": "support_relation_verified" if success else "support_relation_unverified",
            "precondition_status": "alignment_and_contact_required",
        }
    return {
        "completed_actions": [task] if success else [],
        "object_state_summary": "state_verified" if success else "state_unverified",
        "relation_summary": "relation_verified" if success else "relation_unverified",
        "precondition_status": "unknown",
    }


def build_reflection_text(task: str, success: bool, steps: int, ep_len: int, mismatch_type: str) -> Dict[str, Any]:
    decision = template_postexec_reflection_decision(task, success, steps, ep_len, mismatch_type)
    return {
        "trust_score": float(decision.trust_score),
        "trust": "high" if decision.trust_score >= 0.67 else "medium" if decision.trust_score >= 0.34 else "low",
        "mismatch_type": str(decision.mismatch_type),
        "correction_direction": str(decision.correction_direction),
        "recoverability": str(decision.recoverability),
        "reflection_text": str(decision.raw_text),
        "explanation": str(decision.explanation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an RLBench-only memory bank from rollout rows and features.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-slow-success", action="store_true")
    parser.add_argument("--recent-k", type=int, default=4)
    parser.add_argument("--task-filter", type=str, default="")
    args = parser.parse_args()

    rollout_rows = load_rollout_rows(args.rollout_jsonl)
    labeled_rows = load_labeled_rows(args.rollout_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_filter = {item.strip() for item in args.task_filter.split(",") if item.strip()}

    success_bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    memory_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    key_current_features: List[np.ndarray] = []
    key_future_features: List[np.ndarray] = []
    key_repair_deltas: List[np.ndarray] = []
    task_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    mismatch_counter = Counter()
    row_ids: List[int] = []
    future_features: List[np.ndarray] = []
    repair_targets: List[np.ndarray] = []
    repair_deltas: List[np.ndarray] = []
    reflection_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(labeled_rows):
        task = str(row["task"])
        if task_filter and task not in task_filter:
            continue
        row_id = int(row.get("row_id", idx))
        rollout = rollout_rows[row_id]
        feat = load_feature(args.feature_root / rollout["feature_path"])
        success = bool(row["success"])
        steps = int(row["steps"])
        ep_len = int(row["ep_len"])
        mismatch_type = infer_mismatch_type(task, success, steps, ep_len)
        current_feature = feat["obs_current_feature"].astype(np.float32)
        future_feature = feat["defi_future_feature"].astype(np.float32)
        target_feature, target_row_id, retrieval_score = nearest_success_target(task, current_feature, success_bank)
        if target_feature is None:
            target_feature = future_feature
            target_row_id = None
            retrieval_score = -1.0
        repair_delta = np.asarray(target_feature, dtype=np.float32) - future_feature
        lang_text = str(row.get("lang_text", row.get("lang_goal", task.replace("_", " "))))
        reflection = build_reflection_text(task, success, steps, ep_len, mismatch_type)

        if success:
            success_bank[task].append(
                {
                    "row_id": row_id,
                    "obs_current_feature": current_feature,
                    "defi_future_feature": future_feature,
                }
            )

        diagnostic_semantics = {
            "is_mismatch": bool(mismatch_type not in {"none", "unknown"}),
            "mismatch_type": mismatch_type,
            "trust_score": float(row.get("trust_score", 0.5)),
            "retrieved_success_row_id": target_row_id,
            "retrieved_success_score": float(retrieval_score),
            "lang_goal": lang_text,
        }
        state_semantics = infer_state_semantics(task, success)

        memory_row = {
            "row_id": row_id,
            "sequence_index": int(row.get("sequence_index", -1)),
            "subtask_index": int(row.get("subtask_index", -1)),
            "task": task,
            "lang_goal": lang_text,
            "outcome": {
                "success": success,
                "steps": steps,
                "ep_len": ep_len,
            },
            "state_semantics": state_semantics,
            "diagnostic_semantics": diagnostic_semantics,
            "mismatch_type": mismatch_type,
            "trust_score": float(reflection["trust_score"]),
            "trust": str(reflection["trust"]),
            "reflection_text": reflection["reflection_text"],
            "correction_direction": reflection["correction_direction"],
            "recoverability": reflection["recoverability"],
            "explanation": reflection["explanation"],
            "repair_delta_norm": float(np.linalg.norm(repair_delta.reshape(-1))),
            "repair_helped": bool(mismatch_type != "none" and target_row_id is not None),
            "feature_path": str(rollout["feature_path"]),
            "retrieved_success_row_id": target_row_id,
            "retrieved_success_score": float(retrieval_score),
        }
        memory_rows.append(memory_row)
        task_to_rows[task].append(memory_row)
        mismatch_counter[mismatch_type] += 1

        keep = mismatch_type != "none" and (args.include_slow_success or mismatch_type != "slow_success")
        if keep:
            key_rows.append(memory_row)
            key_current_features.append(current_feature)
            key_future_features.append(future_feature)
            key_repair_deltas.append(repair_delta)

        row_ids.append(row_id)
        future_features.append(future_feature)
        repair_targets.append(np.asarray(target_feature, dtype=np.float32))
        repair_deltas.append(repair_delta.astype(np.float32))
        reflection_rows.append(
            {
                "row_id": row_id,
                "task": task,
                "lang_goal": lang_text,
                "success": success,
                "steps": steps,
                "ep_len": ep_len,
                "mismatch_type": mismatch_type,
                "trust_score": float(reflection["trust_score"]),
                "trust": str(reflection["trust"]),
                "correction_direction": reflection["correction_direction"],
                "recoverability": reflection["recoverability"],
                "reflection_text": reflection["reflection_text"],
                "explanation": reflection["explanation"],
                "retrieved_success_row_id": target_row_id,
                "retrieved_success_score": float(retrieval_score),
                "feature_path": str(rollout["feature_path"]),
            }
        )

    with (args.output_dir / "memory_rows.jsonl").open("w") as handle:
        for row in memory_rows:
            handle.write(json.dumps(row) + "\n")

    with (args.output_dir / "key_memory.jsonl").open("w") as handle:
        for row in key_rows:
            handle.write(json.dumps(row) + "\n")

    with (args.output_dir / "reflection_rows.jsonl").open("w") as handle:
        for row in reflection_rows:
            handle.write(json.dumps(row) + "\n")

    np.savez_compressed(
        args.output_dir / "key_memory_features.npz",
        row_ids=np.asarray([int(row["row_id"]) for row in key_rows], dtype=np.int64),
        tasks=np.asarray([str(row["task"]) for row in key_rows], dtype=object),
        mismatch_types=np.asarray([str(row["mismatch_type"]) for row in key_rows], dtype=object),
        lang_goals=np.asarray([str(row.get("lang_goal", "")) for row in key_rows], dtype=object),
        current_features=np.stack(key_current_features, axis=0).astype(np.float32) if key_current_features else np.zeros((0, 1), dtype=np.float32),
        future_features=np.stack(key_future_features, axis=0).astype(np.float32) if key_future_features else np.zeros((0, 1), dtype=np.float32),
        repair_deltas=np.stack(key_repair_deltas, axis=0).astype(np.float32) if key_repair_deltas else np.zeros((0, 1), dtype=np.float32),
        repair_targets=np.stack(repair_targets, axis=0).astype(np.float32) if repair_targets else np.zeros((0, 1), dtype=np.float32),
    )

    rule_memory = {}
    for task, rows in sorted(task_to_rows.items()):
        mismatch_counts = Counter(str(row["mismatch_type"]) for row in rows if str(row["mismatch_type"]) != "none")
        correction_counts = Counter(str(row.get("correction_direction", "")).strip() for row in rows if str(row.get("correction_direction", "")).strip())
        rule_memory[task] = {
            "rule_text": f"Preserve task progress for {task}; trust success histories more than failure futures.",
            "common_mismatch_type": mismatch_counts.most_common(1)[0][0] if mismatch_counts else "none",
            "common_correction_direction": correction_counts.most_common(1)[0][0] if correction_counts else "stabilize contact",
            "count": len(rows),
        }
    rule_memory["default"] = "Prefer conservative future calibration and preserve verified progress."
    (args.output_dir / "repair_rule_memory.json").write_text(json.dumps(rule_memory, indent=2) + "\n")
    (args.output_dir / "recent_memory_spec.json").write_text(json.dumps({"type": "online_queue", "recent_k": int(args.recent_k)}, indent=2) + "\n")

    summary = {
        "num_memory_rows": len(memory_rows),
        "num_key_rows": len(key_rows),
        "num_reflection_rows": len(reflection_rows),
        "recent_k": int(args.recent_k),
        "include_slow_success": bool(args.include_slow_success),
        "mismatch_counts": dict(mismatch_counter),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
