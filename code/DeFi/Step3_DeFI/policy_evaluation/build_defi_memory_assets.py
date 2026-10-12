import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import load_feature, load_labeled_rows, load_rollout_rows
from policy_evaluation.oracle_hypothesis_rollout import RULE_MEMORY


def infer_state_semantics(task: str, success: bool) -> Dict[str, Any]:
    tokens = task.split("_")
    completed_actions: List[str] = [task] if success else []
    object_state_summary = "state_unknown"
    relation_summary = "relation_unknown"
    precondition_status = "unknown"

    if task == "open_drawer":
        object_state_summary = "drawer=open" if success else "drawer=not_open"
        precondition_status = "drawer_access_required"
    elif task == "close_drawer":
        object_state_summary = "drawer=closed" if success else "drawer=not_closed"
    elif task == "move_slider_left":
        object_state_summary = "slider=left" if success else "slider=not_left"
    elif task == "move_slider_right":
        object_state_summary = "slider=right" if success else "slider=not_right"
    elif task == "turn_on_lightbulb":
        object_state_summary = "lightbulb=on" if success else "lightbulb=off"
        relation_summary = "switch_contact_established" if success else "switch_contact_missing"
    elif task == "turn_off_lightbulb":
        object_state_summary = "lightbulb=off" if success else "lightbulb=on"
        relation_summary = "switch_contact_established" if success else "switch_contact_missing"
    elif task == "turn_on_led":
        object_state_summary = "led=on" if success else "led=off"
    elif task == "turn_off_led":
        object_state_summary = "led=off" if success else "led=on"
    elif task == "stack_block":
        object_state_summary = "stack=assembled" if success else "stack=not_assembled"
        relation_summary = "block_on_block" if success else "block_alignment_missing"
        precondition_status = "grasp_and_alignment_required"
    elif task == "unstack_block":
        object_state_summary = "stack=disassembled" if success else "stack=still_assembled"
        relation_summary = "block_separated" if success else "block_still_contacting"
    elif task.startswith("push_"):
        obj = tokens[1] if len(tokens) > 1 else "object"
        target = tokens[-1] if len(tokens) > 2 else "target"
        object_state_summary = f"{obj}_block={target}" if success else f"{obj}_block=not_{target}"
        relation_summary = "gripper_object_contact" if success else "contact_weak_or_missing"
        precondition_status = "contact_required"
    elif task.startswith("lift_"):
        obj = tokens[1] if len(tokens) > 1 else "object"
        src = tokens[-1] if len(tokens) > 2 else "surface"
        object_state_summary = f"{obj}_block=lifted_from_{src}" if success else f"{obj}_block=not_lifted_from_{src}"
        relation_summary = "object_in_gripper" if success else "gripper_object_grasp_missing"
        precondition_status = "stable_grasp_required"
    elif task.startswith("place_in_"):
        target = tokens[-1] if len(tokens) > 2 else "container"
        object_state_summary = f"object=in_{target}" if success else f"object=not_in_{target}"
        relation_summary = "released_into_target" if success else "release_or_alignment_failed"
        precondition_status = "grasp_then_alignment_required"
    elif task.startswith("rotate_"):
        obj = tokens[1] if len(tokens) > 1 else "object"
        direction = tokens[-1] if len(tokens) > 2 else "target_orientation"
        object_state_summary = f"{obj}_block=rotated_{direction}" if success else f"{obj}_block=not_rotated_{direction}"
        relation_summary = "controlled_rotation" if success else "rotation_control_missing"
        precondition_status = "stable_contact_required"

    return {
        "completed_actions": completed_actions,
        "object_state_summary": object_state_summary,
        "relation_summary": relation_summary,
        "precondition_status": precondition_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline memory assets from DeFi single-future rollout labels.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--labeled-jsonl", type=Path, required=True)
    parser.add_argument("--repair-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recent-k", type=int, default=4)
    parser.add_argument("--include-slow-success", action="store_true")
    args = parser.parse_args()

    rollout_rows = load_rollout_rows(args.rollout_jsonl)
    labeled_rows = load_labeled_rows(args.labeled_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repair_npz = np.load(args.repair_targets, allow_pickle=True)
    row_ids = repair_npz["row_ids"].astype(np.int64)
    repair_deltas = repair_npz["repair_deltas"].astype(np.float32)
    repair_targets = repair_npz["repair_targets"].astype(np.float32)
    row_to_idx = {int(row_id): idx for idx, row_id in enumerate(row_ids.tolist())}

    memory_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    key_current_features = []
    key_future_features = []
    key_repair_deltas = []
    task_to_rows = defaultdict(list)
    mismatch_counter = Counter()

    for labeled in labeled_rows:
        row_id = int(labeled["row_id"])
        rollout = rollout_rows[row_id]
        feat = load_feature(args.feature_root / rollout["feature_path"])
        repair_idx = row_to_idx[row_id]
        repair_delta = repair_deltas[repair_idx]
        repaired_target = repair_targets[repair_idx]
        mismatch_type = str(labeled["mismatch_type"])
        success = bool(labeled["success"])
        repair_helped_proxy = bool(mismatch_type != "none" and labeled.get("retrieved_success_row_id") is not None and float(np.linalg.norm(repair_delta.reshape(-1))) > 1e-6)
        state_semantics = infer_state_semantics(str(labeled["task"]), success)
        diagnostic_semantics = {
            "is_mismatch": bool(mismatch_type not in {"none", "unknown"}),
            "mismatch_type": mismatch_type,
            "trust_score": float(labeled.get("trust_score", 0.5)),
            "correction_direction": str(labeled.get("correction_direction", "")),
            "recoverability": str(labeled.get("recoverability", "")),
            "repair_helped": repair_helped_proxy,
            "retrieved_success_row_id": labeled.get("retrieved_success_row_id"),
            "retrieved_success_score": float(labeled.get("retrieved_success_score", -1.0)),
        }
        row = {
            "row_id": row_id,
            "sequence_index": int(labeled["sequence_index"]),
            "subtask_index": int(labeled["subtask_index"]),
            "task": str(labeled["task"]),
            "outcome": {
                "success": success,
                "steps": int(labeled["steps"]),
                "ep_len": int(labeled["ep_len"]),
            },
            "state_semantics": state_semantics,
            "diagnostic_semantics": diagnostic_semantics,
            "mismatch_type": mismatch_type,
            "trust_score": float(labeled.get("trust_score", 0.5)),
            "reflection_text": str(labeled.get("reflection_text", "")),
            "correction_direction": str(labeled.get("correction_direction", "")),
            "recoverability": str(labeled.get("recoverability", "")),
            "repair_delta_norm": float(np.linalg.norm(repair_delta.reshape(-1))),
            "repair_helped": repair_helped_proxy,
            "repair_helped_is_proxy": True,
            "retrieved_success_row_id": labeled.get("retrieved_success_row_id"),
            "retrieved_success_score": float(labeled.get("retrieved_success_score", -1.0)),
            "feature_path": str(rollout["feature_path"]),
        }
        memory_rows.append(row)
        task_to_rows[row["task"]].append(row)
        mismatch_counter[mismatch_type] += 1

        keep = mismatch_type != "none" and (args.include_slow_success or mismatch_type != "slow_success")
        if keep:
            key_rows.append(row)
            key_current_features.append(feat["obs_current_feature"].astype(np.float32))
            key_future_features.append(feat["defi_future_feature"].astype(np.float32))
            key_repair_deltas.append(repair_delta.astype(np.float32))

    with (args.output_dir / "memory_rows.jsonl").open("w") as handle:
        for row in memory_rows:
            handle.write(json.dumps(row) + "\n")

    with (args.output_dir / "key_memory.jsonl").open("w") as handle:
        for row in key_rows:
            handle.write(json.dumps(row) + "\n")

    np.savez_compressed(
        args.output_dir / "key_memory_features.npz",
        row_ids=np.asarray([int(row["row_id"]) for row in key_rows], dtype=np.int64),
        tasks=np.asarray([str(row["task"]) for row in key_rows], dtype=object),
        mismatch_types=np.asarray([str(row["mismatch_type"]) for row in key_rows], dtype=object),
        current_features=np.stack(key_current_features, axis=0).astype(np.float32) if key_current_features else np.zeros((0, 1), dtype=np.float32),
        future_features=np.stack(key_future_features, axis=0).astype(np.float32) if key_future_features else np.zeros((0, 1), dtype=np.float32),
        repair_deltas=np.stack(key_repair_deltas, axis=0).astype(np.float32) if key_repair_deltas else np.zeros((0, 1), dtype=np.float32),
        repair_targets=np.stack([repair_targets[row_to_idx[int(row["row_id"])]] for row in key_rows], axis=0).astype(np.float32) if key_rows else np.zeros((0, 1), dtype=np.float32),
    )

    rule_memory = {}
    for task, rows in sorted(task_to_rows.items()):
        mismatch_counts = Counter(str(row["mismatch_type"]) for row in rows if str(row["mismatch_type"]) != "none")
        correction_counts = Counter(str(row["correction_direction"]) for row in rows if str(row["correction_direction"]).strip())
        rule_memory[task] = {
            "rule_text": RULE_MEMORY.get(task, "Prefer conservative future calibration and preserve verified progress."),
            "common_mismatch_type": mismatch_counts.most_common(1)[0][0] if mismatch_counts else "none",
            "common_correction_direction": correction_counts.most_common(1)[0][0] if correction_counts else "stabilize contact",
            "count": len(rows),
        }
    (args.output_dir / "repair_rule_memory.json").write_text(json.dumps(rule_memory, indent=2) + "\n")
    (args.output_dir / "recent_memory_spec.json").write_text(
        json.dumps({"type": "online_queue", "recent_k": int(args.recent_k)}, indent=2) + "\n"
    )

    summary = {
        "num_memory_rows": len(memory_rows),
        "num_key_rows": len(key_rows),
        "recent_k": int(args.recent_k),
        "include_slow_success": bool(args.include_slow_success),
        "mismatch_counts": dict(mismatch_counter),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
