import argparse
import json
from pathlib import Path
from typing import Any, Dict

from policy_evaluation.defi_memory_models import iter_jsonl
from policy_evaluation.oracle_hypothesis_rollout import trust_score_to_label


def heuristic_trust_score(row: Dict[str, Any]) -> float:
    success = bool(row.get("success", False))
    mismatch = str(row.get("mismatch_type", "uncertain"))
    steps = int(row.get("steps", 0) or 0)
    ep_len = max(int(row.get("ep_len", 160) or 160), 1)
    alignment = float(row.get("future_alignment_to_actual", 0.5) or 0.5)
    step_ratio = float(steps) / float(ep_len)

    if success and mismatch == "none":
        return 0.85 if alignment >= 0.6 or step_ratio <= 0.5 else 0.7
    if success:
        return 0.5
    if mismatch in {
        "contact_not_realized",
        "object_displacement_overestimated",
        "object_displacement_underestimated",
        "wrong_gripper_state",
        "wrong_object_relation",
        "precondition_not_satisfied",
        "future_temporally_inconsistent",
        "goal_progress_hallucinated",
        "wrong_object_identity",
        "drawer_progress_hallucinated",
        "future_outcome_mismatch",
    }:
        return 0.2
    return 0.35


def build_reflection_text(row: Dict[str, Any]) -> str:
    trust = str(row["trust"])
    mismatch = str(row["mismatch_type"])
    correction = str(row["correction_direction"])
    return f"trust: {trust} | mismatch_type: {mismatch} | correction_direction: {correction}"


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    trust_score = heuristic_trust_score(row)
    normalized = dict(row)
    normalized["trust_score_original"] = float(row.get("trust_score", 0.5) or 0.5)
    normalized["trust"] = trust_score_to_label(trust_score)
    normalized["trust_score"] = trust_score
    normalized["mismatch_type"] = str(row.get("mismatch_type", "uncertain"))[:128]
    normalized["correction_direction"] = str(row.get("correction_direction", "stabilize contact"))[:256]
    normalized["reflection_text"] = build_reflection_text(normalized)
    normalized.pop("recoverability", None)
    normalized.pop("explanation", None)
    return normalized


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert reflection labels to minimal 3-field schema.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--key-memory-jsonl", type=Path, default=None)
    parser.add_argument("--output-labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-key-memory-jsonl", type=Path, default=None)
    args = parser.parse_args()

    label_rows = [normalize_row(row) for row in iter_jsonl(args.labels_jsonl)]
    write_jsonl(args.output_labels_jsonl, label_rows)

    key_rows_written = 0
    if args.key_memory_jsonl is not None and args.output_key_memory_jsonl is not None:
        key_rows = [normalize_row(row) for row in iter_jsonl(args.key_memory_jsonl)]
        write_jsonl(args.output_key_memory_jsonl, key_rows)
        key_rows_written = len(key_rows)

    summary = {
        "label_rows": len(label_rows),
        "key_memory_rows": key_rows_written,
        "output_labels_jsonl": str(args.output_labels_jsonl),
        "output_key_memory_jsonl": str(args.output_key_memory_jsonl) if args.output_key_memory_jsonl is not None else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
