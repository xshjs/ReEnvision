from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from policy_evaluation.defi_memory_models import iter_jsonl, load_feature, load_rollout_rows
from policy_evaluation.memory_bank import MemoryBank
from policy_evaluation.prepare_reflection_labels_minimal import heuristic_trust_score
SYSTEM_PROMPT = (
    "You are a memory-grounded robot manipulation reflector. "
    "Read retrieved memory evidence, select the most relevant memory cases, assign normalized weights, "
    "diagnose the mismatch, and return one compact JSON object only with keys: "
    '{"selected_memory_ids": [string], "memory_weights": {string: number}, "evidence_summary": string, '
    '"trust_score": number, "mismatch_type": string, "correction_direction": string, "recoverability": string}.'
)


def load_feature_root(path: Path, row: Dict[str, Any], rollout_rows: Dict[int, Dict[str, Any]]) -> np.ndarray:
    feature_path = row.get("feature_path")
    if feature_path is None:
        rollout = rollout_rows.get(int(row["row_id"]))
        if rollout is None:
            raise KeyError(f"missing rollout row for row_id={row['row_id']}")
        feature_path = rollout.get("feature_path")
    if feature_path is None:
        raise KeyError(f"feature_path is missing for row_id={row['row_id']}")
    feature = load_feature(path / str(feature_path))
    return feature["defi_future_feature"].astype(np.float32).reshape(-1)


def normalize_weights(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float32)
    arr = np.clip(arr, 1e-6, None)
    arr = arr / float(arr.sum())
    return arr.tolist()


def recoverability_label(row: Dict[str, Any]) -> str:
    if bool(row.get("success", False)):
        return "not_needed"
    mismatch = str(row.get("mismatch_type", "unknown"))
    if mismatch in {"contact_not_realized", "object_displacement_overestimated", "future_outcome_mismatch"}:
        return "retry_needed"
    return "uncertain"


def correction_label(row: Dict[str, Any]) -> str:
    return str(row.get("correction_direction", "stabilize contact"))


def build_user_prompt(row: Dict[str, Any], evidence_payload: Dict[str, Any]) -> str:
    payload = {
        "task": str(row["task"]),
        "success": bool(row.get("success", False)),
        "steps": int(row.get("steps", 0)),
        "ep_len": int(row.get("ep_len", 160)),
        "future_alignment_to_actual": round(float(row.get("future_alignment_to_actual", 0.5)), 4),
        "retrieved_state_memory": {
            "recent_candidates": [
                {
                    "memory_id": item["memory_id"],
                    "task": item.get("task"),
                    "state_semantics": item.get("state_semantics", {}),
                    "similarity_score": item["similarity_score"],
                }
                for item in evidence_payload["recent_candidates"]
            ],
            "key_candidates": [
                {
                    "memory_id": item["memory_id"],
                    "task": item.get("task"),
                    "state_semantics": item.get("state_semantics", {}),
                    "similarity_score": item["similarity_score"],
                }
                for item in evidence_payload["key_candidates"]
            ],
        },
        "retrieved_diagnostic_memory": evidence_payload,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_assistant_target(row: Dict[str, Any], evidence_payload: Dict[str, Any]) -> str:
    all_candidates = evidence_payload["recent_candidates"] + evidence_payload["key_candidates"]
    selected = all_candidates[: min(3, len(all_candidates))]
    selected_ids = [item["memory_id"] for item in selected]
    weights = normalize_weights([float(item["similarity_score"]) for item in selected])
    memory_weights = {memory_id: round(float(weight), 4) for memory_id, weight in zip(selected_ids, weights)}
    evidence_summary = " ; ".join(
        [
            f"{item['memory_id']} state={item.get('state_semantics', {}).get('object_state_summary', 'state_unknown')} "
            f"supports {item['mismatch_type']} with similarity {float(item['similarity_score']):.2f}"
            for item in selected
        ]
    )[:300]
    target = {
        "selected_memory_ids": selected_ids,
        "memory_weights": memory_weights,
        "evidence_summary": evidence_summary or "no strong evidence",
        "trust_score": round(float(heuristic_trust_score(row)), 4),
        "mismatch_type": str(row.get("mismatch_type", "unknown")),
        "correction_direction": correction_label(row),
        "recoverability": recoverability_label(row),
    }
    return json.dumps(target, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memory-grounded reflection SFT JSONL.")
    parser.add_argument("--labeled-jsonl", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--key-memory-jsonl", type=Path, required=True)
    parser.add_argument("--recent-memory-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk-recent", type=int, default=2)
    parser.add_argument("--topk-key", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    memory_bank = MemoryBank.from_jsonl(
        key_memory_path=args.key_memory_jsonl,
        recent_memory_path=args.recent_memory_jsonl,
    )
    rollout_rows = load_rollout_rows(args.rollout_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as handle:
        for idx, row in enumerate(iter_jsonl(args.labeled_jsonl)):
            future_feature = load_feature_root(args.feature_root, row, rollout_rows)
            retrieved = memory_bank.retrieve(
                future_feature=future_feature,
                task=str(row["task"]),
                topk_recent=args.topk_recent,
                topk_key=args.topk_key,
            )
            evidence_payload = retrieved.to_prompt_payload()
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row, evidence_payload)},
                    {"role": "assistant", "content": build_assistant_target(row, evidence_payload)},
                ],
                "metadata": {
                    "row_id": int(row.get("row_id", idx)),
                    "task": str(row["task"]),
                    "source": "memory_grounded_reflection_sft",
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            if args.limit > 0 and written >= args.limit:
                break
    print(json.dumps({"written": written, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
