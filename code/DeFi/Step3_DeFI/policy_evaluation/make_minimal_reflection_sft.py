from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from policy_evaluation.defi_memory_models import iter_jsonl
from policy_evaluation.prepare_reflection_labels_minimal import normalize_row


SYSTEM_PROMPT = (
    "You are a robot manipulation reflection module. "
    "Given one executed DeFi subtask and its future-vs-outcome diagnostics, "
    "return only one compact JSON object with keys: "
    '{"trust": "high|medium|low", "mismatch_type": string, "correction_direction": string}.'
)


def build_user_prompt(row: dict[str, Any]) -> str:
    payload = {
        "task": str(row["task"]),
        "success": bool(row["success"]),
        "steps": int(row["steps"]),
        "ep_len": int(row["ep_len"]),
        "future_alignment_to_actual": round(float(row.get("future_alignment_to_actual", 0.5)), 4),
        "retrieved_success_row_id": row.get("retrieved_success_row_id"),
        "retrieved_success_score": row.get("retrieved_success_score"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_assistant_target(row: dict[str, Any]) -> str:
    target = {
        "trust": str(row["trust"]),
        "mismatch_type": str(row["mismatch_type"]),
        "correction_direction": str(row["correction_direction"]),
    }
    return json.dumps(target, ensure_ascii=False)


def load_rows(paths: list[Path], limit_per_file: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        count = 0
        for raw in iter_jsonl(path):
            rows.append(normalize_row(raw))
            count += 1
            if limit_per_file > 0 and count >= limit_per_file:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimal 3-field reflection SFT JSONL.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-file", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.inputs, args.limit_per_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as handle:
        for row in rows:
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                    {"role": "assistant", "content": build_assistant_target(row)},
                ],
                "metadata": {
                    "row_id": int(row["row_id"]),
                    "sequence_index": int(row["sequence_index"]),
                    "subtask_index": int(row["subtask_index"]),
                    "task": str(row["task"]),
                    "source": "minimal_reflection_sft",
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
