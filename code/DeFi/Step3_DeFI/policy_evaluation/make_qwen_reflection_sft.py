from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a robot manipulation reflection planner. "
    "Given the current CALVIN task, previous executed subtasks, and MoE expert router diagnostics, "
    "write a compact JSON reflection for choosing safer future hypotheses. "
    "Return only JSON with keys: progress_status, failure_type, prefer_next, avoid_next, "
    "correction_direction, memory_query."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_reflections(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    items: dict[tuple[int, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        items[key] = row
    return items


def load_universal_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        base = root / "collect_moe_universal_self_dataset"
        index = base / "index.jsonl"
        if not index.exists():
            raise FileNotFoundError(index)
        sequence_offset = 0
        metadata = root / "metadata.json"
        if metadata.exists():
            sequence_offset = int(json.loads(metadata.read_text()).get("sequence_offset", 0))
        for row in read_jsonl(index):
            row = dict(row)
            row["global_sequence_index"] = sequence_offset + int(row.get("sequence_index", 0))
            rows.append(row)
    rows.sort(key=lambda x: (int(x["global_sequence_index"]), int(x["subtask_index"])))
    return rows


def branch_summary(row: dict[str, Any]) -> list[dict[str, Any]]:
    branches = []
    for item in sorted(row.get("branches", []), key=lambda b: int(b.get("router_rank", 999))):
        branches.append(
            {
                "expert": int(item["expert"]),
                "router_rank": int(item.get("router_rank", -1)),
                "router_logit": round(float(item.get("router_logit", 0.0)), 4),
                "router_score": None,
                "vjepa_score": round(float(item.get("vjepa_score", 0.0)), 4),
                "contradiction": round(float(item.get("contradiction", 0.0)), 5),
                "demo_index": int(item.get("demo_index", -1)),
            }
        )
    if row.get("router_order"):
        order_to_score = {int(expert): idx for idx, expert in enumerate(row["router_order"])}
        for item in branches:
            item["router_order_index"] = order_to_score.get(int(item["expert"]))
    return branches


def history_for(row: dict[str, Any], rows_by_seq: dict[int, list[dict[str, Any]]], max_history: int) -> list[dict[str, Any]]:
    seq = int(row["global_sequence_index"])
    subtask_idx = int(row["subtask_index"])
    previous = [x for x in rows_by_seq[seq] if int(x["subtask_index"]) < subtask_idx]
    previous = previous[-max_history:]
    return [
        {
            "subtask_index": int(item["subtask_index"]),
            "task": str(item["task"]),
            "chosen_expert": int(item.get("chosen_expert", -1)),
            "success": bool(item.get("success", False)),
            "steps": int(item.get("steps", 160)),
        }
        for item in previous
    ]


def target_json(reflection: dict[str, Any]) -> str:
    keys = [
        "progress_status",
        "failure_type",
        "prefer_next",
        "avoid_next",
        "correction_direction",
        "memory_query",
    ]
    return json.dumps({key: str(reflection.get(key, "uncertain")) for key in keys}, ensure_ascii=False)


def user_prompt(row: dict[str, Any], rows_by_seq: dict[int, list[dict[str, Any]]], max_history: int) -> str:
    payload = {
        "global_sequence_index": int(row["global_sequence_index"]),
        "subtask_index": int(row["subtask_index"]),
        "task": str(row["task"]),
        "router_order": [int(x) for x in row.get("router_order", [])],
        "previous_subtasks": history_for(row, rows_by_seq, max_history),
        "experts": branch_summary(row),
        "instruction": (
            "Infer progress, risk, preferred next expert/strategy, avoided expert/strategy, "
            "and correction direction. Do not use hidden rollout labels for the current experts."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen reflection SFT JSONL from Universal MoE rows and Qwen reflection targets.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--reflections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--format", choices=["messages", "sharegpt"], default="messages")
    args = parser.parse_args()

    rows = load_universal_rows(args.roots)
    reflections = load_reflections(args.reflections)
    rows_by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_seq[int(row["global_sequence_index"])].append(row)

    written = 0
    missing = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
            reflection = reflections.get(key)
            if reflection is None:
                missing += 1
                continue
            if args.format == "messages":
                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(row, rows_by_seq, args.max_history)},
                        {"role": "assistant", "content": target_json(reflection)},
                    ],
                    "metadata": {
                        "global_sequence_index": key[0],
                        "subtask_index": key[1],
                        "task": str(row["task"]),
                        "source": "qwen_reflection_distillation",
                    },
                }
            else:
                sample = {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": user_prompt(row, rows_by_seq, args.max_history)},
                        {"from": "gpt", "value": target_json(reflection)},
                    ]
                }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"rows": len(rows), "written": written, "missing_reflections": missing, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
