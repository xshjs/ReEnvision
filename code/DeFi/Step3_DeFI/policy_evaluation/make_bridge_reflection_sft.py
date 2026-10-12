from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

STEP3_ROOT = Path(__file__).resolve().parents[1]
if str(STEP3_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP3_ROOT))

from policy_evaluation.build_rlbench_memory_bank import infer_mismatch_type
from policy_evaluation.prepare_reflection_labels_minimal import heuristic_trust_score, normalize_row


SYSTEM_PROMPT = (
    "You are a robot manipulation reflection module. "
    "Given one executed task outcome, return one compact JSON object only with keys: "
    '{"trust": "high|medium|low", "mismatch_type": string, "correction_direction": string}.'
)

CANDIDATE_FILENAMES = (
    "rollout_rows.jsonl",
    "reflection_rows.jsonl",
    "memory_rows.jsonl",
    "memory_rollout_rows.jsonl",
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_input_files(inputs: List[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for item in inputs:
        if not item.exists():
            continue
        if item.is_file():
            if item.name in CANDIDATE_FILENAMES:
                resolved = item.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield item
            continue
        for name in CANDIDATE_FILENAMES:
            for path in sorted(item.rglob(name)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def infer_task(row: Dict[str, Any]) -> str:
    for key in ("task", "task_name", "task_hint"):
        task = safe_str(row.get(key))
        if task:
            return task
    return "unknown_task"


def infer_lang_goal(row: Dict[str, Any], task: str) -> str:
    for key in ("lang_goal", "lang_text", "retry_lang_goal"):
        lang = safe_str(row.get(key))
        if lang:
            return lang
    return task.replace("_", " ")


def infer_steps(row: Dict[str, Any]) -> int:
    for key in ("steps", "baseline_steps", "episode_length"):
        if row.get(key) is not None:
            return max(safe_int(row.get(key)), 0)
    return 0


def infer_ep_len(row: Dict[str, Any], steps: int) -> int:
    if row.get("ep_len") is not None:
        return max(safe_int(row.get("ep_len")), 1)
    if row.get("episode_length") is not None:
        return max(safe_int(row.get("episode_length")), 1)
    if "sequence_index" in row or "subtask_index" in row:
        return 160
    if steps > 0:
        return max(steps, 25)
    return 25


def trust_score_to_label(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def infer_correction_direction(row: Dict[str, Any], success: bool, mismatch_type: str) -> str:
    existing = safe_str(row.get("correction_direction"))
    if existing:
        return existing[:256]
    if success:
        return "preserve progress"
    if mismatch_type == "drawer_progress_hallucinated":
        return "delay drawer progress and verify opening contact"
    if mismatch_type == "object_displacement_overestimated":
        return "reduce predicted object motion and strengthen lateral contact"
    if mismatch_type == "slow_success":
        return "be conservative and preserve verified progress"
    return "stabilize contact and reduce over-optimistic future progress"


def infer_recoverability(success: bool, mismatch_type: str) -> str:
    if success:
        return "not_needed"
    if mismatch_type in {
        "contact_not_realized",
        "object_displacement_overestimated",
        "future_outcome_mismatch",
        "drawer_progress_hallucinated",
    }:
        return "repairable"
    if mismatch_type == "slow_success":
        return "uncertain"
    return "uncertain"


def normalize_minimal_row(row: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
    task = infer_task(row)
    lang_goal = infer_lang_goal(row, task)
    success = bool(row.get("success", False))
    steps = infer_steps(row)
    ep_len = infer_ep_len(row, steps)
    mismatch_type = safe_str(row.get("mismatch_type")) or infer_mismatch_type(task, success, steps, ep_len)
    correction_direction = infer_correction_direction(row, success, mismatch_type)
    trust_score = safe_float(row.get("trust_score"), heuristic_trust_score({
        "task": task,
        "success": success,
        "steps": steps,
        "ep_len": ep_len,
        "future_alignment_to_actual": row.get("future_alignment_to_actual", 0.5),
        "mismatch_type": mismatch_type,
    }))
    trust = safe_str(row.get("trust")) or trust_score_to_label(trust_score)
    recoverability = safe_str(row.get("recoverability")) or infer_recoverability(success, mismatch_type)

    normalized = {
        "row_id": safe_int(row.get("row_id"), -1),
        "sequence_index": safe_int(row.get("sequence_index"), -1),
        "subtask_index": safe_int(row.get("subtask_index"), -1),
        "episode": safe_int(row.get("episode"), -1),
        "attempt_idx": safe_int(row.get("attempt_idx"), -1),
        "attempt_type": safe_str(row.get("attempt_type"), "baseline"),
        "task": task,
        "lang_goal": lang_goal,
        "success": success,
        "steps": steps,
        "ep_len": ep_len,
        "trust_score": float(trust_score),
        "trust": trust,
        "mismatch_type": mismatch_type,
        "correction_direction": correction_direction,
        "recoverability": recoverability,
        "reflection_text": safe_str(row.get("reflection_text")),
        "source_file": str(source_file.resolve()),
    }

    if not normalized["reflection_text"]:
        normalized["reflection_text"] = (
            f"trust: {normalized['trust']} | "
            f"mismatch_type: {normalized['mismatch_type']} | "
            f"correction_direction: {normalized['correction_direction']}"
        )

    # Keep any extra evidence fields for traceability, but do not force them into the prompt.
    for key in ("state_semantics", "diagnostic_semantics", "rule_text", "recent_summary", "key_summary", "important_summary"):
        if key in row:
            normalized[key] = row[key]
    return normalized


def build_user_prompt(row: Dict[str, Any]) -> str:
    payload = {
        "task": row["task"],
        "lang_goal": row["lang_goal"],
        "success": bool(row["success"]),
        "steps": int(row["steps"]),
        "ep_len": int(row["ep_len"]),
        "attempt_type": row["attempt_type"],
        "source_file": Path(row["source_file"]).name,
    }
    if "state_semantics" in row:
        payload["state_semantics"] = row["state_semantics"]
    if "diagnostic_semantics" in row:
        payload["diagnostic_semantics"] = row["diagnostic_semantics"]
    if "rule_text" in row:
        payload["rule_text"] = row["rule_text"]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_assistant_target(row: Dict[str, Any]) -> str:
    target = {
        "trust": row["trust"],
        "mismatch_type": row["mismatch_type"],
        "correction_direction": row["correction_direction"],
    }
    return json.dumps(target, ensure_ascii=False)


def row_dedupe_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row["task"],
        row["lang_goal"],
        bool(row["success"]),
        int(row["steps"]),
        int(row["ep_len"]),
        row["trust"],
        row["mismatch_type"],
        row["correction_direction"],
        row["reflection_text"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge existing reflection rows into Qwen SFT messages.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-duplicates", action="store_true")
    args = parser.parse_args()

    sources = list(iter_input_files(args.inputs))
    if not sources:
        raise FileNotFoundError("No reflection row files found in the provided inputs.")

    rows: list[Dict[str, Any]] = []
    seen_keys: set[Tuple[Any, ...]] = set()
    source_counts: Dict[str, int] = {}
    task_counts: Dict[str, int] = {}

    for source in sources:
        for raw in iter_jsonl(source):
            normalized = normalize_minimal_row(raw, source)
            key = row_dedupe_key(normalized)
            if not args.allow_duplicates and key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(normalized)
            source_counts[str(source.resolve())] = source_counts.get(str(source.resolve()), 0) + 1
            task_counts[normalized["task"]] = task_counts.get(normalized["task"], 0) + 1

    rows.sort(
        key=lambda row: (
            row["task"],
            row["lang_goal"],
            int(row["sequence_index"]),
            int(row["subtask_index"]),
            int(row["episode"]),
            int(row["attempt_idx"]),
            row["source_file"],
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                    {"role": "assistant", "content": build_assistant_target(row)},
                ],
                "metadata": {
                    "task": row["task"],
                    "lang_goal": row["lang_goal"],
                    "success": bool(row["success"]),
                    "steps": int(row["steps"]),
                    "ep_len": int(row["ep_len"]),
                    "trust_score": float(row["trust_score"]),
                    "mismatch_type": row["mismatch_type"],
                    "correction_direction": row["correction_direction"],
                    "recoverability": row["recoverability"],
                    "source_file": row["source_file"],
                    "source": "bridge_reflection_sft",
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "sources": len(sources),
        "rows": len(rows),
        "tasks": len(task_counts),
        "success_rows": sum(1 for row in rows if bool(row["success"])),
        "failure_rows": sum(1 for row in rows if not bool(row["success"])),
        "source_counts": source_counts,
        "task_counts": dict(sorted(task_counts.items(), key=lambda item: (-item[1], item[0]))),
        "output": str(args.output),
    }
    (args.output.parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
