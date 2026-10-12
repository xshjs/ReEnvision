from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "object_displacement_underestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
    "goal_progress_hallucinated": "goal_completion",
}

VALID_FACTORS = {
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
    "none",
}


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def memory_id_to_row_id(memory_id: Any) -> int | None:
    text = str(memory_id)
    if text.startswith("KEY_"):
        text = text[4:]
    try:
        return int(text)
    except ValueError:
        return None


def canonical_factor(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "drawer_progress": "drawer_slider_progress",
        "slider_progress": "drawer_slider_progress",
        "drawer_slider": "drawer_slider_progress",
        "displacement": "object_displacement",
        "object_pose": "object_displacement",
        "identity": "object_identity",
        "completion": "goal_completion",
        "goal_progress": "goal_completion",
        "future_outcome": "goal_completion",
        "no_failure": "none",
        "no_edit": "none",
        "": "",
    }
    text = aliases.get(text, text)
    return text if text in VALID_FACTORS else ""


def load_memory_rows(memory_jsonl: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for idx, row in enumerate(iter_jsonl(memory_jsonl)):
        row_id = int(row.get("row_id", idx))
        rows[row_id] = row
    return rows


def qwen_factor(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("failure_factor", "selected_failure_factor"):
        factor = canonical_factor(row.get(key))
        if factor:
            return factor, key
    return "", ""


def memory_vote(row: dict[str, Any], memory_by_id: dict[int, dict[str, Any]]) -> tuple[str, float, dict[str, float], list[dict[str, Any]]]:
    votes: dict[str, float] = defaultdict(float)
    evidence = []
    ids = row.get("adapter_memory_ids") or []
    scores = row.get("adapter_memory_scores") or []
    for memory_id, score in zip(ids, scores):
        row_id = memory_id_to_row_id(memory_id)
        if row_id is None or row_id not in memory_by_id:
            continue
        memory_row = memory_by_id[row_id]
        mismatch = str(memory_row.get("mismatch_type", ""))
        factor = MISMATCH_TO_FACTOR.get(mismatch, "")
        if not factor:
            continue
        weight = max(float(score), 0.0)
        votes[factor] += weight
        evidence.append(
            {
                "memory_id": str(memory_id),
                "row_id": row_id,
                "score": float(score),
                "memory_task": str(memory_row.get("task", "")),
                "mismatch_type": mismatch,
                "failure_factor": factor,
            }
        )
    if not votes:
        return "", 0.0, {}, evidence
    factor, weight = max(votes.items(), key=lambda item: item[1])
    total = sum(votes.values())
    return factor, float(weight / max(total, 1e-8)), {k: float(v) for k, v in sorted(votes.items())}, evidence


def task_prior_factor(task: str, lang_text: str = "") -> tuple[str, float, str]:
    text = f"{task} {lang_text}".lower()
    if any(x in text for x in ("drawer", "slider", "sliding door")):
        return "drawer_slider_progress", 0.65, "drawer_or_slider_task"
    if any(x in text for x in ("led", "lightbulb", "light bulb")):
        return "goal_completion", 0.60, "binary_state_task"
    if any(x in text for x in ("push", "move", "place in slider", "place_in_slider")):
        return "object_displacement", 0.55, "object_motion_task"
    if any(x in text for x in ("lift", "stack", "grasp", "pick")):
        return "contact", 0.55, "grasp_or_stack_task"
    if any(x in text for x in ("red", "blue", "pink", "color")):
        return "object_identity", 0.35, "color_object_task"
    return "", 0.0, ""


def infer_object(task: str, lang_text: str = "", row: dict[str, Any] | None = None) -> str:
    if row:
        obj = str(row.get("object", "") or "").strip()
        if obj and obj != "none":
            return obj
    text = f"{task} {lang_text}".lower()
    for color in ("red", "blue", "pink"):
        if color in text and "block" in text:
            return f"{color} block"
    if "drawer" in text:
        return "drawer"
    if "slider" in text or "sliding door" in text:
        return "slider"
    if "led" in text:
        return "led"
    if "lightbulb" in text or "light bulb" in text:
        return "lightbulb"
    if "block" in text:
        return "block"
    return "unknown"


def choose_label(
    row: dict[str, Any],
    memory_by_id: dict[int, dict[str, Any]],
    label_success_as_none: bool,
    prefer_qwen: bool,
) -> dict[str, Any]:
    success = bool(row.get("success", False))
    if label_success_as_none and success:
        return {
            "factor_label": "none",
            "label_source": "success_none",
            "label_confidence": 1.0,
            "qwen_factor": qwen_factor(row)[0],
            "memory_factor": "",
            "memory_confidence": 0.0,
            "task_prior_factor": "",
            "task_prior_confidence": 0.0,
            "task_prior_reason": "",
            "factor_votes": {},
            "adapter_memory_evidence": [],
        }

    q_factor, q_source = qwen_factor(row)
    m_factor, m_conf, votes, evidence = memory_vote(row, memory_by_id)
    t_factor, t_conf, t_reason = task_prior_factor(str(row.get("task", "")), str(row.get("lang_text", "")))

    if prefer_qwen and q_factor and q_factor != "none":
        factor, source, conf = q_factor, f"qwen:{q_source}", 0.85
    elif m_factor:
        factor, source, conf = m_factor, "memory_vote", m_conf
    elif q_factor:
        factor, source, conf = q_factor, f"qwen:{q_source}", 0.60 if q_factor != "none" else 1.0
    elif t_factor:
        factor, source, conf = t_factor, f"task_prior:{t_reason}", t_conf
    else:
        factor, source, conf = "goal_completion", "fallback", 0.20

    return {
        "factor_label": factor,
        "label_source": source,
        "label_confidence": float(conf),
        "qwen_factor": q_factor,
        "memory_factor": m_factor,
        "memory_confidence": float(m_conf),
        "task_prior_factor": t_factor,
        "task_prior_confidence": float(t_conf),
        "task_prior_reason": t_reason,
        "factor_votes": votes,
        "adapter_memory_evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak factor supervision rows for CALVIN future-latent intervention.")
    parser.add_argument("--rollout-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-success-none", action="store_true")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--prefer-memory", action="store_true", help="Use memory vote before Qwen explicit factor.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    memory_by_id = load_memory_rows(args.memory_jsonl)
    rows = []
    dropped = Counter()
    source_counts = Counter()
    factor_counts = Counter()

    for rollout_path in args.rollout_jsonl:
        for idx, row in enumerate(iter_jsonl(rollout_path)):
            success = bool(row.get("success", False))
            if args.failures_only and success:
                dropped["success"] += 1
                continue
            label = choose_label(
                row,
                memory_by_id,
                label_success_as_none=bool(args.include_success_none),
                prefer_qwen=not bool(args.prefer_memory),
            )
            if float(label["label_confidence"]) < args.min_confidence:
                dropped["low_confidence"] += 1
                continue
            trace_path = str(row.get("target_proxy_trace_path", "") or "")
            out = {
                "row_index": len(rows),
                "source_rollout_jsonl": str(rollout_path),
                "source_line_index": idx,
                "sequence_index": row.get("sequence_index"),
                "subtask_index": row.get("subtask_index"),
                "task": str(row.get("task", "")),
                "lang_text": str(row.get("lang_text", "")),
                "success": success,
                "steps": row.get("steps"),
                "factor_label": label["factor_label"],
                "label_source": label["label_source"],
                "label_confidence": label["label_confidence"],
                "qwen_factor": label["qwen_factor"],
                "memory_factor": label["memory_factor"],
                "memory_confidence": label["memory_confidence"],
                "task_prior_factor": label["task_prior_factor"],
                "task_prior_confidence": label["task_prior_confidence"],
                "task_prior_reason": label["task_prior_reason"],
                "object": infer_object(str(row.get("task", "")), str(row.get("lang_text", "")), row),
                "target_proxy_trace_path": trace_path,
                "has_future_trace": bool(trace_path and Path(trace_path).exists()),
                "base_future_norm": row.get("base_future_norm"),
                "proposal_future_norm": row.get("proposal_future_norm"),
                "counterfactual_gate": row.get("counterfactual_gate"),
                "counterfactual_shift_norm": row.get("counterfactual_shift_norm"),
                "adapter_delta_norm": row.get("adapter_delta_norm"),
                "applied_factor_mask": row.get("applied_factor_mask"),
                "selected_failure_factor": row.get("selected_failure_factor"),
                "failure_factor_raw": row.get("failure_factor"),
                "mismatch_type_raw": row.get("mismatch_type"),
                "factor_votes": label["factor_votes"],
                "adapter_memory_evidence": label["adapter_memory_evidence"],
            }
            rows.append(out)
            source_counts[out["label_source"]] += 1
            factor_counts[out["factor_label"]] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "factor_supervision_rows.jsonl"
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_factor_success = {}
    for factor in sorted(set(row["factor_label"] for row in rows)):
        subset = [row for row in rows if row["factor_label"] == factor]
        by_factor_success[factor] = {
            "total": len(subset),
            "success": sum(1 for row in subset if row["success"]),
            "failure": sum(1 for row in subset if not row["success"]),
            "has_future_trace": sum(1 for row in subset if row["has_future_trace"]),
        }

    summary = {
        "num_rows": len(rows),
        "rollout_jsonl": [str(path) for path in args.rollout_jsonl],
        "memory_jsonl": str(args.memory_jsonl),
        "include_success_none": bool(args.include_success_none),
        "failures_only": bool(args.failures_only),
        "prefer_memory": bool(args.prefer_memory),
        "min_confidence": float(args.min_confidence),
        "dropped": dict(dropped),
        "factor_counts": dict(factor_counts.most_common()),
        "label_source_counts": dict(source_counts.most_common()),
        "by_factor": by_factor_success,
        "has_future_trace": sum(1 for row in rows if row["has_future_trace"]),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
