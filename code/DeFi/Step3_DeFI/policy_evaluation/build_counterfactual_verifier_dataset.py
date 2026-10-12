from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1).astype(np.float32)))


def resolve_trace_path(row: dict[str, Any], trace_dir: Path | None) -> Path | None:
    path_text = str(row.get("target_proxy_trace_path") or "").strip()
    if path_text:
        path = Path(path_text)
        if path.exists():
            return path
        if trace_dir is not None:
            alt = trace_dir / path.name
            if alt.exists():
                return alt
    if trace_dir is None:
        return None
    seq = row.get("sequence_index")
    sub = row.get("subtask_index")
    # Fall back to row order naming only when caller prefilled trace_file.
    trace_file = row.get("_trace_file")
    if trace_file:
        alt = trace_dir / str(trace_file)
        if alt.exists():
            return alt
    return None


def build_example(row: dict[str, Any], trace_path: Path, min_cos_gain: float) -> dict[str, Any] | None:
    try:
        trace = np.load(trace_path)
    except Exception:
        return None
    required = ["base_future", "proposal_future", "gated_future", "final_future", "target_proxy_future"]
    if any(key not in trace.files for key in required):
        return None

    base = trace["base_future"]
    proposal = trace["proposal_future"]
    gated = trace["gated_future"]
    final = trace["final_future"]
    target = trace["target_proxy_future"]

    base_cos = cosine(base, target)
    proposal_cos = cosine(proposal, target)
    gated_cos = cosine(gated, target)
    final_cos = cosine(final, target)
    proposal_gain = proposal_cos - base_cos
    gated_gain = gated_cos - base_cos
    final_gain = final_cos - base_cos
    success = bool(row.get("success", False))

    # First scorer label: accept edits that both improved the proxy future and did not fail rollout.
    label_improve = int(success and final_gain > min_cos_gain)
    label_harm = int((not success) or final_gain < -min_cos_gain)

    memory_scores = [safe_float(x) for x in (row.get("adapter_memory_scores") or [])]
    return {
        "sequence_index": row.get("sequence_index"),
        "subtask_index": row.get("subtask_index"),
        "task": row.get("task"),
        "success": success,
        "steps": row.get("steps"),
        "failure_factor": row.get("failure_factor") or row.get("selected_failure_factor") or row.get("applied_factor_mask"),
        "applied_factor_mask": row.get("applied_factor_mask"),
        "factor_mask_applied": bool(row.get("factor_mask_applied", False)),
        "trace_path": str(trace_path),
        "base_to_target_cos": base_cos,
        "proposal_to_target_cos": proposal_cos,
        "gated_to_target_cos": gated_cos,
        "final_to_target_cos": final_cos,
        "proposal_cos_gain": proposal_gain,
        "gated_cos_gain": gated_gain,
        "final_cos_gain": final_gain,
        "proposal_delta_norm": norm(proposal - base),
        "gated_delta_norm": norm(gated - base),
        "final_delta_norm": norm(final - base),
        "target_delta_norm": norm(target - base),
        "proposal_to_target_delta_cos": cosine(proposal - base, target - base),
        "gated_to_target_delta_cos": cosine(gated - base, target - base),
        "final_to_target_delta_cos": cosine(final - base, target - base),
        "counterfactual_gate": safe_float(row.get("counterfactual_gate")),
        "counterfactual_memory_support": safe_float(row.get("counterfactual_memory_support")),
        "counterfactual_alignment": safe_float(row.get("counterfactual_alignment")),
        "counterfactual_residual_norm": safe_float(row.get("counterfactual_residual_norm")),
        "counterfactual_shift_norm": safe_float(row.get("counterfactual_shift_norm")),
        "adapter_delta_norm": safe_float(row.get("adapter_delta_norm")),
        "adapter_gate_mean": safe_float(row.get("adapter_gate_mean")),
        "adapter_weight_mean": safe_float(row.get("adapter_weight_mean")),
        "memory_score_mean": float(np.mean(memory_scores)) if memory_scores else 0.0,
        "memory_score_max": float(np.max(memory_scores)) if memory_scores else 0.0,
        "label_improve": label_improve,
        "label_harm": label_harm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--future-trace-dir", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--min-cos-gain", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.rollout_jsonl))
    num_examples = 0
    missing_trace = 0
    invalid_trace = 0
    success_count = 0
    label_improve_count = 0
    label_harm_count = 0
    final_cos_gain_sum = 0.0
    proposal_cos_gain_sum = 0.0
    factor_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for idx, row in enumerate(rows):
            row["_trace_file"] = f"future_trace_{idx:04d}.npz"
            trace_path = resolve_trace_path(row, args.future_trace_dir)
            if trace_path is None:
                missing_trace += 1
            else:
                ex = build_example(row, trace_path, args.min_cos_gain)
                if ex is None:
                    invalid_trace += 1
                else:
                    num_examples += 1
                    success_count += int(bool(ex["success"]))
                    label_improve_count += int(ex["label_improve"])
                    label_harm_count += int(ex["label_harm"])
                    final_cos_gain_sum += float(ex["final_cos_gain"])
                    proposal_cos_gain_sum += float(ex["proposal_cos_gain"])
                    factor_counts[str(ex.get("failure_factor"))] += 1
                    task_counts[str(ex.get("task"))] += 1
                    handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and ((idx + 1) % args.progress_every == 0 or idx + 1 == len(rows)):
                handle.flush()
                print(
                    json.dumps(
                        {
                            "progress": "build_verifier_dataset",
                            "done": idx + 1,
                            "total": len(rows),
                            "examples": num_examples,
                            "missing_trace": missing_trace,
                            "invalid_trace": invalid_trace,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    summary = {
        "rollout_jsonl": str(args.rollout_jsonl),
        "future_trace_dir": str(args.future_trace_dir) if args.future_trace_dir else None,
        "output_jsonl": str(args.output_jsonl),
        "num_rollout_rows": len(rows),
        "num_examples": num_examples,
        "missing_trace": missing_trace,
        "invalid_trace": invalid_trace,
        "success_count": int(success_count),
        "failure_count": int(num_examples - success_count),
        "label_improve_count": int(label_improve_count),
        "label_harm_count": int(label_harm_count),
        "factor_counts": dict(factor_counts),
        "top_tasks": dict(task_counts.most_common(20)),
        "mean_final_cos_gain": float(final_cos_gain_sum / max(num_examples, 1)),
        "mean_proposal_cos_gain": float(proposal_cos_gain_sum / max(num_examples, 1)),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
