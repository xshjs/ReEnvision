from __future__ import annotations

import argparse
import json
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1).astype(np.float32)))


def load_trace(path: str | Path) -> dict[str, np.ndarray] | None:
    try:
        data = np.load(str(path))
    except Exception:
        return None
    required = ["base_future", "proposal_future", "gated_future", "final_future", "target_proxy_future"]
    if any(key not in data.files for key in required):
        return None
    return {key: np.asarray(data[key], dtype=np.float32) for key in required}


def load_masks(path: Path | None, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    data = np.load(str(path))
    masks = {}
    for key in data.files:
        if not key.startswith("mask_"):
            continue
        mask = np.asarray(data[key], dtype=np.float32)
        if mask.shape == shape:
            masks[key.replace("mask_", "")] = mask
    return masks


def topk_mask(mask: np.ndarray, density: float) -> np.ndarray:
    if mask.size == 0 or float(mask.max()) <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    k = max(1, int(round(mask.size * float(density))))
    flat = mask.reshape(-1)
    threshold = np.partition(flat, -k)[-k]
    return (mask >= threshold).astype(np.float32)


def factor_from_row(row: dict[str, Any]) -> str:
    factor = str(row.get("failure_factor") or row.get("applied_factor_mask") or "goal_completion")
    if factor in {"", "none", "unknown"}:
        return "goal_completion"
    return factor


def build_beams(trace: dict[str, np.ndarray], masks: dict[str, np.ndarray], factor: str, scrub_density: float) -> dict[str, np.ndarray]:
    base = trace["base_future"]
    proposal = trace["proposal_future"]
    gated = trace["gated_future"]
    final = trace["final_future"]
    target = trace["target_proxy_future"]
    beams = {
        "base": base,
        "proposal": proposal,
        "gated": gated,
        "final": final,
    }
    global_mask = masks.get("global")
    factor_mask = masks.get(factor)
    if factor_mask is None:
        factor_mask = masks.get("goal_completion")
    if global_mask is not None:
        hard = topk_mask(global_mask, scrub_density)
        beams["scrub_global_target"] = base + hard * (target - base)
        beams["scrub_global_proposal"] = base + hard * (proposal - base)
    if factor_mask is not None:
        hard = topk_mask(factor_mask, scrub_density)
        beams[f"scrub_{factor}_target"] = base + hard * (target - base)
        beams[f"scrub_{factor}_proposal"] = base + hard * (proposal - base)
    return beams


def pairwise_diversity(beams: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(beams)
    cosines, rmses = [], []
    for i, lhs in enumerate(names):
        for rhs in names[i + 1 :]:
            a = beams[lhs]
            b = beams[rhs]
            cosines.append(cosine(a, b))
            rmses.append(float(np.sqrt(np.mean((a - b) ** 2))))
    return {
        "pairwise_cos_mean": float(np.mean(cosines)) if cosines else 1.0,
        "pairwise_cos_min": float(np.min(cosines)) if cosines else 1.0,
        "pairwise_rmse_mean": float(np.mean(rmses)) if rmses else 0.0,
        "pairwise_rmse_max": float(np.max(rmses)) if rmses else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe semantic future beams from existing CALVIN future traces.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--scrub-mask-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--only-harm", action="store_true")
    parser.add_argument("--scrub-density", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in iter_jsonl(args.dataset_jsonl):
        if args.only_harm and not bool(row.get("label_harm", 0)):
            continue
        if not row.get("trace_path") or not Path(str(row["trace_path"])).exists():
            continue
        rows.append(row)
        if args.max_examples > 0 and len(rows) >= args.max_examples:
            break
    if not rows:
        raise ValueError("no usable rows")

    first_trace = load_trace(rows[0]["trace_path"])
    if first_trace is None:
        raise ValueError("invalid first trace")
    masks = load_masks(args.scrub_mask_npz, tuple(first_trace["base_future"].shape))

    beam_wins: Counter[str] = Counter()
    factor_counts: Counter[str] = Counter()
    gain_by_beam: dict[str, list[float]] = {}
    diversity_rows = []
    with (args.output_dir / "semantic_future_beam_rows.jsonl").open("w") as handle:
        for idx, row in enumerate(rows):
            trace = load_trace(row["trace_path"])
            if trace is None:
                continue
            factor = factor_from_row(row)
            factor_counts[factor] += 1
            beams = build_beams(trace, masks, factor, args.scrub_density)
            target = trace["target_proxy_future"]
            base_score = cosine(trace["base_future"], target)
            scores = {name: cosine(future, target) for name, future in beams.items()}
            gains = {name: score - base_score for name, score in scores.items()}
            best = max(scores, key=scores.get)
            beam_wins[best] += 1
            for name, gain in gains.items():
                gain_by_beam.setdefault(name, []).append(float(gain))
            div = pairwise_diversity(beams)
            diversity_rows.append(div)
            out = {
                "row_index": idx,
                "sequence_index": row.get("sequence_index"),
                "subtask_index": row.get("subtask_index"),
                "task": row.get("task"),
                "success": row.get("success"),
                "factor": factor,
                "best_beam": best,
                "best_gain": float(gains[best]),
                "scores": scores,
                "gains": gains,
                **div,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "num_rows": len(rows),
        "only_harm": bool(args.only_harm),
        "scrub_density": float(args.scrub_density),
        "beam_wins": dict(beam_wins),
        "factor_counts": dict(factor_counts),
        "mean_gain_by_beam": {name: float(np.mean(vals)) for name, vals in sorted(gain_by_beam.items())},
        "positive_gain_rate_by_beam": {name: float(np.mean(np.asarray(vals) > 0)) for name, vals in sorted(gain_by_beam.items())},
        "diversity": {
            key: float(np.mean([row[key] for row in diversity_rows]))
            for key in ["pairwise_cos_mean", "pairwise_cos_min", "pairwise_rmse_mean", "pairwise_rmse_max"]
        },
        "output_rows": str(args.output_dir / "semantic_future_beam_rows.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
