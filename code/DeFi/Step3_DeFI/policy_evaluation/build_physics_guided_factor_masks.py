#!/usr/bin/env python3
"""Build factor masks from physics-probe correlations.

This is a lightweight bridge from measurable CALVIN state changes to DeFI
future-latent masks. It does not claim causal semantics; it creates masks that
focus editing on channels correlated with gripper/object/contact proxies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FACTORS = [
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
]


def metric_scores(summary: dict, metric: str, num_groups: int) -> np.ndarray:
    scores = np.zeros(num_groups, dtype=np.float32)
    corr = summary.get("correlations", {}).get(metric, {})
    for side in ("top_positive", "top_negative"):
        for item in corr.get(side, []) or []:
            group = int(item.get("group", -1))
            if 0 <= group < num_groups:
                # Magnitude is more useful than sign for choosing editable subspace.
                scores[group] = max(scores[group], abs(float(item.get("corr", 0.0))))
    return scores


def top_groups(scores: np.ndarray, k: int, min_corr: float) -> list[int]:
    order = np.argsort(-scores)
    picked = [int(i) for i in order[:k] if float(scores[i]) >= min_corr]
    if not picked:
        picked = [int(order[0])]
    return picked


def group_mask(groups: list[int], shape: tuple[int, int], group_size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    for group in groups:
        start = group * group_size
        end = min(start + group_size, shape[1])
        if start < shape[1]:
            mask[:, start:end] = 1.0
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--channel-dim", type=int, default=2560)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--topk-groups", type=int, default=6)
    parser.add_argument("--min-corr", type=float, default=0.08)
    args = parser.parse_args()

    summary = json.load(open(args.physics_summary, "r", encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape = (int(args.token_dim), int(args.channel_dim))
    num_groups = int(np.ceil(shape[1] / int(args.group_size)))

    contact_score = np.maximum(
        metric_scores(summary, "contact_delta", num_groups),
        metric_scores(summary, "gripper_delta", num_groups),
    )
    object_score = metric_scores(summary, "object_motion", num_groups)
    failure_score = metric_scores(summary, "failure", num_groups)

    selected = {
        "contact": top_groups(contact_score, args.topk_groups, args.min_corr),
        "object_displacement": top_groups(object_score, args.topk_groups, args.min_corr),
        # No reliable object identity metric in current strong-GT; keep it conservative.
        "object_identity": top_groups(object_score, max(1, args.topk_groups // 2), args.min_corr),
        # Current probe had no useful progress signal; use failure-risk groups as fallback.
        "drawer_slider_progress": top_groups(failure_score, args.topk_groups, args.min_corr),
        "goal_completion": top_groups(failure_score, args.topk_groups, args.min_corr),
    }

    arrays = {
        "factor_names": np.asarray(FACTORS, dtype=object),
        "mismatch_to_factor": np.asarray(
            {
                "contact_not_realized": "contact",
                "object_displacement_overestimated": "object_displacement",
                "wrong_object_identity": "object_identity",
                "drawer_progress_hallucinated": "drawer_slider_progress",
                "future_outcome_mismatch": "goal_completion",
                "goal_progress_hallucinated": "goal_completion",
            },
            dtype=object,
        ),
    }
    for factor in FACTORS:
        arrays[f"mask_{factor}"] = group_mask(selected[factor], shape, args.group_size)

    output_npz = out_dir / "physics_guided_factor_masks.npz"
    np.savez_compressed(output_npz, **arrays)

    report = {
        "physics_summary": str(args.physics_summary),
        "output_npz": str(output_npz),
        "shape": list(shape),
        "group_size": int(args.group_size),
        "topk_groups": int(args.topk_groups),
        "min_corr": float(args.min_corr),
        "selected_groups": selected,
        "density": {
            factor: float(arrays[f"mask_{factor}"].mean())
            for factor in FACTORS
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
