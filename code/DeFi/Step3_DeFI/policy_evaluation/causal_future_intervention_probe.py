#!/usr/bin/env python3
"""Offline probe for factor-time causal future intervention.

For each failed future, choose a high-risk factor/time node from a weak causal
graph, then apply a masked success-attraction/failure-repulsion direction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FACTOR_ORDER = ["contact", "object", "motion", "goal"]
SEGMENT_ORDER = ["early", "mid", "late"]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        with np.load(str(path)) as data:
            if key not in data:
                return None
            x = data[key].astype(np.float32)
    except Exception:
        return None
    return x if x.ndim == 2 else None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_factor_masks(path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    zeros = np.zeros(shape, dtype=np.float32)
    masks = {
        "contact": data["mask_contact"].astype(np.float32) if "mask_contact" in data else zeros,
        "goal": data["mask_goal_completion"].astype(np.float32) if "mask_goal_completion" in data else zeros,
    }
    object_parts = []
    motion_parts = []
    for key in ("mask_object_displacement", "mask_object_identity"):
        if key in data:
            object_parts.append(data[key].astype(np.float32))
    for key in ("mask_drawer_slider_progress", "mask_object_displacement"):
        if key in data:
            motion_parts.append(data[key].astype(np.float32))
    masks["object"] = np.maximum.reduce(object_parts) if object_parts else zeros
    masks["motion"] = np.maximum.reduce(motion_parts) if motion_parts else zeros
    for key, mask in list(masks.items()):
        if mask.shape != shape:
            raise ValueError(f"mask {key} shape {mask.shape} != future shape {shape}")
        masks[key] = (mask > 0).astype(np.float32)
    return masks


def segment_mask(segment: str, shape: tuple[int, int]) -> np.ndarray:
    t, d = shape
    mask = np.zeros(shape, dtype=np.float32)
    one = max(1, t // 3)
    if segment == "early":
        mask[:one] = 1.0
    elif segment == "mid":
        mask[one : max(one + 1, 2 * one)] = 1.0
    elif segment == "late":
        mask[max(0, 2 * one) :] = 1.0
    else:
        raise ValueError(segment)
    return mask


def choose_node(causal_row: dict[str, Any], mode: str) -> tuple[str, str, float]:
    candidates = []
    for segment in SEGMENT_ORDER:
        for factor in FACTOR_ORDER:
            key = f"{segment}_{factor}_risk"
            candidates.append((segment, factor, float(causal_row.get(key, 0.0))))
    if mode == "max_positive":
        segment, factor, score = max(candidates, key=lambda x: x[2])
        if score <= 0:
            return "none", "none", score
        return segment, factor, score
    if mode == "absmax":
        return max(candidates, key=lambda x: abs(x[2]))
    raise ValueError(mode)


def wrong_factor(factor: str) -> str:
    order = FACTOR_ORDER
    if factor not in order:
        return "motion"
    return order[(order.index(factor) + 1) % len(order)]


def random_factor_segment(rng: np.random.Generator) -> tuple[str, str]:
    return str(rng.choice(SEGMENT_ORDER)), str(rng.choice(FACTOR_ORDER))


def nearest_indices(
    idx: int,
    pooled_norm: np.ndarray,
    candidates: list[int],
    k: int,
) -> tuple[list[int], np.ndarray]:
    sims = pooled_norm[idx][None, :] @ pooled_norm[candidates].T
    order = np.argsort(-sims.reshape(-1))[: max(1, min(k, len(candidates)))]
    chosen = [candidates[int(i)] for i in order]
    weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones(len(chosen), dtype=np.float32) / max(len(chosen), 1)
    else:
        weights /= weights.sum()
    return chosen, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--causal-rows-jsonl", type=Path, required=True)
    parser.add_argument("--factor-mask-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", default="original_base_future")
    parser.add_argument("--gate", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--node-mode", choices=["max_positive", "absmax"], default="max_positive")
    parser.add_argument("--mask-mode", choices=["correct", "wrong", "random", "global"], default="correct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-failures", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    futures: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for row in iter_jsonl(args.dataset_jsonl):
        future = load_future(row.get("trace_path", ""), args.future_key)
        if future is None:
            continue
        rows.append(row)
        futures.append(future)
        pooled.append(future.mean(axis=0).astype(np.float32))

    causal_by_idx = {}
    for row in iter_jsonl(args.causal_rows_jsonl):
        causal_by_idx[int(row["idx"])] = row

    if not rows:
        raise RuntimeError("no future rows")
    shape = futures[0].shape
    factor_masks = load_factor_masks(args.factor_mask_npz, shape)
    time_masks = {segment: segment_mask(segment, shape) for segment in SEGMENT_ORDER}
    global_mask = np.ones(shape, dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    pooled_arr = np.stack(pooled)
    pooled_norm = normalize_rows(pooled_arr)
    tasks = [str(row.get("task") or "") for row in rows]
    success_indices = [i for i, row in enumerate(rows) if bool(row.get("success"))]
    failure_indices = [i for i, row in enumerate(rows) if not bool(row.get("success"))]
    if args.max_failures:
        failure_indices = failure_indices[: args.max_failures]

    out_rows = []
    success_gains = []
    failure_repulsions = []
    margin_gains = []
    mask_densities = []
    node_counts = Counter()
    skipped = 0

    for idx in failure_indices:
        causal = causal_by_idx.get(idx)
        if causal is None:
            skipped += 1
            continue
        segment, factor, node_score = choose_node(causal, args.node_mode)
        if factor == "none":
            skipped += 1
            continue
        selected_segment = segment
        selected_factor = factor
        if args.mask_mode == "wrong":
            selected_factor = wrong_factor(factor)
        elif args.mask_mode == "random":
            selected_segment, selected_factor = random_factor_segment(rng)
        candidates_success = [j for j in success_indices if (not args.same_task_only or tasks[j] == tasks[idx])]
        candidates_failure = [j for j in failure_indices if j != idx and (not args.same_task_only or tasks[j] == tasks[idx])]
        if not candidates_success:
            candidates_success = success_indices
        if not candidates_failure:
            candidates_failure = [j for j in failure_indices if j != idx]
        if not candidates_success or not candidates_failure:
            skipped += 1
            continue

        succ_ids, succ_w = nearest_indices(idx, pooled_norm, candidates_success, args.k)
        fail_ids, fail_w = nearest_indices(idx, pooled_norm, candidates_failure, args.k)
        succ_center = np.tensordot(succ_w, np.stack([futures[j] for j in succ_ids]), axes=(0, 0)).astype(np.float32)
        fail_center = np.tensordot(fail_w, np.stack([futures[j] for j in fail_ids]), axes=(0, 0)).astype(np.float32)
        direction = succ_center - fail_center
        if args.mask_mode == "global":
            mask = global_mask
        else:
            mask = factor_masks[selected_factor] * time_masks[selected_segment]
        masked_direction = direction * mask
        norm = float(np.linalg.norm(masked_direction))
        if norm <= 1e-8:
            skipped += 1
            continue
        edit = futures[idx] + float(args.gate) * masked_direction / norm

        base_success_cos = cosine(futures[idx], succ_center)
        edit_success_cos = cosine(edit, succ_center)
        base_failure_cos = cosine(futures[idx], fail_center)
        edit_failure_cos = cosine(edit, fail_center)
        sgain = edit_success_cos - base_success_cos
        frepel = base_failure_cos - edit_failure_cos
        mgain = (edit_success_cos - edit_failure_cos) - (base_success_cos - base_failure_cos)
        success_gains.append(sgain)
        failure_repulsions.append(frepel)
        margin_gains.append(mgain)
        mask_densities.append(float(mask.mean()))
        node_counts[f"{segment}_{factor}"] += 1

        out = {
            "idx": int(idx),
            "sequence_index": rows[idx].get("sequence_index"),
            "subtask_index": rows[idx].get("subtask_index"),
            "task": tasks[idx],
            "diagnosed_segment": segment,
            "diagnosed_factor": factor,
            "segment": selected_segment,
            "factor": selected_factor,
            "node": f"{segment}_{factor}",
            "applied_node": "global" if args.mask_mode == "global" else f"{selected_segment}_{selected_factor}",
            "mask_mode": args.mask_mode,
            "node_score": float(node_score),
            "gate": float(args.gate),
            "mask_density": float(mask.mean()),
            "direction_norm": norm,
            "nearest_success_tasks": [tasks[j] for j in succ_ids],
            "nearest_failure_tasks": [tasks[j] for j in fail_ids],
            "base_success_cos": base_success_cos,
            "edit_success_cos": edit_success_cos,
            "success_cos_gain": sgain,
            "base_failure_cos": base_failure_cos,
            "edit_failure_cos": edit_failure_cos,
            "failure_repulsion_gain": frepel,
            "risk_margin_change": mgain,
        }
        out_rows.append(out)

    rows_path = args.output_dir / "causal_intervention_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows), encoding="utf-8")
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "causal_rows_jsonl": str(args.causal_rows_jsonl),
        "factor_mask_npz": str(args.factor_mask_npz),
        "future_key": args.future_key,
        "num_rows": len(rows),
        "num_success": len(success_indices),
        "num_failure": len(failure_indices),
        "num_intervened_failures": len(out_rows),
        "skipped": skipped,
        "gate": float(args.gate),
        "k": int(args.k),
        "same_task_only": bool(args.same_task_only),
        "node_mode": args.node_mode,
        "mask_mode": args.mask_mode,
        "seed": int(args.seed),
        "node_counts": dict(node_counts),
        "mask_density": summarize(mask_densities),
        "success_cos_gain": summarize(success_gains),
        "failure_repulsion_gain": summarize(failure_repulsions),
        "risk_margin_change": summarize(margin_gains),
        "positive_success_gain_rate": float(np.mean(np.asarray(success_gains) > 0)) if success_gains else 0.0,
        "positive_failure_repulsion_rate": float(np.mean(np.asarray(failure_repulsions) > 0)) if failure_repulsions else 0.0,
        "positive_margin_change_rate": float(np.mean(np.asarray(margin_gains) > 0)) if margin_gains else 0.0,
        "output_rows": str(rows_path),
    }
    summary_path = args.output_dir / "causal_intervention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
