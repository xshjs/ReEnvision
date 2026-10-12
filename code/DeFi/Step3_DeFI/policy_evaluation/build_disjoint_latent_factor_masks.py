from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "object_displacement_underestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
    "goal_progress_hallucinated": "goal_completion",
}


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_mask(score: np.ndarray, token_topk: int, channel_topk: int, soft: bool) -> np.ndarray:
    token_score = score.mean(axis=1)
    channel_score = score.mean(axis=0)
    token_idx = np.argsort(-token_score)[:token_topk]
    channel_idx = np.argsort(-channel_score)[:channel_topk]
    mask = np.zeros_like(score, dtype=np.float32)
    if soft:
        token_weight = token_score[token_idx] / max(float(token_score[token_idx].max()), 1e-8)
        channel_weight = channel_score[channel_idx] / max(float(channel_score[channel_idx].max()), 1e-8)
        for i, tok in enumerate(token_idx):
            for j, ch in enumerate(channel_idx):
                mask[int(tok), int(ch)] = float(token_weight[i] * channel_weight[j])
    else:
        mask[np.ix_(token_idx, channel_idx)] = 1.0
    return mask


def mask_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    a = (mask_a > 0).astype(np.float32)
    b = (mask_b > 0).astype(np.float32)
    inter = float((a * b).sum())
    union = float(((a + b) > 0).sum())
    return {
        "jaccard": inter / max(union, 1.0),
        "overlap_a": inter / max(float(a.sum()), 1.0),
        "overlap_b": inter / max(float(b.sum()), 1.0),
    }


def load_factor_scores_from_memory(memory_jsonl: Path, memory_npz: Path) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, Any]]:
    rows = list(iter_jsonl(memory_jsonl))
    arrays = np.load(memory_npz, allow_pickle=True)
    repair_deltas = arrays["repair_deltas"].astype(np.float32)
    if len(rows) != repair_deltas.shape[0]:
        raise ValueError(f"Row/repair_delta mismatch: rows={len(rows)} repair_deltas={repair_deltas.shape}")

    grouped: dict[str, list[int]] = {}
    mismatch_counts = Counter()
    for idx, row in enumerate(rows):
        mismatch = str(row.get("mismatch_type", "unknown"))
        mismatch_counts[mismatch] += 1
        factor = MISMATCH_TO_FACTOR.get(mismatch)
        if factor:
            grouped.setdefault(factor, []).append(idx)

    scores = {}
    counts = {}
    for factor, indices in sorted(grouped.items()):
        scores[factor] = np.abs(repair_deltas[indices]).mean(axis=0).astype(np.float32)
        counts[factor] = len(indices)
    return scores, counts, {"num_rows": len(rows), "mismatch_counts": dict(mismatch_counts.most_common())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disjoint weak latent factor masks by suppressing shared repair-delta activations.")
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--token-topk", type=int, default=32)
    parser.add_argument("--channel-topk", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--soft", action="store_true")
    args = parser.parse_args()

    raw_scores, counts, base_summary = load_factor_scores_from_memory(args.memory_jsonl, args.memory_npz)
    factors = sorted(raw_scores)
    if len(factors) < 2:
        raise ValueError(f"Need at least two factors, got {factors}")

    masks = {}
    raw_masks = {}
    summary = {
        **base_summary,
        "factor_counts": counts,
        "token_topk": int(args.token_topk),
        "channel_topk": int(args.channel_topk),
        "alpha": float(args.alpha),
        "temperature": float(args.temperature),
        "soft": bool(args.soft),
        "factors": {},
        "pairwise_overlap": {},
        "raw_pairwise_overlap": {},
    }

    for factor in factors:
        own = raw_scores[factor]
        others = [raw_scores[name] for name in factors if name != factor]
        other_mean = np.stack(others, axis=0).mean(axis=0)
        specific = own - float(args.alpha) * other_mean
        specific = np.maximum(specific, 0.0)
        if args.temperature != 1.0:
            specific = np.power(specific, float(args.temperature)).astype(np.float32)
        if float(specific.max()) <= 0:
            specific = own
        mask = build_mask(specific, args.token_topk, args.channel_topk, args.soft)
        raw_mask = build_mask(own, args.token_topk, args.channel_topk, args.soft)
        masks[f"mask_{factor}"] = mask.astype(np.float32)
        raw_masks[factor] = raw_mask.astype(np.float32)

        token_score = specific.mean(axis=1)
        channel_score = specific.mean(axis=0)
        summary["factors"][factor] = {
            "num_examples": int(counts[factor]),
            "specific_positive_fraction": float((specific > 0).mean()),
            "raw_score_norm": float(np.linalg.norm(own.reshape(-1), ord=2)),
            "specific_score_norm": float(np.linalg.norm(specific.reshape(-1), ord=2)),
            "mask_density": float((mask > 0).mean()),
            "mask_mean": float(mask.mean()),
            "top_tokens": [int(x) for x in np.argsort(-token_score)[: min(args.token_topk, 16)]],
            "top_channels": [int(x) for x in np.argsort(-channel_score)[: min(args.channel_topk, 16)]],
        }

    for i, factor_a in enumerate(factors):
        for factor_b in factors[i + 1 :]:
            key = f"{factor_a}__{factor_b}"
            summary["pairwise_overlap"][key] = mask_overlap(masks[f"mask_{factor_a}"], masks[f"mask_{factor_b}"])
            summary["raw_pairwise_overlap"][key] = mask_overlap(raw_masks[factor_a], raw_masks[factor_b])

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        factor_names=np.asarray(factors, dtype=object),
        mismatch_to_factor=np.asarray(json.dumps(MISMATCH_TO_FACTOR), dtype=object),
        **masks,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
