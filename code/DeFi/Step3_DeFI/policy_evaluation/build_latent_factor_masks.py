import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
}


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rule-based latent factor masks from repair deltas.")
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--token-topk", type=int, default=32)
    parser.add_argument("--channel-topk", type=int, default=128)
    parser.add_argument("--soft", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.memory_jsonl)
    arrays = np.load(args.memory_npz, allow_pickle=True)
    repair_deltas = arrays["repair_deltas"].astype(np.float32)
    if len(rows) != repair_deltas.shape[0]:
        raise ValueError(f"Row/repair_delta mismatch: rows={len(rows)} repair_deltas={repair_deltas.shape}")

    labels = [str(row.get("mismatch_type", "unknown")) for row in rows]
    factor_to_indices: Dict[str, List[int]] = {}
    for idx, mismatch in enumerate(labels):
        factor = MISMATCH_TO_FACTOR.get(mismatch)
        if factor is not None:
            factor_to_indices.setdefault(factor, []).append(idx)

    masks = {}
    summary = {
        "num_examples": len(rows),
        "token_topk": args.token_topk,
        "channel_topk": args.channel_topk,
        "soft": bool(args.soft),
        "mismatch_counts": dict(Counter(labels).most_common()),
        "factors": {},
    }
    for factor, indices in sorted(factor_to_indices.items()):
        delta_abs = np.abs(repair_deltas[indices]).mean(axis=0)
        mask = build_mask(delta_abs, args.token_topk, args.channel_topk, args.soft)
        token_score = delta_abs.mean(axis=1)
        channel_score = delta_abs.mean(axis=0)
        masks[f"mask_{factor}"] = mask.astype(np.float32)
        summary["factors"][factor] = {
            "num_examples": len(indices),
            "mismatch_types": sorted(set(labels[i] for i in indices)),
            "top_tokens": [int(x) for x in np.argsort(-token_score)[: min(args.token_topk, 16)]],
            "top_channels": [int(x) for x in np.argsort(-channel_score)[: min(args.channel_topk, 16)]],
            "mask_density": float((mask > 0).mean()),
            "mean_repair_delta_norm": float(np.linalg.norm(repair_deltas[indices].reshape(len(indices), -1), axis=1).mean()),
        }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        factor_names=np.asarray(sorted(factor_to_indices.keys()), dtype=object),
        mismatch_to_factor=np.asarray(json.dumps(MISMATCH_TO_FACTOR), dtype=object),
        **masks,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
