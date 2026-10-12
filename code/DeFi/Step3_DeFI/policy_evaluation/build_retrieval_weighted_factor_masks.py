from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
            if line.strip():
                yield json.loads(line)


def memory_id_to_row_id(memory_id: Any) -> int | None:
    text = str(memory_id)
    if text.startswith("KEY_"):
        text = text[4:]
    try:
        return int(text)
    except ValueError:
        return None


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


def load_memory(memory_jsonl: Path, memory_npz: Path) -> tuple[dict[int, dict[str, Any]], dict[int, int], np.ndarray]:
    rows = [row for row in iter_jsonl(memory_jsonl)]
    arrays = np.load(memory_npz, allow_pickle=True)
    repair_deltas = arrays["repair_deltas"].astype(np.float32)
    if len(rows) != repair_deltas.shape[0]:
        raise ValueError(f"Row/repair_delta mismatch: rows={len(rows)} repair_deltas={repair_deltas.shape}")
    row_by_id = {int(row["row_id"]): row for row in rows}
    index_by_row_id = {int(row["row_id"]): idx for idx, row in enumerate(rows)}
    return row_by_id, index_by_row_id, repair_deltas


def pseudo_label_rollout_row(row: dict[str, Any], row_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    factor_votes: dict[str, float] = defaultdict(float)
    mismatch_votes: dict[str, float] = defaultdict(float)
    evidence = []
    for memory_id, score in zip(row.get("adapter_memory_ids") or [], row.get("adapter_memory_scores") or []):
        row_id = memory_id_to_row_id(memory_id)
        if row_id is None or row_id not in row_by_id:
            continue
        memory_row = row_by_id[row_id]
        mismatch = str(memory_row.get("mismatch_type", "unknown"))
        factor = MISMATCH_TO_FACTOR.get(mismatch, "unknown")
        weight = max(float(score), 0.0)
        factor_votes[factor] += weight
        mismatch_votes[mismatch] += weight
        evidence.append(
            {
                "memory_id": str(memory_id),
                "row_id": row_id,
                "score": float(score),
                "mismatch_type": mismatch,
                "failure_factor": factor,
                "memory_task": str(memory_row.get("task", "")),
            }
        )
    factor = max(factor_votes.items(), key=lambda item: item[1])[0] if factor_votes else "unknown"
    mismatch = max(mismatch_votes.items(), key=lambda item: item[1])[0] if mismatch_votes else "unknown"
    vote_sum = sum(factor_votes.values())
    confidence = float(factor_votes.get(factor, 0.0) / vote_sum) if vote_sum > 0 else 0.0
    return {
        "sequence_index": int(row.get("sequence_index", -1)),
        "subtask_index": int(row.get("subtask_index", -1)),
        "task": str(row.get("task", "")),
        "success": bool(row.get("success", False)),
        "steps": int(row.get("steps", 0) or 0),
        "pseudo_mismatch_type": mismatch,
        "pseudo_failure_factor": factor,
        "pseudo_confidence": confidence,
        "factor_votes": {key: float(value) for key, value in sorted(factor_votes.items())},
        "mismatch_votes": {key: float(value) for key, value in sorted(mismatch_votes.items())},
        "adapter_memory_evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval-weighted latent factor pseudo labels and masks from rollout logs.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-topk", type=int, default=32)
    parser.add_argument("--channel-topk", type=int, default=128)
    parser.add_argument("--soft", action="store_true")
    args = parser.parse_args()

    row_by_id, index_by_row_id, repair_deltas = load_memory(args.memory_jsonl, args.memory_npz)
    pseudo_rows = []
    factor_weighted_scores: dict[str, np.ndarray] = {}
    factor_weight_sum: dict[str, float] = defaultdict(float)
    memory_hit_counts: Counter[int] = Counter()
    memory_hit_weight: Counter[int] = Counter()

    for row in iter_jsonl(args.rollout_jsonl):
        pseudo = pseudo_label_rollout_row(row, row_by_id)
        pseudo_rows.append(pseudo)
        for evidence in pseudo["adapter_memory_evidence"]:
            factor = evidence["failure_factor"]
            row_id = int(evidence["row_id"])
            weight = float(evidence["score"])
            if factor == "unknown" or row_id not in index_by_row_id:
                continue
            idx = index_by_row_id[row_id]
            score = np.abs(repair_deltas[idx])
            if factor not in factor_weighted_scores:
                factor_weighted_scores[factor] = np.zeros_like(score, dtype=np.float64)
            factor_weighted_scores[factor] += score.astype(np.float64) * weight
            factor_weight_sum[factor] += weight
            memory_hit_counts[row_id] += 1
            memory_hit_weight[row_id] += weight

    masks = {}
    summary = {
        "rollout_rows": len(pseudo_rows),
        "memory_rows": len(row_by_id),
        "token_topk": args.token_topk,
        "channel_topk": args.channel_topk,
        "soft": bool(args.soft),
        "pseudo_factor_counts": dict(Counter(row["pseudo_failure_factor"] for row in pseudo_rows).most_common()),
        "pseudo_mismatch_counts": dict(Counter(row["pseudo_mismatch_type"] for row in pseudo_rows).most_common()),
        "success_by_factor": {},
        "factors": {},
        "top_memory_hits": [],
    }
    by_factor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pseudo_rows:
        by_factor[row["pseudo_failure_factor"]].append(row)
    for factor, rows in sorted(by_factor.items()):
        summary["success_by_factor"][factor] = {
            "total": len(rows),
            "success": sum(1 for row in rows if row["success"]),
            "success_rate": float(sum(1 for row in rows if row["success"]) / max(len(rows), 1)),
        }
    for factor, weighted_score in sorted(factor_weighted_scores.items()):
        score = (weighted_score / max(float(factor_weight_sum[factor]), 1e-8)).astype(np.float32)
        mask = build_mask(score, args.token_topk, args.channel_topk, args.soft)
        token_score = score.mean(axis=1)
        channel_score = score.mean(axis=0)
        masks[f"mask_{factor}"] = mask.astype(np.float32)
        summary["factors"][factor] = {
            "retrieval_weight_sum": float(factor_weight_sum[factor]),
            "top_tokens": [int(x) for x in np.argsort(-token_score)[: min(args.token_topk, 16)]],
            "top_channels": [int(x) for x in np.argsort(-channel_score)[: min(args.channel_topk, 16)]],
            "mask_density": float((mask > 0).mean()),
            "mean_weighted_delta_norm": float(np.linalg.norm(score.reshape(-1), ord=2)),
        }
    for row_id, count in memory_hit_counts.most_common(20):
        row = row_by_id[row_id]
        summary["top_memory_hits"].append(
            {
                "row_id": int(row_id),
                "hits": int(count),
                "weight": float(memory_hit_weight[row_id]),
                "task": str(row.get("task", "")),
                "mismatch_type": str(row.get("mismatch_type", "")),
                "failure_factor": MISMATCH_TO_FACTOR.get(str(row.get("mismatch_type", "")), "unknown"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = args.output_dir / "pseudo_factor_rollout_rows.jsonl"
    with pseudo_path.open("w") as handle:
        for row in pseudo_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(
        args.output_dir / "retrieval_weighted_factor_masks.npz",
        factor_names=np.asarray(sorted(factor_weighted_scores.keys()), dtype=object),
        mismatch_to_factor=np.asarray(json.dumps(MISMATCH_TO_FACTOR), dtype=object),
        **masks,
    )
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
