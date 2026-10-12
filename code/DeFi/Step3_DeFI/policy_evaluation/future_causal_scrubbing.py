from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    av = a.reshape(-1).astype(np.float32)
    bv = b.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def safe_factor(row: dict[str, Any]) -> str:
    factor = str(row.get("failure_factor") or row.get("applied_factor_mask") or "unknown")
    if factor in {"", "none"}:
        return "unknown"
    return factor


def load_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        data = np.load(str(path))
    except Exception:
        return None
    required = ["base_future", "proposal_future", "target_proxy_future"]
    if any(key not in data.files for key in required):
        return None
    return (
        np.asarray(data["base_future"], dtype=np.float32),
        np.asarray(data["proposal_future"], dtype=np.float32),
        np.asarray(data["target_proxy_future"], dtype=np.float32),
    )


def block_slices(size: int, block: int):
    start = 0
    while start < size:
        end = min(size, start + block)
        yield start, end
        start = end


def patch_gain(
    base: np.ndarray,
    donor: np.ndarray,
    target: np.ndarray,
    token_slice: tuple[int, int],
    channel_slice: tuple[int, int],
    base_dot: float,
    base_norm_sq: float,
    target_norm: float,
    base_cos: float,
) -> float:
    ts, te = token_slice
    cs, ce = channel_slice
    old = base[ts:te, cs:ce].reshape(-1).astype(np.float32)
    new = donor[ts:te, cs:ce].reshape(-1).astype(np.float32)
    tgt = target[ts:te, cs:ce].reshape(-1).astype(np.float32)
    delta_dot = float(np.dot(new - old, tgt))
    delta_norm_sq = float(np.dot(new, new) - np.dot(old, old))
    patched_norm_sq = max(base_norm_sq + delta_norm_sq, 1e-12)
    patched_cos = (base_dot + delta_dot) / (float(np.sqrt(patched_norm_sq)) * target_norm + 1e-12)
    return float(patched_cos - base_cos)


def top_patch_search(
    base: np.ndarray,
    donor: np.ndarray,
    target: np.ndarray,
    token_block: int,
    channel_block: int,
    topk: int,
) -> list[dict[str, Any]]:
    candidates = []
    base_flat = base.reshape(-1).astype(np.float32)
    target_flat = target.reshape(-1).astype(np.float32)
    base_dot = float(np.dot(base_flat, target_flat))
    base_norm_sq = float(np.dot(base_flat, base_flat))
    target_norm = float(np.linalg.norm(target_flat))
    base_cos = base_dot / (float(np.sqrt(max(base_norm_sq, 1e-12))) * target_norm + 1e-12)
    for ts, te in block_slices(base.shape[0], token_block):
        for cs, ce in block_slices(base.shape[1], channel_block):
            gain = patch_gain(base, donor, target, (ts, te), (cs, ce), base_dot, base_norm_sq, target_norm, base_cos)
            candidates.append(
                {
                    "token_start": int(ts),
                    "token_end": int(te),
                    "channel_start": int(cs),
                    "channel_end": int(ce),
                    "gain": float(gain),
                }
            )
    candidates.sort(key=lambda item: item["gain"], reverse=True)
    return candidates[:topk]


def add_patch_to_mask(mask: np.ndarray, patch: dict[str, Any], weight: float) -> None:
    ts, te = int(patch["token_start"]), int(patch["token_end"])
    cs, ce = int(patch["channel_start"]), int(patch["channel_end"])
    mask[ts:te, cs:ce] += float(weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline causal scrubbing over DeFI future latents.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--donor", choices=["target_proxy", "proposal"], default="target_proxy")
    parser.add_argument("--token-block", type=int, default=16)
    parser.add_argument("--channel-block", type=int, default=160)
    parser.add_argument("--topk-patches", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--only-harm", action="store_true")
    parser.add_argument("--only-failures", action="store_true")
    parser.add_argument("--min-gain", type=float, default=1e-5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in iter_jsonl(args.dataset_jsonl):
        if args.only_harm and not bool(row.get("label_harm", 0)):
            continue
        if args.only_failures and bool(row.get("success", False)):
            continue
        trace_path = str(row.get("trace_path", "") or "")
        if not trace_path or not Path(trace_path).exists():
            continue
        rows.append(row)
        if args.max_examples > 0 and len(rows) >= args.max_examples:
            break
    if not rows:
        raise ValueError("no usable rows")

    first = load_trace(rows[0]["trace_path"])
    if first is None:
        raise ValueError("first trace is invalid")
    token_dim, channel_dim = first[0].shape
    global_mask = np.zeros((token_dim, channel_dim), dtype=np.float32)
    factor_masks: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((token_dim, channel_dim), dtype=np.float32))
    factor_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()

    patch_rows = []
    gain_values = []
    with (args.output_dir / "causal_scrubbing_patches.jsonl").open("w") as handle:
        for idx, row in enumerate(rows):
            loaded = load_trace(row["trace_path"])
            if loaded is None:
                continue
            base, proposal, target = loaded
            donor = target if args.donor == "target_proxy" else proposal
            base_cos = cosine(base, target)
            donor_cos = cosine(donor, target)
            patches = top_patch_search(base, donor, target, args.token_block, args.channel_block, args.topk_patches)
            factor = safe_factor(row)
            factor_counts[factor] += 1
            task_counts[str(row.get("task", ""))] += 1
            accepted = [patch for patch in patches if float(patch["gain"]) >= args.min_gain]
            for rank, patch in enumerate(accepted):
                weight = max(float(patch["gain"]), 0.0)
                add_patch_to_mask(global_mask, patch, weight)
                add_patch_to_mask(factor_masks[factor], patch, weight)
                gain_values.append(weight)
                out = {
                    "row_index": idx,
                    "sequence_index": row.get("sequence_index"),
                    "subtask_index": row.get("subtask_index"),
                    "task": row.get("task"),
                    "success": row.get("success"),
                    "failure_factor": factor,
                    "base_to_target_cos": base_cos,
                    "donor_to_target_cos": donor_cos,
                    "rank": rank,
                    **patch,
                }
                patch_rows.append(out)
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
        handle.flush()

    def normalize(mask: np.ndarray) -> np.ndarray:
        if float(mask.max()) <= 0:
            return mask
        return (mask / float(mask.max())).astype(np.float32)

    save_kwargs = {"mask_global": normalize(global_mask)}
    for factor, mask in factor_masks.items():
        save_kwargs[f"mask_{factor}"] = normalize(mask)
    np.savez_compressed(args.output_dir / "causal_scrubbing_masks.npz", **save_kwargs)

    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_rows": int(len(rows)),
        "num_patches": int(len(patch_rows)),
        "donor": args.donor,
        "token_block": int(args.token_block),
        "channel_block": int(args.channel_block),
        "topk_patches": int(args.topk_patches),
        "min_gain": float(args.min_gain),
        "mean_patch_gain": float(np.mean(gain_values)) if gain_values else 0.0,
        "max_patch_gain": float(np.max(gain_values)) if gain_values else 0.0,
        "factor_counts": dict(factor_counts),
        "task_counts_top20": dict(task_counts.most_common(20)),
        "output_patches": str(args.output_dir / "causal_scrubbing_patches.jsonl"),
        "output_masks": str(args.output_dir / "causal_scrubbing_masks.npz"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
