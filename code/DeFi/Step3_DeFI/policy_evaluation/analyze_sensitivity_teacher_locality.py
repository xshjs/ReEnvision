from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _resolve_feature_path(rows_path: Path, feature_path: str) -> Path:
    p = Path(feature_path)
    if p.is_absolute():
        return p
    return rows_path.parent / p


def _segments(mask: np.ndarray) -> int:
    count = 0
    active = False
    for x in mask.astype(bool).reshape(-1):
        if x and not active:
            count += 1
            active = True
        elif not x:
            active = False
    return count


def _pattern(mask: np.ndarray) -> str:
    return "".join("H" if x else "L" for x in mask.astype(bool).reshape(-1))


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values.astype(np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether finite-difference GIDM sensitivity is temporally local."
    )
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.80)
    parser.add_argument("--min-peak-over-mean", type=float, default=1.5)
    parser.add_argument("--max-local-fraction", type=float, default=0.40)
    parser.add_argument("--max-local-segments", type=int, default=1)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    rows = _read_jsonl(args.rows_jsonl)
    score_rows: list[np.ndarray] = []
    kept: list[dict] = []
    for row in rows:
        feature = _resolve_feature_path(args.rows_jsonl, str(row.get("feature_path", "")))
        if not feature.exists():
            continue
        with np.load(feature, allow_pickle=True) as data:
            if "teacher_step_scores" in data:
                scores = np.asarray(data["teacher_step_scores"], dtype=np.float32).reshape(-1)
            elif "teacher_suffix_delta_action" in data:
                delta = np.asarray(data["teacher_suffix_delta_action"], dtype=np.float32)
                scores = np.linalg.norm(delta.reshape(delta.shape[0], -1, delta.shape[-1]), axis=-1).reshape(-1)
            elif "teacher_delta_action" in data:
                delta = np.asarray(data["teacher_delta_action"], dtype=np.float32)
                scores = np.linalg.norm(delta.reshape(delta.shape[0], -1, delta.shape[-1]), axis=-1).reshape(-1)
            else:
                continue
        if scores.size == 0 or not np.isfinite(scores).all():
            continue
        score_rows.append(scores)
        kept.append(row)

    if not score_rows:
        raise ValueError("no valid teacher score rows found")

    flat_scores = np.concatenate(score_rows, axis=0)
    threshold = float(args.threshold)
    if threshold < 0:
        threshold = _safe_quantile(flat_scores, args.threshold_quantile)

    examples: list[dict] = []
    affected_fracs = []
    segment_counts = []
    peak_over_mean = []
    local_flags = []
    pattern_counts: Counter[str] = Counter()
    for row, scores in zip(kept, score_rows):
        mask = scores > threshold
        frac = float(mask.mean())
        segs = _segments(mask)
        peak = float(scores.max())
        mean = float(scores.mean())
        ratio = peak / max(mean, 1e-12)
        is_local = (
            mask.any()
            and ratio >= float(args.min_peak_over_mean)
            and frac <= float(args.max_local_fraction)
            and segs <= int(args.max_local_segments)
        )
        affected_fracs.append(frac)
        segment_counts.append(segs)
        peak_over_mean.append(ratio)
        local_flags.append(is_local)
        pattern = _pattern(mask)
        pattern_counts[pattern] += 1
        if is_local and len(examples) < int(args.examples):
            examples.append(
                {
                    "row_id": int(row.get("row_id", -1)),
                    "sequence_index": int(row.get("sequence_index", -1)),
                    "subtask_index": int(row.get("subtask_index", -1)),
                    "task": str(row.get("task", "")),
                    "step": int(row.get("step", -1)),
                    "scores": [float(x) for x in scores.tolist()],
                    "mask": [int(x) for x in mask.astype(np.int32).tolist()],
                    "pattern": pattern,
                    "peak_over_mean": ratio,
                    "affected_fraction": frac,
                    "segments": segs,
                }
            )

    affected_fracs_np = np.asarray(affected_fracs, dtype=np.float32)
    segment_counts_np = np.asarray(segment_counts, dtype=np.float32)
    peak_over_mean_np = np.asarray(peak_over_mean, dtype=np.float32)
    local_np = np.asarray(local_flags, dtype=bool)
    result = {
        "rows_jsonl": str(args.rows_jsonl),
        "num_rows_total": int(len(rows)),
        "num_rows_valid": int(len(score_rows)),
        "chunk_len_min": int(min(s.size for s in score_rows)),
        "chunk_len_max": int(max(s.size for s in score_rows)),
        "threshold": threshold,
        "threshold_source": "fixed" if args.threshold >= 0 else f"global_q{args.threshold_quantile}",
        "score_p50": _safe_quantile(flat_scores, 0.50),
        "score_p80": _safe_quantile(flat_scores, 0.80),
        "score_p90": _safe_quantile(flat_scores, 0.90),
        "score_p99": _safe_quantile(flat_scores, 0.99),
        "affected_fraction_mean": float(affected_fracs_np.mean()),
        "affected_fraction_p50": _safe_quantile(affected_fracs_np, 0.50),
        "affected_fraction_p90": _safe_quantile(affected_fracs_np, 0.90),
        "segments_mean": float(segment_counts_np.mean()),
        "segments_p50": _safe_quantile(segment_counts_np, 0.50),
        "peak_over_mean_mean": float(peak_over_mean_np.mean()),
        "peak_over_mean_p50": _safe_quantile(peak_over_mean_np, 0.50),
        "peak_over_mean_p90": _safe_quantile(peak_over_mean_np, 0.90),
        "local_rows": int(local_np.sum()),
        "local_ratio": float(local_np.mean()),
        "top_patterns": pattern_counts.most_common(20),
        "examples": examples,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
