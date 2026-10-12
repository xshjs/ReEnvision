#!/usr/bin/env python3
"""Build a weak causal transition graph from saved DeFI future traces.

This is intentionally lightweight: it discovers factor/time risk states from
success/failure centroids and estimates temporal edges by correlation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FACTOR_ORDER = ["contact", "object", "motion", "goal"]
SEGMENT_ORDER = ["early", "mid", "late"]


def stable_text_features(text: str, dim: int) -> np.ndarray:
    import hashlib

    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "little", signed=False) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / max(float(norm), 1e-6)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def infer_factor(row: dict[str, Any]) -> str:
    factor = str(row.get("energy_factor") or row.get("future_energy_selected_factor") or "").strip()
    if factor in FACTOR_ORDER:
        return factor
    task = str(row.get("task") or "")
    if any(x in task for x in ["drawer", "slider", "push"]):
        return "motion"
    if any(x in task for x in ["lift", "stack", "place_in"]):
        return "contact"
    if any(x in task for x in ["rotate", "block"]):
        return "object"
    if any(x in task for x in ["led", "lightbulb"]):
        return "goal"
    return "goal"


def segment_future(future: np.ndarray) -> dict[str, np.ndarray]:
    n = future.shape[0]
    a = max(1, n // 3)
    segments = {
        "early": future[:a],
        "mid": future[a : max(a + 1, 2 * a)],
        "late": future[max(0, 2 * a) :],
    }
    out = {}
    for name, seg in segments.items():
        mean = seg.mean(axis=0)
        std = seg.std(axis=0)
        delta = seg[-1] - seg[0]
        out[name] = np.concatenate([mean, std, delta], axis=0).astype(np.float32)
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-6)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def load_examples(dataset_jsonl: Path, max_rows: int | None) -> list[dict[str, Any]]:
    examples = []
    for row in iter_jsonl(dataset_jsonl):
        path = Path(row.get("trace_path", ""))
        if not path.exists():
            continue
        examples.append(row)
        if max_rows is not None and len(examples) >= max_rows:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", default="original_base_future")
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--min-factor-failures", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_examples(args.dataset_jsonl, args.max_rows)
    if not rows:
        raise RuntimeError(f"no valid rows in {args.dataset_jsonl}")

    records = []
    for row in rows:
        with np.load(row["trace_path"]) as data:
            if args.future_key not in data:
                continue
            future = data[args.future_key].astype(np.float32)
        segments = segment_future(future)
        task = str(row.get("task") or "")
        task_vec = stable_text_features(task, args.task_dim)
        records.append(
            {
                "row": row,
                "success": int(bool(row.get("success"))),
                "fail": int(not bool(row.get("success"))),
                "task": task,
                "factor": infer_factor(row),
                "segments": {k: np.concatenate([v, task_vec], axis=0).astype(np.float32) for k, v in segments.items()},
            }
        )

    factor_counts = Counter(r["factor"] for r in records)
    factor_fail_counts = Counter(r["factor"] for r in records if r["fail"])

    centroids: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    risk_scores = {
        factor: {seg: np.zeros(len(records), dtype=np.float32) for seg in SEGMENT_ORDER}
        for factor in FACTOR_ORDER
    }

    for factor in FACTOR_ORDER:
        factor_records = [r for r in records if r["factor"] == factor]
        if factor_fail_counts[factor] < args.min_factor_failures:
            factor_records = records
        for seg in SEGMENT_ORDER:
            success_feats = [r["segments"][seg] for r in factor_records if r["success"]]
            fail_feats = [r["segments"][seg] for r in factor_records if r["fail"]]
            if not success_feats or not fail_feats:
                success_feats = [r["segments"][seg] for r in records if r["success"]]
                fail_feats = [r["segments"][seg] for r in records if r["fail"]]
            succ = l2_normalize(np.stack(success_feats).mean(axis=0))
            fail = l2_normalize(np.stack(fail_feats).mean(axis=0))
            centroids[factor][seg] = {"success": succ, "failure": fail}
            for idx, r in enumerate(records):
                feat = l2_normalize(r["segments"][seg])
                risk_scores[factor][seg][idx] = float(feat @ fail - feat @ succ)

    y_fail = np.asarray([r["fail"] for r in records], dtype=np.int64)
    node_metrics: dict[str, Any] = {}
    for factor in FACTOR_ORDER:
        for seg in SEGMENT_ORDER:
            scores = risk_scores[factor][seg]
            node = f"{seg}_{factor}"
            node_metrics[node] = {
                "auc_fail": auc_score(y_fail, scores),
                "mean_success_risk": float(scores[y_fail == 0].mean()) if (y_fail == 0).any() else 0.0,
                "mean_failure_risk": float(scores[y_fail == 1].mean()) if (y_fail == 1).any() else 0.0,
                "corr_with_failure": pearson(scores, y_fail.astype(np.float32)),
            }

    edges = []
    segment_pairs = [("early", "mid"), ("mid", "late")]
    for src_seg, dst_seg in segment_pairs:
        for src_factor in FACTOR_ORDER:
            src = risk_scores[src_factor][src_seg]
            for dst_factor in FACTOR_ORDER:
                dst = risk_scores[dst_factor][dst_seg]
                edge_strength = pearson(src, dst)
                failure_path_strength = pearson(src * dst, y_fail.astype(np.float32))
                edges.append(
                    {
                        "src": f"{src_seg}_{src_factor}",
                        "dst": f"{dst_seg}_{dst_factor}",
                        "edge_corr": edge_strength,
                        "failure_path_corr": failure_path_strength,
                    }
                )
    edges = sorted(edges, key=lambda x: abs(x["failure_path_corr"]), reverse=True)

    top_paths = []
    for f1 in FACTOR_ORDER:
        early = risk_scores[f1]["early"]
        for f2 in FACTOR_ORDER:
            mid = risk_scores[f2]["mid"]
            for f3 in FACTOR_ORDER:
                late = risk_scores[f3]["late"]
                path_score = early * mid * late
                top_paths.append(
                    {
                        "path": [f"early_{f1}", f"mid_{f2}", f"late_{f3}"],
                        "failure_corr": pearson(path_score, y_fail.astype(np.float32)),
                        "auc_fail": auc_score(y_fail, path_score),
                    }
                )
    top_paths = sorted(top_paths, key=lambda x: abs(x["failure_corr"]), reverse=True)

    row_output = args.output_dir / "weak_causal_future_rows.jsonl"
    with row_output.open("w", encoding="utf-8") as handle:
        for idx, r in enumerate(records):
            row_scores = {
                f"{seg}_{factor}_risk": float(risk_scores[factor][seg][idx])
                for factor in FACTOR_ORDER
                for seg in SEGMENT_ORDER
            }
            handle.write(
                json.dumps(
                    {
                        "idx": idx,
                        "sequence_index": r["row"].get("sequence_index"),
                        "subtask_index": r["row"].get("subtask_index"),
                        "task": r["task"],
                        "success": bool(r["success"]),
                        "factor": r["factor"],
                        **row_scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "future_key": args.future_key,
        "num_records": len(records),
        "success_count": int((y_fail == 0).sum()),
        "failure_count": int((y_fail == 1).sum()),
        "factor_counts": dict(factor_counts),
        "factor_failure_counts": dict(factor_fail_counts),
        "node_metrics": dict(sorted(node_metrics.items(), key=lambda kv: kv[1]["auc_fail"], reverse=True)),
        "top_edges": edges[:20],
        "top_paths": top_paths[:20],
        "row_output": str(row_output),
    }
    report_path = args.output_dir / "weak_causal_future_graph.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
