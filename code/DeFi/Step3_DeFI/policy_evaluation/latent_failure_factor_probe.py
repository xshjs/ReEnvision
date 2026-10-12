import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FACTOR_NAMES = {
    "contact_not_realized": "contact_establishment_factor",
    "object_displacement_overestimated": "object_displacement_factor",
    "wrong_object_identity": "object_identity_factor",
    "drawer_progress_hallucinated": "drawer_slider_progress_factor",
    "future_outcome_mismatch": "goal_completion_hallucination_factor",
}


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def pooled_features(future: np.ndarray, delta: np.ndarray, mode: str) -> np.ndarray:
    future_pool = future.mean(axis=1)
    delta_pool = delta.mean(axis=1)
    delta_abs_pool = np.abs(delta).mean(axis=1)
    if mode == "future":
        return future_pool
    if mode == "delta":
        return delta_pool
    if mode == "abs_delta":
        return delta_abs_pool
    if mode == "future_abs_delta":
        return np.concatenate([future_pool, delta_abs_pool], axis=1)
    raise ValueError(f"Unsupported feature mode: {mode}")


def nearest_centroid_leave_one_out(x: np.ndarray, labels: List[str]) -> Tuple[float, List[Dict]]:
    preds = []
    labels_arr = np.asarray(labels)
    for idx in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[idx] = False
        centroids = {}
        for label in sorted(set(labels_arr[train_mask].tolist())):
            members = x[train_mask & (labels_arr == label)]
            if len(members):
                centroids[label] = members.mean(axis=0)
        scores = {
            label: float(np.dot(x[idx], centroid) / (np.linalg.norm(x[idx]) * np.linalg.norm(centroid) + 1e-8))
            for label, centroid in centroids.items()
        }
        pred = max(scores, key=scores.get)
        preds.append(
            {
                "index": int(idx),
                "gold": str(labels[idx]),
                "pred": str(pred),
                "correct": bool(pred == labels[idx]),
                "score": float(scores[pred]),
            }
        )
    acc = sum(item["correct"] for item in preds) / max(1, len(preds))
    return float(acc), preds


def top_latent_factors(delta: np.ndarray, labels: List[str], rows: List[Dict], topk: int) -> Dict[str, Dict]:
    token_norm = np.linalg.norm(delta, axis=2)
    channel_norm = np.linalg.norm(delta, axis=1)
    report = {}
    for label in sorted(set(labels)):
        idx = [i for i, item in enumerate(labels) if item == label]
        if not idx:
            continue
        tok = token_norm[idx].mean(axis=0)
        ch = channel_norm[idx].mean(axis=0)
        task_counts = Counter(str(rows[i].get("task", "unknown")) for i in idx)
        delta_norms = np.linalg.norm(delta[idx].reshape(len(idx), -1), axis=1)
        report[label] = {
            "factor_name": FACTOR_NAMES.get(label, f"{label}_factor"),
            "num_examples": int(len(idx)),
            "top_tokens": [int(x) for x in np.argsort(-tok)[:topk]],
            "top_channels": [int(x) for x in np.argsort(-ch)[:topk]],
            "mean_repair_delta_norm": float(delta_norms.mean()),
            "tasks": dict(task_counts.most_common()),
        }
    return report


def confusion_matrix(labels: List[str], preds: List[Dict]) -> Dict[str, Dict[str, int]]:
    matrix = defaultdict(Counter)
    for item in preds:
        matrix[item["gold"]][item["pred"]] += 1
    return {gold: dict(counter) for gold, counter in matrix.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe latent failure factors from DeFi memory future repair deltas.")
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", default="future_abs_delta", choices=["future", "delta", "abs_delta", "future_abs_delta"])
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    rows = load_jsonl(args.memory_jsonl)
    arrays = np.load(args.memory_npz, allow_pickle=True)
    future = arrays["future_features"].astype(np.float32)
    delta = arrays["repair_deltas"].astype(np.float32)
    labels = [str(item.get("mismatch_type", "unknown")) for item in rows]

    if len(rows) != future.shape[0] or len(rows) != delta.shape[0]:
        raise ValueError(f"Row/array mismatch: rows={len(rows)} future={future.shape} delta={delta.shape}")

    x = normalize_features(pooled_features(future, delta, args.feature_mode))
    acc, preds = nearest_centroid_leave_one_out(x, labels)
    factor_report = top_latent_factors(delta, labels, rows, args.topk)

    summary = {
        "num_examples": int(len(rows)),
        "feature_mode": args.feature_mode,
        "leave_one_out_nearest_centroid_acc": acc,
        "label_counts": dict(Counter(labels).most_common()),
        "confusion": confusion_matrix(labels, preds),
        "latent_factors": factor_report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "latent_failure_factor_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "latent_failure_factor_predictions.jsonl").open("w") as handle:
        for item in preds:
            handle.write(json.dumps(item) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
