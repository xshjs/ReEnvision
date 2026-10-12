import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


BRANCHES = ["continue", "retry_demo", "recover"]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def branch_map(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["name"]: item for item in row["branches"]}


def immediate_value(row: Dict[str, Any], ep_len: int) -> np.ndarray:
    """Current-subtask proxy: success is valuable, faster success is slightly better."""
    by_name = branch_map(row)
    values = []
    for name in BRANCHES:
        item = by_name.get(name)
        if item is None or not item["success"]:
            values.append(0.0)
        else:
            values.append(1.0 + (ep_len - float(item["steps"])) / ep_len)
    return np.asarray(values, dtype=np.float64)


def zscore_features(rows: List[Dict[str, Any]]) -> np.ndarray:
    x = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.linalg.norm(a - b) < 1e-12 else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-neighbor value-vector consistency for selector examples.")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-len", type=int, default=120)
    args = parser.parse_args()

    rows = load_rows(args.examples)
    features = zscore_features(rows)
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)
    labels = np.asarray([row["label"] for row in rows])
    tasks = np.asarray([row["task"] for row in rows])
    values = np.stack([immediate_value(row, args.ep_len) for row in rows], axis=0)
    value_nn = values[nn]
    label_agree = labels == labels[nn]
    same_task = tasks == tasks[nn]
    l2 = np.linalg.norm(values - value_nn, axis=1)
    cosine = np.sum(values * value_nn, axis=1) / (
        np.linalg.norm(values, axis=1) * np.linalg.norm(value_nn, axis=1) + 1e-12
    )
    corr = np.asarray([safe_corr(values[i], value_nn[i]) for i in range(len(rows))], dtype=np.float64)

    summary = {
        "num_examples": len(rows),
        "branches": BRANCHES,
        "value_definition": "immediate current-subtask proxy: 0 if branch fails, else 1 + (ep_len - steps)/ep_len",
        "label_distribution": dict(Counter(labels.tolist())),
        "nn_label_agreement": float(label_agree.mean()),
        "nn_same_task_rate": float(same_task.mean()),
        "same_task_nn_label_agreement": float(label_agree[same_task].mean()) if same_task.any() else None,
        "value_l2_mean": float(l2.mean()),
        "value_l2_median": float(np.median(l2)),
        "value_cosine_mean": float(cosine.mean()),
        "value_cosine_median": float(np.median(cosine)),
        "value_corr_mean": float(corr.mean()),
        "value_corr_median": float(np.median(corr)),
        "same_task_value_corr_mean": float(corr[same_task].mean()) if same_task.any() else None,
        "cross_task_value_corr_mean": float(corr[~same_task].mean()) if (~same_task).any() else None,
        "high_value_corr_rate_ge_0_8": float(np.mean(corr >= 0.8)),
        "low_value_corr_rate_lt_0": float(np.mean(corr < 0.0)),
        "top_mismatch_examples": [],
    }
    mismatch = np.flatnonzero(~label_agree)
    order = mismatch[np.argsort(-sim[mismatch, nn[mismatch]])]
    for i in order[:20]:
        j = int(nn[i])
        summary["top_mismatch_examples"].append(
            {
                "index": int(i),
                "label": str(labels[i]),
                "task": str(tasks[i]),
                "nn_index": j,
                "nn_label": str(labels[j]),
                "nn_task": str(tasks[j]),
                "state_cosine": float(sim[i, j]),
                "value": values[i].tolist(),
                "nn_value": values[j].tolist(),
                "value_corr": float(corr[i]),
                "value_l2": float(l2[i]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
