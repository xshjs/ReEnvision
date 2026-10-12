import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


BRANCHES = ["continue", "retry_demo", "recover"]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    out = np.empty_like(x, dtype=np.float64)
    out[order] = np.arange(len(x), dtype=np.float64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.examples)
    states = np.asarray([row["state_feature"] for row in rows], dtype=np.float64)
    states = (states - states.mean(axis=0, keepdims=True)) / (states.std(axis=0, keepdims=True) + 1e-6)
    states = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-12)
    sim = states @ states.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)

    values = np.asarray(
        [[float(row["hypothesis_values"].get(branch, 0.0)) for branch in BRANCHES] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([row["label"] for row in rows])
    pearson = np.asarray([safe_corr(values[i], values[nn[i]]) for i in range(len(rows))])
    spearman = np.asarray([safe_corr(ranks(values[i]), ranks(values[nn[i]])) for i in range(len(rows))])
    cosine = np.sum(values * values[nn], axis=1) / (
        np.linalg.norm(values, axis=1) * np.linalg.norm(values[nn], axis=1) + 1e-12
    )

    summary = {
        "num_decisions": len(rows),
        "branches": BRANCHES,
        "value_definition": "observed rollout value for each hypothesis branch",
        "nn_label_agreement": float(np.mean(labels == labels[nn])),
        "recoverability_pearson_mean": float(pearson.mean()),
        "recoverability_pearson_median": float(np.median(pearson)),
        "recoverability_spearman_mean": float(spearman.mean()),
        "recoverability_spearman_median": float(np.median(spearman)),
        "recoverability_cosine_mean": float(cosine.mean()),
        "recoverability_cosine_median": float(np.median(cosine)),
        "spearman_ge_0_7_rate": float(np.mean(spearman >= 0.7)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
