import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6), mean, std


def pca(x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_center = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x_center, full_matrices=False)
    components = vt[:n_components].astype(np.float32)
    scores = x_center @ components.T
    explained = (singular_values[:n_components] ** 2) / np.maximum((singular_values**2).sum(), 1e-8)
    return scores.astype(np.float32), components, explained.astype(np.float32)


def remove_top_pcs(x: np.ndarray, n_remove: int) -> np.ndarray:
    if n_remove <= 0:
        return x
    x_center = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_center, full_matrices=False)
    basis = vt[: min(n_remove, vt.shape[0])]
    projection = (x_center @ basis.T) @ basis
    return (x_center - projection).astype(np.float32)


def kmeans(x: np.ndarray, k: int, seed: int, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.zeros((len(x),), dtype=np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = x[labels == idx]
            if len(members):
                centroids[idx] = members.mean(axis=0)
    return labels, centroids


def purity(labels: np.ndarray, names: List[str]) -> float:
    total = 0
    for label in sorted(set(labels.tolist())):
        idx = np.nonzero(labels == label)[0]
        if len(idx):
            total += Counter(names[i] for i in idx).most_common(1)[0][1]
    return float(total / max(1, len(labels)))


def top_indices(score: np.ndarray, n: int) -> List[int]:
    return [int(x) for x in np.argsort(-score)[:n]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover unsupervised latent repair factors from DeFi repair deltas.")
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--remove-top-pcs", type=int, default=0)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    rows = load_jsonl(args.memory_jsonl)
    arrays = np.load(args.memory_npz, allow_pickle=True)
    repair_deltas = arrays["repair_deltas"].astype(np.float32)
    if len(rows) != repair_deltas.shape[0]:
        raise ValueError(f"Row/repair_delta mismatch: rows={len(rows)} repair_deltas={repair_deltas.shape}")

    tasks = [str(row.get("task", "unknown")) for row in rows]
    mismatches = [str(row.get("mismatch_type", "unknown")) for row in rows]
    pooled = np.abs(repair_deltas).mean(axis=1)
    x, feature_mean, feature_std = standardize(pooled)
    x_for_cluster = remove_top_pcs(x, args.remove_top_pcs)
    pca_scores, pca_components, explained = pca(x_for_cluster, min(args.pca_dim, x.shape[0], x.shape[1]))
    cluster_input = pca_scores[:, : min(args.pca_dim, pca_scores.shape[1])]
    labels, centroids = kmeans(cluster_input, args.k, args.seed, args.kmeans_iters)

    report = {
        "num_examples": int(len(rows)),
        "k": int(args.k),
        "pca_dim": int(cluster_input.shape[1]),
        "remove_top_pcs": int(args.remove_top_pcs),
        "pca_explained_top": [float(x) for x in explained[: min(10, len(explained))]],
        "pca_explained_sum": float(explained.sum()),
        "mismatch_purity": purity(labels, mismatches),
        "task_purity": purity(labels, tasks),
        "clusters": {},
    }
    factor_basis = []
    factor_masks = []
    for cluster in range(args.k):
        idx = np.nonzero(labels == cluster)[0]
        if len(idx) == 0:
            continue
        mean_delta = repair_deltas[idx].mean(axis=0)
        abs_delta = np.abs(repair_deltas[idx]).mean(axis=0)
        token_score = abs_delta.mean(axis=1)
        channel_score = abs_delta.mean(axis=0)
        top_tokens = top_indices(token_score, args.topk)
        top_channels = top_indices(channel_score, args.topk)
        mask = np.zeros_like(abs_delta, dtype=np.float32)
        mask[np.ix_(top_tokens, top_channels)] = 1.0
        factor_basis.append(mean_delta.astype(np.float32))
        factor_masks.append(mask)
        report["clusters"][str(cluster)] = {
            "num_examples": int(len(idx)),
            "task_counts": dict(Counter(tasks[i] for i in idx).most_common(12)),
            "mismatch_counts": dict(Counter(mismatches[i] for i in idx).most_common(12)),
            "top_tokens": top_tokens,
            "top_channels": top_channels,
            "mean_repair_delta_norm": float(np.linalg.norm(repair_deltas[idx].reshape(len(idx), -1), axis=1).mean()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "latent_factor_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    np.savez_compressed(
        args.output_dir / "latent_factor_basis.npz",
        labels=labels.astype(np.int64),
        pca_components=pca_components,
        pca_explained=explained,
        kmeans_centroids=centroids.astype(np.float32),
        factor_basis=np.stack(factor_basis, axis=0).astype(np.float32),
        factor_masks=np.stack(factor_masks, axis=0).astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
