#!/usr/bin/env python3
"""Visualize CALVIN/DeFI future latent traces as heatmaps and token trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_latent(path: Path, key: str) -> np.ndarray:
    data = np.load(str(path))
    if key not in data.files:
        raise KeyError(f"{key} not found in {path}; available={data.files}")
    x = np.asarray(data[key], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D latent for {key}, got shape={x.shape}")
    return x


def zscore_clip(x: np.ndarray, clip: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if std > 1e-8:
        y = (y - mean) / std
    if clip > 0:
        y = np.clip(y, -clip, clip)
    return y


def plot_heatmap(x: np.ndarray, title: str, path: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(14, 5))
    plt.imshow(x, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("latent channel")
    plt.ylabel("future token / time")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_line(values: np.ndarray, title: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(values)), values)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bar(values: np.ndarray, title: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(14, 4))
    plt.bar(np.arange(len(values)), values)
    plt.title(title)
    plt.xlabel("channel group")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def pca_2d(x: np.ndarray) -> np.ndarray:
    y = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(y, full_matrices=False)
    return y @ vt[:2].T


def plot_pca(tokens: np.ndarray, title: str, path: Path) -> None:
    xy = pca_2d(tokens)
    plt.figure(figsize=(6, 6))
    plt.scatter(xy[:, 0], xy[:, 1], c=np.arange(xy.shape[0]), s=12, cmap="viridis")
    plt.plot(xy[:, 0], xy[:, 1], alpha=0.35)
    plt.colorbar(label="future token / time")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def normalize_image(arr: np.ndarray) -> np.ndarray | None:
    img = np.asarray(arr)
    if img.ndim == 4:
        img = img[0]
    if img.ndim != 3:
        return None
    if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        return None
    img = img.astype(np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def save_trace_images(trace_path: Path, output_dir: Path) -> list[str]:
    data = np.load(str(trace_path))
    saved = []
    for key in data.files:
        if not (key.startswith("decision_obs") or "rgb" in key.lower() or "image" in key.lower()):
            continue
        img = normalize_image(data[key])
        if img is None:
            continue
        out = output_dir / f"{trace_path.stem}_{key}.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(key)
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        plt.close()
        saved.append(str(out))
    return saved


def group_energy(x: np.ndarray, group_size: int) -> np.ndarray:
    vals = []
    for start in range(0, x.shape[1], group_size):
        vals.append(float(np.linalg.norm(x[:, start : start + group_size])))
    return np.asarray(vals, dtype=np.float32)


def summarize(x: np.ndarray, group_size: int, topk: int) -> dict:
    token_energy = np.linalg.norm(x, axis=1)
    channel_abs = np.mean(np.abs(x), axis=0)
    groups = group_energy(x, group_size)
    top_tokens = np.argsort(-token_energy)[:topk]
    top_channels = np.argsort(-channel_abs)[:topk]
    top_groups = np.argsort(-groups)[:topk]
    return {
        "shape": list(x.shape),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "norm": float(np.linalg.norm(x.reshape(-1))),
        "top_tokens": [{"idx": int(i), "value": float(token_energy[i])} for i in top_tokens],
        "top_channels": [{"idx": int(i), "value": float(channel_abs[i])} for i in top_channels],
        "top_channel_groups": [{"idx": int(i), "value": float(groups[i])} for i in top_groups],
    }


def visualize_one(trace_path: Path, output_dir: Path, key: str, group_size: int, clip: float, topk: int) -> dict:
    stem = trace_path.stem
    x = load_latent(trace_path, key)
    prefix = output_dir / f"{stem}_{key}"
    plot_heatmap(zscore_clip(x, clip), f"{stem} {key} latent zscore", prefix.with_name(prefix.name + "_heatmap.png"))
    plot_line(np.linalg.norm(x, axis=1), f"{stem} {key} token energy", prefix.with_name(prefix.name + "_token_energy.png"), "L2")
    plot_bar(group_energy(x, group_size), f"{stem} {key} channel group energy", prefix.with_name(prefix.name + "_channel_groups.png"), "L2")
    plot_pca(x, f"{stem} {key} token PCA trajectory", prefix.with_name(prefix.name + "_pca_tokens.png"))
    return summarize(x, group_size=group_size, topk=topk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--max-traces", type=int, default=4)
    parser.add_argument("--keys", type=str, default="base_future,final_future,edit_direction")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    traces = list(args.trace)
    if args.trace_dir is not None:
        traces.extend(sorted(args.trace_dir.glob("future_trace_*.npz"))[: args.max_traces])
    if not traces:
        raise ValueError("provide --trace or --trace-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    summary: dict[str, dict] = {}
    for trace in traces[: args.max_traces]:
        available = np.load(str(trace)).files
        trace_summary = {}
        image_paths = save_trace_images(trace, args.output_dir)
        if image_paths:
            trace_summary["images"] = image_paths
        for key in keys:
            if key not in available:
                continue
            trace_summary[key] = visualize_one(
                trace,
                args.output_dir,
                key,
                group_size=args.group_size,
                clip=args.clip,
                topk=args.topk,
            )
        summary[str(trace)] = trace_summary

    with (args.output_dir / "latent_visualization_summary.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": str(args.output_dir), "num_traces": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
