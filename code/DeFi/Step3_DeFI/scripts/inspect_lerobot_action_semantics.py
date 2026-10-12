#!/usr/bin/env python3
"""Inspect LeRobot V30 action/state semantics before real-robot deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    return json.loads(info_path.read_text())


def _as_array(column) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float32)


def _print_named_indices(title: str, names: list[str], limit: int) -> None:
    print(f"\n{title} first {limit}:")
    for idx, name in enumerate(names[:limit]):
        print(f"  {idx:03d}: {name}")
    grippers = [(idx, name) for idx, name in enumerate(names) if "gripper" in name.lower()]
    if grippers:
        print(f"{title} gripper-like indices:")
        for idx, name in grippers:
            print(f"  {idx:03d}: {name}")


def _mean_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, axis=1).mean())


def inspect(dataset_root: Path, max_episodes: int, name_limit: int) -> None:
    info = _load_info(dataset_root)
    action_names = info["features"]["action"].get("names") or []
    state_names = info["features"]["observation.state"].get("names") or []

    print(f"dataset: {dataset_root}")
    print(f"robot_type: {info.get('robot_type')}")
    print(f"fps: {info.get('fps')}")
    print(f"action_dim: {info['features']['action'].get('shape')}")
    print(f"state_dim: {info['features']['observation.state'].get('shape')}")
    _print_named_indices("action", action_names, name_limit)
    _print_named_indices("state", state_names, name_limit)

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    printed = 0
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "action", "observation.state"])
        episode_ids = np.asarray(table["episode_index"].to_pylist())
        frame_ids = np.asarray(table["frame_index"].to_pylist())
        actions = _as_array(table["action"])
        states = _as_array(table["observation.state"])

        for episode_id in np.unique(episode_ids):
            if printed >= max_episodes:
                return
            mask = episode_ids == episode_id
            if int(mask.sum()) < 3:
                continue
            ep_actions = actions[mask]
            ep_states = states[mask]
            ep_frames = frame_ids[mask]
            n = min(len(ep_actions), len(ep_states)) - 1

            print(f"\nepisode {int(episode_id)} frames={len(ep_actions)} frame_range={int(ep_frames[0])}-{int(ep_frames[-1])}")
            print(f"  mean |action_xyz_t - state_xyz_t|:   {_mean_norm(ep_actions[:n, :3] - ep_states[:n, :3]):.6f}")
            print(f"  mean |action_xyz_t - state_xyz_t+1|: {_mean_norm(ep_actions[:n, :3] - ep_states[1 : n + 1, :3]):.6f}")
            print(f"  mean |delta action_xyz|:             {_mean_norm(np.diff(ep_actions[:, :3], axis=0)):.6f}")
            print(f"  mean |delta state_xyz|:              {_mean_norm(np.diff(ep_states[:, :3], axis=0)):.6f}")
            print(f"  first action[:8]: {ep_actions[0, :8]}")
            print(f"  first state[:8]:  {ep_states[0, :8]}")
            printed += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LeRobot V30 dataset root.")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--name_limit", type=int, default=20)
    args = parser.parse_args()
    inspect(args.input, args.max_episodes, args.name_limit)


if __name__ == "__main__":
    main()
