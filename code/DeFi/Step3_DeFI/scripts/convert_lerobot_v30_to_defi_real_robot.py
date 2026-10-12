#!/usr/bin/env python3
"""Convert a LeRobot V30 dataset into DeFi's real-robot NPZ layout.

This is a pragmatic bridge for smoke-testing DeFi on real robot LeRobot data.
It can either preserve the original continuous action/state vectors by slicing
them, or convert absolute TCP pose targets into DeFi-style 7D relative actions:
delta_xyz + delta_euler_xyz + gripper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


POSE_SUFFIXES = ("x", "y", "z", "qx", "qy", "qz", "qw")


def _remove_tree(path: Path, retries: int = 5, delay_s: float = 0.5) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            # Some shared filesystems can briefly report "Directory not empty"
            # while directory entries are still being removed.
            if exc.errno != 39 or attempt == retries - 1:
                break
            time.sleep(delay_s * (attempt + 1))
    if path.exists():
        fallback = path.with_name(f"{path.name}.stale.{int(time.time())}")
        os.replace(path, fallback)
        shutil.rmtree(fallback, ignore_errors=True)
        return
    if last_error is not None:
        raise last_error


def _read_tasks(root: Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {0: "perform the task"}
    table = pq.read_table(tasks_path).to_pydict()
    names = table.get("__index_level_0__", [])
    indices = table.get("task_index", list(range(len(names))))
    return {int(idx): str(name).replace("_", " ") for idx, name in zip(indices, names)}


def _read_feature_names(root: Path) -> tuple[list[str], list[str]]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return [], []
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names") or []
    action_names = features.get("action", {}).get("names") or []
    return list(state_names), list(action_names)


def _read_rows(root: Path) -> dict[int, list[dict]]:
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    episodes: dict[int, list[dict]] = {}
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        columns = table.to_pydict()
        row_count = table.num_rows
        for row_idx in range(row_count):
            episode_index = int(columns["episode_index"][row_idx])
            episodes.setdefault(episode_index, []).append(
                {
                    "frame_index": int(columns["frame_index"][row_idx]),
                    "index": int(columns["index"][row_idx]),
                    "task_index": int(columns.get("task_index", [0] * row_count)[row_idx]),
                    "state": np.asarray(columns["observation.state"][row_idx], dtype=np.float32),
                    "action": np.asarray(columns["action"][row_idx], dtype=np.float32),
                }
            )

    for rows in episodes.values():
        rows.sort(key=lambda row: row["frame_index"])
    return episodes


def _video_map(root: Path, feature: str) -> list[Path]:
    feature_dir = root / "videos" / feature
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Missing video feature directory: {feature_dir}")
    out: list[Path] = []
    for path in sorted(feature_dir.glob("chunk-*/*")):
        if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        out.append(path)
    if not out:
        raise FileNotFoundError(f"No videos found in {feature_dir}")
    return out


def _read_video(path: Path, resize: int | None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize is not None and (frame_rgb.shape[0] != resize or frame_rgb.shape[1] != resize):
            frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb.astype(np.uint8, copy=False))
    cap.release()
    return frames


def _read_video_frames_by_global_indices(
    video_paths: list[Path],
    global_indices: list[int],
    resize: int | None,
) -> dict[int, np.ndarray]:
    if not global_indices:
        return {}

    sorted_unique = sorted(set(int(idx) for idx in global_indices))
    out: dict[int, np.ndarray] = {}
    remaining = sorted_unique
    base = 0
    for path in video_paths:
        cap = cv2.VideoCapture(path.as_posix())
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        chunk_start = base
        chunk_end = base + frame_count
        local_targets = [idx - chunk_start for idx in remaining if chunk_start <= idx < chunk_end]
        if not local_targets:
            cap.release()
            base = chunk_end
            continue

        target_ptr = 0
        next_target = local_targets[target_ptr]
        local_index = 0
        while target_ptr < len(local_targets):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if local_index == next_target:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if resize is not None and (frame_rgb.shape[0] != resize or frame_rgb.shape[1] != resize):
                    frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
                out[chunk_start + local_index] = frame_rgb.astype(np.uint8, copy=False)
                target_ptr += 1
                if target_ptr < len(local_targets):
                    next_target = local_targets[target_ptr]
            local_index += 1
        cap.release()
        if target_ptr != len(local_targets):
            missing = [chunk_start + idx for idx in local_targets[target_ptr:]]
            raise ValueError(f"Video {path} ended early; missing global indices {missing[:5]}")
        remaining = [idx for idx in remaining if idx >= chunk_end]
        base = chunk_end
        if not remaining:
            break

    if len(out) != len(sorted_unique):
        missing = [idx for idx in sorted_unique if idx not in out]
        raise ValueError(f"Could not resolve all requested frames from videos; missing {missing[:5]}")
    return out


def _safe_slice(vec: np.ndarray, dim: int, key: str) -> np.ndarray:
    if vec.shape[0] < dim:
        raise ValueError(f"{key} has dim {vec.shape[0]}, expected at least {dim}")
    return vec[:dim].astype(np.float32, copy=False)


def _pad_slice(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    take = min(dim, int(vec.shape[0]))
    out[:take] = np.asarray(vec[:take], dtype=np.float32)
    return out


def _angle_wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _index_by_name(names: list[str], name: str, key: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise KeyError(f"Could not find '{name}' in {key} feature names") from exc


def _pose_indices(names: list[str], arm: str, key: str) -> list[int]:
    return [_index_by_name(names, f"{arm}_{suffix}", key) for suffix in POSE_SUFFIXES]


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat = quat / norm
    if quat[-1] < 0:
        quat = -quat
    return quat.astype(np.float32)


def _gripper_value(raw: float, mode: str, gripper_min: float, gripper_max: float) -> np.float32:
    if mode == "raw":
        return np.float32(raw)
    if mode == "binary_signed":
        return np.float32(1.0 if raw > 0.5 else -1.0)
    if mode == "continuous_range":
        if gripper_max <= gripper_min:
            raise ValueError("--gripper_max must be greater than --gripper_min")
        clipped = min(max(float(raw), float(gripper_min)), float(gripper_max))
        norm01 = (clipped - float(gripper_min)) / (float(gripper_max) - float(gripper_min))
        return np.float32(norm01 * 2.0 - 1.0)
    raise ValueError(f"Unsupported gripper mode: {mode}")


def _build_delta_pose_gripper_action(
    state: np.ndarray,
    action: np.ndarray,
    state_pose_idx: list[int],
    action_pose_idx: list[int],
    action_gripper_idx: int,
    max_pos: float,
    max_orn: float,
    gripper_mode: str,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    if max_pos <= 0.0 or max_orn <= 0.0:
        raise ValueError("--max_pos and --max_orn must be positive for delta action conversion")
    curr_pose = state[state_pose_idx].astype(np.float32, copy=False)
    target_pose = action[action_pose_idx].astype(np.float32, copy=False)
    curr_euler = R.from_quat(_normalize_quat_xyzw(curr_pose[3:7])).as_euler("xyz", degrees=False)
    target_euler = R.from_quat(_normalize_quat_xyzw(target_pose[3:7])).as_euler("xyz", degrees=False)
    rel_pos = np.clip(target_pose[:3] - curr_pose[:3], -max_pos, max_pos) / max_pos
    rel_orn = np.clip(_angle_wrap(target_euler - curr_euler), -max_orn, max_orn) / max_orn
    gripper = np.asarray(
        [_gripper_value(float(action[action_gripper_idx]), gripper_mode, gripper_min, gripper_max)],
        dtype=np.float32,
    )
    return np.concatenate([rel_pos, rel_orn, gripper], axis=0).astype(np.float32)


def _build_delta_pose_gripper_robot_obs(
    state: np.ndarray,
    state_dim: int,
    state_pose_idx: list[int],
    state_gripper_idx: int,
) -> np.ndarray:
    if state_dim < 15:
        raise ValueError("DeFi CALVIN proprioception config expects state_dim >= 15")
    robot_obs = np.zeros((state_dim,), dtype=np.float32)
    robot_obs[:7] = state[state_pose_idx].astype(np.float32, copy=False)
    robot_obs[14] = np.float32(state[state_gripper_idx])
    return robot_obs


def _build_pose_gripper_robot_obs_from_pose(
    pose_xyzw: np.ndarray,
    gripper: float,
    state_dim: int,
) -> np.ndarray:
    if state_dim < 15:
        raise ValueError("DeFi CALVIN proprioception config expects state_dim >= 15")
    robot_obs = np.zeros((state_dim,), dtype=np.float32)
    robot_obs[:7] = np.asarray(pose_xyzw, dtype=np.float32)
    robot_obs[14] = np.float32(gripper)
    return robot_obs


def _build_joint_robot_obs(state: np.ndarray, state_dim: int) -> np.ndarray:
    robot_obs = _pad_slice(state, state_dim)
    # Keep compatibility with the existing CALVIN proprio slicing, which expects
    # the gripper scalar to also live at robot_obs[14].
    if state_dim >= 15 and state.shape[0] >= 7:
        robot_obs[14] = np.float32(state[6])
    return robot_obs


def _build_joint_delta_action(
    state: np.ndarray,
    future_state: np.ndarray,
    action_dim: int,
    max_joint_delta: float,
    gripper_mode: str,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    if action_dim < 1:
        raise ValueError("--action_dim must be >= 1")
    if max_joint_delta <= 0.0:
        raise ValueError("--max_joint_delta must be positive for joint_delta conversion")
    rel = np.zeros((action_dim,), dtype=np.float32)
    joint_dims = min(max(action_dim - 1, 0), int(state.shape[0]), int(future_state.shape[0]))
    if joint_dims > 0:
        delta = np.asarray(future_state[:joint_dims] - state[:joint_dims], dtype=np.float32)
        rel[:joint_dims] = np.clip(delta / max_joint_delta, -1.0, 1.0)
    if action_dim <= int(state.shape[0]) and action_dim <= int(future_state.shape[0]):
        rel[action_dim - 1] = _gripper_value(
            float(future_state[action_dim - 1]), gripper_mode, gripper_min, gripper_max
        )
    return rel


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy.tolist()
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _parse_xyz_rpy(node: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.fromstring(node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(node.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    return xyz, rpy


def _load_fk_chain(urdf_path: Path) -> list[dict]:
    root = ET.parse(urdf_path).getroot()
    joints: list[dict] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type")
        if joint_type not in {"revolute", "continuous"}:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        child = joint.find("child")
        if origin is None or axis is None or child is None:
            raise ValueError(f"Joint {joint.attrib.get('name', '<unnamed>')} is missing origin/axis/child in {urdf_path}")
        xyz, rpy = _parse_xyz_rpy(origin)
        axis_xyz = np.fromstring(axis.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
        axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)
        joints.append(
            {
                "name": joint.attrib["name"],
                "child": child.attrib["link"],
                "origin_xyz": xyz,
                "origin_rpy": rpy,
                "axis": axis_xyz,
            }
        )
    if len(joints) < 6:
        raise ValueError(f"Expected at least 6 revolute joints in {urdf_path}, found {len(joints)}")
    return joints[:6]


def _fk_pose_xyzw(chain: list[dict], joint_values: np.ndarray) -> np.ndarray:
    if joint_values.shape[0] < len(chain):
        raise ValueError(f"Need at least {len(chain)} joint values, got {joint_values.shape[0]}")
    transform = np.eye(4, dtype=np.float64)
    for joint, angle in zip(chain, joint_values[: len(chain)], strict=True):
        origin_rotation = _rpy_to_matrix(joint["origin_rpy"])
        origin_transform = _make_transform(origin_rotation, joint["origin_xyz"])
        motion_transform = _make_transform(R.from_rotvec(joint["axis"] * float(angle)).as_matrix(), np.zeros(3, dtype=np.float64))
        transform = transform @ origin_transform @ motion_transform
    quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat()
    return np.concatenate([transform[:3, 3], quat_xyzw], axis=0).astype(np.float32)


def _build_fk_delta_pose_gripper_action(
    curr_state: np.ndarray,
    future_state: np.ndarray,
    fk_chain: list[dict],
    max_pos: float,
    max_orn: float,
    gripper_mode: str,
    gripper_min: float,
    gripper_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    curr_pose = _fk_pose_xyzw(fk_chain, curr_state[:6])
    future_pose = _fk_pose_xyzw(fk_chain, future_state[:6])
    curr_euler = R.from_quat(_normalize_quat_xyzw(curr_pose[3:7])).as_euler("xyz", degrees=False)
    future_euler = R.from_quat(_normalize_quat_xyzw(future_pose[3:7])).as_euler("xyz", degrees=False)
    rel_pos = np.clip(future_pose[:3] - curr_pose[:3], -max_pos, max_pos) / max_pos
    rel_orn = np.clip(_angle_wrap(future_euler - curr_euler), -max_orn, max_orn) / max_orn
    gripper = np.asarray(
        [_gripper_value(float(future_state[6]), gripper_mode, gripper_min, gripper_max)],
        dtype=np.float32,
    )
    return curr_pose, np.concatenate([rel_pos, rel_orn, gripper], axis=0).astype(np.float32)


def _write_lang_ann(split_dir: Path, ranges: list[tuple[int, int]], texts: list[str], tasks: list[str]) -> None:
    emb = np.zeros((len(texts), 1, 1024), dtype=np.float32)
    payload = {
        "info": {"indx": ranges},
        "language": {
            "ann": texts,
            "task": tasks,
            "emb": emb,
        },
    }
    np.save(split_dir / "auto_lang_ann.npy", payload, allow_pickle=True)


def _write_split(
    split_dir: Path,
    source_root: Path,
    episode_ids: list[int],
    episodes: dict[int, list[dict]],
    tasks: dict[int, str],
    left_videos: list[Path],
    right_videos: list[Path],
    state_dim: int,
    action_dim: int,
    action_mode: str,
    arm: str,
    state_names: list[str],
    action_names: list[str],
    max_pos: float,
    max_orn: float,
    max_joint_delta: float,
    future_state_offset: int,
    gripper_mode: str,
    gripper_min: float,
    gripper_max: float,
    fk_chain: list[dict] | None,
    resize: int | None,
    frame_stride: int,
    skip_broken_episodes: bool,
) -> dict:
    split_dir.mkdir(parents=True, exist_ok=True)
    ranges: list[tuple[int, int]] = []
    texts: list[str] = []
    task_names: list[str] = []
    out_index = 0
    written_episodes = 0
    written_frames = 0
    if action_mode == "delta_pose_gripper":
        state_pose_idx = _pose_indices(state_names, arm, "observation.state")
        action_pose_idx = _pose_indices(action_names, arm, "action")
        state_gripper_idx = _index_by_name(state_names, f"gripper_{arm}", "observation.state")
        action_gripper_idx = _index_by_name(action_names, f"gripper_{arm}", "action")
    else:
        state_pose_idx = action_pose_idx = []
        state_gripper_idx = action_gripper_idx = -1

    for episode_id in episode_ids:
        print(f"[convert] split={split_dir.name} episode={episode_id} start", flush=True)
        rows = episodes[episode_id]

        start = out_index
        sampled_rows = rows[::frame_stride]
        sampled_global_indices = [int(row["index"]) for row in sampled_rows]
        try:
            left_frames = _read_video_frames_by_global_indices(left_videos, sampled_global_indices, resize)
            right_frames = _read_video_frames_by_global_indices(right_videos, sampled_global_indices, resize)
        except Exception as exc:
            if not skip_broken_episodes:
                raise
            print(
                f"[convert] split={split_dir.name} episode={episode_id} skipped due to video read failure: {exc}",
                flush=True,
            )
            continue
        for row_idx, row in enumerate(sampled_rows):
            global_index = int(row["index"])
            task_text = tasks.get(row["task_index"], "perform the task")
            raw_action = _safe_slice(row["action"], action_dim, "action")
            if action_mode == "raw_slice":
                robot_obs = _build_joint_robot_obs(row["state"], state_dim)
                rel_actions = raw_action
            elif action_mode == "joint_delta":
                robot_obs = _build_joint_robot_obs(row["state"], state_dim)
                future_row = sampled_rows[min(row_idx + future_state_offset, len(sampled_rows) - 1)]
                rel_actions = _build_joint_delta_action(
                    row["state"],
                    future_row["state"],
                    action_dim,
                    max_joint_delta,
                    gripper_mode,
                    gripper_min,
                    gripper_max,
                )
            elif action_mode == "fk_delta_pose_gripper":
                if fk_chain is None:
                    raise ValueError("--urdf is required for fk_delta_pose_gripper")
                future_row = sampled_rows[min(row_idx + future_state_offset, len(sampled_rows) - 1)]
                curr_pose, rel_actions = _build_fk_delta_pose_gripper_action(
                    row["state"],
                    future_row["state"],
                    fk_chain,
                    max_pos,
                    max_orn,
                    gripper_mode,
                    gripper_min,
                    gripper_max,
                )
                robot_obs = _build_pose_gripper_robot_obs_from_pose(
                    curr_pose,
                    float(row["state"][6]),
                    state_dim,
                )
            elif action_mode == "delta_pose_gripper":
                robot_obs = _build_delta_pose_gripper_robot_obs(
                    row["state"],
                    state_dim,
                    state_pose_idx,
                    state_gripper_idx,
                )
                rel_actions = _build_delta_pose_gripper_action(
                    row["state"],
                    row["action"],
                    state_pose_idx,
                    action_pose_idx,
                    action_gripper_idx,
                    max_pos,
                    max_orn,
                    gripper_mode,
                    gripper_min,
                    gripper_max,
                )
            else:
                raise ValueError(f"Unsupported action mode: {action_mode}")
            np.savez_compressed(
                split_dir / f"episode_{out_index:07d}.npz",
                rgb_static=left_frames[global_index],
                rgb_gripper=right_frames[global_index],
                robot_obs=robot_obs,
                scene_obs=np.zeros((1,), dtype=np.float32),
                actions=raw_action,
                rel_actions=rel_actions,
            )
            out_index += 1
        end = out_index - 1
        if end >= start:
            ranges.append((start, end))
            texts.append(task_text)
            task_names.append(task_text.replace(" ", "_"))
            written_episodes += 1
            written_frames += end - start + 1
            print(
                f"[convert] split={split_dir.name} episode={episode_id} wrote={end - start + 1} total_frames={written_frames}",
                flush=True,
            )

    np.save(split_dir / "ep_start_end_ids.npy", np.asarray(ranges, dtype=np.int64))
    _write_lang_ann(split_dir, ranges, texts, task_names)
    return {"episodes": written_episodes, "frames": written_frames, "dir": split_dir.as_posix()}


def _split_episode_ids(episode_ids: list[int], val_ratio: float) -> tuple[list[int], list[int]]:
    if not episode_ids:
        return [], []
    if len(episode_ids) == 1:
        return episode_ids, episode_ids
    val_count = max(1, int(round(len(episode_ids) * val_ratio)))
    val_count = min(val_count, len(episode_ids) - 1)
    return episode_ids[:-val_count], episode_ids[-val_count:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="LeRobot V30 dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Output DeFi dataset root")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for smoke tests")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--state_dim", type=int, default=15, help="DeFi CALVIN config indexes robot_obs[14]")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument(
        "--action_mode",
        choices=["raw_slice", "joint_delta", "delta_pose_gripper", "fk_delta_pose_gripper"],
        default="joint_delta",
        help="raw_slice keeps action[:action_dim]; joint_delta writes normalized state[t+offset]-state[t]; delta_pose_gripper writes [dxyz, deuler_xyz, gripper] from source TCP pose; fk_delta_pose_gripper derives EE pose from joints via URDF and writes [dxyz, deuler_xyz, gripper].",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="left", help="Arm used for delta_pose_gripper")
    parser.add_argument("--urdf", type=Path, default=None, help="Required for fk_delta_pose_gripper")
    parser.add_argument("--max_pos", type=float, default=0.05, help="Meters used to normalize/clamp delta xyz")
    parser.add_argument("--max_orn", type=float, default=0.50, help="Radians used to normalize/clamp delta Euler xyz")
    parser.add_argument("--max_joint_delta", type=float, default=0.05, help="Magnitude used to normalize/clamp per-step joint deltas")
    parser.add_argument("--future_state_offset", type=int, default=1, help="For joint_delta, use state[t+offset] - state[t]")
    parser.add_argument(
        "--gripper_mode",
        choices=["binary_signed", "raw", "continuous_range"],
        default="binary_signed",
        help="binary_signed maps >0.5 to 1 else -1; raw preserves the source gripper scalar; continuous_range linearly maps [gripper_min, gripper_max] to [-1, 1].",
    )
    parser.add_argument("--gripper_min", type=float, default=-3.5, help="Closed gripper value for continuous_range normalization.")
    parser.add_argument("--gripper_max", type=float, default=0.0, help="Open gripper value for continuous_range normalization.")
    parser.add_argument("--static_video_key", default="observation.images.cam_left", help="Source video feature mapped to rgb_static")
    parser.add_argument("--gripper_video_key", default="observation.images.cam_right", help="Source video feature mapped to rgb_gripper")
    parser.add_argument("--resize", type=int, default=224, help="Output square RGB size; set <=0 to keep source")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--skip_broken_episodes", action="store_true", help="Skip episodes whose source videos are truncated/corrupted.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    if args.future_state_offset < 1:
        raise ValueError("--future_state_offset must be >= 1")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
        _remove_tree(args.output.resolve())

    source_root = args.input.resolve()
    output_root = args.output.resolve()
    resize = None if args.resize <= 0 else int(args.resize)

    tasks = _read_tasks(source_root)
    state_names, action_names = _read_feature_names(source_root)
    episodes = _read_rows(source_root)
    left_videos = _video_map(source_root, args.static_video_key)
    right_videos = _video_map(source_root, args.gripper_video_key)
    fk_chain = None if args.urdf is None else _load_fk_chain(args.urdf.resolve())

    episode_ids = sorted(episodes)
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    train_ids, val_ids = _split_episode_ids(episode_ids, args.val_ratio)

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": source_root.as_posix(),
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "action_mode": args.action_mode,
        "arm": args.arm,
        "urdf": None if args.urdf is None else args.urdf.resolve().as_posix(),
        "max_pos": args.max_pos,
        "max_orn": args.max_orn,
        "max_joint_delta": args.max_joint_delta,
        "future_state_offset": args.future_state_offset,
        "gripper_mode": args.gripper_mode,
        "gripper_min": args.gripper_min,
        "gripper_max": args.gripper_max,
        "static_video_key": args.static_video_key,
        "gripper_video_key": args.gripper_video_key,
        "frame_stride": args.frame_stride,
        "skip_broken_episodes": bool(args.skip_broken_episodes),
        "train": _write_split(
            output_root / "training",
            source_root,
            train_ids,
            episodes,
            tasks,
            left_videos,
            right_videos,
            args.state_dim,
            args.action_dim,
            args.action_mode,
            args.arm,
            state_names,
            action_names,
            args.max_pos,
            args.max_orn,
            args.max_joint_delta,
            args.future_state_offset,
            args.gripper_mode,
            args.gripper_min,
            args.gripper_max,
            fk_chain,
            resize,
            args.frame_stride,
            args.skip_broken_episodes,
        ),
        "validation": _write_split(
            output_root / "validation",
            source_root,
            val_ids,
            episodes,
            tasks,
            left_videos,
            right_videos,
            args.state_dim,
            args.action_dim,
            args.action_mode,
            args.arm,
            state_names,
            action_names,
            args.max_pos,
            args.max_orn,
            args.max_joint_delta,
            args.future_state_offset,
            args.gripper_mode,
            args.gripper_min,
            args.gripper_max,
            fk_chain,
            resize,
            args.frame_stride,
            args.skip_broken_episodes,
        ),
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
