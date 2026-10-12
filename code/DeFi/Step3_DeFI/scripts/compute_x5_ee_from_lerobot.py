#!/usr/bin/env python3
"""Compute ARX X5 end-effector poses from a LeRobot V3.0 dataset."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def _load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    return json.loads(info_path.read_text())


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy.tolist()
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _parse_xyz_rpy(node: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.fromstring(node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(node.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    return xyz, rpy


def _load_chain(urdf_path: Path) -> list[dict]:
    root = ET.parse(urdf_path).getroot()
    joints: list[dict] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type")
        name = joint.attrib["name"]
        if joint_type not in {"revolute", "continuous"}:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        child = joint.find("child")
        if origin is None or axis is None or child is None:
            raise ValueError(f"Joint {name} is missing origin/axis/child in {urdf_path}")
        xyz, rpy = _parse_xyz_rpy(origin)
        axis_xyz = np.fromstring(axis.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
        axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)
        joints.append(
            {
                "name": name,
                "child": child.attrib["link"],
                "origin_xyz": xyz,
                "origin_rpy": rpy,
                "axis": axis_xyz,
            }
        )
    if len(joints) != 6:
        raise ValueError(f"Expected 6 revolute joints in {urdf_path}, found {len(joints)}")
    return joints


def _forward_kinematics(chain: list[dict], joint_values: np.ndarray) -> np.ndarray:
    if joint_values.shape[0] < len(chain):
        raise ValueError(f"Need at least {len(chain)} joint values, got {joint_values.shape[0]}")
    transform = np.eye(4, dtype=np.float64)
    for joint, angle in zip(chain, joint_values[: len(chain)], strict=True):
        origin_rotation = _rpy_to_matrix(joint["origin_rpy"])
        origin_transform = _make_transform(origin_rotation, joint["origin_xyz"])
        motion_transform = _make_transform(_axis_angle_to_matrix(joint["axis"], float(angle)), np.zeros(3, dtype=np.float64))
        transform = transform @ origin_transform @ motion_transform
    return transform


def _iter_rows(dataset_root: Path):
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "observation.state"])
        columns = table.to_pydict()
        row_count = len(columns["episode_index"])
        for row_idx in range(row_count):
            yield {
                "episode_index": int(columns["episode_index"][row_idx]),
                "frame_index": int(columns["frame_index"][row_idx]),
                "state": np.asarray(columns["observation.state"][row_idx], dtype=np.float64),
            }


def _group_by_episode(dataset_root: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for row in _iter_rows(dataset_root):
        episodes.setdefault(row["episode_index"], []).append(row)
    for rows in episodes.values():
        rows.sort(key=lambda row: row["frame_index"])
    return episodes


def compute(
    dataset_root: Path,
    urdf_path: Path,
    output_path: Path | None,
    future_offset: int,
    max_episodes: int | None,
) -> dict:
    info = _load_info(dataset_root)
    state_names = info["features"]["observation.state"].get("names") or []
    if len(state_names) < 7:
        raise ValueError("Expected 7D state with 6 joints + gripper")

    chain = _load_chain(urdf_path)
    episodes = _group_by_episode(dataset_root)

    summary = {
        "dataset_root": str(dataset_root),
        "urdf_path": str(urdf_path),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "state_names": state_names,
        "joint_names_from_urdf": [joint["name"] for joint in chain],
        "ee_link": chain[-1]["child"],
        "future_offset": future_offset,
        "episodes": [],
    }
    ee_rows: list[np.ndarray] = []

    episode_ids = sorted(episodes)
    if max_episodes is not None:
        episode_ids = episode_ids[:max_episodes]

    for episode_id in episode_ids:
        rows = episodes[episode_id]
        ee_positions = []
        ee_eulers = []
        grippers = []
        frame_indices = []
        for row in rows:
            transform = _forward_kinematics(chain, row["state"][:6])
            ee_positions.append(transform[:3, 3])
            ee_eulers.append(Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"))
            grippers.append(float(row["state"][6]))
            frame_indices.append(int(row["frame_index"]))

        pos = np.asarray(ee_positions, dtype=np.float64)
        euler = np.asarray(ee_eulers, dtype=np.float64)
        grip = np.asarray(grippers, dtype=np.float64)
        frames = np.asarray(frame_indices, dtype=np.int32)

        next_idx = np.minimum(np.arange(len(rows)) + future_offset, len(rows) - 1)
        delta_pos = pos[next_idx] - pos
        delta_euler = euler[next_idx] - euler
        delta_euler = (delta_euler + np.pi) % (2 * np.pi) - np.pi
        future_grip = grip[next_idx]

        episode_summary = {
            "episode_index": int(episode_id),
            "num_frames": int(len(rows)),
            "frame_start": int(frames[0]),
            "frame_end": int(frames[-1]),
            "ee_pos_min": pos.min(axis=0).round(6).tolist(),
            "ee_pos_max": pos.max(axis=0).round(6).tolist(),
            "mean_abs_delta_pos": np.abs(delta_pos).mean(axis=0).round(6).tolist(),
            "mean_abs_delta_euler_xyz": np.abs(delta_euler).mean(axis=0).round(6).tolist(),
            "gripper_minmax": [round(float(grip.min()), 6), round(float(grip.max()), 6)],
            "sample_first": {
                "frame_index": int(frames[0]),
                "joint_rad": rows[0]["state"][:6].round(6).tolist(),
                "gripper": round(float(grip[0]), 6),
                "ee_pos": pos[0].round(6).tolist(),
                "ee_euler_xyz": euler[0].round(6).tolist(),
                "delta_pos_to_future": delta_pos[0].round(6).tolist(),
                "delta_euler_xyz_to_future": delta_euler[0].round(6).tolist(),
                "future_gripper": round(float(future_grip[0]), 6),
            },
        }
        summary["episodes"].append(episode_summary)

        for idx in range(len(rows)):
            ee_rows.append(
                np.concatenate(
                    [
                        np.array([episode_id, frames[idx]], dtype=np.float64),
                        rows[idx]["state"][:6],
                        np.array([grip[idx]], dtype=np.float64),
                        pos[idx],
                        euler[idx],
                        delta_pos[idx],
                        delta_euler[idx],
                        np.array([future_grip[idx]], dtype=np.float64),
                    ]
                )
            )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "episode_index,frame_index,"
            "joint1,joint2,joint3,joint4,joint5,joint6,gripper,"
            "ee_x,ee_y,ee_z,ee_roll,ee_pitch,ee_yaw,"
            f"delta{future_offset}_x,delta{future_offset}_y,delta{future_offset}_z,"
            f"delta{future_offset}_roll,delta{future_offset}_pitch,delta{future_offset}_yaw,"
            f"future{future_offset}_gripper"
        )
        np.savetxt(output_path, np.asarray(ee_rows, dtype=np.float64), delimiter=",", header=header, comments="")
        summary["csv_output"] = str(output_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LeRobot V3 dataset root.")
    parser.add_argument("--urdf", type=Path, required=True, help="ARX X5 URDF path.")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional CSV export path.")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional JSON summary output.")
    parser.add_argument("--future_offset", type=int, default=5, help="Offset in frames for EE delta statistics.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit processed episodes for quick inspection.")
    args = parser.parse_args()

    summary = compute(args.input, args.urdf, args.output_csv, args.future_offset, args.max_episodes)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
