from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def load_action_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if "pred" in data:
            arr = data["pred"]
        elif "actions" in data:
            arr = data["actions"]
        elif "state" in data:
            arr = data["state"]
        else:
            raise KeyError(f"{path} must contain one of: pred, actions, state")
    elif path.suffix == ".json":
        arr = np.asarray(json.loads(path.read_text()), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file suffix: {path.suffix}")
    return np.asarray(arr, dtype=np.float64)


def load_action_array_json(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=np.float64)


def convert_rel_action_to_x5_ee(
    actions: np.ndarray,
    max_pos: float = 0.05,
    max_orn: float = 0.50,
    binary_gripper: bool = False,
    gripper_close_value: float = -1.0,
    gripper_open_value: float = 1.0,
) -> np.ndarray:
    # This is the direct denormalization path used when the execution side accepts EE deltas.
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected last dim 7, got {actions.shape}")
    out = np.array(actions, dtype=np.float32, copy=True)
    out[..., :3] = np.clip(out[..., :3], -1.0, 1.0) * max_pos
    out[..., 3:6] = np.clip(out[..., 3:6], -1.0, 1.0) * max_orn
    if binary_gripper:
        out[..., 6] = np.where(out[..., 6] > 0.0, 1.0, -1.0)
    else:
        if gripper_open_value < gripper_close_value:
            raise ValueError("gripper_open_value must be >= gripper_close_value")
        norm01 = (np.clip(out[..., 6], -1.0, 1.0) + 1.0) * 0.5
        out[..., 6] = gripper_close_value + norm01 * (gripper_open_value - gripper_close_value)
    return out


def load_joint_state_from_dataset(dataset_root: Path, episode_index: int, frame_index: int) -> np.ndarray:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "observation.state"])
        data = table.to_pydict()
        for ep, frame, state in zip(data["episode_index"], data["frame_index"], data["observation.state"], strict=True):
            if int(ep) == episode_index and int(frame) == frame_index:
                return np.asarray(state, dtype=np.float64)
    raise FileNotFoundError(
        f"Could not find episode_index={episode_index}, frame_index={frame_index} in {dataset_root}"
    )


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
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _parse_xyz_rpy(node: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.fromstring(node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(node.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    return xyz, rpy


def load_x5_fk_chain(urdf_path: Path) -> list[dict]:
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
            raise ValueError(f"Joint {joint.attrib.get('name', '<unnamed>')} missing origin/axis/child")
        xyz, rpy = _parse_xyz_rpy(origin)
        axis_xyz = np.fromstring(axis.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
        axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)
        joints.append(
            {
                "name": joint.attrib["name"],
                "origin_xyz": xyz,
                "origin_rpy": rpy,
                "axis": axis_xyz,
                "child": child.attrib["link"],
            }
        )
    if len(joints) < 6:
        raise ValueError(f"Expected at least 6 revolute joints in {urdf_path}, found {len(joints)}")
    return joints[:6]


def _forward_kinematics(chain: list[dict], joint_values: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for joint, angle in zip(chain, joint_values[: len(chain)], strict=True):
        origin_rotation = _rpy_to_matrix(joint["origin_rpy"])
        origin_transform = _make_transform(origin_rotation, joint["origin_xyz"])
        motion_transform = _make_transform(
            Rotation.from_rotvec(joint["axis"] * float(angle)).as_matrix(),
            np.zeros(3, dtype=np.float64),
        )
        transform = transform @ origin_transform @ motion_transform
    return transform


def _pose_vec(chain: list[dict], joint_values: np.ndarray) -> np.ndarray:
    tf = _forward_kinematics(chain, joint_values)
    euler = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([tf[:3, 3], euler], axis=0)


def _wrap_angle(diff: np.ndarray) -> np.ndarray:
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _numerical_jacobian(chain: list[dict], joints: np.ndarray, eps: float) -> np.ndarray:
    base = _pose_vec(chain, joints)
    jac = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        perturbed = joints.copy()
        perturbed[i] += eps
        pose_eps = _pose_vec(chain, perturbed)
        delta = pose_eps - base
        delta[3:6] = _wrap_angle(delta[3:6])
        jac[:, i] = delta / eps
    return jac


def _damped_ls_solve(jac: np.ndarray, target_delta: np.ndarray, damping: float) -> np.ndarray:
    jjt = jac @ jac.T
    reg = (damping ** 2) * np.eye(jjt.shape[0], dtype=np.float64)
    return jac.T @ np.linalg.solve(jjt + reg, target_delta)


def convert_rel_action_to_x5_joint(
    actions: np.ndarray,
    joints: np.ndarray,
    fk_chain: list[dict],
    max_pos: float = 0.05,
    max_orn: float = 0.50,
    damping: float = 0.05,
    jac_eps: float = 1e-4,
    max_joint_delta: float = 0.05,
    binary_gripper: bool = False,
    gripper_close_value: float = -1.0,
    gripper_open_value: float = 1.0,
) -> np.ndarray:
    # This path approximates EE->joint conversion with a numerical Jacobian around the current joint state.
    actions = np.asarray(actions, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected action last dim 7, got {actions.shape}")
    if joints.shape[-1] < 6:
        raise ValueError(f"Expected joint state last dim >= 6, got {joints.shape}")

    flat_actions = actions.reshape(-1, 7)
    flat_joints = joints.reshape(-1, joints.shape[-1])
    if flat_joints.shape[0] == 1 and flat_actions.shape[0] > 1:
        flat_joints = np.repeat(flat_joints, flat_actions.shape[0], axis=0)
    if flat_joints.shape[0] != flat_actions.shape[0]:
        raise ValueError(f"Action count {flat_actions.shape[0]} != joint-state count {flat_joints.shape[0]}")

    out = np.zeros((flat_actions.shape[0], 7), dtype=np.float64)
    for i, (a, q) in enumerate(zip(flat_actions, flat_joints, strict=True)):
        ee_delta = np.zeros((6,), dtype=np.float64)
        ee_delta[:3] = np.clip(a[:3], -1.0, 1.0) * max_pos
        ee_delta[3:6] = np.clip(a[3:6], -1.0, 1.0) * max_orn
        jac = _numerical_jacobian(fk_chain, q[:6], jac_eps)
        dq = _damped_ls_solve(jac, ee_delta, damping)
        dq = np.clip(dq, -max_joint_delta, max_joint_delta)
        out[i, :6] = dq
        if binary_gripper:
            out[i, 6] = 1.0 if a[6] > 0.0 else -1.0
        else:
            if gripper_open_value < gripper_close_value:
                raise ValueError("gripper_open_value must be >= gripper_close_value")
            norm01 = (float(np.clip(a[6], -1.0, 1.0)) + 1.0) * 0.5
            out[i, 6] = gripper_close_value + norm01 * (gripper_open_value - gripper_close_value)
    return out.reshape(actions.shape).astype(np.float32)
