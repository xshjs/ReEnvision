import argparse
import io
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_models.utils.rlbench_3d import image_to_float_array, pointcloud_from_depth_and_camera_params


SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6], dtype=np.float32)
N_TRANS_BINS = 100
N_ROT_BINS = 72


def _register_rlbench_pickle_stubs():
    if "rlbench" in sys.modules:
        return

    rlbench = types.ModuleType("rlbench")
    demo_mod = types.ModuleType("rlbench.demo")
    backend_mod = types.ModuleType("rlbench.backend")
    obs_mod = types.ModuleType("rlbench.backend.observation")

    class Demo(list):
        pass

    class Observation:
        pass

    demo_mod.Demo = Demo
    obs_mod.Observation = Observation
    backend_mod.observation = obs_mod
    rlbench.demo = demo_mod
    rlbench.backend = backend_mod
    sys.modules["rlbench"] = rlbench
    sys.modules["rlbench.demo"] = demo_mod
    sys.modules["rlbench.backend"] = backend_mod
    sys.modules["rlbench.backend.observation"] = obs_mod


_register_rlbench_pickle_stubs()


def read_pickle_from_zip(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as handle:
        return pickle.load(handle)


def read_rgb_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as handle:
        return np.array(Image.open(io.BytesIO(handle.read())).convert("RGB"))


def try_read_rgb_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray | None:
    try:
        return read_rgb_from_zip(zf, member)
    except KeyError:
        return None


def read_depth_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as handle:
        image = Image.open(io.BytesIO(handle.read()))
        return image_to_float_array(image).astype(np.float32)


def try_read_depth_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray | None:
    try:
        return read_depth_from_zip(zf, member)
    except KeyError:
        return None


def build_camera_depth_and_pointcloud(obs, camera_name: str, depth_norm: np.ndarray | None):
    if depth_norm is None:
        return {}
    misc = getattr(obs, "misc", {})
    near = float(misc[f"{camera_name}_camera_near"])
    far = float(misc[f"{camera_name}_camera_far"])
    intr = np.asarray(misc[f"{camera_name}_camera_intrinsics"], dtype=np.float32)
    extr = np.asarray(misc[f"{camera_name}_camera_extrinsics"], dtype=np.float32)
    depth_m = near + depth_norm * (far - near)
    point_cloud = pointcloud_from_depth_and_camera_params(depth_m, extr, intr)
    return {
        f"depth_{camera_name}": depth_m.astype(np.float32),
        f"pcd_{camera_name}": point_cloud.astype(np.float32),
        f"{camera_name}_camera_intrinsics": intr.astype(np.float32),
        f"{camera_name}_camera_extrinsics": extr.astype(np.float32),
    }


def quat_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32)


def angle_wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def build_robot_obs(obs) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    euler = quat_to_euler_xyz(pose[3:7])
    gripper = np.float32(obs.gripper_open)
    ignore_collision = np.float32(get_ignore_collision(obs))
    return np.concatenate([pos, euler, [gripper, ignore_collision]], axis=0).astype(np.float32)


def build_abs_action(target_obs) -> np.ndarray:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    euler = quat_to_euler_xyz(pose[3:7])
    gripper = np.array([1.0 if float(target_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    return np.concatenate([pos, euler, gripper], axis=0).astype(np.float32)


def build_abs_action_quat(target_obs) -> np.ndarray:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    quat = pose[3:7]
    quat = quat / np.linalg.norm(quat).clip(min=1e-6)
    gripper = np.array([1.0 if float(target_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    return np.concatenate([pos, quat, gripper], axis=0).astype(np.float32)


def get_ignore_collision(obs) -> float:
    value = getattr(obs, "ignore_collisions", None)
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0] > 0.5)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    quat = quat / np.linalg.norm(quat).clip(min=1e-6)
    if quat[-1] < 0:
        quat = -quat
    return quat.astype(np.float32)


def quaternion_to_discrete_euler(quat_xyzw: np.ndarray, n_rot_bins: int = N_ROT_BINS) -> np.ndarray:
    quat = normalize_quaternion(quat_xyzw)
    euler = R.from_quat(quat).as_euler("xyz", degrees=True)
    euler = np.mod(np.round(euler), 360.0)
    resolution = 360.0 / float(n_rot_bins)
    disc = np.floor(euler / resolution + 0.5).astype(np.int64) % n_rot_bins
    return disc


def point_to_axis_bins(pos_xyz: np.ndarray, scene_bounds: np.ndarray = SCENE_BOUNDS, n_trans_bins: int = N_TRANS_BINS) -> np.ndarray:
    pos = np.asarray(pos_xyz, dtype=np.float32).reshape(3)
    mins = scene_bounds[:3]
    maxs = scene_bounds[3:]
    scaled = (pos - mins) / (maxs - mins).clip(min=1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * (n_trans_bins - 1)).astype(np.int64)


def build_abs_action_rlbench(target_obs, collision_obs=None) -> np.ndarray:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    quat = normalize_quaternion(pose[3:7])
    gripper = np.array([1.0 if float(target_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    if collision_obs is None:
        collision_obs = target_obs
    ignore_collision = np.array([get_ignore_collision(collision_obs)], dtype=np.float32)
    return np.concatenate([pos, quat, gripper, ignore_collision], axis=0).astype(np.float32)


def build_bridge_action(target_obs, collision_obs=None) -> np.ndarray:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    pos_idx = point_to_axis_bins(pose[:3])
    rot_idx = quaternion_to_discrete_euler(pose[3:7])
    gripper = np.int64(float(target_obs.gripper_open) > 0.5)
    if collision_obs is None:
        collision_obs = target_obs
    ignore_collision = np.int64(get_ignore_collision(collision_obs) > 0.5)
    return np.concatenate(
        [pos_idx, rot_idx, np.array([gripper, ignore_collision], dtype=np.int64)],
        axis=0,
    ).astype(np.int64)


def build_bridge_supervision(target_obs, collision_obs=None) -> dict[str, np.ndarray]:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    trans_idx = point_to_axis_bins(pose[:3]).astype(np.int32)
    rot_idx = quaternion_to_discrete_euler(pose[3:7]).astype(np.int32)
    gripper = np.int32(float(target_obs.gripper_open) > 0.5)
    if collision_obs is None:
        collision_obs = target_obs
    ignore_collision = np.int32(get_ignore_collision(collision_obs) > 0.5)
    rot_grip = np.concatenate([rot_idx, np.array([gripper], dtype=np.int32)], axis=0)
    return {
        "trans_action_indicies": trans_idx,
        "rot_grip_action_indicies": rot_grip,
        "ignore_collisions": np.asarray([ignore_collision], dtype=np.int32),
        "gripper_pose": np.concatenate([pose[:3], normalize_quaternion(pose[3:7])], axis=0).astype(np.float32),
    }


def build_rel_action_to_target(curr_obs, target_obs, max_pos: float, max_orn: float) -> np.ndarray:
    curr_pose = np.asarray(curr_obs.gripper_pose, dtype=np.float32)
    target_pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    curr_pos, target_pos = curr_pose[:3], target_pose[:3]
    curr_euler = quat_to_euler_xyz(curr_pose[3:7])
    target_euler = quat_to_euler_xyz(target_pose[3:7])
    rel_pos = np.clip(target_pos - curr_pos, -max_pos, max_pos) / max_pos
    rel_orn = np.clip(angle_wrap(target_euler - curr_euler), -max_orn, max_orn) / max_orn
    gripper = np.array([1.0 if float(target_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    return np.concatenate([rel_pos, rel_orn, gripper], axis=0).astype(np.float32)


def build_rel_action(curr_obs, next_obs, max_pos: float, max_orn: float) -> np.ndarray:
    return build_rel_action_to_target(curr_obs, next_obs, max_pos=max_pos, max_orn=max_orn)


def is_stopped(obs, stopped_buffer: int, delta: float = 0.1) -> bool:
    joint_velocities = getattr(obs, "joint_velocities", None)
    if joint_velocities is None:
        return False
    if stopped_buffer <= 0:
        return False
    joint_velocities = np.asarray(joint_velocities, dtype=np.float32)
    return bool(np.max(np.abs(joint_velocities)) < delta)


def discover_keypoints(obs_seq) -> list[int]:
    """PerAct-style keypoints: gripper changes, robot stops, and final frame."""
    keypoints = []
    stopped_buffer = 0
    previous_gripper_open = float(getattr(obs_seq[0], "gripper_open", 1.0))
    for idx, obs in enumerate(obs_seq):
        stopped = is_stopped(obs, stopped_buffer)
        stopped_buffer = 4 if stopped else max(0, stopped_buffer - 1)
        gripper_open = float(getattr(obs, "gripper_open", previous_gripper_open))
        gripper_changed = gripper_open != previous_gripper_open
        last = idx == len(obs_seq) - 1
        if idx > 0 and (gripper_changed or stopped or last):
            if not keypoints or keypoints[-1] != idx:
                keypoints.append(idx)
        previous_gripper_open = gripper_open
    if not keypoints or keypoints[-1] != len(obs_seq) - 1:
        keypoints.append(len(obs_seq) - 1)
    return keypoints


def detect_first_close_step(obs_seq) -> int | None:
    """Return the first step where gripper transitions from open to closed."""
    previous_open = float(getattr(obs_seq[0], "gripper_open", 1.0))
    for idx, obs in enumerate(obs_seq[1:], start=1):
        gripper_open = float(getattr(obs, "gripper_open", previous_open))
        if previous_open > 0.5 and gripper_open <= 0.5:
            return idx
        previous_open = gripper_open
    return None


def build_stage_labels(obs_seq) -> np.ndarray:
    """Build a coarse CALVIN-like stage label for every frame.

    Stages:
      0: approach
      1: pre_contact
      2: contact_execute
      3: release / post_contact
    """
    episode_len = len(obs_seq)
    first_close = detect_first_close_step(obs_seq)
    if first_close is None:
        pivot = max(1, int(round(0.65 * (episode_len - 1))))
        first_close = pivot
    window = max(4, min(8, episode_len // 12))
    pre_contact_start = max(0, first_close - window)
    contact_end = min(episode_len, first_close + window)
    stages = np.zeros((episode_len,), dtype=np.int64)
    for idx in range(episode_len):
        if idx < pre_contact_start:
            stages[idx] = 0
        elif idx < first_close:
            stages[idx] = 1
        elif idx < contact_end:
            stages[idx] = 2
        else:
            stages[idx] = 3
    return stages


def select_target_index(step_idx: int, obs_seq, keypoints: list[int], mode: str, stride: int) -> int:
    if mode == "next":
        return min(step_idx + 1, len(obs_seq) - 1)
    if mode == "stride":
        return min(step_idx + stride, len(obs_seq) - 1)
    if mode == "keypoint":
        for keypoint in keypoints:
            if keypoint > step_idx:
                return keypoint
        return len(obs_seq) - 1
    raise ValueError(f"unknown action target mode: {mode}")


def iter_episode_roots(zf: zipfile.ZipFile):
    members = [name for name in zf.namelist() if name.endswith("low_dim_obs.pkl")]
    for member in sorted(members):
        yield member[: -len("low_dim_obs.pkl")]


def extract_obs_seq(demo):
    if hasattr(demo, "_observations"):
        return demo._observations
    return demo


def find_variation_description(zf: zipfile.ZipFile, episode_root: str):
    parts = Path(episode_root).parts
    variation_dir = None
    for idx, part in enumerate(parts):
        if part.startswith("variation"):
            variation_dir = "/".join(parts[: idx + 1])
            break
    if variation_dir is None:
        return ""
    desc_member = f"{variation_dir}/variation_descriptions.pkl"
    descriptions = read_pickle_from_zip(zf, desc_member)
    if isinstance(descriptions, (list, tuple)) and descriptions:
        return str(descriptions[0])
    return ""


def save_npz_frame(
    out_dir: Path,
    frame_idx: int,
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    robot_obs: np.ndarray,
    rel_action: np.ndarray,
    abs_action: np.ndarray,
    abs_action_quat: np.ndarray,
    abs_action_rlbench: np.ndarray,
    bridge_action: np.ndarray,
    bridge_supervision: dict[str, np.ndarray],
    stage: int,
    target_index: int,
    extra_rgb: dict[str, np.ndarray],
    extra_3d: dict[str, np.ndarray],
):
    payload = {
        "rgb_static": rgb_static,
        "rgb_gripper": rgb_gripper,
        "robot_obs": robot_obs,
        "scene_obs": np.zeros((1,), dtype=np.float32),
        "rel_actions": rel_action,
        "actions": abs_action,
        "actions_quat": abs_action_quat,
        "actions_rlbench": abs_action_rlbench,
        "actions_bridge": bridge_action,
        "stage": np.asarray(stage, dtype=np.int64),
        "rlbench_target_index": np.asarray([target_index], dtype=np.int32),
    }
    payload.update(bridge_supervision)
    payload.update(extra_rgb)
    payload.update(extra_3d)
    np.savez(out_dir / f"episode_{frame_idx:07d}.npz", **payload)


def convert_zip_to_split(
    zip_path: Path,
    split_dir: Path,
    frame_offset: int,
    episode_offset: int,
    max_episodes: int | None,
    max_pos: float,
    max_orn: float,
    action_target: str,
    action_stride: int,
    split_name: str,
):
    lang_ann = []
    lang_task = []
    lang_emb = []
    lang_indx = []
    global_frame_idx = frame_offset
    global_episode_count = episode_offset
    task_name = zip_path.stem

    with zipfile.ZipFile(zip_path) as zf:
        episode_roots = list(iter_episode_roots(zf))
        if max_episodes is not None:
            episode_roots = episode_roots[:max_episodes]
        for episode_root in episode_roots:
            obs_seq = extract_obs_seq(read_pickle_from_zip(zf, f"{episode_root}low_dim_obs.pkl"))
            if len(obs_seq) < 2:
                continue

            language = find_variation_description(zf, episode_root) or task_name.replace("_", " ")
            start_idx = global_frame_idx
            keypoints = discover_keypoints(obs_seq)
            stage_labels = build_stage_labels(obs_seq)

            for step_idx, curr_obs in enumerate(obs_seq):
                target_idx = select_target_index(step_idx, obs_seq, keypoints, action_target, action_stride)
                target_obs = obs_seq[target_idx]
                rgb_static = read_rgb_from_zip(zf, f"{episode_root}front_rgb/{step_idx}.png")
                rgb_gripper = read_rgb_from_zip(zf, f"{episode_root}wrist_rgb/{step_idx}.png")
                rgb_left = try_read_rgb_from_zip(zf, f"{episode_root}left_shoulder_rgb/{step_idx}.png")
                rgb_right = try_read_rgb_from_zip(zf, f"{episode_root}right_shoulder_rgb/{step_idx}.png")
                depth_front = try_read_depth_from_zip(zf, f"{episode_root}front_depth/{step_idx}.png")
                depth_wrist = try_read_depth_from_zip(zf, f"{episode_root}wrist_depth/{step_idx}.png")
                depth_left = try_read_depth_from_zip(zf, f"{episode_root}left_shoulder_depth/{step_idx}.png")
                depth_right = try_read_depth_from_zip(zf, f"{episode_root}right_shoulder_depth/{step_idx}.png")
                robot_obs = build_robot_obs(curr_obs)
                rel_action = build_rel_action_to_target(curr_obs, target_obs, max_pos=max_pos, max_orn=max_orn)
                abs_action = build_abs_action(target_obs)
                abs_action_quat = build_abs_action_quat(target_obs)
                collision_obs = obs_seq[max(0, target_idx - 1)]
                abs_action_rlbench = build_abs_action_rlbench(target_obs, collision_obs=collision_obs)
                bridge_action = build_bridge_action(target_obs, collision_obs=collision_obs)
                bridge_supervision = build_bridge_supervision(target_obs, collision_obs=collision_obs)
                extra_rgb = {}
                if rgb_left is not None:
                    extra_rgb["rgb_left_shoulder"] = rgb_left
                if rgb_right is not None:
                    extra_rgb["rgb_right_shoulder"] = rgb_right
                extra_3d = {}
                extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "front", depth_front))
                extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "wrist", depth_wrist))
                extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "left_shoulder", depth_left))
                extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "right_shoulder", depth_right))
                save_npz_frame(
                    split_dir,
                    global_frame_idx,
                    rgb_static,
                    rgb_gripper,
                    robot_obs,
                    rel_action,
                    abs_action,
                    abs_action_quat,
                    abs_action_rlbench,
                    bridge_action,
                    bridge_supervision,
                    int(stage_labels[step_idx]),
                    target_idx,
                    extra_rgb,
                    extra_3d,
                )
                global_frame_idx += 1

            end_idx = global_frame_idx - 1
            lang_ann.append(language)
            lang_task.append(task_name)
            lang_emb.append(np.zeros((1, 1024), dtype=np.float32))
            lang_indx.append((start_idx, end_idx))
            global_episode_count += 1

    return {
        "frame_offset": global_frame_idx,
        "episode_count": global_episode_count,
        "lang_ann": lang_ann,
        "lang_task": lang_task,
        "lang_emb": lang_emb,
        "lang_indx": lang_indx,
    }


def save_lang_bundle(out_dir: Path, lang_ann, lang_task, lang_emb, lang_indx):
    lang_dir = out_dir / "lang_clip_resnet50"
    lang_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {"episodes": [], "indx": lang_indx},
        "language": {
            "ann": lang_ann,
            "task": lang_task,
            "emb": np.asarray(lang_emb, dtype=np.float32),
        },
    }
    np.save(lang_dir / "auto_lang_ann.npy", payload, allow_pickle=True)
    np.save(out_dir / "auto_lang_ann.npy", payload, allow_pickle=True)
    np.save(out_dir / "ep_start_end_ids.npy", np.asarray(lang_indx, dtype=np.int64))

    compat_dir = out_dir / "lang_annotations"
    compat_dir.mkdir(parents=True, exist_ok=True)
    np.save(compat_dir / "auto_lang_ann.npy", payload, allow_pickle=True)


def save_statistics(out_dir: Path, frame_count: int):
    sums: dict[str, np.ndarray] = {}
    sq_sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for frame_idx in range(frame_count):
        data = np.load(out_dir / f"episode_{frame_idx:07d}.npz")
        for key in ("robot_obs", "scene_obs", "rel_actions", "actions"):
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
            sums[key] = sums.get(key, np.zeros_like(arr)) + arr
            sq_sums[key] = sq_sums.get(key, np.zeros_like(arr)) + arr * arr
            counts[key] = counts.get(key, 0) + 1

    lines = []
    target_map = {
        "robot_obs": "policy_models.utils.transforms.NormalizeVector",
        "scene_obs": "policy_models.utils.transforms.NormalizeVector",
    }
    for key in ("robot_obs", "scene_obs"):
        if key not in sums:
            continue
        mean = sums[key] / counts[key]
        var = np.maximum(sq_sums[key] / counts[key] - mean * mean, 1e-12)
        std = np.sqrt(var)
        lines.extend(
            [
                f"{key}:",
                f"  - _target_: {target_map[key]}",
                f"    mean: {[float(x) for x in mean]}",
                f"    std: {[float(x) for x in std]}",
            ]
        )
    if lines:
        (out_dir / "statistics.yaml").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert RLBench task zips into a merged DeFi dataset.")
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-pos", type=float, default=0.05)
    parser.add_argument("--max-orn", type=float, default=0.50)
    parser.add_argument("--action-target", choices=["next", "stride", "keypoint"], default="keypoint")
    parser.add_argument("--action-stride", type=int, default=5)
    parser.add_argument("--skip-statistics", action="store_true")
    args = parser.parse_args()

    zip_paths = sorted(args.zip_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found in {args.zip_dir}")
    if args.tasks:
        requested = {f"{task.strip()}.zip" for task in args.tasks.split(",") if task.strip()}
        zip_paths = [path for path in zip_paths if path.name in requested]
        if not zip_paths:
            raise FileNotFoundError(f"No requested task zips found for tasks={sorted(requested)} in {args.zip_dir}")

    train_dir = args.out_root / "training"
    val_dir = args.out_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_frame_offset = 0
    val_frame_offset = 0
    train_episode_offset = 0
    val_episode_offset = 0
    train_lang_ann = []
    train_lang_task = []
    train_lang_emb = []
    train_lang_indx = []
    val_lang_ann = []
    val_lang_task = []
    val_lang_emb = []
    val_lang_indx = []

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            episode_roots = list(iter_episode_roots(zf))
        if args.max_episodes is not None:
            episode_roots = episode_roots[:args.max_episodes]
        if not episode_roots:
            continue

        num_val = max(1, int(round(len(episode_roots) * args.val_ratio)))
        num_val = min(num_val, len(episode_roots) - 1) if len(episode_roots) > 1 else 0
        val_episode_set = set(episode_roots[-num_val:]) if num_val > 0 else set()

        def _convert_subset(split_dir: Path, selected_roots: set[str], frame_offset: int, episode_offset: int):
            lang_ann = []
            lang_task = []
            lang_emb = []
            lang_indx = []
            global_frame_idx = frame_offset
            global_episode_count = episode_offset

            with zipfile.ZipFile(zip_path) as zf:
                for episode_root in episode_roots:
                    if episode_root not in selected_roots:
                        continue
                    obs_seq = extract_obs_seq(read_pickle_from_zip(zf, f"{episode_root}low_dim_obs.pkl"))
                    if len(obs_seq) < 2:
                        continue
                    task_name = zip_path.stem
                    language = find_variation_description(zf, episode_root) or task_name.replace("_", " ")
                    start_idx = global_frame_idx
                    keypoints = discover_keypoints(obs_seq)
                    stage_labels = build_stage_labels(obs_seq)
                    for step_idx, curr_obs in enumerate(obs_seq):
                        target_idx = select_target_index(
                            step_idx,
                            obs_seq,
                            keypoints,
                            args.action_target,
                            args.action_stride,
                        )
                        target_obs = obs_seq[target_idx]
                        rgb_static = read_rgb_from_zip(zf, f"{episode_root}front_rgb/{step_idx}.png")
                        rgb_gripper = read_rgb_from_zip(zf, f"{episode_root}wrist_rgb/{step_idx}.png")
                        rgb_left = try_read_rgb_from_zip(zf, f"{episode_root}left_shoulder_rgb/{step_idx}.png")
                        rgb_right = try_read_rgb_from_zip(zf, f"{episode_root}right_shoulder_rgb/{step_idx}.png")
                        depth_front = try_read_depth_from_zip(zf, f"{episode_root}front_depth/{step_idx}.png")
                        depth_wrist = try_read_depth_from_zip(zf, f"{episode_root}wrist_depth/{step_idx}.png")
                        depth_left = try_read_depth_from_zip(zf, f"{episode_root}left_shoulder_depth/{step_idx}.png")
                        depth_right = try_read_depth_from_zip(zf, f"{episode_root}right_shoulder_depth/{step_idx}.png")
                        robot_obs = build_robot_obs(curr_obs)
                        rel_action = build_rel_action_to_target(curr_obs, target_obs, max_pos=args.max_pos, max_orn=args.max_orn)
                        abs_action = build_abs_action(target_obs)
                        abs_action_quat = build_abs_action_quat(target_obs)
                        collision_obs = obs_seq[max(0, target_idx - 1)]
                        abs_action_rlbench = build_abs_action_rlbench(target_obs, collision_obs=collision_obs)
                        bridge_action = build_bridge_action(target_obs, collision_obs=collision_obs)
                        bridge_supervision = build_bridge_supervision(target_obs, collision_obs=collision_obs)
                        extra_rgb = {}
                        if rgb_left is not None:
                            extra_rgb["rgb_left_shoulder"] = rgb_left
                        if rgb_right is not None:
                            extra_rgb["rgb_right_shoulder"] = rgb_right
                        extra_3d = {}
                        extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "front", depth_front))
                        extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "wrist", depth_wrist))
                        extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "left_shoulder", depth_left))
                        extra_3d.update(build_camera_depth_and_pointcloud(curr_obs, "right_shoulder", depth_right))
                        save_npz_frame(
                            split_dir,
                            global_frame_idx,
                            rgb_static,
                            rgb_gripper,
                            robot_obs,
                            rel_action,
                            abs_action,
                            abs_action_quat,
                            abs_action_rlbench,
                            bridge_action,
                            bridge_supervision,
                            int(stage_labels[step_idx]),
                            target_idx,
                            extra_rgb,
                            extra_3d,
                        )
                        global_frame_idx += 1
                    end_idx = global_frame_idx - 1
                    lang_ann.append(language)
                    lang_task.append(task_name)
                    lang_emb.append(np.zeros((1, 1024), dtype=np.float32))
                    lang_indx.append((start_idx, end_idx))
                    global_episode_count += 1
            return global_frame_idx, global_episode_count, lang_ann, lang_task, lang_emb, lang_indx

        train_roots = set(episode_roots) - val_episode_set
        train_frame_offset, train_episode_offset, ta, tt, te, ti = _convert_subset(
            train_dir, train_roots, train_frame_offset, train_episode_offset
        )
        val_frame_offset, val_episode_offset, va, vt, ve, vi = _convert_subset(
            val_dir, val_episode_set, val_frame_offset, val_episode_offset
        )
        train_lang_ann.extend(ta)
        train_lang_task.extend(tt)
        train_lang_emb.extend(te)
        train_lang_indx.extend(ti)
        val_lang_ann.extend(va)
        val_lang_task.extend(vt)
        val_lang_emb.extend(ve)
        val_lang_indx.extend(vi)

        print(
            f"{zip_path.name}: train_episodes={len(train_roots)}, val_episodes={len(val_episode_set)}, "
            f"train_frames={train_frame_offset}, val_frames={val_frame_offset}"
        )

    save_lang_bundle(train_dir, train_lang_ann, train_lang_task, train_lang_emb, train_lang_indx)
    save_lang_bundle(val_dir, val_lang_ann, val_lang_task, val_lang_emb, val_lang_indx)
    if not args.skip_statistics:
        save_statistics(train_dir, train_frame_offset)
        save_statistics(val_dir, val_frame_offset)

    summary = {
        "training": {
            "episodes": len(train_lang_indx),
            "frames": train_frame_offset,
        },
        "validation": {
            "episodes": len(val_lang_indx),
            "frames": val_frame_offset,
        },
        "zips": [p.name for p in zip_paths],
        "action_target": args.action_target,
        "action_stride": args.action_stride,
        "max_pos": args.max_pos,
        "max_orn": args.max_orn,
        "skip_statistics": bool(args.skip_statistics),
    }
    np.save(args.out_root / "summary.npy", summary, allow_pickle=True)
    print(summary)


if __name__ == "__main__":
    main()
