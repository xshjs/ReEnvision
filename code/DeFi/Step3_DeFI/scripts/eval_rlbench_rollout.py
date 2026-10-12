import argparse
import importlib
import io
import os
import pickle
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow reusing DeFi's existing ML stack only when the interpreter ABI matches.
# DeFi/.venv is Python 3.10; mounting it into a different interpreter (e.g.
# bridgevla's Python 3.9) breaks binary extensions such as scipy/PIL/cv2.
DEFI_SITE_PACKAGES = ROOT.parents[0] / ".venv" / "lib" / "python3.10" / "site-packages"
if (
    sys.version_info[:2] == (3, 10)
    and DEFI_SITE_PACKAGES.is_dir()
    and str(DEFI_SITE_PACKAGES) not in sys.path
):
    sys.path.append(str(DEFI_SITE_PACKAGES))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from scipy.spatial.transform import Rotation as R

from policy_models.utils.rlbench_3d import image_to_float_array, pointcloud_from_depth_and_camera_params

BRIDGEVLA_FINETUNE_ROOT = ROOT.parents[1] / "BridgeVLA" / "finetune"
BRIDGEVLA_LIBS = BRIDGEVLA_FINETUNE_ROOT / "bridgevla" / "libs"
for extra_path in [
    BRIDGEVLA_LIBS / "PyRep",
    BRIDGEVLA_LIBS / "RLBench",
    BRIDGEVLA_LIBS / "YARR",
]:
    if extra_path.is_dir() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

COPPELIASIM_ROOT = BRIDGEVLA_FINETUNE_ROOT / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if COPPELIASIM_ROOT.is_dir():
    os.environ.setdefault("COPPELIASIM_ROOT", str(COPPELIASIM_ROOT))
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(COPPELIASIM_ROOT) not in current_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{COPPELIASIM_ROOT}:{current_ld}" if current_ld else str(COPPELIASIM_ROOT)
        )
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(COPPELIASIM_ROOT))

import hydra
import torch
from hydra import compose, initialize
from PIL import Image
import torchvision


def _patch_compat():
    try:
        import huggingface_hub as hf_hub

        if not hasattr(hf_hub, "cached_download") and hasattr(hf_hub, "hf_hub_download"):
            hf_hub.cached_download = hf_hub.hf_hub_download
    except Exception:
        pass

    try:
        import transformers.utils as tf_utils

        if not hasattr(tf_utils, "FLAX_WEIGHTS_NAME"):
            tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        if hasattr(tf_utils, "check_torch_load_is_safe"):
            tf_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


_patch_compat()

from policy_models.datasets.utils.episode_utils import (  # noqa: E402
    process_camera,
    process_depth,
    process_point_cloud,
    process_rgb,
    process_state,
)


def build_eval_transforms(cfg):
    transforms_cfg = cfg.datamodule.transforms.val
    transforms = {}
    for key in transforms_cfg:
        ops = []
        for transform in transforms_cfg[key]:
            if transform._target_ == "torchvision.transforms.ColorJitter":
                op = torchvision.transforms.ColorJitter(
                    brightness=transform.brightness,
                    contrast=tuple(transform.contrast),
                    saturation=tuple(transform.saturation),
                )
            else:
                op = hydra.utils.instantiate(transform)
            ops.append(op)
        transforms[key] = torchvision.transforms.Compose(ops)
    return transforms


def filter_state_dict_by_shape(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, f"{tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        filtered[key] = value
    return filtered, skipped


def snake_to_camel(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


def import_task(task_name: str):
    module = importlib.import_module(f"rlbench.tasks.{task_name}")
    return getattr(module, snake_to_camel(task_name))


def load_demo_from_zip(zip_dir: str, task_name: str, demo_index: int):
    zip_path = Path(zip_dir) / f"{task_name}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"demo zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith("low_dim_obs.pkl"))
        if not names:
            raise FileNotFoundError(f"no low_dim_obs.pkl entries in {zip_path}")
        member = names[demo_index % len(names)]
        with zf.open(member) as handle:
            demo = pickle.load(handle)
    if len(demo) > 0 and hasattr(demo[0], "misc") and "variation_index" not in demo[0].misc:
        demo[0].misc["variation_index"] = 0
    return demo, member


def load_demo_language_from_zip(zip_dir: str, task_name: str, demo_member: str) -> list[str]:
    zip_path = Path(zip_dir) / f"{task_name}.zip"
    episode_root = demo_member[: -len("low_dim_obs.pkl")]
    lang_member = f"{episode_root}variation_descriptions.pkl"
    if not zip_path.exists():
        return []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if lang_member not in zf.namelist():
                return []
            with zf.open(lang_member) as handle:
                descriptions = pickle.load(handle)
    except Exception:
        return []

    if isinstance(descriptions, str):
        return [descriptions]
    if isinstance(descriptions, (list, tuple)):
        return [str(x) for x in descriptions if str(x).strip()]
    return []


def read_rgb_from_zip(zip_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return np.array(Image.open(handle).convert("RGB"))


def read_depth_from_zip(zip_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            return image_to_float_array(Image.open(io.BytesIO(handle.read()))).astype(np.float32)


def build_obs_config(image_size: int):
    from rlbench.observation_config import CameraConfig, ObservationConfig

    cam = CameraConfig(
        rgb=True,
        depth=True,
        point_cloud=True,
        mask=False,
        image_size=(image_size, image_size),
    )
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera = cam
    obs_config.left_shoulder_camera = cam
    obs_config.right_shoulder_camera = cam
    obs_config.wrist_camera = cam
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.record_ignore_collisions = True
    return obs_config


def build_action_mode(kind: str):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.action_modes.arm_action_modes import Scene

    class EndEffectorPoseViaPlanningClamped(EndEffectorPoseViaPlanning):
        def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
            action[:3] = np.clip(
                action[:3],
                np.array([scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]) + 1e-7,
                np.array([scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]) - 1e-7,
            )
            super().action(scene, action, ignore_collisions)

    if kind == "ik":
        arm = EndEffectorPoseViaIK(absolute_mode=True, collision_checking=True)
    elif kind == "planning":
        arm = EndEffectorPoseViaPlanningClamped(absolute_mode=True, collision_checking=True)
    else:
        raise ValueError(f"unknown action mode: {kind}")
    return MoveArmThenGripper(arm, Discrete())


def quat_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quat = quat / norm
    return R.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)


def get_ignore_collision(obs) -> float:
    value = getattr(obs, "ignore_collisions", None)
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0] > 0.5)


def build_robot_obs(obs) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    euler = quat_to_euler_xyz(pose[3:7])
    gripper = np.float32(obs.gripper_open)
    ignore_collision = np.float32(get_ignore_collision(obs))
    return np.concatenate([pos, euler, [gripper, ignore_collision]], axis=0).astype(np.float32)


def build_rel_action(curr_obs, next_obs, max_pos: float, max_orn: float) -> np.ndarray:
    curr_pose = np.asarray(curr_obs.gripper_pose, dtype=np.float32)
    next_pose = np.asarray(next_obs.gripper_pose, dtype=np.float32)
    curr_pos, next_pos = curr_pose[:3], next_pose[:3]
    curr_euler = quat_to_euler_xyz(curr_pose[3:7])
    next_euler = quat_to_euler_xyz(next_pose[3:7])
    rel_pos = np.clip(next_pos - curr_pos, -max_pos, max_pos) / max_pos
    rel_orn = np.clip((next_euler - curr_euler + np.pi) % (2 * np.pi) - np.pi, -max_orn, max_orn) / max_orn
    gripper = np.array([1.0 if float(next_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    return np.concatenate([rel_pos, rel_orn, gripper], axis=0).astype(np.float32)


def build_abs_action_quat(target_obs) -> np.ndarray:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    quat = pose[3:7]
    quat = quat / np.linalg.norm(quat).clip(min=1e-6)
    gripper = np.array([1.0 if float(target_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    return np.concatenate([pos, quat, gripper], axis=0).astype(np.float32)


def build_abs_action_rlbench(target_obs) -> np.ndarray:
    pose = np.asarray(target_obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    quat = pose[3:7]
    quat = quat / np.linalg.norm(quat).clip(min=1e-6)
    gripper = np.array([1.0 if float(target_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    ignore_collision = np.array([get_ignore_collision(target_obs)], dtype=np.float32)
    return np.concatenate([pos, quat, gripper, ignore_collision], axis=0).astype(np.float32)


def normalize_quaternion_xyzw(quat_xyzw: np.ndarray, fallback_xyzw: np.ndarray | None = None) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    if quat.shape[0] != 4:
        if fallback_xyzw is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return normalize_quaternion_xyzw(fallback_xyzw, None)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        if fallback_xyzw is not None:
            return normalize_quaternion_xyzw(fallback_xyzw, None)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat = quat / norm
    if quat[-1] < 0:
        quat = -quat
    return quat.astype(np.float32)


def to_uint8_rgb(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_episode_frame(frame_dir: Path, step: int, obs) -> None:
    from PIL import Image

    frame_dir.mkdir(parents=True, exist_ok=True)
    image = to_uint8_rgb(obs.front_rgb)
    Image.fromarray(image).save(frame_dir / f"{step:04d}.png")


def render_episode_video(frame_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[video] ffmpeg not found; frames kept at {frame_dir}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[video] saved {output_path}")


def demo_obs_to_policy_batch(obs, cfg, transforms, device):
    episode = {
        "rgb_static": to_uint8_rgb(obs.front_rgb),
        "rgb_left_shoulder": to_uint8_rgb(obs.left_shoulder_rgb),
        "rgb_right_shoulder": to_uint8_rgb(obs.right_shoulder_rgb),
        "rgb_gripper": to_uint8_rgb(obs.wrist_rgb),
        "depth_front": np.asarray(obs.front_depth, dtype=np.float32),
        "depth_left_shoulder": np.asarray(obs.left_shoulder_depth, dtype=np.float32),
        "depth_right_shoulder": np.asarray(obs.right_shoulder_depth, dtype=np.float32),
        "depth_wrist": np.asarray(obs.wrist_depth, dtype=np.float32),
        "pcd_front": np.asarray(obs.front_point_cloud, dtype=np.float32),
        "pcd_left_shoulder": np.asarray(obs.left_shoulder_point_cloud, dtype=np.float32),
        "pcd_right_shoulder": np.asarray(obs.right_shoulder_point_cloud, dtype=np.float32),
        "pcd_wrist": np.asarray(obs.wrist_point_cloud, dtype=np.float32),
        "front_camera_intrinsics": np.asarray(obs.misc["front_camera_intrinsics"], dtype=np.float32),
        "front_camera_extrinsics": np.asarray(obs.misc["front_camera_extrinsics"], dtype=np.float32),
        "left_shoulder_camera_intrinsics": np.asarray(obs.misc["left_shoulder_camera_intrinsics"], dtype=np.float32),
        "left_shoulder_camera_extrinsics": np.asarray(obs.misc["left_shoulder_camera_extrinsics"], dtype=np.float32),
        "right_shoulder_camera_intrinsics": np.asarray(obs.misc["right_shoulder_camera_intrinsics"], dtype=np.float32),
        "right_shoulder_camera_extrinsics": np.asarray(obs.misc["right_shoulder_camera_extrinsics"], dtype=np.float32),
        "wrist_camera_intrinsics": np.asarray(obs.misc["wrist_camera_intrinsics"], dtype=np.float32),
        "wrist_camera_extrinsics": np.asarray(obs.misc["wrist_camera_extrinsics"], dtype=np.float32),
        "robot_obs": build_robot_obs(obs),
    }
    rgb = process_rgb(episode, cfg.datamodule.observation_space, transforms)
    depth = process_depth(episode, cfg.datamodule.observation_space, transforms)
    point_cloud = process_point_cloud(episode, cfg.datamodule.observation_space, transforms)
    camera = process_camera(episode, cfg.datamodule.observation_space)
    state = process_state(episode, cfg.datamodule.observation_space, transforms, cfg.datamodule.proprioception_dims)
    return {
        "rgb_obs": {k: v.unsqueeze(0).to(device) for k, v in rgb["rgb_obs"].items() if not k.endswith("_origin")},
        "depth_obs": {k: v.unsqueeze(0).to(device) for k, v in depth["depth_obs"].items()},
        "point_cloud_obs": {k: v.unsqueeze(0).to(device) for k, v in point_cloud["point_cloud_obs"].items()},
        "camera_obs": {k: v.unsqueeze(0).to(device) for k, v in camera["camera_obs"].items()},
        "robot_obs": state["robot_obs"].unsqueeze(0).to(device),
    }


def demo_step_to_policy_batch(zip_dir: str, task_name: str, demo_member: str, demo_obs, step: int, cfg, transforms, device):
    episode_root = demo_member[: -len("low_dim_obs.pkl")]
    zip_path = Path(zip_dir) / f"{task_name}.zip"
    front_depth = read_depth_from_zip(zip_path, f"{episode_root}front_depth/{step}.png")
    left_depth = read_depth_from_zip(zip_path, f"{episode_root}left_shoulder_depth/{step}.png")
    right_depth = read_depth_from_zip(zip_path, f"{episode_root}right_shoulder_depth/{step}.png")
    wrist_depth = read_depth_from_zip(zip_path, f"{episode_root}wrist_depth/{step}.png")
    front_depth_m = float(demo_obs.misc["front_camera_near"]) + front_depth * float(demo_obs.misc["front_camera_far"] - demo_obs.misc["front_camera_near"])
    left_depth_m = float(demo_obs.misc["left_shoulder_camera_near"]) + left_depth * float(demo_obs.misc["left_shoulder_camera_far"] - demo_obs.misc["left_shoulder_camera_near"])
    right_depth_m = float(demo_obs.misc["right_shoulder_camera_near"]) + right_depth * float(demo_obs.misc["right_shoulder_camera_far"] - demo_obs.misc["right_shoulder_camera_near"])
    wrist_depth_m = float(demo_obs.misc["wrist_camera_near"]) + wrist_depth * float(demo_obs.misc["wrist_camera_far"] - demo_obs.misc["wrist_camera_near"])
    episode = {
        "rgb_static": read_rgb_from_zip(zip_path, f"{episode_root}front_rgb/{step}.png"),
        "rgb_left_shoulder": read_rgb_from_zip(zip_path, f"{episode_root}left_shoulder_rgb/{step}.png"),
        "rgb_right_shoulder": read_rgb_from_zip(zip_path, f"{episode_root}right_shoulder_rgb/{step}.png"),
        "rgb_gripper": read_rgb_from_zip(zip_path, f"{episode_root}wrist_rgb/{step}.png"),
        "depth_front": front_depth_m.astype(np.float32),
        "depth_left_shoulder": left_depth_m.astype(np.float32),
        "depth_right_shoulder": right_depth_m.astype(np.float32),
        "depth_wrist": wrist_depth_m.astype(np.float32),
        "pcd_front": pointcloud_from_depth_and_camera_params(front_depth_m, demo_obs.misc["front_camera_extrinsics"], demo_obs.misc["front_camera_intrinsics"]),
        "pcd_left_shoulder": pointcloud_from_depth_and_camera_params(left_depth_m, demo_obs.misc["left_shoulder_camera_extrinsics"], demo_obs.misc["left_shoulder_camera_intrinsics"]),
        "pcd_right_shoulder": pointcloud_from_depth_and_camera_params(right_depth_m, demo_obs.misc["right_shoulder_camera_extrinsics"], demo_obs.misc["right_shoulder_camera_intrinsics"]),
        "pcd_wrist": pointcloud_from_depth_and_camera_params(wrist_depth_m, demo_obs.misc["wrist_camera_extrinsics"], demo_obs.misc["wrist_camera_intrinsics"]),
        "front_camera_intrinsics": np.asarray(demo_obs.misc["front_camera_intrinsics"], dtype=np.float32),
        "front_camera_extrinsics": np.asarray(demo_obs.misc["front_camera_extrinsics"], dtype=np.float32),
        "left_shoulder_camera_intrinsics": np.asarray(demo_obs.misc["left_shoulder_camera_intrinsics"], dtype=np.float32),
        "left_shoulder_camera_extrinsics": np.asarray(demo_obs.misc["left_shoulder_camera_extrinsics"], dtype=np.float32),
        "right_shoulder_camera_intrinsics": np.asarray(demo_obs.misc["right_shoulder_camera_intrinsics"], dtype=np.float32),
        "right_shoulder_camera_extrinsics": np.asarray(demo_obs.misc["right_shoulder_camera_extrinsics"], dtype=np.float32),
        "wrist_camera_intrinsics": np.asarray(demo_obs.misc["wrist_camera_intrinsics"], dtype=np.float32),
        "wrist_camera_extrinsics": np.asarray(demo_obs.misc["wrist_camera_extrinsics"], dtype=np.float32),
        "robot_obs": build_robot_obs(demo_obs),
    }
    rgb = process_rgb(episode, cfg.datamodule.observation_space, transforms)
    depth = process_depth(episode, cfg.datamodule.observation_space, transforms)
    point_cloud = process_point_cloud(episode, cfg.datamodule.observation_space, transforms)
    camera = process_camera(episode, cfg.datamodule.observation_space)
    state = process_state(episode, cfg.datamodule.observation_space, transforms, cfg.datamodule.proprioception_dims)
    return {
        "rgb_obs": {k: v.unsqueeze(0).to(device) for k, v in rgb["rgb_obs"].items() if not k.endswith("_origin")},
        "depth_obs": {k: v.unsqueeze(0).to(device) for k, v in depth["depth_obs"].items()},
        "point_cloud_obs": {k: v.unsqueeze(0).to(device) for k, v in point_cloud["point_cloud_obs"].items()},
        "camera_obs": {k: v.unsqueeze(0).to(device) for k, v in camera["camera_obs"].items()},
        "robot_obs": state["robot_obs"].unsqueeze(0).to(device),
    }


def rlbench_obs_to_policy_batch(obs, cfg, transforms, device):
    episode = {
        "rgb_static": to_uint8_rgb(obs.front_rgb),
        "rgb_left_shoulder": to_uint8_rgb(obs.left_shoulder_rgb),
        "rgb_right_shoulder": to_uint8_rgb(obs.right_shoulder_rgb),
        "rgb_gripper": to_uint8_rgb(obs.wrist_rgb),
        "depth_front": np.asarray(obs.front_depth, dtype=np.float32),
        "depth_left_shoulder": np.asarray(obs.left_shoulder_depth, dtype=np.float32),
        "depth_right_shoulder": np.asarray(obs.right_shoulder_depth, dtype=np.float32),
        "depth_wrist": np.asarray(obs.wrist_depth, dtype=np.float32),
        "pcd_front": np.asarray(obs.front_point_cloud, dtype=np.float32),
        "pcd_left_shoulder": np.asarray(obs.left_shoulder_point_cloud, dtype=np.float32),
        "pcd_right_shoulder": np.asarray(obs.right_shoulder_point_cloud, dtype=np.float32),
        "pcd_wrist": np.asarray(obs.wrist_point_cloud, dtype=np.float32),
        "front_camera_intrinsics": np.asarray(obs.misc["front_camera_intrinsics"], dtype=np.float32),
        "front_camera_extrinsics": np.asarray(obs.misc["front_camera_extrinsics"], dtype=np.float32),
        "left_shoulder_camera_intrinsics": np.asarray(obs.misc["left_shoulder_camera_intrinsics"], dtype=np.float32),
        "left_shoulder_camera_extrinsics": np.asarray(obs.misc["left_shoulder_camera_extrinsics"], dtype=np.float32),
        "right_shoulder_camera_intrinsics": np.asarray(obs.misc["right_shoulder_camera_intrinsics"], dtype=np.float32),
        "right_shoulder_camera_extrinsics": np.asarray(obs.misc["right_shoulder_camera_extrinsics"], dtype=np.float32),
        "wrist_camera_intrinsics": np.asarray(obs.misc["wrist_camera_intrinsics"], dtype=np.float32),
        "wrist_camera_extrinsics": np.asarray(obs.misc["wrist_camera_extrinsics"], dtype=np.float32),
        "robot_obs": build_robot_obs(obs),
    }
    rgb = process_rgb(episode, cfg.datamodule.observation_space, transforms)
    depth = process_depth(episode, cfg.datamodule.observation_space, transforms)
    point_cloud = process_point_cloud(episode, cfg.datamodule.observation_space, transforms)
    camera = process_camera(episode, cfg.datamodule.observation_space)
    state = process_state(episode, cfg.datamodule.observation_space, transforms, cfg.datamodule.proprioception_dims)
    return {
        "rgb_obs": {k: v.unsqueeze(0).to(device) for k, v in rgb["rgb_obs"].items() if not k.endswith("_origin")},
        "depth_obs": {k: v.unsqueeze(0).to(device) for k, v in depth["depth_obs"].items()},
        "point_cloud_obs": {k: v.unsqueeze(0).to(device) for k, v in point_cloud["point_cloud_obs"].items()},
        "camera_obs": {k: v.unsqueeze(0).to(device) for k, v in camera["camera_obs"].items()},
        "robot_obs": state["robot_obs"].unsqueeze(0).to(device),
    }


def policy_action_to_rlbench(
    action,
    obs,
    max_pos: float,
    max_orn: float,
    invert_gripper: bool = False,
    force_close: bool = False,
):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action = np.clip(action, -1.0, 1.0)
    curr_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    curr_pos = curr_pose[:3]
    curr_euler = quat_to_euler_xyz(curr_pose[3:7])
    next_pos = curr_pos + action[:3] * max_pos
    next_euler = curr_euler + action[3:6] * max_orn
    next_quat = R.from_euler("xyz", next_euler, degrees=False).as_quat().astype(np.float32)
    gripper_score = -action[6] if invert_gripper else action[6]
    gripper_open = np.float32(0.0 if force_close else (1.0 if gripper_score > 0 else 0.0))
    return np.concatenate([next_pos.astype(np.float32), next_quat, [gripper_open]], axis=0)


def smooth_action(raw_action: np.ndarray, prev_action: np.ndarray | None, alpha: float) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if prev_action is None or alpha <= 0.0:
        return action
    prev = np.asarray(prev_action, dtype=np.float32).reshape(-1)
    return (1.0 - alpha) * action + alpha * prev


def clamp_position_to_workspace(task_env, target_pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(target_pos, dtype=np.float32).copy()
    try:
        boundary = task_env._scene._workspace_boundary
        mins = np.asarray(boundary.get_bounding_box()[:3], dtype=np.float32)
        maxs = np.asarray(boundary.get_bounding_box()[3:], dtype=np.float32)
        return np.clip(pos, mins, maxs)
    except Exception:
        return pos


def limit_waypoint_step(action: np.ndarray, obs, max_pos_step: float, max_orn_step: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if max_pos_step <= 0.0 and max_orn_step <= 0.0:
        return action

    curr_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    curr_pos = curr_pose[:3]
    target_pos = action[:3]
    if max_pos_step > 0.0:
        delta = target_pos - curr_pos
        dist = float(np.linalg.norm(delta))
        if np.isfinite(dist) and dist > max_pos_step:
            action[:3] = curr_pos + delta / max(dist, 1e-6) * max_pos_step

    if max_orn_step > 0.0:
        curr_quat = normalize_quaternion_xyzw(curr_pose[3:7])
        target_quat = normalize_quaternion_xyzw(action[3:7], fallback_xyzw=curr_quat)
        rel = R.from_quat(target_quat) * R.from_quat(curr_quat).inv()
        rotvec = rel.as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if np.isfinite(angle) and angle > max_orn_step:
            limited = R.from_rotvec(rotvec / max(angle, 1e-6) * max_orn_step) * R.from_quat(curr_quat)
            action[3:7] = limited.as_quat().astype(np.float32)

    return action


def load_policy(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args")
    model_folder = os.path.join(os.path.dirname(args.ckpt))
    exp_cfg_file = os.path.join(model_folder, "exp_cfg.yaml")
    mvt_cfg_file = os.path.join(model_folder, "mvt_cfg.yaml")

    if ckpt_args is not None:
        cfg = ckpt_args
    else:
        with initialize(config_path="../policy_conf", job_name="eval_rlbench_rollout"):
            cfg = compose(config_name="VPP_RLBench_train")
    exp_cfg_path = exp_cfg_file if os.path.isfile(exp_cfg_file) else None
    mvt_cfg_path = mvt_cfg_file if os.path.isfile(mvt_cfg_file) else None
    if exp_cfg_path is not None:
        try:
            cfg.merge_from_file(exp_cfg_path)
        except Exception:
            pass
    if mvt_cfg_path is not None:
        try:
            cfg.model.mvt_cfg_path = mvt_cfg_path
        except Exception:
            pass
    if args.root_data_dir:
        cfg.root_data_dir = args.root_data_dir
        cfg.datamodule.root_data_dir = args.root_data_dir
    if getattr(args, "video_model_path", ""):
        cfg.model.pretrained_model_path = args.video_model_path
    if getattr(args, "text_encoder_path", ""):
        cfg.model.text_encoder_path = args.text_encoder_path
    if getattr(args, "t5_model_path", ""):
        cfg.model.t5_model_path = args.t5_model_path
    if getattr(args, "language_goal_path", ""):
        cfg.model.language_goal_path = args.language_goal_path
    if args.action_head_type:
        cfg.model.action_head_type = args.action_head_type
    if args.policy_output in {"waypoint_quat", "native_quat_collision"}:
        # Waypoint checkpoints predict absolute pose + quat + gripper.
        # Keep the old delta path at 7 dims; use 8/9 dims for native RLBench eval.
        cfg.model.action_dim = 9 if args.policy_output == "native_quat_collision" else 8
    elif hasattr(cfg.model, "action_dim") and cfg.model.action_dim is None:
        cfg.model.action_dim = 7

    cfg.batch_size = 1
    cfg.num_workers = 0
    transforms = build_eval_transforms(cfg)

    model = hydra.utils.instantiate(cfg.model)
    state = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    state, skipped = filter_state_dict_by_shape(model, state)
    if skipped:
        print(f"[load] skipped {len(skipped)} incompatible checkpoint keys")
        for key, reason in skipped[:10]:
            print(f"[load] skip key={key} reason={reason}")
    load_result = model.load_state_dict(state, strict=False)
    missing_keys = set(getattr(load_result, "missing_keys", []))
    if any(k.startswith("waypoint_head.stage_head") for k in missing_keys):
        model.use_stage_gate = False
    model.to(device)
    model.process_device()
    model.eval()
    return model, cfg, transforms, device


def infer_policy_output(args, cfg) -> str:
    if args.policy_output and args.policy_output != "auto":
        return args.policy_output
    model_target = str(cfg.model.get("_target_", ""))
    action_head_type = str(cfg.model.get("action_head_type", ""))
    action_dim = int(cfg.model.get("action_dim", 7))
    if (
        "RLBenchNativePolicy" in model_target
        or "RLBenchBridgePolicy" in model_target
        or action_head_type == "rlbench_native"
        or action_head_type == "rlbench_bridge"
        or action_dim == 9
    ):
        return "native_quat_collision"
    if (
        "RLBenchWaypointPolicy" in model_target
        or action_head_type == "rlbench_waypoint"
        or action_dim == 8
    ):
        return "waypoint_quat"
    return "delta"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tasks", nargs="+", default=["close_jar", "open_drawer", "push_buttons"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--action-mode", choices=["ik", "planning"], default="ik")
    parser.add_argument("--max-pos", type=float, default=0.05)
    parser.add_argument("--max-orn", type=float, default=0.50)
    parser.add_argument("--debug-steps", type=int, default=0)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--force-close-after", type=int, default=-1)
    parser.add_argument("--force-open-until", type=int, default=-1)
    parser.add_argument("--exec-horizon", type=int, default=10)
    parser.add_argument("--reset-mode", choices=["live", "demo"], default="live")
    parser.add_argument(
        "--demo-zip-dir",
        default="/mnt/data/manipulation/datasets/rlbench-18-tasks-30gb/data/train",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--root-data-dir", default="")
    parser.add_argument("--language-source", choices=["task", "rlbench"], default="task")
    parser.add_argument("--compare-demo-actions", action="store_true")
    parser.add_argument("--offline-demo-compare-only", action="store_true")
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument("--clamp-workspace", action="store_true")
    parser.add_argument(
        "--action-head-type",
        choices=["diffusion", "discrete", "rlbench_waypoint", "rlbench_native", "rlbench_bridge"],
        default="",
    )
    parser.add_argument("--policy-output", choices=["auto", "delta", "waypoint_quat", "native_quat_collision"], default="auto")
    parser.add_argument("--waypoint-max-pos-step", type=float, default=0.0)
    parser.add_argument("--waypoint-max-orn-step", type=float, default=0.0)
    parser.add_argument("--keep-current-rotation", action="store_true")
    parser.add_argument("--offline-compare-stride", type=int, default=5)
    parser.add_argument("--save-video-dir", default="")
    parser.add_argument("--save-video-fps", type=int, default=8)
    args = parser.parse_args()

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] tasks={args.tasks} episodes={args.episodes} max_steps={args.max_steps}")
    model, cfg, transforms, device = load_policy(args)
    args.policy_output = infer_policy_output(args, cfg)
    model.multistep = max(1, int(args.exec_horizon))
    print(f"[eval] policy_output={args.policy_output}")
    print(f"[eval] policy loaded on {device}")

    results = {}
    if args.offline_demo_compare_only:
        for task_name in args.tasks:
            successes = []
            for ep in range(args.episodes):
                if args.reset_mode != "demo":
                    raise ValueError("--offline-demo-compare-only requires --reset-mode demo")
                demo, demo_member = load_demo_from_zip(args.demo_zip_dir, task_name, ep)
                demo_langs = load_demo_language_from_zip(args.demo_zip_dir, task_name, demo_member)
                if args.language_source == "rlbench" and demo_langs:
                    lang = demo_langs[0]
                else:
                    lang = task_name.replace("_", " ")
                model.rollout_step_counter = 0
                for step in range(min(args.max_steps, len(demo) - 1)):
                    batch = demo_step_to_policy_batch(
                        args.demo_zip_dir,
                        task_name,
                        demo_member,
                        demo[step],
                        step,
                        cfg,
                        transforms,
                        device,
                    )
                    with torch.no_grad():
                        action = model.step(batch, {"lang_text": lang})
                    pred_action = np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                    if args.policy_output in {"waypoint_quat", "native_quat_collision"}:
                        target_step = min(step + max(1, args.offline_compare_stride), len(demo) - 1)
                        demo_action = (
                            build_abs_action_rlbench(demo[target_step])
                            if args.policy_output == "native_quat_collision"
                            else build_abs_action_quat(demo[target_step])
                        )
                        pred_compare = pred_action[: demo_action.shape[0]].copy()
                        pred_compare[3:7] = pred_compare[3:7] / np.linalg.norm(pred_compare[3:7]).clip(min=1e-6)
                        if float(np.dot(pred_compare[3:7], demo_action[3:7])) < 0.0:
                            pred_compare[3:7] *= -1.0
                        if args.invert_gripper:
                            pred_compare[7] *= -1.0
                    else:
                        target_step = step + 1
                        demo_action = build_rel_action(demo[step], demo[target_step], args.max_pos, args.max_orn)
                        pred_compare = np.clip(pred_action, -1.0, 1.0)
                    delta = pred_compare - demo_action
                    print(
                        f"[offline-compare] task={task_name} ep={ep} step={step} "
                        f"target_step={target_step} "
                        f"pred={np.array2string(pred_compare, precision=3, suppress_small=True)} "
                        f"demo={np.array2string(demo_action, precision=3, suppress_small=True)} "
                        f"delta={np.array2string(delta, precision=3, suppress_small=True)} "
                        f"l2={float(np.linalg.norm(delta)):.4f}"
                    )
                successes.append(False)
                print(
                    f"[episode] task={task_name} ep={ep} success=0 invalid_actions=0 "
                    f"lang={lang!r} demo={demo_member!r} offline_compare_only=1"
                )
            rate = float(np.mean(successes)) if successes else 0.0
            results[task_name] = rate
            print(f"[result] task={task_name} success_rate={rate:.3f} ({sum(successes)}/{len(successes)})")
        print("[summary]")
        for task_name, rate in results.items():
            print(f"{task_name}: {rate:.3f}")
        return

    from rlbench.environment import Environment
    from rlbench.backend.exceptions import InvalidActionError

    env = Environment(
        action_mode=build_action_mode(args.action_mode),
        obs_config=build_obs_config(args.image_size),
        headless=args.headless,
    )
    env.launch()
    try:
        for task_name in args.tasks:
            task_cls = import_task(task_name)
            task_env = env.get_task(task_cls)
            successes = []
            for ep in range(args.episodes):
                demo_member = ""
                demo_langs = []
                video_frame_dir = None
                video_output_path = None
                if args.reset_mode == "demo":
                    demo, demo_member = load_demo_from_zip(args.demo_zip_dir, task_name, ep)
                    demo_langs = load_demo_language_from_zip(args.demo_zip_dir, task_name, demo_member)
                    descriptions, obs = task_env.reset_to_demo(demo)
                else:
                    demo = None
                    descriptions, obs = task_env.reset()
                if args.save_video_dir:
                    safe_lang = ""
                    video_frame_dir = Path(args.save_video_dir) / task_name / f"episode_{ep}" / "frames"
                    video_output_path = Path(args.save_video_dir) / task_name / f"episode_{ep}.mp4"
                    save_episode_frame(video_frame_dir, 0, obs)
                if args.language_source == "rlbench":
                    if demo_langs:
                        lang = demo_langs[0]
                    elif descriptions:
                        lang = descriptions[0]
                    else:
                        lang = task_name.replace("_", " ")
                else:
                    lang = task_name.replace("_", " ")
                model.rollout_step_counter = 0
                success = False
                invalid_actions = 0
                prev_action = None
                for step in range(args.max_steps):
                    batch = rlbench_obs_to_policy_batch(obs, cfg, transforms, device)
                    with torch.no_grad():
                        action = model.step(batch, {"lang_text": lang})
                    raw_action = action.detach().cpu().numpy()
                    raw_action = smooth_action(raw_action, prev_action, args.action_smoothing)
                    prev_action = raw_action
                    if args.compare_demo_actions and demo is not None and step < len(demo) - 1:
                        pred_action = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(-1), -1.0, 1.0)
                        if args.policy_output in {"waypoint_quat", "native_quat_collision"}:
                            target_step = step + 1
                            demo_action = (
                                build_abs_action_rlbench(demo[target_step])
                                if args.policy_output == "native_quat_collision"
                                else build_abs_action_quat(demo[target_step])
                            )
                            pred_action = pred_action[: demo_action.shape[0]].copy()
                            pred_action[3:7] = pred_action[3:7] / np.linalg.norm(pred_action[3:7]).clip(min=1e-6)
                            if float(np.dot(pred_action[3:7], demo_action[3:7])) < 0.0:
                                pred_action[3:7] *= -1.0
                            if args.invert_gripper:
                                pred_action[7] *= -1.0
                        else:
                            demo_action = build_rel_action(demo[step], demo[step + 1], args.max_pos, args.max_orn)
                        delta = pred_action - demo_action
                        print(
                            f"[compare] task={task_name} ep={ep} step={step} "
                            f"pred={np.array2string(pred_action, precision=3, suppress_small=True)} "
                            f"demo={np.array2string(demo_action, precision=3, suppress_small=True)} "
                            f"delta={np.array2string(delta, precision=3, suppress_small=True)} "
                            f"l2={float(np.linalg.norm(delta)):.4f}"
                        )
                    force_close = args.force_close_after >= 0 and step >= args.force_close_after
                    force_open = args.force_open_until >= 0 and step < args.force_open_until
                    if args.policy_output in {"waypoint_quat", "native_quat_collision"}:
                        waypoint = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                        if args.keep_current_rotation:
                            quat = np.asarray(obs.gripper_pose[3:7], dtype=np.float32).copy()
                            quat = quat / np.linalg.norm(quat).clip(min=1e-6)
                        else:
                            quat = waypoint[3:7]
                            quat = quat / np.linalg.norm(quat).clip(min=1e-6)
                        gripper_score = -waypoint[7] if args.invert_gripper else waypoint[7]
                        gripper_open = np.float32(
                            1.0 if force_open else (0.0 if force_close else (1.0 if gripper_score > 0 else 0.0))
                        )
                        if args.policy_output == "native_quat_collision":
                            ignore_collision = np.float32(1.0 if waypoint.shape[0] > 8 and waypoint[8] > 0.5 else 0.0)
                            rlbench_action = np.concatenate([waypoint[:3], quat, [gripper_open, ignore_collision]]).astype(np.float32)
                        else:
                            rlbench_action = np.concatenate([waypoint[:3], quat, [gripper_open]]).astype(np.float32)
                        rlbench_action = limit_waypoint_step(
                            rlbench_action,
                            obs,
                            args.waypoint_max_pos_step,
                            args.waypoint_max_orn_step,
                        )
                    else:
                        rlbench_action = policy_action_to_rlbench(
                            raw_action,
                            obs,
                            args.max_pos,
                            args.max_orn,
                            invert_gripper=args.invert_gripper,
                            force_close=force_close,
                        )
                        if force_open:
                            rlbench_action[-1] = 1.0
                    if args.clamp_workspace:
                        rlbench_action[:3] = clamp_position_to_workspace(task_env, rlbench_action[:3])
                    if step < args.debug_steps:
                        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                        clipped = np.clip(raw, -1.0, 1.0)
                        pose = np.asarray(obs.gripper_pose, dtype=np.float32)
                        print(
                            f"[debug] task={task_name} ep={ep} step={step} "
                            f"raw={np.array2string(raw, precision=3, suppress_small=True)} "
                            f"clip={np.array2string(clipped, precision=3, suppress_small=True)} "
                            f"pos={np.array2string(pose[:3], precision=3, suppress_small=True)} "
                            f"target_pos={np.array2string(rlbench_action[:3], precision=3, suppress_small=True)} "
                            f"grip={rlbench_action[-1]:.0f}"
                        )
                    try:
                        obs, reward, terminate = task_env.step(rlbench_action)
                        if video_frame_dir is not None:
                            save_episode_frame(video_frame_dir, step + 1, obs)
                    except InvalidActionError as exc:
                        invalid_actions += 1
                        if invalid_actions <= 5:
                            print(f"[warn] {task_name} ep={ep} step={step} invalid action: {exc}")
                        terminate = False
                        reward = 0.0
                    success = bool(reward > 0.0)
                    if success or terminate:
                        break
                successes.append(success)
                if video_frame_dir is not None and video_output_path is not None:
                    render_episode_video(video_frame_dir, video_output_path, args.save_video_fps)
                print(
                    f"[episode] task={task_name} ep={ep} success={int(success)} "
                    f"invalid_actions={invalid_actions} lang={lang!r} demo={demo_member!r}"
                )
            rate = float(np.mean(successes)) if successes else 0.0
            results[task_name] = rate
            print(f"[result] task={task_name} success_rate={rate:.3f} ({sum(successes)}/{len(successes)})")
    finally:
        env.shutdown()

    print("[summary]")
    for task_name, rate in results.items():
        print(f"{task_name}: {rate:.3f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
