import argparse
import io
import pickle
import sys
import types
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


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


def quat_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32)


def angle_wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def build_robot_obs(obs) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    pos = pose[:3]
    euler = quat_to_euler_xyz(pose[3:7])
    gripper = np.float32(obs.gripper_open)
    return np.concatenate([pos, euler, [gripper, 0.0]], axis=0).astype(np.float32)


def build_rel_action(curr_obs, next_obs, max_pos: float, max_orn: float) -> np.ndarray:
    curr_pose = np.asarray(curr_obs.gripper_pose, dtype=np.float32)
    next_pose = np.asarray(next_obs.gripper_pose, dtype=np.float32)
    curr_pos, next_pos = curr_pose[:3], next_pose[:3]
    curr_euler = quat_to_euler_xyz(curr_pose[3:7])
    next_euler = quat_to_euler_xyz(next_pose[3:7])
    rel_pos = np.clip(next_pos - curr_pos, -max_pos, max_pos) / max_pos
    rel_orn = np.clip(angle_wrap(next_euler - curr_euler), -max_orn, max_orn) / max_orn
    gripper = np.array([1.0 if float(next_obs.gripper_open) > 0.5 else -1.0], dtype=np.float32)
    return np.concatenate([rel_pos, rel_orn, gripper], axis=0).astype(np.float32)


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


def save_npz_frame(out_dir: Path, frame_idx: int, rgb_static: np.ndarray, rgb_gripper: np.ndarray,
                   robot_obs: np.ndarray, rel_action: np.ndarray):
    np.savez(
        out_dir / f"episode_{frame_idx:07d}.npz",
        rgb_static=rgb_static,
        rgb_gripper=rgb_gripper,
        robot_obs=robot_obs,
        scene_obs=np.zeros((1,), dtype=np.float32),
        rel_actions=rel_action,
    )


def convert_zip(zip_path: Path, out_dir: Path, max_episodes: int | None, max_pos: float, max_orn: float):
    lang_ann = []
    lang_task = []
    lang_emb = []
    lang_indx = []
    global_frame_idx = 0
    task_name = zip_path.stem

    with zipfile.ZipFile(zip_path) as zf:
        for episode_idx, episode_root in enumerate(iter_episode_roots(zf)):
            if max_episodes is not None and episode_idx >= max_episodes:
                break

            obs_seq = extract_obs_seq(read_pickle_from_zip(zf, f"{episode_root}low_dim_obs.pkl"))
            if len(obs_seq) < 2:
                continue

            language = find_variation_description(zf, episode_root) or task_name.replace("_", " ")
            start_idx = global_frame_idx

            for step_idx, curr_obs in enumerate(obs_seq):
                next_obs = obs_seq[min(step_idx + 1, len(obs_seq) - 1)]
                rgb_static = read_rgb_from_zip(zf, f"{episode_root}front_rgb/{step_idx}.png")
                rgb_gripper = read_rgb_from_zip(zf, f"{episode_root}wrist_rgb/{step_idx}.png")
                robot_obs = build_robot_obs(curr_obs)
                rel_action = build_rel_action(curr_obs, next_obs, max_pos=max_pos, max_orn=max_orn)
                save_npz_frame(out_dir, global_frame_idx, rgb_static, rgb_gripper, robot_obs, rel_action)
                global_frame_idx += 1

            end_idx = global_frame_idx - 1
            lang_ann.append(language)
            lang_task.append(task_name)
            lang_emb.append(np.zeros((1, 1024), dtype=np.float32))
            lang_indx.append((start_idx, end_idx))

    lang_dir = out_dir / "lang_clip_resnet50"
    lang_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        lang_dir / "auto_lang_ann.npy",
        {
            "info": {"episodes": [], "indx": lang_indx},
            "language": {"ann": lang_ann, "task": lang_task, "emb": np.asarray(lang_emb, dtype=np.float32)},
        },
        allow_pickle=True,
    )
    np.save(out_dir / "ep_start_end_ids.npy", np.asarray(lang_indx, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description="Convert PerAct-style RLBench task zip to DeFi Stage3 dataset format.")
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-pos", type=float, default=0.05)
    parser.add_argument("--max-orn", type=float, default=0.50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    convert_zip(
        zip_path=args.zip_path,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
        max_pos=args.max_pos,
        max_orn=args.max_orn,
    )


if __name__ == "__main__":
    main()
