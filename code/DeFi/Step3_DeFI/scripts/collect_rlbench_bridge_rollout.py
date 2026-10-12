import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from eval_rlbench_rollout import (
    build_action_mode,
    build_obs_config,
    build_robot_obs,
    clamp_position_to_workspace,
    import_task,
    infer_policy_output,
    load_demo_from_zip,
    load_demo_language_from_zip,
    load_policy,
    limit_waypoint_step,
    policy_action_to_rlbench,
    render_episode_video,
    rlbench_obs_to_policy_batch,
    save_episode_frame,
)
from policy_evaluation.oracle_hypothesis_rollout import obs_current_feature
from policy_evaluation.label_defi_future_mismatch import infer_mismatch_type
from policy_evaluation.oracle_hypothesis_rollout import template_postexec_reflection_decision


def _save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            ready[key] = value.detach().cpu().numpy()
        else:
            ready[key] = np.asarray(value)
    np.savez_compressed(path, **ready)


def _normalize_quaternion_xyzw(quat_xyzw, fallback_xyzw=None):
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    if quat.shape[0] != 4:
        if fallback_xyzw is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return _normalize_quaternion_xyzw(fallback_xyzw, None)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        if fallback_xyzw is not None:
            return _normalize_quaternion_xyzw(fallback_xyzw, None)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat = quat / norm
    if quat[-1] < 0:
        quat = -quat
    return quat.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Bridge RLBench rollout rows and features.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tasks", nargs="+", default=["open_drawer"])
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--action-mode", choices=["ik", "planning"], default="planning")
    parser.add_argument("--max-pos", type=float, default=0.05)
    parser.add_argument("--max-orn", type=float, default=0.50)
    parser.add_argument("--reset-mode", choices=["live", "demo"], default="demo")
    parser.add_argument("--demo-zip-dir", default="/mnt/data/datasets/rlbench-18-tasks/data/train")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-data-dir", default="")
    parser.add_argument("--video-model-path", default="/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm")
    parser.add_argument("--text-encoder-path", default="/mnt/data/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32")
    parser.add_argument("--t5-model-path", default="/mnt/workspace/manipulation/DeFi/ckpts/t5_base")
    parser.add_argument("--language-goal-path", default="ViT-B/32")
    parser.add_argument("--language-source", choices=["task", "rlbench"], default="rlbench")
    parser.add_argument("--action-head-type", default="rlbench_bridge")
    parser.add_argument("--policy-output", default="native_quat_collision")
    parser.add_argument("--waypoint-max-pos-step", type=float, default=0.02)
    parser.add_argument("--waypoint-max-orn-step", type=float, default=0.0)
    parser.add_argument("--keep-current-rotation", action="store_true")
    clamp_group = parser.add_mutually_exclusive_group()
    clamp_group.add_argument(
        "--clamp-workspace",
        dest="clamp_workspace",
        action="store_true",
        help="Clamp RLBench waypoints to the task workspace before env.step().",
    )
    clamp_group.add_argument(
        "--no-clamp-workspace",
        dest="clamp_workspace",
        action="store_false",
        help="Disable workspace clamping.",
    )
    parser.set_defaults(clamp_workspace=True)
    parser.add_argument("--save-video-dir", default="")
    parser.add_argument("--save-video-fps", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model, cfg, transforms, device = load_policy(args)
    args.policy_output = infer_policy_output(args, cfg)
    model.multistep = 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "policy_output": args.policy_output,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rollout_rows.jsonl"

    from rlbench.environment import Environment
    from rlbench.backend.exceptions import InvalidActionError

    env = Environment(
        action_mode=build_action_mode(args.action_mode),
        obs_config=build_obs_config(args.image_size),
        headless=args.headless,
    )
    env.launch()
    row_id = 0
    try:
        with rows_path.open("w") as row_handle:
            for task_name in args.tasks:
                task_cls = import_task(task_name)
                task_env = env.get_task(task_cls)
                for ep in tqdm(range(args.episodes), desc=f"collect_{task_name}"):
                    if args.reset_mode == "demo":
                        demo, demo_member = load_demo_from_zip(args.demo_zip_dir, task_name, ep)
                        demo_langs = load_demo_language_from_zip(args.demo_zip_dir, task_name, demo_member)
                        descriptions, obs = task_env.reset_to_demo(demo)
                    else:
                        demo = None
                        demo_member = ""
                        demo_langs = []
                        descriptions, obs = task_env.reset()

                    if args.language_source == "rlbench":
                        if demo_langs:
                            lang = demo_langs[0]
                        elif descriptions:
                            lang = descriptions[0]
                        else:
                            lang = task_name.replace("_", " ")
                    else:
                        lang = task_name.replace("_", " ")

                    video_frame_dir = None
                    video_output_path = None
                    if args.save_video_dir:
                        video_frame_dir = Path(args.save_video_dir) / task_name / f"episode_{ep}" / "frames"
                        video_output_path = Path(args.save_video_dir) / task_name / f"episode_{ep}.mp4"
                        save_episode_frame(video_frame_dir, 0, obs)

                    current_feature = obs_current_feature(model, rlbench_obs_to_policy_batch(obs, cfg, transforms, device), lang)
                    success = False
                    invalid_actions = 0
                    actions = []
                    rewards = []
                    for step in range(args.max_steps):
                        batch = rlbench_obs_to_policy_batch(obs, cfg, transforms, device)
                        with torch.no_grad():
                            action = model.step(batch, {"lang_text": lang})
                        raw_action = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
                        if args.policy_output in {"waypoint_quat", "native_quat_collision"}:
                            waypoint = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                            if args.keep_current_rotation:
                                quat = _normalize_quaternion_xyzw(obs.gripper_pose[3:7])
                            else:
                                quat = _normalize_quaternion_xyzw(waypoint[3:7], fallback_xyzw=obs.gripper_pose[3:7])
                            gripper_open = np.float32(1.0 if waypoint.shape[0] > 7 and waypoint[7] > 0 else 0.0)
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
                            )
                        if args.clamp_workspace:
                            rlbench_action[:3] = clamp_position_to_workspace(task_env, rlbench_action[:3])
                        try:
                            obs, reward, terminate = task_env.step(rlbench_action)
                            if video_frame_dir is not None:
                                save_episode_frame(video_frame_dir, step + 1, obs)
                            actions.append(rlbench_action.tolist())
                            rewards.append(float(reward))
                        except InvalidActionError as exc:
                            invalid_actions += 1
                            if invalid_actions <= 5:
                                print(f"[warn] {task_name} ep={ep} step={step} invalid action: {exc}")
                            terminate = False
                            reward = 0.0
                        success = bool(reward > 0.0)
                        if success or terminate:
                            break

                    end_feature = obs_current_feature(
                        model,
                        rlbench_obs_to_policy_batch(obs, cfg, transforms, device),
                        lang,
                    )
                    mismatch_type = infer_mismatch_type(
                        task_name,
                        bool(success),
                        int(len(actions)),
                        int(args.max_steps),
                    )
                    decision = template_postexec_reflection_decision(
                        task_name,
                        bool(success),
                        int(len(actions)),
                        int(args.max_steps),
                        mismatch_type,
                    )
                    feature_path = Path("features") / f"row_{row_id:07d}.npz"
                    _save_npz(
                        output_dir / feature_path,
                        {
                            "obs_current_feature": current_feature.detach().cpu(),
                            "defi_future_feature": end_feature.detach().cpu(),
                            "end_obs_current_feature": end_feature.detach().cpu(),
                            "actions": np.asarray(actions, dtype=np.float32),
                            "rewards": np.asarray(rewards, dtype=np.float32),
                            "success": np.asarray([int(success)], dtype=np.int64),
                            "steps": np.asarray([int(len(actions))], dtype=np.int64),
                            "robot_obs": build_robot_obs(obs),
                        },
                    )
                    row = {
                        "row_id": row_id,
                        "sequence_index": ep,
                        "subtask_index": 0,
                        "task": task_name,
                        "lang_goal": lang,
                        "lang_text": lang,
                        "success": bool(success),
                        "steps": int(len(actions)),
                        "ep_len": int(args.max_steps),
                        "feature_path": str(feature_path),
                        "demo_member": demo_member,
                        "invalid_actions": int(invalid_actions),
                        "gate_label_hint": int(not success),
                        "mismatch_type": mismatch_type,
                        "correction_direction": decision.correction_direction,
                        "recoverability": decision.recoverability,
                        "retrieved_success_row_id": -1,
                        "retrieved_success_score": -1.0,
                        "reflection_text": decision.explanation,
                        "trust_score": float(decision.trust_score),
                    }
                    row_handle.write(json.dumps(row) + "\n")
                    row_handle.flush()
                    row_id += 1

                    if video_frame_dir is not None and video_output_path is not None:
                        render_episode_video(video_frame_dir, video_output_path, args.save_video_fps)

                    print(
                        f"[episode] task={task_name} ep={ep} success={int(success)} "
                        f"invalid_actions={invalid_actions} lang={lang!r} demo={demo_member!r}"
                    )
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
