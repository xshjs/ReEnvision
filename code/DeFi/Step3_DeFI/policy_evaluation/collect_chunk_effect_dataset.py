from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.oracle_hypothesis_rollout import (
    _apply_checkpoint_model_compat_overrides,
    compact_obs,
    count_success,
    defi_future_feature,
    ensure_train_folder,
    get_default_beso_and_env,
    get_env_state_for_initial_condition,
    goal_to_device,
    load_eval_sequences,
    load_model,
    raw_env_obs,
)


def _log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[CHUNK_EFFECT][{now}] {message}", flush=True)


def save_npz(path: Path, payload: Dict[str, Any]) -> None:
    ready = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            ready[key] = value.detach().cpu().numpy()
        else:
            ready[key] = np.asarray(value)
    np.savez_compressed(path, **ready)


def build_summary(results: Sequence[int], logs: Sequence[Dict[str, Any]], task_total: Counter, task_success: Counter) -> Dict[str, Any]:
    return {
        "num_sequences": len(results),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": {str(idx + 1): value for idx, value in enumerate(count_success(results))},
        "results": list(results),
        "task_info": {
            task: {"success": int(task_success[task]), "total": int(total)}
            for task, total in sorted(task_total.items())
        },
        "row_count": int(sum(len(item.get("subtasks", [])) for item in logs)),
    }


def pad_remaining_action(actions: List[np.ndarray], chunk_len: int, action_dim: int = 7) -> np.ndarray:
    out = np.zeros((int(chunk_len), int(action_dim)), dtype=np.float32)
    if not actions:
        return out
    arr = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
    take = min(int(chunk_len), int(arr.shape[0]))
    out[:take, : min(int(action_dim), int(arr.shape[1]))] = arr[:take, : min(int(action_dim), int(arr.shape[1]))]
    return out


def sample_cut_index(chunk_len: int, rng: random.Random) -> int:
    if chunk_len <= 1:
        return 0
    return int(rng.randint(0, chunk_len - 1))


def rollout_collect_chunk_effect(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    chunk_len: int,
    rng: random.Random,
) -> Tuple[bool, int, Dict[str, Any] | None]:
    model.reset()
    model.future_feature_mode = "gfdm"
    model.override_future_feature = None
    model.override_latent_motion_tokens_up = None

    obs = env.get_obs()
    start_info = env.get_info()
    planned_actions: List[np.ndarray] = []
    planned_tensors: List[torch.Tensor] = []
    cut_index = sample_cut_index(int(chunk_len), rng)

    step = 0
    while step < int(chunk_len):
        if step < 2:
            _log(f"rollout step={step} before model.step")
        action = model.step(obs, goal)
        if step < 2:
            _log(f"rollout step={step} after model.step")
        planned_tensors.append(action.detach().clone())
        planned_actions.append(np.asarray(action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1).copy())
        if step == cut_index:
            _log(f"rollout hit cut_index={cut_index} at step={step}")
            break
        if step < 2:
            _log(f"rollout step={step} before env.step")
        obs, _, _, current_info = env.step(action)
        if step < 2:
            _log(f"rollout step={step} after env.step")
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            final_raw = raw_env_obs(env)
            final_info = env.get_info()
            _log(f"rollout early success at step={step}")
            return True, step + 1, {
                "cut_index": int(step + 1),
                "remaining_action_chunk": np.zeros((chunk_len, 7), dtype=np.float32),
                "summary_t": defi_future_feature(model, obs, goal["lang_text"]).detach().cpu().numpy().astype(np.float32),
                "summary_end": defi_future_feature(model, obs, goal["lang_text"]).detach().cpu().numpy().astype(np.float32),
                "d_remain": np.zeros_like(defi_future_feature(model, obs, goal["lang_text"]).detach().cpu().numpy().astype(np.float32)),
                "state_t": compact_obs(final_raw),
                "state_end": compact_obs(final_raw),
                "success_remain": True,
                "remain_steps_executed": 0,
                "end_info": final_info,
            }
        step += 1

    raw_cut = raw_env_obs(env)
    compact_cut = compact_obs(raw_cut)
    _log("before summary_t")
    summary_t = defi_future_feature(model, obs, goal["lang_text"]).detach().cpu().numpy().astype(np.float32)
    _log("after summary_t")
    remain_tensors = planned_tensors[cut_index + 1 :]
    remain_actions = planned_actions[cut_index + 1 :]
    remain_start_info = env.get_info()

    remain_executed = 0
    success_remain = False
    current_info = remain_start_info
    for action in remain_tensors:
        if remain_executed < 2:
            _log(f"remain step={remain_executed} before env.step")
        obs, _, _, current_info = env.step(action)
        if remain_executed < 2:
            _log(f"remain step={remain_executed} after env.step")
        remain_executed += 1
        current_task_info = task_oracle.get_task_info_for_set(remain_start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            success_remain = True
            _log(f"remain success at remain_step={remain_executed}")
            break

    raw_end = raw_env_obs(env)
    compact_end = compact_obs(raw_end)
    _log("before summary_end")
    summary_end = defi_future_feature(model, obs, goal["lang_text"]).detach().cpu().numpy().astype(np.float32)
    _log("after summary_end")
    d_remain = (summary_end - summary_t).astype(np.float32)

    success_full = bool(success_remain)
    if not success_full:
        full_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        success_full = len(full_task_info) > 0
    return success_full, cut_index + 1 + remain_executed, {
        "cut_index": int(cut_index + 1),
        "remaining_action_chunk": pad_remaining_action(remain_actions, int(chunk_len)),
        "summary_t": summary_t,
        "summary_end": summary_end,
        "d_remain": d_remain,
        "state_t": compact_cut,
        "state_end": compact_end,
        "success_remain": bool(success_remain),
        "remain_steps_executed": int(remain_executed),
        "end_info": current_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect chunk-effect dataset for P_phi: (state_t, summary_t, A_remain) -> d_remain.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--eval-sequences-path", type=Path, default=None)
    parser.add_argument("--eval-sequence-len", type=int, default=None)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-tasks", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    rng = random.Random(args.seed)
    _log(f"start checkpoint={args.checkpoint} num_sequences={args.num_sequences} chunk_len={args.chunk_len}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.output_dir / "chunk_effect_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../policy_conf", job_name="collect_chunk_effect_dataset"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    _log("hydra compose done")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.train_folder = str(ensure_train_folder(args.output_dir, args.checkpoint))
    cfg.ep_len = args.ep_len
    cfg.num_sequences = args.num_sequences
    cfg.log_wandb = False
    with open_dict(cfg):
        cfg.future_feature_mode = "gfdm"
        cfg.future_frame_idx = 5
    _apply_checkpoint_model_compat_overrides(cfg, args.checkpoint)
    _log("checkpoint compat overrides done")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    _log(f"build env begin device={device}")
    env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder,
        cfg.root_data_dir,
        args.checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=device.index if device.type == "cuda" and device.index is not None else "cpu",
        cfg=cfg,
    )
    _log("build env done")
    _log("load model begin")
    model = load_model(cfg, args.checkpoint, device)
    _log("load model done")
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    _log("task oracle done")

    requested_num_sequences = args.num_sequences + args.sequence_offset
    if args.eval_sequences_path is not None and args.focus_tasks.strip():
        generated_sequences = load_eval_sequences(None, args.eval_sequences_path, args.eval_sequence_len)
    else:
        generated_sequences = load_eval_sequences(requested_num_sequences, args.eval_sequences_path, args.eval_sequence_len)
    if generated_sequences is None:
        generated_sequences = get_sequences(requested_num_sequences)
    _log(f"loaded sequences total={len(generated_sequences)}")
    focus_tasks = {task.strip() for task in args.focus_tasks.split(",") if task.strip()}
    total_sequences_before_focus = len(generated_sequences)
    if focus_tasks:
        generated_sequences = [
            (initial_state, seq)
            for initial_state, seq in generated_sequences
            if any(task in focus_tasks for task in seq)
        ]
    total_sequences_after_focus = len(generated_sequences)
    sequences = generated_sequences[args.sequence_offset : args.sequence_offset + args.num_sequences]

    results: List[int] = []
    logs: List[Dict[str, Any]] = []
    task_total = Counter()
    task_success = Counter()
    row_id = 0
    rows_path = args.output_dir / "chunk_effect_rows.jsonl"

    with rows_path.open("w", encoding="utf-8") as row_handle:
        for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="collect_chunk_effect_dataset")):
            if seq_idx == 0:
                _log("seq0 reset begin")
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            if seq_idx == 0:
                _log("seq0 reset done")
            success_counter = 0
            subtasks = []
            for sub_idx, subtask in enumerate(eval_sequence):
                if seq_idx == 0 and sub_idx == 0:
                    _log(f"seq0 sub0 get_lang_goal task={subtask}")
                lang_text = cfg.annotations[subtask][0]
                goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
                goal["lang_text"] = lang_text
                if seq_idx == 0 and sub_idx == 0:
                    _log("seq0 sub0 rollout begin")

                success, steps, payload = rollout_collect_chunk_effect(
                    env,
                    model,
                    task_oracle,
                    cfg,
                    subtask,
                    goal,
                    int(args.chunk_len),
                    rng,
                )
                if seq_idx == 0 and sub_idx == 0:
                    _log(f"seq0 sub0 rollout done success={success} steps={steps} payload_none={payload is None}")
                if payload is None:
                    break

                feature_rel = Path("chunk_effect_features") / f"row_{row_id:07d}.npz"
                save_npz(
                    args.output_dir / feature_rel,
                    {
                        "remaining_action_chunk": payload["remaining_action_chunk"],
                        "summary_t": payload["summary_t"],
                        "summary_end": payload["summary_end"],
                        "d_remain": payload["d_remain"],
                        "rgb_static_t": payload["state_t"]["rgb_static"],
                        "rgb_gripper_t": payload["state_t"]["rgb_gripper"],
                        "robot_obs_t": payload["state_t"]["robot_obs"],
                        "scene_obs_t": payload["state_t"]["scene_obs"],
                        "rgb_static_end": payload["state_end"]["rgb_static"],
                        "rgb_gripper_end": payload["state_end"]["rgb_gripper"],
                        "robot_obs_end": payload["state_end"]["robot_obs"],
                        "scene_obs_end": payload["state_end"]["scene_obs"],
                        "success_remain": np.asarray([int(payload["success_remain"])], dtype=np.int64),
                        "remain_steps_executed": np.asarray([int(payload["remain_steps_executed"])], dtype=np.int64),
                        "cut_index": np.asarray([int(payload["cut_index"])], dtype=np.int64),
                        "steps": np.asarray([int(steps)], dtype=np.int64),
                        "success_full": np.asarray([int(success)], dtype=np.int64),
                    },
                )

                row = {
                    "row_id": row_id,
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "lang_text": lang_text,
                    "success": bool(success),
                    "success_remain": bool(payload["success_remain"]),
                    "steps": int(steps),
                    "ep_len": int(cfg.ep_len),
                    "chunk_len": int(args.chunk_len),
                    "cut_index": int(payload["cut_index"]),
                    "remain_steps_executed": int(payload["remain_steps_executed"]),
                    "feature_path": str(feature_rel),
                }
                row_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_handle.flush()

                task_total[subtask] += 1
                task_success[subtask] += int(success)
                subtasks.append(row)
                row_id += 1
                if success:
                    success_counter += 1
                else:
                    break

            results.append(success_counter)
            logs.append(
                {
                    "sequence_index": seq_idx,
                    "eval_sequence": list(eval_sequence),
                    "success_counter": success_counter,
                    "subtasks": subtasks,
                }
            )

    summary = build_summary(results, logs, task_total, task_success)
    metadata = {
        "num_sequences": args.num_sequences,
        "sequence_offset": args.sequence_offset,
        "eval_sequences_path": str(args.eval_sequences_path) if args.eval_sequences_path is not None else None,
        "eval_sequence_len": args.eval_sequence_len,
        "focus_tasks": sorted(focus_tasks),
        "total_sequences_before_focus": total_sequences_before_focus,
        "total_sequences_after_focus": total_sequences_after_focus,
        "selected_sequences": len(sequences),
        "ep_len": args.ep_len,
        "chunk_len": int(args.chunk_len),
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "sequences.jsonl").write_text("\n".join(json.dumps(row) for row in logs) + ("\n" if logs else ""))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
