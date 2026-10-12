from __future__ import annotations

import argparse
import json
import sys
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
    _apply_checkpoint_model_compat_overrides,
)


def save_npz(path: Path, payload: Dict[str, Any]) -> None:
    ready = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            ready[key] = value.detach().cpu().numpy()
        else:
            ready[key] = np.asarray(value)
    np.savez_compressed(path, **ready)


def pad_action_chunk(actions: List[np.ndarray], chunk_len: int, action_dim: int = 7) -> np.ndarray:
    out = np.zeros((int(chunk_len), int(action_dim)), dtype=np.float32)
    if not actions:
        return out
    arr = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
    take = min(int(chunk_len), int(arr.shape[0]))
    out[:take, : min(int(action_dim), int(arr.shape[1]))] = arr[:take, : min(int(action_dim), int(arr.shape[1]))]
    return out


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


def rollout_collect_action_chunk(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    start_info: Dict[str, Any],
    chunk_len: int,
    lang_text: str,
    debug_prefix: str = "",
) -> Tuple[bool, int, Dict[str, np.ndarray], List[np.ndarray], List[Dict[str, np.ndarray]], List[np.ndarray]]:
    model.reset()
    model.future_feature_mode = "gfdm"
    model.override_future_feature = None
    model.override_latent_motion_tokens_up = None
    if debug_prefix:
        print(f"{debug_prefix} after model.reset", flush=True)
    obs = env.get_obs()
    if debug_prefix:
        print(f"{debug_prefix} after env.get_obs", flush=True)
    compact_trace: List[Dict[str, np.ndarray]] = [compact_obs(raw_env_obs(env))]
    summary_trace: List[np.ndarray] = [defi_future_feature(model, obs, lang_text).detach().float().cpu().numpy()]
    actions: List[np.ndarray] = []
    for step in range(cfg.ep_len):
        if debug_prefix and step < 2:
            print(f"{debug_prefix} before model.step step={step}", flush=True)
        action = model.step(obs, goal)
        if debug_prefix and step < 2:
            print(f"{debug_prefix} after model.step step={step}", flush=True)
        if len(actions) < int(chunk_len):
            actions.append(np.asarray(action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1).copy())
        if debug_prefix and step < 2:
            print(f"{debug_prefix} before env.step step={step}", flush=True)
        obs, _, _, current_info = env.step(action)
        if debug_prefix and step < 2:
            print(f"{debug_prefix} after env.step step={step}", flush=True)
        if len(actions) <= int(chunk_len):
            compact_trace.append(compact_obs(raw_env_obs(env)))
            summary_trace.append(defi_future_feature(model, obs, lang_text).detach().float().cpu().numpy())
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            final_raw = raw_env_obs(env)
            if debug_prefix:
                print(f"{debug_prefix} success after raw_env_obs(end)", flush=True)
            return True, step + 1, final_raw, actions, compact_trace, summary_trace
    final_raw = raw_env_obs(env)
    if debug_prefix:
        print(f"{debug_prefix} fail after raw_env_obs(end)", flush=True)
    return False, cfg.ep_len, final_raw, actions, compact_trace, summary_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real rollout action chunks from baseline DeFi policy.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=110)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--eval-sequences-path", type=Path, default=None)
    parser.add_argument("--eval-sequence-len", type=int, default=None)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-tasks", type=str, default="")
    parser.add_argument("--phi-output-dir", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.output_dir / "action_chunks"
    features_dir.mkdir(parents=True, exist_ok=True)
    phi_output_dir = args.phi_output_dir or (args.output_dir / "phi_proxy")
    phi_output_dir.mkdir(parents=True, exist_ok=True)
    phi_features_dir = phi_output_dir / "features"
    phi_features_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../policy_conf", job_name="collect_baseline_action_chunks"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
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

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
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
    model = load_model(cfg, args.checkpoint, device)
    task_oracle = hydra.utils.instantiate(cfg.tasks)

    requested_num_sequences = args.num_sequences + args.sequence_offset
    if args.eval_sequences_path is not None and args.focus_tasks.strip():
        generated_sequences = load_eval_sequences(None, args.eval_sequences_path, args.eval_sequence_len)
    else:
        generated_sequences = load_eval_sequences(requested_num_sequences, args.eval_sequences_path, args.eval_sequence_len)
    if generated_sequences is None:
        generated_sequences = get_sequences(requested_num_sequences)
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
    phi_row_id = 0
    rows_path = args.output_dir / "action_chunk_rows.jsonl"
    phi_rows_path = phi_output_dir / "phi_rows.jsonl"

    with rows_path.open("w", encoding="utf-8") as row_handle, phi_rows_path.open("w", encoding="utf-8") as phi_row_handle:
        for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="collect_baseline_action_chunks")):
            if seq_idx == 0:
                print(f"[COLLECT] seq={seq_idx} before get_env_state", flush=True)
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            if seq_idx == 0:
                print(f"[COLLECT] seq={seq_idx} before env.reset", flush=True)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            if seq_idx == 0:
                print(f"[COLLECT] seq={seq_idx} after env.reset", flush=True)
            success_counter = 0
            subtasks = []
            for sub_idx, subtask in enumerate(eval_sequence):
                debug_prefix = f"[COLLECT] seq={seq_idx} sub={sub_idx} task={subtask}"
                if seq_idx == 0 and sub_idx == 0:
                    print(f"{debug_prefix} before get_lang_goal", flush=True)
                lang_text = cfg.annotations[subtask][0]
                goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
                goal["lang_text"] = lang_text
                if seq_idx == 0 and sub_idx == 0:
                    print(f"{debug_prefix} after get_lang_goal", flush=True)
                start_info = env.get_info()
                if seq_idx == 0 and sub_idx == 0:
                    print(f"{debug_prefix} after env.get_info", flush=True)
                raw_start = raw_env_obs(env)
                if seq_idx == 0 and sub_idx == 0:
                    print(f"{debug_prefix} after raw_env_obs(start)", flush=True)
                compact_start = compact_obs(raw_start)
                if seq_idx == 0 and sub_idx == 0:
                    print(f"{debug_prefix} after compact_obs(start)", flush=True)

                success, steps, raw_end, actions, compact_trace, summary_trace = rollout_collect_action_chunk(
                    env,
                    model,
                    task_oracle,
                    cfg,
                    subtask,
                    goal,
                    start_info,
                    int(args.chunk_len),
                    lang_text,
                    debug_prefix=debug_prefix if seq_idx == 0 and sub_idx == 0 else "",
                )

                compact_end = compact_obs(raw_end)
                feature_rel = Path("action_chunks") / f"row_{row_id:07d}.npz"
                save_npz(
                    args.output_dir / feature_rel,
                    {
                        "action_chunk": pad_action_chunk(actions, int(args.chunk_len)),
                        "rgb_static_start": compact_start["rgb_static"],
                        "rgb_gripper_start": compact_start["rgb_gripper"],
                        "robot_obs_start": compact_start["robot_obs"],
                        "scene_obs_start": compact_start["scene_obs"],
                        "rgb_static_end": compact_end["rgb_static"],
                        "rgb_gripper_end": compact_end["rgb_gripper"],
                        "robot_obs_end": compact_end["robot_obs"],
                        "scene_obs_end": compact_end["scene_obs"],
                        "success": np.asarray([int(success)], dtype=np.int64),
                        "steps": np.asarray([int(steps)], dtype=np.int64),
                    },
                )

                row = {
                    "row_id": row_id,
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "lang_text": lang_text,
                    "next_task": eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None,
                    "success": bool(success),
                    "steps": int(steps),
                    "ep_len": int(cfg.ep_len),
                    "chunk_len": int(args.chunk_len),
                    "feature_path": str(feature_rel),
                }
                row_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_handle.flush()

                valid_steps = min(len(actions), max(len(compact_trace) - 1, 0), max(len(summary_trace) - 1, 0))
                if valid_steps > 0:
                    compact_end_chunk = compact_trace[valid_steps]
                    summary_end_chunk = np.asarray(summary_trace[valid_steps], dtype=np.float32)
                    for cut_idx in range(valid_steps):
                        compact_current = compact_trace[cut_idx]
                        summary_current = np.asarray(summary_trace[cut_idx], dtype=np.float32)
                        remain_actions = actions[cut_idx:valid_steps]
                        remain_path = Path("features") / f"row_{phi_row_id:07d}.npz"
                        save_npz(
                            phi_output_dir / remain_path,
                            {
                                "action_chunk": pad_action_chunk(remain_actions, int(args.chunk_len)),
                                "state_start": np.concatenate(
                                    [
                                        np.asarray(compact_current["robot_obs"], dtype=np.float32).reshape(-1),
                                        np.asarray(compact_current["scene_obs"], dtype=np.float32).reshape(-1),
                                    ],
                                    axis=0,
                                ),
                                "state_end": np.concatenate(
                                    [
                                        np.asarray(compact_end_chunk["robot_obs"], dtype=np.float32).reshape(-1),
                                        np.asarray(compact_end_chunk["scene_obs"], dtype=np.float32).reshape(-1),
                                    ],
                                    axis=0,
                                ),
                                "base_summary_exec": summary_current,
                                "target_summary_exec": summary_end_chunk,
                                "success": np.asarray([int(success)], dtype=np.int64),
                                "steps": np.asarray([int(valid_steps - cut_idx)], dtype=np.int64),
                                "cut_idx": np.asarray([int(cut_idx)], dtype=np.int64),
                                "full_chunk_len": np.asarray([int(valid_steps)], dtype=np.int64),
                            },
                        )
                        phi_row = {
                            "row_id": phi_row_id,
                            "sequence_index": seq_idx,
                            "subtask_index": sub_idx,
                            "task": subtask,
                            "lang_text": lang_text,
                            "success": bool(success),
                            "cut_idx": int(cut_idx),
                            "remaining_steps": int(valid_steps - cut_idx),
                            "full_chunk_len": int(valid_steps),
                            "feature_path": str(remain_path),
                        }
                        phi_row_handle.write(json.dumps(phi_row, ensure_ascii=False) + "\n")
                        phi_row_handle.flush()
                        phi_row_id += 1

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
        "phi_output_dir": str(phi_output_dir),
        "phi_row_count": int(phi_row_id),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "sequences.jsonl").write_text("\n".join(json.dumps(row) for row in logs) + ("\n" if logs else ""))
    (phi_output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
