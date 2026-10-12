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
    load_eval_sequences,
    get_default_beso_and_env,
    get_env_state_for_initial_condition,
    goal_to_device,
    load_model,
    obs_current_feature,
    raw_env_obs,
    rollout_once,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Frozen DeFi single-future rollout rows.")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--focus-tasks", type=str, default="")
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../policy_conf", job_name="collect_defi_single_future_rollout"):
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
        # For focused collection, load the whole split first, then filter, then slice by offset.
        # Otherwise later rounds can exhaust the partially loaded prefix and return near-zero rows.
        generated_sequences = load_eval_sequences(
            None,
            args.eval_sequences_path,
            args.eval_sequence_len,
        )
    else:
        generated_sequences = load_eval_sequences(
            requested_num_sequences,
            args.eval_sequences_path,
            args.eval_sequence_len,
        )
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
    rows_path = args.output_dir / "rollout_rows.jsonl"

    with rows_path.open("w") as row_handle:
        for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="collect_defi_single_future_rollout")):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success_counter = 0
            subtasks = []
            for sub_idx, subtask in enumerate(eval_sequence):
                lang_text = cfg.annotations[subtask][0]
                goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
                goal["lang_text"] = lang_text
                start_info = env.get_info()
                obs = env.get_obs()
                raw_start = raw_env_obs(env)
                current_feature = obs_current_feature(model, obs, lang_text)
                future_feature = defi_future_feature(model, obs, lang_text)
                compact_start = compact_obs(raw_start)

                success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")

                end_obs = env.get_obs()
                raw_end = raw_env_obs(env)
                end_feature = obs_current_feature(model, end_obs, lang_text)
                compact_end = compact_obs(raw_end)
                feature_rel = Path("features") / f"row_{row_id:07d}.npz"
                save_npz(
                    args.output_dir / feature_rel,
                    {
                        "obs_current_feature": current_feature,
                        "defi_future_feature": future_feature,
                        "end_obs_current_feature": end_feature,
                        "rgb_static_start": compact_start["rgb_static"],
                        "rgb_gripper_start": compact_start["rgb_gripper"],
                        "rgb_static_end": compact_end["rgb_static"],
                        "rgb_gripper_end": compact_end["rgb_gripper"],
                        "robot_obs_start": compact_start["robot_obs"],
                        "scene_obs_start": compact_start["scene_obs"],
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
                    "feature_path": str(feature_rel),
                }
                row_handle.write(json.dumps(row) + "\n")
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
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "sequences.jsonl").write_text("\n".join(json.dumps(row) for row in logs) + ("\n" if logs else ""))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
