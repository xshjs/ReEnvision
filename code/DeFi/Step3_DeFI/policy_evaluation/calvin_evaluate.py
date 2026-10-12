from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import torch.distributed as dist
from hydra import compose, initialize
import argparse
from omegaconf import OmegaConf, open_dict
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.oracle_hypothesis_rollout import compact_obs, defi_future_feature, raw_env_obs
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from policy_models.utils.utils import get_last_checkpoint, get_all_checkpoints
from policy_models.rollout.rollout_video import RolloutVideo


logger = logging.getLogger(__name__)


def pad_action_chunk(actions, chunk_len: int, action_dim: int = 7):
    out = np.zeros((int(chunk_len), int(action_dim)), dtype=np.float32)
    if not actions:
        return out
    arr = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
    take = min(int(chunk_len), int(arr.shape[0]))
    out[:take, : min(int(action_dim), int(arr.shape[1]))] = arr[:take, : min(int(action_dim), int(arr.shape[1]))]
    return out


def action_to_numpy(action):
    if isinstance(action, torch.Tensor):
        return action.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def build_phi_proxy_items(sequence_index, subtask_index, subtask, lang_annotation, success, traced_actions, compact_trace, future_trace, action_trace_steps):
    valid_steps = min(len(traced_actions), max(len(compact_trace) - 1, 0), max(len(future_trace) - 1, 0), int(action_trace_steps))
    if valid_steps <= 0:
        return []
    compact_end = compact_trace[valid_steps]
    future_end = np.asarray(future_trace[valid_steps], dtype=np.float32).reshape(-1)
    rows = []
    for cut_idx in range(valid_steps):
        compact_cur = compact_trace[cut_idx]
        future_cur = np.asarray(future_trace[cut_idx], dtype=np.float32).reshape(-1)
        state_start = np.concatenate(
            [
                np.asarray(compact_cur["robot_obs"], dtype=np.float32).reshape(-1),
                np.asarray(compact_cur["scene_obs"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        state_end = np.concatenate(
            [
                np.asarray(compact_end["robot_obs"], dtype=np.float32).reshape(-1),
                np.asarray(compact_end["scene_obs"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        rows.append(
            {
                "sequence_index": int(sequence_index),
                "subtask_index": int(subtask_index),
                "task": str(subtask),
                "lang_text": str(lang_annotation),
                "success": bool(success),
                "cut_idx": int(cut_idx),
                "remaining_steps": int(valid_steps - cut_idx),
                "full_chunk_len": int(valid_steps),
                "action_chunk": pad_action_chunk(traced_actions[cut_idx:valid_steps], int(action_trace_steps)),
                "state_start": state_start,
                "state_end": state_end,
                "base_summary_exec": future_cur,
                "target_summary_exec": future_end,
            }
        )
    return rows


def _apply_checkpoint_model_compat_overrides(cfg, checkpoint_path: str) -> None:
    state = torch.load(str(checkpoint_path), map_location="cpu")
    train_args = state.get("args")
    if train_args is None:
        return
    if not isinstance(train_args, dict):
        train_args = OmegaConf.to_container(train_args, resolve=True)
    if not isinstance(train_args, dict):
        return

    model_args = train_args.get("model", {}) or {}
    act_seq_len = train_args.get("act_seq_len", train_args.get("action_seq_len"))
    num_latents = train_args.get("num_latents", model_args.get("num_latents"))

    with open_dict(cfg):
        if act_seq_len is not None:
            act_seq_len = int(act_seq_len)
            cfg.act_seq_len = act_seq_len
            cfg.model.action_seq_len = act_seq_len
            cfg.model.act_window_size = act_seq_len
            cfg.model.multistep = act_seq_len
            cfg.multistep = act_seq_len
            print(f"[INFO] Applied checkpoint act_seq_len override: {act_seq_len}", flush=True)
        if num_latents is not None:
            cfg.model.num_latents = int(num_latents)
            print(f"[INFO] Applied checkpoint num_latents override: {int(num_latents)}", flush=True)


def load_eval_sequences(eval_sequences_path, num_sequences=None):
    with open(eval_sequences_path, "r") as handle:
        sequences = json.load(handle)
    if num_sequences is not None:
        sequences = sequences[:num_sequences]
    return sequences


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


def get_eval_horizon(cfg, sequences=None):
    if getattr(cfg, "eval_horizon", None) is not None:
        return int(cfg.eval_horizon)
    if sequences:
        return max(len(sequence) for _, sequence in sequences)
    return 5


def count_success(results, eval_horizon):
    count = Counter(results)
    step_success = []
    for i in range(1, eval_horizon + 1):
        n_success = sum(count[j] for j in reversed(range(i, eval_horizon + 1)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = cfg.eval_sequences if "eval_sequences" in cfg and cfg.eval_sequences is not None else get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    eval_horizon = get_eval_horizon(cfg, sequences)
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results, eval_horizon))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")


def evaluate_policy(model, env, lang_embeddings, cfg, 
                    num_procs, procs_id, checkpoint,
                    save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    rollout_video = RolloutVideo(
        logger=logger,
        empty_cache=False,
        log_to_file=True,
        save_dir=save_dir,
        resolution_scale=1,
    )

    eval_sequences = get_sequences(cfg.num_sequences)
    num_seq_per_procs = cfg.num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]
    progress_path = Path(save_dir) / "sequence_progress.jsonl" if save_dir is not None else None
    partial_results_path = Path(save_dir) / "partial_results.json" if save_dir is not None else None
    collect_action_chunks = bool(getattr(cfg, "collect_action_chunks", False))
    action_chunk_rows_path = Path(save_dir) / "action_chunk_rows.jsonl" if save_dir is not None and collect_action_chunks else None
    action_chunk_dir = Path(save_dir) / "action_chunks" if save_dir is not None and collect_action_chunks else None
    action_trace_steps = int(getattr(cfg, "action_chunk_steps", getattr(cfg, "multistep", 10)))
    action_row_id = 0
    collect_phi_proxy = bool(getattr(cfg, "collect_phi_proxy", False))
    phi_rows_path = Path(save_dir) / "phi_rows.jsonl" if save_dir is not None and collect_phi_proxy else None
    phi_dir = Path(save_dir) / "phi_proxy" if save_dir is not None and collect_phi_proxy else None
    phi_row_id = 0

    results = []

    def append_progress(row):
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    def write_partial_results(results):
        if partial_results_path is None:
            return
        eval_horizon = get_eval_horizon(cfg, eval_sequences)
        payload = {
            "checkpoint": str(checkpoint),
            "num_sequences_total": len(eval_sequences),
            "num_sequences_finished": len(results),
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": count_success(results, eval_horizon) if results else [],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with partial_results_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()

    def write_action_chunk(row):
        nonlocal action_row_id
        if action_chunk_rows_path is None or action_chunk_dir is None:
            return
        action_chunk_rows_path.parent.mkdir(parents=True, exist_ok=True)
        action_chunk_dir.mkdir(parents=True, exist_ok=True)
        row = dict(row)
        row["row_id"] = int(action_row_id)
        feature_rel = Path("action_chunks") / f"row_{action_row_id:07d}.npz"
        np.savez_compressed(
            Path(save_dir) / feature_rel,
            action_chunk=np.asarray(row.pop("action_chunk"), dtype=np.float32),
        )
        row["feature_path"] = str(feature_rel)
        with action_chunk_rows_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        action_row_id += 1

    def write_phi_items(items):
        nonlocal phi_row_id
        if phi_rows_path is None or phi_dir is None or not items:
            return
        phi_rows_path.parent.mkdir(parents=True, exist_ok=True)
        phi_dir.mkdir(parents=True, exist_ok=True)
        for row in items:
            row = dict(row)
            row["row_id"] = int(phi_row_id)
            feature_rel = Path("phi_proxy") / f"row_{phi_row_id:07d}.npz"
            np.savez_compressed(
                Path(save_dir) / feature_rel,
                action_chunk=np.asarray(row.pop("action_chunk"), dtype=np.float32),
                state_start=np.asarray(row.pop("state_start"), dtype=np.float32),
                state_end=np.asarray(row.pop("state_end"), dtype=np.float32),
                base_summary_exec=np.asarray(row.pop("base_summary_exec"), dtype=np.float32),
                target_summary_exec=np.asarray(row.pop("target_summary_exec"), dtype=np.float32),
                success=np.asarray([int(row["success"])], dtype=np.int64),
                steps=np.asarray([int(row["remaining_steps"])], dtype=np.int64),
                cut_idx=np.asarray([int(row["cut_idx"])], dtype=np.int64),
                full_chunk_len=np.asarray([int(row["full_chunk_len"])], dtype=np.int64),
            )
            row["feature_path"] = str(feature_rel)
            with phi_rows_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            phi_row_id += 1

    print(
        f"[BASELINE_EVAL] checkpoint={checkpoint} save_dir={save_dir} num_sequences={len(eval_sequences)} "
        f"num_procs={num_procs} proc_id={procs_id}",
        flush=True,
    )

    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seq_start_time = time.time()
        task_names = [str(task) for task in eval_sequence]
        print(f"[BASELINE_EVAL][START] seq={i} tasks={task_names}", flush=True)
        append_progress(
            {
                "event": "start",
                "sequence_index": int(i),
                "tasks": task_names,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        # TODO 这里控制是否存下来eval的视频
        record = False
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i,
            write_action_chunk, action_trace_steps, write_phi_items,
        )
        results.append(result)
        duration = time.time() - seq_start_time
        print(
            f"[BASELINE_EVAL][DONE] seq={i} success_len={result}/{len(eval_sequence)} duration_sec={duration:.2f}",
            flush=True,
        )
        append_progress(
            {
                "event": "done",
                "sequence_index": int(i),
                "tasks": task_names,
                "success_len": int(result),
                "sequence_len": int(len(eval_sequence)),
                "duration_sec": float(duration),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        write_partial_results(results)
        if record:
            rollout_video.write_to_tmp()

        eval_horizon = get_eval_horizon(cfg, eval_sequences)
        success_rates = count_success(results, eval_horizon)
        average_rate = sum(success_rates) / len(success_rates) * eval_horizon
        description = " ".join([f"{i + 1}/{eval_horizon} : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
        description += f" Average: {average_rate:.1f} |"
        eval_sequences.set_description(description)

        if record:
            rollout_video._log_currentvideos_to_file(i, save_as_video=True)

    results_dict = {}
    results_dict[checkpoint] = results
    print_and_save(results_dict, cfg, log_dir=save_dir)
    return results


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i,
    action_chunk_writer=None, action_trace_steps=10, phi_writer=None,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0

    for sub_idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()
        success = rollout(
            env,
            model,
            task_checker,
            cfg,
            subtask,
            lang_embeddings,
            val_annotations,
            record,
            rollout_video,
            sequence_index=int(i),
            subtask_index=int(sub_idx),
            action_chunk_writer=action_chunk_writer,
            action_trace_steps=int(action_trace_steps),
            phi_writer=phi_writer,
        )
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    model,
    task_oracle,
    cfg,
    subtask,
    lang_embeddings,
    val_annotations,
    record=False,
    rollout_video=None,
    sequence_index=0,
    subtask_index=0,
    action_chunk_writer=None,
    action_trace_steps=10,
    phi_writer=None,
):

    debug_rollout = bool(getattr(cfg, "debug_rollout", False))
    obs = env.get_obs()
    if debug_rollout:
        print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} after env.get_obs", flush=True)
    
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    if debug_rollout:
        print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} after get_lang_goal", flush=True)
    
    model.reset()
    if debug_rollout:
        print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} after model.reset", flush=True)
    start_info = env.get_info()
    if debug_rollout:
        print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} after env.get_info", flush=True)
    traced_actions = []
    compact_trace = []
    future_trace = []
    if phi_writer is not None:
        compact_trace.append(compact_obs(raw_env_obs(env)))
        future_trace.append(defi_future_feature(model, obs, lang_annotation).detach().float().cpu().numpy())

    for step in range(cfg.ep_len):  # 360
        if debug_rollout and step < 3:
            print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} before model.step step={step}", flush=True)
        action = model.step(obs, goal)  # torch.Size([7])
        if debug_rollout and step < 3:
            print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} after model.step step={step}", flush=True)
        if len(traced_actions) < int(action_trace_steps):
            traced_actions.append(action_to_numpy(action).copy())

        # 在env中执行action
        if debug_rollout and step < 3:
            print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} before env.step step={step}", flush=True)
        obs, _, _, current_info = env.step(action)
        if debug_rollout and step < 3:
            print(f"[BASELINE_DEBUG] seq={sequence_index} sub={subtask_index} task={subtask} after env.step step={step}", flush=True)
        if phi_writer is not None and len(traced_actions) <= int(action_trace_steps):
            compact_trace.append(compact_obs(raw_env_obs(env)))
            future_trace.append(defi_future_feature(model, obs, lang_annotation).detach().float().cpu().numpy())
        
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if phi_writer is not None:
                phi_writer(
                    build_phi_proxy_items(
                        sequence_index,
                        subtask_index,
                        subtask,
                        lang_annotation,
                        True,
                        traced_actions,
                        compact_trace,
                        future_trace,
                        action_trace_steps,
                    )
                )
            if action_chunk_writer is not None:
                action_chunk_writer(
                    {
                        "sequence_index": int(sequence_index),
                        "subtask_index": int(subtask_index),
                        "task": str(subtask),
                        "lang_text": str(lang_annotation),
                        "success": True,
                        "steps": int(step + 1),
                        "ep_len": int(cfg.ep_len),
                        "action_chunk_steps": int(action_trace_steps),
                        "action_chunk": pad_action_chunk(traced_actions, int(action_trace_steps)),
                    }
                )
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True  # 从这里返回就是rollout成功
    
    if phi_writer is not None:
        phi_writer(
            build_phi_proxy_items(
                sequence_index,
                subtask_index,
                subtask,
                lang_annotation,
                False,
                traced_actions,
                compact_trace,
                future_trace,
                action_trace_steps,
            )
        )
    if action_chunk_writer is not None:
        action_chunk_writer(
            {
                "sequence_index": int(sequence_index),
                "subtask_index": int(subtask_index),
                "task": str(subtask),
                "lang_text": str(lang_annotation),
                "success": False,
                "steps": int(cfg.ep_len),
                "ep_len": int(cfg.ep_len),
                "action_chunk_steps": int(action_trace_steps),
                "action_chunk": pad_action_chunk(traced_actions, int(action_trace_steps)),
            }
        )
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False  # 从这里返回就是rollout失败


def main(cfg):
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device

    log_wandb = cfg.log_wandb
    seed_everything(0, workers=True)
    checkpoints = get_all_checkpoints(Path(cfg.train_folder))
    lang_embeddings = None
    env = None
    results = {}

    print('train_folder',cfg.train_folder)
    print("\n" + "="*80)
    print(f"Found {len(checkpoints)} checkpoint(s) for evaluation:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  [{i}] {str(ckpt)}")
    print("="*80 + "\n")

    for checkpoint in checkpoints:
        _apply_checkpoint_model_compat_overrides(cfg, str(checkpoint))
        print(device)
        print(f"\n{'*'*80}")
        print(f"Processing checkpoint: {str(checkpoint)}")
        print(f"{'*'*80}")
        device_id = device.index if device.type == "cuda" and device.index is not None else 0
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=device_id,
            cfg=cfg,
        )

        print(f"Loading model from {str(checkpoint)}")
        state_dict = torch.load(str(checkpoint), map_location="cpu")
        print(f"✓ Successfully loaded checkpoint from: {str(checkpoint)}")

        model = hydra.utils.instantiate(cfg.model)

        model.load_state_dict(state_dict['model'],strict = False)
        model.freeze()

        print(cfg.num_sampling_steps, cfg.sampler_type, cfg.multistep, cfg.sigma_min, cfg.sigma_max, cfg.noise_scheduler)
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        model = acc.prepare(model, device_placement=[True])
        model.process_device()
        model.eval()
        if log_wandb:
            log_dir = get_log_dir(cfg.train_folder)
            os.makedirs(log_dir / "wandb", exist_ok=True)

            results[checkpoint] = evaluate_policy(
                model, env, lang_embeddings, cfg, 
                acc.num_processes, acc.process_index, checkpoint,
                save_dir=Path(log_dir))
            avg_reward = torch.tensor(results[checkpoint]).float().mean().to(device)
            acc.wait_for_everyone()
            avg_reward = acc.gather_for_metrics(avg_reward).mean()
            if acc.is_main_process:
                print('average success rate ', avg_reward)
        
        # 显式清理
        model.to("cpu")
        model.process_device()
        del model
        del env
        del lang_embeddings
        lang_embeddings = None
        env = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--t5_model_path", type=str, default="")
    parser.add_argument("--language_goal_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    parser.add_argument("--eval_sequences_path", type=str, default="")
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=None)
    parser.add_argument("--use_default_sequences", action="store_true")
    parser.add_argument("--default_sequences_path", type=str, default="")
    parser.add_argument("--default_sequence_len", type=int, default=None)
    parser.add_argument("--collect_action_chunks", action="store_true")
    parser.add_argument("--action_chunk_steps", type=int, default=None)
    parser.add_argument("--collect_phi_proxy", action="store_true")
    parser.add_argument("--debug_rollout", action="store_true")
    
    args = parser.parse_args()
    
    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
        print(f"[INFO] Overriding cfg.model.t5_model_path with args.t5_model_path: {cfg.model.t5_model_path}")
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
        print(f"[INFO] Overriding cfg.model.language_goal_path with args.language_goal_path: {cfg.model.language_goal_path}")
    cfg.root_data_dir = args.calvin_abc_dir
    with open_dict(cfg):
        if args.eval_horizon is not None:
            cfg.eval_horizon = int(args.eval_horizon)
        if args.num_sequences is not None:
            cfg.num_sequences = int(args.num_sequences)
        cfg.collect_action_chunks = bool(args.collect_action_chunks)
        cfg.collect_phi_proxy = bool(args.collect_phi_proxy)
        cfg.debug_rollout = bool(args.debug_rollout)
        if args.action_chunk_steps is not None:
            cfg.action_chunk_steps = int(args.action_chunk_steps)
    if args.default_sequences_path:
        os.environ["DEFI_CUSTOM_EVAL_SEQUENCE_PATH"] = args.default_sequences_path
    if args.default_sequence_len is not None:
        os.environ["DEFI_CUSTOM_EVAL_SEQUENCE_LEN"] = str(int(args.default_sequence_len))
    if args.eval_sequences_path:
        with open_dict(cfg):
            cfg.eval_sequences = load_eval_sequences(args.eval_sequences_path, num_sequences=args.num_sequences)
            cfg.num_sequences = len(cfg.eval_sequences)
            if args.eval_horizon is None and cfg.eval_sequences:
                cfg.eval_horizon = max(len(sequence) for _, sequence in cfg.eval_sequences)
    elif args.use_default_sequences and args.eval_horizon is None:
        with open_dict(cfg):
            cfg.eval_horizon = args.default_sequence_len if args.default_sequence_len is not None else 5
    main(cfg)
