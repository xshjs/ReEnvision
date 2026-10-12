import argparse
import json
import os
import shutil
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

if "distutils.spawn" not in sys.modules:
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_module

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from policy_evaluation.calvin_evaluate import count_success
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition


def ensure_train_folder(output_dir: Path, checkpoint: Path) -> Path:
    train_folder = output_dir / "eval_train_folder"
    saved_models = train_folder / "saved_models"
    saved_models.mkdir(parents=True, exist_ok=True)
    link_path = saved_models / checkpoint.name
    if not link_path.exists():
        try:
            link_path.symlink_to(checkpoint.resolve())
        except OSError:
            shutil.copy2(checkpoint, link_path)
    return train_folder


def load_model(cfg: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(state_dict["model"], strict=False)
    model.freeze()
    model.future_feature_mode = getattr(cfg, "future_feature_mode", "gfdm")
    model.future_frame_idx = getattr(cfg, "future_frame_idx", 5)
    print(f"future_feature_mode={model.future_feature_mode}, future_frame_idx={model.future_frame_idx}")
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model = model.to(device)
    model.process_device()
    model.eval()
    return model


def rollout_once(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    start_info: Dict[str, Any],
) -> Tuple[bool, int]:
    model.reset()
    obs = env.get_obs()
    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, step + 1
    return False, cfg.ep_len


def rollout_with_retry(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    lang_embeddings: Any,
    val_annotations: Any,
    max_retries: int,
) -> Tuple[bool, Dict[str, Any]]:
    lang_text = val_annotations[subtask][0]
    goal = lang_embeddings.get_lang_goal(lang_text)
    goal["lang_text"] = lang_text
    start_info = env.get_info()
    attempts = []
    total_steps = 0
    for attempt_idx in range(max_retries + 1):
        success, steps = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info)
        attempts.append({"attempt": attempt_idx, "success": success, "steps": steps})
        total_steps += steps
        if success:
            return True, {"attempts": attempts, "total_steps": total_steps, "retried": attempt_idx > 0}
    return False, {"attempts": attempts, "total_steps": total_steps, "retried": max_retries > 0}


def evaluate_variant(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    max_retries: int,
    output_dir: Path,
    variant_name: str,
) -> Dict[str, Any]:
    val_annotations = cfg.annotations
    results = []
    task_success = Counter()
    task_total = Counter()
    task_retry_success = Counter()
    task_retried = Counter()
    sequence_logs = []
    pbar = tqdm(list(enumerate(sequences)), desc=variant_name)

    for seq_idx, (initial_state, eval_sequence) in pbar:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtask_logs = []
        for subtask_idx, subtask in enumerate(eval_sequence):
            success, info = rollout_with_retry(
                env,
                model,
                task_oracle,
                cfg,
                subtask,
                lang_embeddings,
                val_annotations,
                max_retries=max_retries,
            )
            task_total[subtask] += 1
            if info["retried"]:
                task_retried[subtask] += 1
            if success:
                task_success[subtask] += 1
                if len(info["attempts"]) > 1:
                    task_retry_success[subtask] += 1
                success_counter += 1
            subtask_logs.append({"subtask_index": subtask_idx, "task": subtask, "success": success, **info})
            if not success:
                break
        results.append(success_counter)
        sequence_logs.append(
            {
                "sequence_index": seq_idx,
                "eval_sequence": list(eval_sequence),
                "success_counter": success_counter,
                "subtasks": subtask_logs,
            }
        )
        success_rates = count_success(results)
        pbar.set_description(
            f"{variant_name} " + " ".join([f"{i + 1}/5={rate * 100:.1f}%" for i, rate in enumerate(success_rates)])
        )

    chain_sr = {str(i + 1): rate for i, rate in enumerate(count_success(results))}
    summary = {
        "variant": variant_name,
        "max_retries": max_retries,
        "num_sequences": len(results),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": chain_sr,
        "results": results,
        "task_info": {
            task: {
                "success": int(task_success[task]),
                "total": int(task_total[task]),
                "retried": int(task_retried[task]),
                "retry_success": int(task_retry_success[task]),
            }
            for task in sorted(task_total)
        },
    }
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (variant_dir / "sequences.jsonl").open("w") as handle:
        for row in sequence_logs:
            handle.write(json.dumps(row) + "\n")
    return summary


def write_comparison(output_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# Contact Verification Retry Evaluation",
        "",
        "Post-skill verifier: simulator task/contact-state oracle. Recovery: retry current skill with the same DeFi policy.",
        "",
        "| Variant | Avg Seq Len | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        chain = summary["chain_sr"]
        lines.append(
            f"| {summary['variant']} | {summary['avg_seq_len']:.3f} | "
            f"{chain['1']:.3f} | {chain['2']:.3f} | {chain['3']:.3f} | {chain['4']:.3f} | {chain['5']:.3f} |"
        )
    if len(summaries) >= 2:
        baseline = summaries[0]
        for summary in summaries[1:]:
            lines.extend(
                [
                    "",
                    f"## Delta: {summary['variant']} - {baseline['variant']}",
                    "",
                    f"- Avg seq len: {summary['avg_seq_len'] - baseline['avg_seq_len']:+.3f}",
                    f"- 5/5 SR: {summary['chain_sr']['5'] - baseline['chain_sr']['5']:+.3f}",
                ]
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps({item["variant"]: item for item in summaries}, indent=2) + "\n")


def parse_retry_counts(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeFi with post-skill oracle verification and retry-current-skill recovery.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--ep-len", type=int, default=120)
    parser.add_argument("--retry-counts", default="0,1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--future-feature-mode", choices=["gfdm", "current", "zero"], default="gfdm")
    parser.add_argument("--future-frame-idx", type=int, default=5)
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retry_counts = parse_retry_counts(args.retry_counts)

    with initialize(config_path="../policy_conf", job_name="contact_retry_evaluate"):
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
        cfg.future_feature_mode = args.future_feature_mode
        cfg.future_frame_idx = args.future_frame_idx

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
    sequences = get_sequences(args.num_sequences)
    metadata = {
        "num_sequences": args.num_sequences,
        "ep_len_per_attempt": args.ep_len,
        "retry_counts": retry_counts,
        "checkpoint": str(args.checkpoint),
        "note": "Post-skill verifier uses simulator task/contact-state oracle; retry re-executes the current skill from the current environment state.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    summaries = []
    for retries in retry_counts:
        variant_name = "defi" if retries == 0 else f"defi_contact_retry{retries}"
        summaries.append(
            evaluate_variant(
                env,
                model,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                max_retries=retries,
                output_dir=args.output_dir,
                variant_name=variant_name,
            )
        )
    write_comparison(args.output_dir, summaries)
    print((args.output_dir / "comparison.md").read_text())


if __name__ == "__main__":
    os.environ.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")
    main()
