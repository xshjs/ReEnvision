from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.calvin_evaluate_with_memory_reflection import (
    _apply_checkpoint_model_compat_overrides,
    _joint_pair_future_summary_exec,
)
from policy_evaluation.defi_memory_models import iter_jsonl
from policy_evaluation.oracle_hypothesis_rollout import goal_to_device, image_to_tensor, load_model
from policy_evaluation.utils import LangEmbeddings


def load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_obs(feature: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    static = image_to_tensor(np.asarray(feature["rgb_static_start"]), device)
    gripper = image_to_tensor(np.asarray(feature["rgb_gripper_start"]), device)
    robot_obs = torch.from_numpy(np.asarray(feature["robot_obs_start"], dtype=np.float32)).unsqueeze(0)
    scene_obs = torch.from_numpy(np.asarray(feature["scene_obs_start"], dtype=np.float32)).unsqueeze(0)
    return {
        "rgb_obs": {
            "rgb_static": static,
            "rgb_gripper": gripper,
        },
        "robot_obs": robot_obs.to(device),
        "scene_obs": scene_obs.to(device),
    }


@torch.no_grad()
def infer_target_future(model: torch.nn.Module, obs: dict[str, Any], lang_text: str) -> np.ndarray:
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    perceptual = model.TVP_encoder(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    perceptual = perceptual.reshape(perceptual.shape[0], perceptual.shape[1], perceptual.shape[2], -1)
    perceptual = perceptual.permute(0, 1, 3, 2).to(torch.float32)
    batch = rgb_static.shape[0]
    perceptual, _ = torch.split(perceptual, [batch, batch], dim=0)
    frame_idx = min(max(int(getattr(model, "future_frame_idx", 5)), 0), perceptual.shape[1] - 1)
    return perceptual[:, frame_idx][0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def infer_action_chunk(model: torch.nn.Module, obs: dict[str, Any], lang_text: str) -> np.ndarray:
    raise NotImplementedError


def infer_action_chunk_with_goal(
    model: torch.nn.Module,
    obs: dict[str, Any],
    goal: dict[str, Any],
) -> np.ndarray:
    model.reset()
    pred = model.eval_forward(obs, goal)
    return pred[0].detach().cpu().numpy().astype(np.float32)


def current_feature_to_base_summary_exec(current_feature: np.ndarray, num_tokens: int) -> np.ndarray:
    current_feature = np.asarray(current_feature, dtype=np.float32).reshape(1, -1)
    pseudo_future = np.repeat(current_feature, repeats=int(num_tokens), axis=0).astype(np.float32)
    return _joint_pair_future_summary_exec(pseudo_future)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a full delta-action repair memory npz from old single-future rollout assets.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--calvin-abc-dir", type=Path, required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--base-summary-mode", choices=["current_repeat", "copy_target", "zeros"], default="current_repeat")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-nonfinite-actions", action="store_true")
    args = parser.parse_args()

    with initialize(config_path="../policy_conf", job_name="rebuild_delta_action_repair_memory"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
    with open_dict(cfg):
        cfg.future_feature_mode = "gfdm"
        cfg.future_frame_idx = 5
        cfg.root_data_dir = str(args.calvin_abc_dir)
    _apply_checkpoint_model_compat_overrides(cfg, str(args.checkpoint))

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    model.eval()
    model.future_frame_idx = 5
    lang_folder = str(cfg.datamodule.datasets.lang_dataset.lang_folder)
    lang_embeddings = LangEmbeddings(args.calvin_abc_dir / "validation", lang_folder, device=device)

    rows = list(iter_jsonl(args.rollout_jsonl))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("no rollout rows found")

    action_chunks = []
    base_summaries = []
    target_summaries = []
    target_deltas = []
    states = []
    state_ends = []
    state_transitions = []
    has_state = []
    success = []
    tasks = []
    lang_texts = []
    sequence_index = []
    subtask_index = []
    row_ids = []
    feature_paths = []
    skipped_nonfinite_actions = 0

    for row in tqdm(rows, desc="rebuild_delta_action_repair_memory"):
        feature_path = args.feature_root / str(row["feature_path"])
        feature = load_feature(feature_path)
        lang_text = str(row.get("lang_text", row.get("task", "perform the task")))
        obs = build_obs(feature, device)
        goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
        goal["lang_text"] = lang_text
        target_future = infer_target_future(model, obs, lang_text)
        target_summary = _joint_pair_future_summary_exec(target_future)
        if args.base_summary_mode == "current_repeat":
            base_summary = current_feature_to_base_summary_exec(
                np.asarray(feature["obs_current_feature"], dtype=np.float32),
                num_tokens=int(target_future.shape[0]),
            )
        elif args.base_summary_mode == "zeros":
            base_summary = np.zeros_like(target_summary, dtype=np.float32)
        else:
            base_summary = target_summary.copy()
        action_chunk = infer_action_chunk_with_goal(model, obs, goal)
        if not np.isfinite(action_chunk).all():
            if args.skip_nonfinite_actions:
                skipped_nonfinite_actions += 1
                continue
            raise ValueError(f"non-finite action_chunk at row_id={row.get('row_id')} task={row.get('task')} lang={lang_text}")
        target_delta = (target_summary - base_summary).astype(np.float32)

        robot_start = np.asarray(feature.get("robot_obs_start", []), dtype=np.float32).reshape(-1)
        scene_start = np.asarray(feature.get("scene_obs_start", []), dtype=np.float32).reshape(-1)
        robot_end = np.asarray(feature.get("robot_obs_end", []), dtype=np.float32).reshape(-1)
        scene_end = np.asarray(feature.get("scene_obs_end", []), dtype=np.float32).reshape(-1)

        if robot_start.size and scene_start.size and robot_end.size and scene_end.size:
            state_start = np.concatenate([robot_start, scene_start], axis=0).astype(np.float32)
            state_end = np.concatenate([robot_end, scene_end], axis=0).astype(np.float32)
            state_transition = (state_end - state_start).astype(np.float32)
            has_state.append(1.0)
        else:
            state_start = np.zeros((0,), dtype=np.float32)
            state_end = np.zeros((0,), dtype=np.float32)
            state_transition = np.zeros((0,), dtype=np.float32)
            has_state.append(0.0)

        action_chunks.append(action_chunk)
        base_summaries.append(base_summary)
        target_summaries.append(target_summary)
        target_deltas.append(target_delta)
        states.append(state_start)
        state_ends.append(state_end)
        state_transitions.append(state_transition)
        success.append(1.0 if bool(row.get("success", False)) else 0.0)
        tasks.append(str(row.get("task", "unknown")))
        lang_texts.append(lang_text)
        sequence_index.append(int(row.get("sequence_index", 0)))
        subtask_index.append(int(row.get("subtask_index", 0)))
        row_ids.append(int(row.get("row_id", len(row_ids))))
        feature_paths.append(str(feature_path))

    state_dim = max((arr.shape[0] for arr in states), default=0)
    if state_dim > 0:
        zero_state = np.zeros((state_dim,), dtype=np.float32)
        state_start_arr = np.stack([(arr if arr.shape[0] == state_dim else zero_state) for arr in states]).astype(np.float32)
        state_end_arr = np.stack([(arr if arr.shape[0] == state_dim else zero_state) for arr in state_ends]).astype(np.float32)
        state_transition_arr = np.stack([(arr if arr.shape[0] == state_dim else zero_state) for arr in state_transitions]).astype(np.float32)
    else:
        state_start_arr = np.zeros((len(rows), 0), dtype=np.float32)
        state_end_arr = np.zeros((len(rows), 0), dtype=np.float32)
        state_transition_arr = np.zeros((len(rows), 0), dtype=np.float32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        row_id=np.asarray(row_ids, dtype=np.int64),
        base_summary_exec=np.stack(base_summaries).astype(np.float32),
        target_summary_exec=np.stack(target_summaries).astype(np.float32),
        target_delta_exec=np.stack(target_deltas).astype(np.float32),
        action_chunk=np.stack(action_chunks).astype(np.float32),
        state_start=state_start_arr,
        state_end=state_end_arr,
        state_transition=state_transition_arr,
        has_state=np.asarray(has_state, dtype=np.float32),
        success=np.asarray(success, dtype=np.float32),
        task=np.asarray(tasks, dtype=object),
        lang_text=np.asarray(lang_texts, dtype=object),
        sequence_index=np.asarray(sequence_index, dtype=np.int32),
        subtask_index=np.asarray(subtask_index, dtype=np.int32),
        feature_path=np.asarray(feature_paths, dtype=object),
    )

    summary = {
        "rollout_jsonl": str(args.rollout_jsonl),
        "feature_root": str(args.feature_root),
        "checkpoint": str(args.checkpoint),
        "output_npz": str(args.output_npz),
        "num_rows_requested": len(rows),
        "num_rows_written": len(action_chunks),
        "skipped_nonfinite_actions": int(skipped_nonfinite_actions),
        "action_shape": list(np.stack(action_chunks).shape[1:]),
        "future_summary_dim": int(np.stack(target_summaries).shape[-1]),
        "state_dim": int(state_start_arr.shape[-1]),
        "base_summary_mode": str(args.base_summary_mode),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
