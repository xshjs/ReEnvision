'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/eval.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import yaml
import csv
import json
import traceback
import torch
import cv2
import shutil
import numpy as np
from multiprocessing import Value
from copy import deepcopy
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.agents.agent import VideoSummary

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.mvt.mvt import MVT
from bridgevla.libs.peract.helpers import utils
from utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from bridgevla.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    RLBENCH_TASKS,
)
from bridgevla.utils.rvt_utils import load_agent as load_agent_state
import os 

try:
    from policy_evaluation.label_defi_future_mismatch import infer_mismatch_type
    from policy_evaluation.oracle_hypothesis_rollout import template_postexec_reflection_decision
except Exception:  # pragma: no cover - fallback for bridge-only runs
    infer_mismatch_type = None
    template_postexec_reflection_decision = None

def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False):
    if torch.cuda.is_available():
        device = str(device)
        if not device.startswith("cuda"):
            device = f"cuda:{device}"
    else:
        device = "cpu"
    assert model_path is not None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path != None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()


    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

    mvt_cfg.freeze()

    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    renderer_device = device
    print(f"[BridgeVLA RLBench] device={device} renderer_device={renderer_device}", flush=True)
    rvt = MVT(
        renderer_device=renderer_device,
        **mvt_cfg,
    )

    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )


    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    bridge_latent_head_ckpt = os.environ.get("BRIDGEVLA_BRIDGE_LATENT_HEAD_CKPT", "").strip()
    if bridge_latent_head_ckpt:
        agent.init_bridge_latent_heads(ckpt_path=bridge_latent_head_ckpt)
    bridge_standalone_gidm_ckpt = os.environ.get("BRIDGEVLA_BRIDGE_STANDALONE_GIDM_CKPT", "").strip()
    if bridge_standalone_gidm_ckpt:
        agent.init_bridge_standalone_gidm(
            ckpt_path=bridge_standalone_gidm_ckpt,
            t5_model_path=os.environ.get(
                "BRIDGEVLA_BRIDGE_STANDALONE_GIDM_T5_PATH",
                "/mnt/workspace/manipulation/DeFi/ckpts/t5_base",
            ),
        )
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent




def _save_episode_rollout_artifacts(
    rollout_root,
    step_feature_dir,
    step_rows_path,
    task_name,
    ep,
    attempt_idx,
    attempt_type,
    episode_rollout,
    agent,
    episode_length,
    lang_goal,
    reward,
):
    if rollout_root is None:
        return None

    feature_dir = rollout_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    first_obs = episode_rollout[0].observation
    last_transition = episode_rollout[-1]
    final_obs = getattr(last_transition, "final_observation", None)
    if final_obs is None:
        final_obs = last_transition.observation

    obs_current_feature = agent.extract_pooled_feature(first_obs, language_goal=lang_goal)
    end_obs_current_feature = agent.extract_pooled_feature(final_obs, language_goal=lang_goal)
    if obs_current_feature is not None:
        obs_current_feature = np.asarray(obs_current_feature)
    if end_obs_current_feature is not None:
        end_obs_current_feature = np.asarray(end_obs_current_feature)

    defi_future_feature = None
    if hasattr(agent, "extract_defi_future_feature"):
        try:
            defi_future_feature = agent.extract_defi_future_feature(final_obs, language_goal=lang_goal)
        except Exception:
            defi_future_feature = None
    if defi_future_feature is None and hasattr(agent, "_last_motion_latent"):
        defi_future_feature = getattr(agent, "_last_motion_latent")
    if isinstance(defi_future_feature, torch.Tensor):
        defi_future_feature = defi_future_feature.detach().cpu().numpy()
    if defi_future_feature is None:
        defi_future_feature = end_obs_current_feature

    defi_future_feature_seq = None
    if hasattr(agent, "get_defi_future_feature_sequence"):
        try:
            defi_future_feature_seq = agent.get_defi_future_feature_sequence()
        except Exception:
            defi_future_feature_seq = None
    if isinstance(defi_future_feature_seq, torch.Tensor):
        defi_future_feature_seq = defi_future_feature_seq.detach().cpu().numpy()

    feature_name = f"row_{task_name}_ep{ep:04d}_att{attempt_idx:02d}.npz"
    feature_path = str(feature_dir / feature_name)
    save_kwargs = {
        "obs_current_feature": obs_current_feature,
        "defi_future_feature": defi_future_feature,
        "end_obs_current_feature": end_obs_current_feature,
    }
    if defi_future_feature_seq is not None:
        save_kwargs["defi_future_feature_seq"] = defi_future_feature_seq
    np.savez_compressed(feature_path, **save_kwargs)

    mismatch_type = "future_outcome_mismatch"
    if infer_mismatch_type is not None:
        try:
            mismatch_type = infer_mismatch_type(task_name, bool(reward > 99), int(len(episode_rollout)), int(episode_length))
        except Exception:
            mismatch_type = "future_outcome_mismatch"
    decision = None
    if template_postexec_reflection_decision is not None:
        try:
            decision = template_postexec_reflection_decision(
                task_name,
                bool(reward > 99),
                int(len(episode_rollout)),
                int(episode_length),
                mismatch_type,
            )
        except Exception:
            decision = None
    with open(rollout_root / "rollout_rows.jsonl", "a", encoding="utf-8") as fp:
        fp.write(
            json.dumps(
                {
                    "task": task_name,
                    "lang_goal": lang_goal,
                    "success": bool(reward > 99),
                    "steps": int(len(episode_rollout)),
                    "ep_len": int(episode_length),
                    "feature_path": feature_path,
                    "episode": int(ep),
                    "attempt_idx": int(attempt_idx),
                    "attempt_type": str(attempt_type),
                    "mismatch_type": mismatch_type,
                    "correction_direction": getattr(decision, "correction_direction", "stabilize contact"),
                    "recoverability": getattr(decision, "recoverability", "uncertain"),
                    "trust_score": float(getattr(decision, "trust_score", 0.5)),
                    "reflection_text": getattr(decision, "explanation", ""),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    if step_feature_dir is not None and step_rows_path is not None:
        for step_idx, replay_transition in enumerate(episode_rollout):
            step_feature_path = step_feature_dir / f"row_{task_name}_ep{ep:04d}_att{attempt_idx:02d}_step{step_idx:03d}.npz"
            try:
                current_obs = replay_transition.observation
                current_feature = agent.extract_pooled_feature(current_obs, language_goal=lang_goal)
                if current_feature is not None and torch.is_tensor(current_feature):
                    current_feature = current_feature.detach().cpu().numpy()
                defi_future_feature_step = None
                if hasattr(agent, "extract_defi_future_feature"):
                    try:
                        defi_future_feature_step = agent.extract_defi_future_feature(current_obs, language_goal=lang_goal)
                    except Exception:
                        defi_future_feature_step = None
                if defi_future_feature_step is None and hasattr(agent, "_last_motion_latent"):
                    defi_future_feature_step = getattr(agent, "_last_motion_latent")
                if torch.is_tensor(defi_future_feature_step):
                    defi_future_feature_step = defi_future_feature_step.detach().cpu().numpy()
                if defi_future_feature_step is None:
                    defi_future_feature_step = np.zeros((1, 768), dtype=np.float32)
                end_obs_feature = current_feature
                if current_feature is None:
                    current_feature = np.zeros((1, 12288), dtype=np.float32)
                info = getattr(replay_transition, "info", {}) or {}
                state_latent = info.get("agent_state_latent", None)
                future_latent = info.get("agent_future_latent", None)
                motion_latent = info.get("agent_motion_latent", None)
                gidm_latent = info.get("agent_gidm_latent", None)
                if torch.is_tensor(state_latent):
                    state_latent = state_latent.detach().cpu().numpy()
                if torch.is_tensor(future_latent):
                    future_latent = future_latent.detach().cpu().numpy()
                if torch.is_tensor(motion_latent):
                    motion_latent = motion_latent.detach().cpu().numpy()
                if torch.is_tensor(gidm_latent):
                    gidm_latent = gidm_latent.detach().cpu().numpy()
                step_save_kwargs = {
                    "obs_current_feature": current_feature,
                    "defi_future_feature": defi_future_feature_step,
                    "end_obs_current_feature": end_obs_feature,
                    "state_latent": np.asarray(state_latent, dtype=np.float32) if state_latent is not None else np.zeros((1, 768), dtype=np.float32),
                    "future_latent": np.asarray(future_latent, dtype=np.float32) if future_latent is not None else np.zeros((1, 768), dtype=np.float32),
                    "motion_latent": np.asarray(motion_latent, dtype=np.float32) if motion_latent is not None else np.zeros((1, 768), dtype=np.float32),
                    "gidm_latent": np.asarray(gidm_latent, dtype=np.float32) if gidm_latent is not None else np.zeros((1, 768), dtype=np.float32),
                    "reward": np.asarray([float(replay_transition.reward)], dtype=np.float32),
                    "final_episode_success": np.asarray([float(reward > 99)], dtype=np.float32),
                    "final_episode_return": np.asarray([float(reward)], dtype=np.float32),
                    "final_episode_length": np.asarray([int(len(episode_rollout))], dtype=np.int32),
                }
                np.savez_compressed(step_feature_path, **step_save_kwargs)
                row = {
                    "task": task_name,
                    "lang_goal": lang_goal,
                    "success": bool(replay_transition.reward > 99),
                    "step": int(step_idx),
                    "steps": int(step_idx + 1),
                    "ep_len": int(episode_length),
                    "feature_path": str(step_feature_path),
                    "episode": int(ep),
                    "attempt_idx": int(attempt_idx),
                    "attempt_type": str(attempt_type),
                    "final_episode_success": bool(reward > 99),
                    "final_episode_return": float(reward),
                    "final_episode_length": int(len(episode_rollout)),
                    "repair_mode_active": bool(getattr(agent, "_repair_mode_active", False)),
                    "repair_mode_reason": str(getattr(agent, "_repair_mode_reason", "")),
                }
                for info_key in [
                    "agent_gidm_source",
                    "agent_gidm_gate",
                    "agent_gidm_latent_norm",
                    "agent_state_latent_norm",
                    "agent_future_latent_norm",
                    "agent_motion_latent_norm",
                    "agent_bridge_latent_head_mode",
                    "agent_joint_selector_selector_used",
                    "agent_joint_selector_selector_applied",
                    "agent_joint_selector_memory_index",
                    "agent_joint_selector_rank",
                    "agent_joint_selector_similarity",
                    "agent_joint_selector_energy_improve",
                    "agent_joint_selector_future_shift",
                    "agent_joint_selector_motion_shift",
                    "agent_joint_selector_final_future_shift",
                    "agent_joint_selector_final_motion_shift",
                    "agent_joint_selector_repair_repair_used",
                    "agent_joint_selector_repair_repair_reason",
                    "agent_joint_selector_repair_future_delta_norm",
                    "agent_joint_selector_repair_motion_delta_norm",
                    "agent_joint_selector_verify_verify_used",
                    "agent_joint_selector_verify_verify_passed",
                    "agent_joint_selector_verify_verify_reason",
                    "agent_joint_selector_verify_verify_c_exp",
                    "agent_joint_selector_verify_verify_c_latent",
                    "agent_joint_selector_verify_verify_c_physical",
                    "agent_joint_selector_verify_acceptance_prob",
                    "agent_suffix_td_active",
                    "agent_suffix_td_requested",
                    "agent_suffix_td_applied",
                    "agent_suffix_td_mode",
                    "agent_suffix_td_patch_prob",
                    "agent_suffix_td_replan_prob",
                    "agent_suffix_td_trigger_reason",
                    "agent_suffix_td_mix",
                    "agent_suffix_td_delta_norm",
                    "agent_suffix_td_apply_enabled",
                ]:
                    if info_key in replay_transition.info:
                        row[info_key] = replay_transition.info[info_key]
                with step_rows_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"[warn] failed to save step rollout for {task_name} ep={ep} att={attempt_idx} step={step_idx}: {exc}")

    return feature_path


def _as_numpy_array(value, dtype=np.float32):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _rollout_requested_suffix_td(episode_rollout):
    for transition in episode_rollout:
        info = getattr(transition, "info", {}) or {}
        if bool(info.get("agent_suffix_td_requested", False)):
            return True
    return False


def _obs_low_dim(obs):
    if not isinstance(obs, dict):
        return np.zeros((0,), dtype=np.float32)
    low = obs.get("low_dim_state", None)
    if torch.is_tensor(low):
        low = low.detach().cpu().numpy()
    if low is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(low, dtype=np.float32).reshape(-1)


def _info_latent(info, key, fallback_shape=(1, 768)):
    value = None if info is None else info.get(key, None)
    arr = _as_numpy_array(value)
    if arr is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _joint_belief_latents(info, next_info=None):
    info = info or {}
    next_info = next_info or {}
    pooled = _info_latent(info, "agent_pooled_latent")
    next_pooled = _info_latent(next_info, "agent_pooled_latent", fallback_shape=pooled.shape)
    future = _as_numpy_array(info.get("agent_future_latent", None))
    action = _as_numpy_array(info.get("agent_motion_latent", None))
    next_future = _as_numpy_array(next_info.get("agent_future_latent", None))
    next_action = _as_numpy_array(next_info.get("agent_motion_latent", None))
    if future is None:
        future = pooled
    if action is None:
        action = pooled
    if next_future is None:
        next_future = next_pooled
    if next_action is None:
        next_action = next_pooled
    return (
        future.astype(np.float32, copy=False),
        action.astype(np.float32, copy=False),
        next_future.astype(np.float32, copy=False),
        next_action.astype(np.float32, copy=False),
        pooled.astype(np.float32, copy=False),
        next_pooled.astype(np.float32, copy=False),
    )


def _pad_action_chunk(actions, start, horizon, action_dim):
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    valid = np.zeros((horizon,), dtype=np.float32)
    if start >= len(actions):
        return chunk, valid
    selected = actions[start : start + horizon]
    if len(selected) == 0:
        return chunk, valid
    arr = np.stack([np.asarray(a, dtype=np.float32).reshape(-1) for a in selected], axis=0)
    width = min(action_dim, arr.shape[1])
    chunk[: arr.shape[0], :width] = arr[:, :width]
    valid[: arr.shape[0]] = 1.0
    return chunk, valid


def _collect_suffix_transition_episode(
    rollout_root,
    task_name,
    ep,
    attempt_idx,
    attempt_type,
    episode_rollout,
    lang_goal,
    episode_length,
    final_success,
):
    if rollout_root is None or not bool(int(os.environ.get("BRIDGEVLA_COLLECT_SUFFIX_DATA", "0"))):
        return
    horizon = int(os.environ.get("BRIDGEVLA_SUFFIX_HORIZON", "10"))
    suffix_dir = rollout_root / "suffix_transition"
    suffix_dir.mkdir(parents=True, exist_ok=True)
    rows_path = rollout_root / "suffix_transition_rows.jsonl"
    actions = [np.asarray(t.action, dtype=np.float32).reshape(-1) for t in episode_rollout]
    if len(actions) < 2:
        return
    action_dim = max(int(a.shape[0]) for a in actions)
    for step_idx, transition in enumerate(episode_rollout[:-1]):
        next_transition = episode_rollout[step_idx + 1]
        info = getattr(transition, "info", {}) or {}
        next_info = getattr(next_transition, "info", {}) or {}
        obs_t = transition.observation
        obs_t1 = info.get("next_observation", None)
        if obs_t1 is None:
            obs_t1 = next_transition.observation
        (
            h_future_t,
            h_action_t,
            h_future_t1,
            h_action_t1,
            pooled_t,
            pooled_t1,
        ) = _joint_belief_latents(info, next_info)
        remain_before, remain_before_mask = _pad_action_chunk(actions, step_idx + 1, horizon, action_dim)
        # After executing a_t, the available suffix is still [a_{t+1}, ...].
        # The one-step-shifted future window is saved separately for analysis.
        remain_after = remain_before.copy()
        remain_after_mask = remain_before_mask.copy()
        next_action_window, next_action_window_mask = _pad_action_chunk(actions, step_idx + 2, horizon, action_dim)
        action_delta_target = remain_after - remain_before
        remaining_count = int(remain_before_mask.sum())
        if bool(final_success):
            mode_label = 0
            mode_reason = "successful_baseline_keep"
        elif remaining_count >= int(os.environ.get("BRIDGEVLA_SUFFIX_MIN_PATCH_REMAIN", "2")):
            mode_label = 1
            mode_reason = "failed_baseline_patch_candidate"
        else:
            mode_label = 2
            mode_reason = "failed_baseline_replan_candidate"
        feature_path = suffix_dir / f"{task_name}_ep{ep:04d}_att{attempt_idx:02d}_step{step_idx:03d}.npz"
        np.savez_compressed(
            feature_path,
            state_before=_obs_low_dim(obs_t),
            state_after=_obs_low_dim(obs_t1),
            executed_action=actions[step_idx].astype(np.float32),
            h_future_before=h_future_t.astype(np.float32),
            h_action_before=h_action_t.astype(np.float32),
            h_future_after=h_future_t1.astype(np.float32),
            h_action_after=h_action_t1.astype(np.float32),
            pooled_latent_before=pooled_t,
            pooled_latent_after=pooled_t1,
            action_remain_before=remain_before,
            action_remain_before_mask=remain_before_mask,
            action_remain_after=remain_after,
            action_remain_after_mask=remain_after_mask,
            next_action_window=next_action_window,
            next_action_window_mask=next_action_window_mask,
            action_delta_target=action_delta_target.astype(np.float32),
            mode_label=np.asarray([mode_label], dtype=np.int64),
            success=np.asarray([float(final_success)], dtype=np.float32),
            step_success=np.asarray([float(transition.reward > 99)], dtype=np.float32),
        )
        row = {
            "task": task_name,
            "lang_goal": lang_goal,
            "episode": int(ep),
            "attempt_idx": int(attempt_idx),
            "attempt_type": str(attempt_type),
            "step": int(step_idx),
            "ep_len": int(episode_length),
            "feature_path": str(feature_path),
            "action_dim": int(action_dim),
            "horizon": int(horizon),
            "mode_label": int(mode_label),
            "mode_reason": mode_reason,
            "success": bool(final_success),
            "step_success": bool(transition.reward > 99),
            "delta_norm": float(np.linalg.norm(action_delta_target)),
        }
        with rows_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    replay_ground_truth=False,
    collect_repair_after_failure=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
    model_name="debug",
    visualize=False,
    visualize_root_dir="",
    save_rollout_dir="",
):
    agent.eval()
    suffix_retry_only = bool(int(os.environ.get("BRIDGEVLA_SUFFIX_RETRY_ONLY", "0")))

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    rollout_root = Path(save_rollout_dir) if save_rollout_dir else None
    step_feature_dir = None
    step_rows_path = None
    if rollout_root is not None:
        step_feature_dir = rollout_root / "step_features"
        step_feature_dir.mkdir(parents=True, exist_ok=True)
        step_rows_path = rollout_root / "step_rollout_rows.jsonl"
        rollout_root.mkdir(parents=True, exist_ok=True)
    for task_id in range(num_tasks):
        task_rewards = []
        language_goals=[]
        episode_records = []
        final_episode_records = []
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            task_name = tasks[task_id]
            if suffix_retry_only and hasattr(agent, "set_suffix_td_apply_enabled"):
                agent.set_suffix_td_apply_enabled(False)
            if not visualize:
                generator = rollout_generator.generator(
                    step_signal=step_signal,
                    env=eval_env,
                    agent=agent,
                    episode_length=episode_length,
                    timesteps=1,
                    eval=True,
                    eval_demo_seed=ep,
                    record_enabled=save_video,
                    replay_ground_truth=replay_ground_truth,
                )
            else:
                task_name = tasks[task_id]
                lang_goal = eval_env._lang_goal
                visualize_save_dir=os.path.join(visualize_root_dir,task_name,f"episode_{ep}")
                if not os.path.exists(visualize_save_dir):
                    os.makedirs(visualize_save_dir)
                generator = rollout_generator.generator_visualize(
                    step_signal=step_signal,
                    env=eval_env,
                    agent=agent,
                    episode_length=episode_length,
                    timesteps=1,
                    eval=True,
                    eval_demo_seed=ep,
                    record_enabled=True,
                    visualize_save_dir=visualize_save_dir,
                    visualize=True,
                )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info.get("active_task_id", task_id)
                if current_task_id != task_id:
                    print(
                        f"[warn] task_id mismatch: expected={task_id} got={current_task_id} "
                        f"task={task_name} ep={ep}"
                    )
                    continue

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            language_goals.append(lang_goal)
            mismatch_type = "future_outcome_mismatch"
            decision = None
            if infer_mismatch_type is not None:
                try:
                    mismatch_type = infer_mismatch_type(
                        task_name,
                        bool(reward > 99),
                        int(len(episode_rollout)),
                        int(episode_length),
                    )
                except Exception:
                    mismatch_type = "future_outcome_mismatch"
            if template_postexec_reflection_decision is not None:
                try:
                    decision = template_postexec_reflection_decision(
                        task_name,
                        bool(reward > 99),
                        int(len(episode_rollout)),
                        int(episode_length),
                        mismatch_type,
                    )
                except Exception:
                    decision = None
            feature_path = None
            if rollout_root is not None:
                feature_dir = rollout_root / "features"
                feature_dir.mkdir(parents=True, exist_ok=True)
                first_obs = episode_rollout[0].observation
                last_transition = episode_rollout[-1]
                final_obs = getattr(last_transition, "final_observation", None)
                if final_obs is None:
                    final_obs = last_transition.observation

                obs_current_feature = agent.extract_pooled_feature(first_obs, language_goal=lang_goal)
                end_obs_current_feature = agent.extract_pooled_feature(final_obs, language_goal=lang_goal)
                if obs_current_feature is not None:
                    obs_current_feature = np.asarray(obs_current_feature)
                if end_obs_current_feature is not None:
                    end_obs_current_feature = np.asarray(end_obs_current_feature)
                defi_future_feature = None
                if hasattr(agent, "extract_defi_future_feature"):
                    try:
                        defi_future_feature = agent.extract_defi_future_feature(final_obs, language_goal=lang_goal)
                    except Exception:
                        defi_future_feature = None
                if defi_future_feature is None and hasattr(agent, "_last_motion_latent"):
                    defi_future_feature = getattr(agent, "_last_motion_latent")
                if isinstance(defi_future_feature, torch.Tensor):
                    defi_future_feature = defi_future_feature.detach().cpu().numpy()
                if defi_future_feature is None:
                    defi_future_feature = end_obs_current_feature

                defi_future_feature_seq = None
                if hasattr(agent, "get_defi_future_feature_sequence"):
                    try:
                        defi_future_feature_seq = agent.get_defi_future_feature_sequence()
                    except Exception:
                        defi_future_feature_seq = None
                if isinstance(defi_future_feature_seq, torch.Tensor):
                    defi_future_feature_seq = defi_future_feature_seq.detach().cpu().numpy()

                feature_path = _save_episode_rollout_artifacts(
                    rollout_root,
                    step_feature_dir,
                    step_rows_path,
                    task_name,
                    ep,
                    0,
                    "baseline",
                    episode_rollout,
                    agent,
                    episode_length,
                    lang_goal,
                    reward,
                )
                _collect_suffix_transition_episode(
                    rollout_root,
                    task_name,
                    ep,
                    0,
                    "baseline",
                    episode_rollout,
                    lang_goal,
                    episode_length,
                    bool(reward > 99),
                )
            baseline_success = bool(reward > 99)
            repair_success = None
            final_reward = float(reward)
            final_length = len(episode_rollout)
            final_attempt_type = "baseline"
            final_success = baseline_success
            episode_records.append(
                {
                    "episode": ep,
                    "reward": float(reward),
                    "length": len(episode_rollout),
                    "lang_goal": lang_goal,
                    "task": task_name,
                    "attempt_type": "baseline",
                    "success": bool(reward > 99),
                    "steps": len(episode_rollout),
                    "ep_len": episode_length,
                    "feature_path": feature_path,
                    "mismatch_type": mismatch_type,
                    "correction_direction": getattr(decision, "correction_direction", "stabilize contact"),
                    "recoverability": getattr(decision, "recoverability", "uncertain"),
                    "trust_score": float(getattr(decision, "trust_score", 0.5)),
                }
            )
            if verbose:
                agent_gidm_source = episode_rollout[-1].info.get("agent_gidm_source", "none") if episode_rollout else "none"
                agent_gidm_gate = episode_rollout[-1].info.get("agent_gidm_gate", None) if episode_rollout else None
                agent_gidm_norm = episode_rollout[-1].info.get("agent_gidm_latent_norm", None) if episode_rollout else None
                agent_state_norm = episode_rollout[-1].info.get("agent_state_latent_norm", None) if episode_rollout else None
                agent_future_norm = episode_rollout[-1].info.get("agent_future_latent_norm", None) if episode_rollout else None
                agent_motion_norm = episode_rollout[-1].info.get("agent_motion_latent_norm", None) if episode_rollout else None
                print(
                    f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )
                print(
                    "[episode-meta] "
                    f"task={task_name} "
                    f"lang_goal={lang_goal!r} "
                    f"attempt=baseline "
                    f"success={bool(reward > 99)} "
                    f"steps={len(episode_rollout)} "
                    f"ep_len={episode_length} "
                    f"feature_path={feature_path} "
                    f"gidm_source={agent_gidm_source} "
                    f"gidm_gate={agent_gidm_gate} "
                    f"gidm_norm={agent_gidm_norm} "
                    f"s_norm={agent_state_norm} "
                    f"zf_norm={agent_future_norm} "
                    f"za_norm={agent_motion_norm}"
                )
            if getattr(agent, "_debug_step_state", False):
                try:
                    _print_episode_state_debug(task_name, eval_env, episode_rollout)
                except Exception as debug_exc:
                    print(f"[debug] failed to dump step state for {task_name} ep={ep}: {debug_exc}")

            suffix_retry_triggered = False
            if (not bool(reward > 99)) and bool(collect_repair_after_failure) and suffix_retry_only:
                suffix_retry_triggered = _rollout_requested_suffix_td(episode_rollout)
                if not suffix_retry_triggered:
                    print(
                        f"[suffix-retry-skip] task={task_name} ep={ep} "
                        "baseline failed but suffix dry-run did not trigger",
                        flush=True,
                    )

            if (
                (not bool(reward > 99))
                and bool(collect_repair_after_failure)
                and ((not suffix_retry_only) or suffix_retry_triggered)
            ):
                if hasattr(agent, "set_suffix_td_apply_enabled"):
                    agent.set_suffix_td_apply_enabled(True)
                if hasattr(agent, "prepare_repair_retry"):
                    try:
                        agent.prepare_repair_retry(
                            reason=f"baseline_fail_{task_name}_ep{ep}",
                            task_name=task_name,
                            lang_goal=lang_goal,
                            success=False,
                            steps=len(episode_rollout),
                            ep_len=episode_length,
                        )
                    except TypeError:
                        agent.prepare_repair_retry(reason=f"baseline_fail_{task_name}_ep{ep}")
                repair_rollout = []
                if not visualize:
                    repair_generator = rollout_generator.generator(
                        step_signal=step_signal,
                        env=eval_env,
                        agent=agent,
                        episode_length=episode_length,
                        timesteps=1,
                        eval=True,
                        eval_demo_seed=ep,
                        record_enabled=save_video,
                        replay_ground_truth=replay_ground_truth,
                    )
                else:
                    repair_visualize_save_dir = os.path.join(visualize_root_dir, task_name, f"episode_{ep}_repair")
                    if not os.path.exists(repair_visualize_save_dir):
                        os.makedirs(repair_visualize_save_dir)
                    repair_generator = rollout_generator.generator_visualize(
                        step_signal=step_signal,
                        env=eval_env,
                        agent=agent,
                        episode_length=episode_length,
                        timesteps=1,
                        eval=True,
                        eval_demo_seed=ep,
                        record_enabled=True,
                        visualize_save_dir=repair_visualize_save_dir,
                        visualize=True,
                    )
                try:
                    for replay_transition in repair_generator:
                        repair_rollout.append(replay_transition)
                except StopIteration:
                    pass
                except Exception as repair_exc:
                    print(f"[warn] repair retry failed for {task_name} ep={ep}: {repair_exc!r}")
                    traceback.print_exc()
                if repair_rollout:
                    repair_reward = repair_rollout[-1].reward
                    repair_lang_goal = getattr(agent, "_language_override", None) or eval_env._lang_goal
                    repair_success = bool(repair_reward > 99)
                    repair_feature_path = _save_episode_rollout_artifacts(
                        rollout_root,
                        step_feature_dir,
                        step_rows_path,
                        task_name,
                        ep,
                        1,
                        "repair",
                        repair_rollout,
                        agent,
                        episode_length,
                        repair_lang_goal,
                        repair_reward,
                    )
                    _collect_suffix_transition_episode(
                        rollout_root,
                        task_name,
                        ep,
                        1,
                        "repair",
                        repair_rollout,
                        repair_lang_goal,
                        episode_length,
                        bool(repair_reward > 99),
                    )
                    episode_records.append(
                        {
                            "episode": ep,
                            "reward": float(repair_reward),
                            "length": len(repair_rollout),
                            "lang_goal": repair_lang_goal,
                            "task": task_name,
                            "attempt_type": "repair",
                            "success": bool(repair_reward > 99),
                            "steps": len(repair_rollout),
                            "ep_len": episode_length,
                            "feature_path": repair_feature_path,
                        }
                    )
                    final_reward = float(repair_reward)
                    final_length = len(repair_rollout)
                    final_attempt_type = "repair" if repair_success else "repair_failed"
                    final_success = baseline_success or repair_success
                    if verbose:
                        print(
                            f"[repair-meta] task={task_name} | Episode {ep} | Score: {repair_reward} | "
                            f"Episode Length: {len(repair_rollout)} | Lang Goal: {repair_lang_goal}"
                        )
                if hasattr(agent, "finish_repair_retry"):
                    try:
                        agent.finish_repair_retry()
                    except Exception as finish_exc:
                        print(f"[warn] failed to clear repair retry state for {task_name} ep={ep}: {finish_exc}")
                if suffix_retry_only and hasattr(agent, "set_suffix_td_apply_enabled"):
                    agent.set_suffix_td_apply_enabled(False)

            final_episode_records.append(
                {
                    "episode": ep,
                    "task": task_name,
                    "lang_goal": lang_goal,
                    "baseline_success": baseline_success,
                    "repair_success": repair_success,
                    "final_success": bool(final_success),
                    "final_reward": float(final_reward),
                    "final_length": int(final_length),
                    "final_attempt_type": final_attempt_type,
                    "suffix_retry_triggered": bool(suffix_retry_triggered),
                }
            )
            if hasattr(agent, "set_episode_result"):
                try:
                    agent.set_episode_result(
                        task_name=task_name,
                        lang_goal=lang_goal,
                        baseline_success=baseline_success,
                        repair_success=repair_success,
                        final_success=bool(final_success),
                        baseline_reward=float(reward),
                        repair_reward=None if repair_success is None else float(final_reward),
                        final_reward=float(final_reward),
                        baseline_length=int(len(episode_rollout)),
                        repair_length=None if repair_success is None else int(final_length),
                        final_length=int(final_length),
                    )
                except Exception as episode_result_exc:
                    print(f"[warn] failed to record episode result for {task_name} ep={ep}: {episode_result_exc}")
            if verbose:
                final_info = episode_rollout[-1].info if episode_rollout else {}
                print(
                    "[episode-final] "
                    f"task={task_name} ep={ep} "
                    f"baseline_success={baseline_success} "
                    f"repair_success={repair_success} "
                    f"final_success={bool(final_success)} "
                    f"final_score={final_reward} "
                    f"final_len={final_length} "
                    f"attempt={final_attempt_type} "
                    f"gidm_source={final_info.get('agent_gidm_source', 'none')} "
                    f"gidm_gate={final_info.get('agent_gidm_gate', None)} "
                    f"gidm_norm={final_info.get('agent_gidm_latent_norm', None)} "
                    f"s_norm={final_info.get('agent_state_latent_norm', None)} "
                    f"zf_norm={final_info.get('agent_future_latent_norm', None)} "
                    f"za_norm={final_info.get('agent_motion_latent_norm', None)}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        final_success_rate = float(np.mean([1.0 if r["final_success"] else 0.0 for r in final_episode_records])) if len(final_episode_records) > 0 else 0.0
        final_length_mean = float(np.mean([r["final_length"] for r in final_episode_records])) if len(final_episode_records) > 0 else 0.0
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_results["success rate"] = final_success_rate
                csv_results["length"] = final_length_mean
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(final_episode_records) > 0:
            task_score = final_success_rate
        elif len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = float(np.mean(task_rewards)) if len(task_rewards) > 0 else "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        success_cases = [r for r in final_episode_records if r["final_success"]]
        fail_cases = [r for r in final_episode_records if not r["final_success"]]
        if len(success_cases) > 0:
            print(f"[Evaluation] {task_name} success cases ({len(success_cases)} final episodes):")
            for r in success_cases:
                print(
                    f"  - ep={r['episode']} final_score={r['final_reward']:.1f} len={r['final_length']} lang='{r['lang_goal']}'"
                )
        if len(fail_cases) > 0:
            print(f"[Evaluation] {task_name} fail cases ({len(fail_cases)} final episodes):")
            for r in fail_cases:
                print(
                    f"  - ep={r['episode']} final_score={r['final_reward']:.1f} len={r['final_length']} lang='{r['lang_goal']}'"
                )

        scores.append(task_score)

        if save_video:
            video_image_folder = f"./tmp/{model_name}/{task_name}"
            palette_image_folder = f"./tmp/{model_name}/palette_folder"
            palette_image_path=os.path.join(palette_image_folder,"palette.png")

            record_fps = 25
            if not visualize:
                record_folder = os.path.join(log_dir, "videos")
            else:
                record_folder = os.path.join(visualize_root_dir,task_name,"videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    lang_goal = language_goals.pop(0)
                    lang_goal=lang_goal.replace(" ", "_")
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    if task_rewards[video_cnt] > 99:
                        video_path = os.path.join(
                            record_folder,
                            f"{lang_goal}_success_{video_success_cnt}.mp4",
                        )
                        video_success_cnt += 1
                    else:
                        video_path = os.path.join(
                            record_folder, f"{lang_goal}_fail_{video_fail_cnt}.mp4"
                        )
                        video_fail_cnt += 1
                    video_cnt += 1
                    os.makedirs(video_image_folder, exist_ok=True)
                    os.makedirs(palette_image_folder, exist_ok=True)
                    for idx in range(len(video) - 10):
                        cv2.imwrite(
                            os.path.join(video_image_folder, f"{idx}.png"), video[idx]
                        )
                    images_path = os.path.join(video_image_folder, r"%d.png")
                    os.system(
                        "ffmpeg -i {} -vf palettegen {} -hide_banner -loglevel error".format(
                            images_path, palette_image_path
                        )
                    )
                    
                    os.system(
                        "ffmpeg -framerate {} -i {} -i {} -lavfi paletteuse {} -hide_banner -loglevel error".format(
                            record_fps, images_path, palette_image_path, video_path
                        )
                    )
                    print(f'video saved - {task_name}')
                    os.remove(palette_image_path)
                    shutil.rmtree(video_image_folder)

    eval_env.shutdown()

    if logging:
        csv_fp.close()


    return scores


def _parse_low_dim_state(low_dim_state):
    if low_dim_state is None:
        return None
    arr = np.asarray(low_dim_state, dtype=np.float32).reshape(-1)
    if arr.size < 23:
        return None
    gripper_open = float(arr[0])
    gripper_pose = arr[15:22]
    pos = gripper_pose[:3]
    quat = gripper_pose[3:7]
    time = float(arr[-1]) if arr.size > 23 else None
    return {
        "gripper_open": gripper_open,
        "pos": pos,
        "quat": quat,
        "time": time,
    }


def _print_episode_state_debug(task_name, eval_env, episode_rollout):
    if not episode_rollout:
        return
    print(f"[debug] {task_name} step state dump:")
    for idx, transition in enumerate(episode_rollout):
        obs = transition.observation
        low = _parse_low_dim_state(obs.get("low_dim_state"))
        drawer_joint = None
        if hasattr(eval_env, "_task") and hasattr(eval_env._task, "_joints"):
            try:
                variation_index = getattr(getattr(eval_env._task, "_scene", None), "_variation_index", 0)
                joints = getattr(eval_env._task, "_joints", [])
                if joints:
                    joint_idx = int(variation_index) % len(joints)
                    drawer_joint = float(joints[joint_idx].get_joint_position())
            except Exception:
                drawer_joint = None
        if low is None:
            print(f"  step={idx:02d} reward={float(transition.reward):.1f} drawer_joint={drawer_joint}")
        else:
            pos = low["pos"]
            quat = low["quat"]
            time = low["time"]
            print(
                "  step={:02d} reward={:.1f} drawer_joint={} pos=[{:.3f},{:.3f},{:.3f}] quat=[{:.3f},{:.3f},{:.3f},{:.3f}] gripper_open={:.3f}{}".format(
                    idx,
                    float(transition.reward),
                    "unknown" if drawer_joint is None else f"{drawer_joint:.4f}",
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                    float(low["gripper_open"]),
                    "" if time is None else f" time={time:.3f}",
                )
            )


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):

    model_paths = []
    assert args.model_name is not None
    model_paths.append(os.path.join(args.model_folder, args.model_name))
    tb = TensorboardManager(args.eval_log_dir)
    for model_path in model_paths:
        tasks_to_eval = deepcopy(args.tasks)
        model_idx = get_model_index(model_path)
        if model_idx is None:
            model_idx = 0

  
        agent = load_agent(
            model_path=model_path,
            exp_cfg_path=args.exp_cfg_path,
            mvt_cfg_path=args.mvt_cfg_path,
            eval_log_dir=args.eval_log_dir,
            device=args.device,
            use_input_place_with_mean=args.use_input_place_with_mean,
        )
        agent._debug_step_state = getattr(args, "debug_step_state", False)

        agent_eval_log_dir = os.path.join(
            args.eval_log_dir, os.path.basename(model_path).split(".")[0]
        )


        os.makedirs(agent_eval_log_dir, exist_ok=True)
        scores = eval(
            agent=agent,
            tasks=tasks_to_eval,
            eval_datafolder=args.eval_datafolder,
            start_episode=args.start_episode,
            eval_episodes=args.eval_episodes,
            episode_length=args.episode_length,
            replay_ground_truth=args.ground_truth,
            collect_repair_after_failure=args.collect_repair_after_failure,
            device=args.device,
            headless=args.headless,
            visualize=args.visualize,
            logging=True,
            log_dir=agent_eval_log_dir,
            verbose=True,
            save_video=args.save_video,
            model_name=args.model_name,
            visualize_root_dir=args.visualize_root_dir,
            save_rollout_dir=args.save_rollout_dir,
        )
        print(f"model {model_path}, scores {scores}")
        task_scores = {}
        for i in range(len(tasks_to_eval)):
            score = scores[i]
            if score == "unknown":
                print(f"skip tensorboard update for {tasks_to_eval[i]}: score is unknown")
                continue
            task_scores[tasks_to_eval[i]] = score

        print("save ", task_scores)
        if task_scores:
            tb.update("eval", model_idx, task_scores)
            tb.writer.flush()

    tb.close()


if __name__ == "__main__":
    parser = get_eval_parser()

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)
    if bool(int(os.environ.get("BRIDGEVLA_COLLECT_SUFFIX_DATA", "0"))) and not args.save_rollout_dir:
        args.save_rollout_dir = os.path.join(args.eval_log_dir, "suffix_collect")
        print(f"[BRIDGE_SUFFIX_COLLECT] save_rollout_dir={args.save_rollout_dir}", flush=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)
