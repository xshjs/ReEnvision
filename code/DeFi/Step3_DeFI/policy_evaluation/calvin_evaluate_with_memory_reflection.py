from collections import Counter
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import timedelta
from typing import Any

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from policy_evaluation.calvin_evaluate import (
    count_success,
    evaluate_policy as original_evaluate_policy,
    get_log_dir,
    get_video_tag,
    load_eval_sequences,
    print_and_save,
    rollout as original_rollout,
)
from policy_evaluation.evaluate_defi_memory_language_conditioned import (
    build_counterfactual_instruction,
    build_counterfactual_reflection_prompt,
    build_augmented_instruction,
    build_postexec_reflection_prompt,
    blend_counterfactual_future,
    infer_mismatch_type,
    mean_pool_feature,
    load_key_memory_rows,
    make_reflection_row,
    retrieve_adapter_memory_topk,
    retrieve_key_memory,
    summarize_important_key_memory,
    summarize_key_memory,
    summarize_recent_memory,
)
from policy_evaluation.defi_memory_models import load_memory_arrays
from policy_evaluation.train_factor_conditioned_future_adapter import FactorConditionedFutureAdapter
from policy_evaluation.train_causal_intervention_adapter import (
    CausalInterventionAdapter,
    FACTORS as CAUSAL_FACTORS,
    SEGMENTS as CAUSAL_SEGMENTS,
)
from policy_evaluation.train_future_manifold_navigator import FutureManifoldNavigator, task_hash
from policy_evaluation.train_top_task_future_manifold import TopTaskFutureManifold
from policy_evaluation.train_learned_future_geometry_field import LearnedFutureGeometryField
from policy_evaluation.train_program_conditioned_mask_head import (
    DEFAULT_TIME_WINDOWS as PROGRAM_MASK_TIME_WINDOWS,
    ProgramConditionedMaskHead,
)
from policy_evaluation.train_unified_future_encoder import UnifiedFutureEncoder
from policy_evaluation.train_joint_future_editor import JointFutureEditor
from policy_evaluation.train_joint_editor_trigger import (
    TriggerMLP as JointEditorTriggerMLP,
    build_feature as build_joint_editor_trigger_feature,
    future_stats as joint_editor_trigger_future_stats,
)
from policy_evaluation.train_token_immune_adapter import TokenImmuneAdapter
from policy_evaluation.train_local_repair_field_scorer import LocalRepairFieldScorer, build_pair_features
from policy_evaluation.train_sufficiency_mask_critic import (
    QCritic as FutureQCritic,
    future_summary_torch as q_future_summary_torch,
    hashed_task as q_hashed_task,
)
from policy_evaluation.train_joint_future_action_energy import JointFutureActionEnergy
from policy_evaluation.train_joint_future_action_policy import JointFutureActionPolicy
from policy_evaluation.train_joint_action_generator_mlp import JointActionGeneratorMLP
from policy_evaluation.train_delta_action_repair_mlp import DeltaActionRepairMLP
from policy_evaluation.train_chunk_effect_phi_mlp import ChunkEffectPhiMLP
from policy_evaluation.train_joint_execution_update_mlp import JointExecutionUpdateMLP
from policy_evaluation.train_suffix_trigger_mlp import SuffixTriggerMLP, build_suffix_trigger_feature
from policy_evaluation.train_factored_belief_action_transition import (
    ActionAdaptationMLP,
    BeliefTransitionMLP,
    ExpectedTransitionMLP,
    FullRSSMTransitionMLP,
    PlanCompatibilityHead,
    RSSMTransitionMLP,
)
from policy_evaluation.train_directional_sensitivity_tcn import DirectionalSensitivityTCN
from policy_evaluation.train_direct_action_correction_tcn import DirectActionCorrectionTCN
from policy_evaluation.train_future_action_dynamics_probe import FutureActionDynamicsProbe
from policy_evaluation.train_joint_future_action_robotics_energy import JointFutureActionRoboticsEnergy
from policy_evaluation.train_joint_hypothesis_repair import JointHypothesisRepairMLP
from policy_evaluation.train_joint_repair_critic import JointRepairCriticMLP
from policy_evaluation.train_future_success_verifier import FutureSuccessVerifier, hashed_text_features, summarize_future
from policy_evaluation.future_energy_model import TaskConditionedFutureEnergy, build_energy_features_torch
from policy_evaluation.score_state_action_dynamics import load_probe as load_state_action_dynamics_probe, score_pair as score_state_action_pair
from policy_evaluation.train_dynamic_coupling_operator import DynamicCouplingOperator
from policy_evaluation.memory_state_guided_adapter import MemoryStateGuidedAdapter
from policy_evaluation.memory_weight_calibrator import MemoryWeightCalibrator, pad_weight_inputs
from policy_evaluation.summary_conditioned_future_adapter import SummaryConditionedFutureAdapter
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
from policy_models.rollout.rollout_video import RolloutVideo
from policy_models.utils.utils import get_all_checkpoints

SUFFIX_MASK_ABLATION_SCHEDULE_SET: set[tuple[int, int, int]] = set()

try:
    from policy_evaluation.failure_warning_rollout import (
        all_object_motion as strong_gt_all_object_motion,
        observed_contact_sets as strong_gt_observed_contact_sets,
        observed_movable_contacts as strong_gt_observed_movable_contacts,
        subtask_diagnostics as strong_gt_subtask_diagnostics,
        )
except Exception:
    strong_gt_all_object_motion = None
    strong_gt_observed_contact_sets = None
    strong_gt_observed_movable_contacts = None
    strong_gt_subtask_diagnostics = None


RELATION_PROBE_TARGETS = ["risk", "contact", "object_motion", "progress", "goal"]
_JOINT_PAIR_SELECTOR_CACHE = {}
_DYNAMIC_COUPLING_CACHE = {}
_JOINT_ACTION_GENERATOR_CACHE = {}
_DELTA_ACTION_REPAIR_CACHE = {}
_FUTURE_ACTION_DYNAMICS_CACHE = {}
_SUFFIX_TRIGGER_CACHE = {}
_SUMMARY_FUTURE_ADAPTER_CACHE = {}
_JOINT_EXECUTION_UPDATE_CACHE = {}
_DIRECTIONAL_SENSITIVITY_TCN_CACHE = {}
_DIRECT_ACTION_CORRECTION_TCN_CACHE = {}


def _structured_binary_mask_from_scores(
    scores: torch.Tensor,
    *,
    patch_cost: float,
    switch_cost: float,
) -> torch.Tensor:
    flat = scores.detach().float().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    keep_cost = flat.cpu().numpy().astype(np.float64)
    patch_c = float(max(0.0, patch_cost))
    switch_c = float(max(0.0, switch_cost))
    dp = np.zeros((n, 2), dtype=np.float64)
    back = np.zeros((n, 2), dtype=np.int64)
    dp[0, 0] = keep_cost[0]
    dp[0, 1] = patch_c
    for i in range(1, n):
        for state in (0, 1):
            unary = keep_cost[i] if state == 0 else patch_c
            stay = dp[i - 1, state]
            switch = dp[i - 1, 1 - state] + switch_c
            if stay <= switch:
                dp[i, state] = stay + unary
                back[i, state] = state
            else:
                dp[i, state] = switch + unary
                back[i, state] = 1 - state
    state = int(np.argmin(dp[-1]))
    mask = np.zeros((n,), dtype=bool)
    for i in range(n - 1, -1, -1):
        mask[i] = state == 1
        state = int(back[i, state])
    return torch.from_numpy(mask).to(device=scores.device).reshape_as(scores)
_JOINT_REPAIR_RUNTIME_CACHE = {}
_JOINT_VALUE_RUNTIME_CACHE = {}


def _call_preserving_rng(fn, *args, **kwargs):
    """Run analysis-only forwards without changing later diffusion sampling."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return fn(*args, **kwargs)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _apply_checkpoint_model_compat_overrides(cfg, checkpoint_path: str) -> None:
    """
    Old Step3 DeFI checkpoints may have been trained with a different action
    chunk length / latent count than the current default config. Read the
    checkpoint training args and override the runtime config before model
    instantiation so loading stays shape-compatible.
    """
    if not checkpoint_path:
        return
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return
    try:
        state = torch.load(str(ckpt), map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Failed to inspect checkpoint for compat overrides: {exc}", flush=True)
        return
    train_args = state.get("args")
    if train_args is None:
        return
    if not isinstance(train_args, dict):
        train_args = OmegaConf.to_container(train_args, resolve=True)
    if not isinstance(train_args, dict):
        return

    act_seq_len = train_args.get("act_seq_len", train_args.get("action_seq_len"))
    num_latents = train_args.get("num_latents")
    with open_dict(cfg):
        if act_seq_len is not None:
            act_seq_len = int(act_seq_len)
            cfg.act_seq_len = act_seq_len
            cfg.model.action_seq_len = act_seq_len
            cfg.model.act_window_size = act_seq_len
            cfg.model.multistep = act_seq_len
            print(f"[INFO] Applied checkpoint act_seq_len override: {act_seq_len}", flush=True)
        if num_latents is not None:
            cfg.model.num_latents = int(num_latents)
            print(f"[INFO] Applied checkpoint num_latents override: {int(num_latents)}", flush=True)


def maybe_make_online_acceptance_row(row):
    if not bool(row.get("joint_pair_selector_used", False)):
        return None
    trace_path = str(
        row.get("collection_trace_path")
        or row.get("target_proxy_trace_path")
        or row.get("joint_pair_selector_trace_path")
        or ""
    )
    if not trace_path:
        return None
    return {
        "sequence_index": int(row.get("sequence_index", -1)),
        "subtask_index": int(row.get("subtask_index", -1)),
        "task": str(row.get("task", "unknown")),
        "success": bool(row.get("success", False)),
        "steps": int(row.get("steps", 0)),
        "used_reflection_retry": bool(row.get("used_reflection_retry", False)),
        "no_qwen_reflection": bool(row.get("no_qwen_reflection", False)),
        "joint_pair_rollout": bool(row.get("joint_pair_rollout", False)),
        "trace_path": trace_path,
        "joint_pair_selector_memory_index": int(row.get("joint_pair_selector_memory_index", -1)),
        "joint_pair_selector_rank": int(row.get("joint_pair_selector_rank", -1)),
        "joint_pair_selector_energy": float(row.get("joint_pair_selector_energy", 0.0)),
        "joint_pair_selector_second_energy": float(row.get("joint_pair_selector_second_energy", 0.0)),
        "joint_pair_selector_energy_margin": float(row.get("joint_pair_selector_energy_margin", 0.0)),
        "joint_pair_selector_fused_score": float(row.get("joint_pair_selector_fused_score", 0.0)),
        "joint_pair_selector_fused_margin": float(row.get("joint_pair_selector_fused_margin", 0.0)),
        "joint_pair_selector_similarity": float(row.get("joint_pair_selector_similarity", 0.0)),
        "joint_pair_selector_used_dynamics": bool(row.get("joint_pair_selector_used_dynamics", False)),
        "joint_pair_selector_dynamics_score": float(row.get("dynamics_score", 0.0)),
        "joint_pair_selector_dynamics_cos": float(row.get("dynamics_cos", 0.0)),
        "joint_pair_selector_dynamics_l1": float(row.get("dynamics_l1", 0.0)),
        "joint_pair_selector_gate_passed": bool(row.get("joint_pair_selector_gate_passed", False)),
        "joint_pair_selector_gate_reason": str(row.get("joint_pair_selector_gate_reason", "")),
        "joint_pair_selector_outcome": str(row.get("joint_pair_selector_outcome", "")),
        "joint_pair_selector_factor": str(row.get("joint_pair_selector_factor", "")),
        "joint_pair_repair_used": bool(row.get("joint_pair_repair_used", False)),
        "joint_pair_repair_future_mix": float(row.get("joint_pair_repair_future_mix", 0.0)),
        "joint_pair_repair_action_mix": float(row.get("joint_pair_repair_action_mix", 0.0)),
        "joint_pair_repair_future_shift_norm": float(row.get("joint_pair_repair_future_shift_norm", 0.0)),
        "joint_pair_repair_action_shift_norm": float(row.get("joint_pair_repair_action_shift_norm", 0.0)),
        "joint_pair_verify_used": bool(row.get("joint_pair_verify_used", False)),
        "joint_pair_verify_passed": bool(row.get("joint_pair_verify_passed", False)),
        "joint_pair_verify_reason": str(row.get("joint_pair_verify_reason", "")),
        "joint_pair_verify_c_exp": float(row.get("joint_pair_verify_c_exp", 0.0)),
        "joint_pair_verify_c_latent": float(row.get("joint_pair_verify_c_latent", 0.0)),
        "joint_pair_verify_c_physical": float(row.get("joint_pair_verify_c_physical", 0.0)),
        "joint_pair_verify_delta_c_exp": float(row.get("joint_pair_verify_delta_c_exp", 0.0)),
        "joint_pair_verify_delta_c_latent": float(row.get("joint_pair_verify_delta_c_latent", 0.0)),
        "joint_pair_verify_delta_c_physical": float(row.get("joint_pair_verify_delta_c_physical", 0.0)),
        "joint_editor_triggered": bool(row.get("joint_editor_triggered", False)),
        "joint_editor_trigger_probability": float(row.get("joint_editor_trigger_probability", 0.0)),
        "joint_editor_trigger_threshold": float(row.get("joint_editor_trigger_threshold", 0.0)),
    }


def _zero_task_vec(dim: int) -> np.ndarray:
    return np.zeros((int(dim),), dtype=np.float32)


def _dynamic_coupling_task_vec(task: str, dim: int) -> np.ndarray:
    vec = np.zeros((int(dim),), dtype=np.float32)
    key = hashlib.md5(str(task).encode("utf-8")).hexdigest()
    hid = int(key[:8], 16) % int(dim)
    vec[hid] = 1.0
    return vec


def load_dynamic_coupling_operator(cfg, device):
    path = str(getattr(cfg, "dynamic_coupling_operator_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _DYNAMIC_COUPLING_CACHE:
        return _DYNAMIC_COUPLING_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    model = DynamicCouplingOperator(
        state_dim=int(ckpt["state_dim"]),
        future_dim=int(ckpt["future_dim"]),
        progress_dim=int(ckpt.get("progress_dim", 1)),
        action_shape=tuple(int(x) for x in ckpt["action_shape"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        history_len=int(ckpt["history_len"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    bundle = {
        "model": model,
        "state_dim": int(ckpt["state_dim"]),
        "future_dim": int(ckpt["future_dim"]),
        "progress_dim": int(ckpt.get("progress_dim", 1)),
        "action_shape": tuple(int(x) for x in ckpt["action_shape"]),
        "task_dim": int(ckpt["task_dim"]),
        "history_len": int(ckpt["history_len"]),
    }
    _DYNAMIC_COUPLING_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded dynamic coupling operator: {path}", flush=True)
    return bundle


def _build_dynamic_coupling_history_actions(recent_actions, history_len: int, chunk_len: int, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    hist = np.zeros((history_len, chunk_len, action_dim), dtype=np.float32)
    mask = np.zeros((history_len,), dtype=np.float32)
    if not recent_actions:
        return hist, mask
    flat = [np.asarray(a, dtype=np.float32).reshape(-1)[:action_dim] for a in recent_actions]
    total_needed = history_len * chunk_len
    flat = flat[-total_needed:]
    num = len(flat)
    for i in range(history_len):
        start = max(0, num - (history_len - i) * chunk_len)
        end = max(0, num - (history_len - i - 1) * chunk_len)
        chunk = flat[start:end]
        if not chunk:
            continue
        chunk_arr = np.zeros((chunk_len, action_dim), dtype=np.float32)
        take = min(len(chunk), chunk_len)
        chunk_arr[-take:] = np.stack(chunk[-take:], axis=0)
        hist[i] = chunk_arr
        mask[i] = 1.0
    return hist, mask


def _coupled_future_delta_from_committed(
    target_key: str,
    dt: torch.Tensor,
    committed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal structural bias for the coupled correction operator.

    We do not treat committed effect as an external module in the execution path.
    Instead, correction is parameterized through:

      d_committed = C_theta(s, H, z_a)
      delta_z_f   = f_theta(d_target - d_committed)

    and the caller applies:

      z_f' = z_f + delta_z_f
    """
    if target_key in {"pending_effect", "residual_to_target"}:
        d_committed = dt - committed
        delta_z_f = committed
    elif target_key == "observed_effect":
        d_committed = committed
        delta_z_f = dt - committed
    elif target_key == "residual_before":
        d_committed = torch.zeros_like(committed)
        delta_z_f = committed
    else:
        d_committed = committed
        delta_z_f = dt - d_committed
    return d_committed, delta_z_f


def apply_dynamic_coupling_future(
    cfg,
    env,
    model,
    obs,
    lang_text: str,
    task: str,
    subtask_index: int,
    slow_target_future: torch.Tensor,
    recent_actions,
):
    bundle = load_dynamic_coupling_operator(cfg, model.device)
    if bundle is None:
        return slow_target_future, None, {"dynamic_coupling_used": False, "dynamic_coupling_reason": "missing_ckpt"}
    from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

    raw = snapshot_env_raw(env)
    current_state = _joint_pair_state_from_raw(raw)
    if current_state is None:
        return slow_target_future, None, {"dynamic_coupling_used": False, "dynamic_coupling_reason": "missing_state"}
    current_future = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
    if bool(getattr(cfg, "dynamic_coupling_gate_with_trigger", False)):
        trigger_row = {
            "task": task,
            "failure_factor": infer_diagnosis_factor_from_task(task),
            "success": False,
            "steps": 0,
        }
        trigger_probability, trigger_info = predict_joint_editor_trigger(cfg, trigger_row, current_future, task)
        threshold = float(getattr(cfg, "dynamic_coupling_trigger_threshold", getattr(cfg, "joint_editor_trigger_threshold", 0.5)))
        if trigger_probability is None or float(trigger_probability) < threshold:
            info = {
                "dynamic_coupling_used": False,
                "dynamic_coupling_applied": False,
                "dynamic_coupling_reason": f"trigger<{threshold:g}",
                "dynamic_coupling_trigger_probability": None if trigger_probability is None else float(trigger_probability),
                "dynamic_coupling_trigger_threshold": float(threshold),
                **trigger_info,
            }
            return slow_target_future, None, info
    current_summary_exec = _joint_pair_future_summary_exec(current_future.detach().float().cpu().numpy().astype(np.float32))
    target_summary_exec = _joint_pair_future_summary_exec(slow_target_future.detach().float().cpu().numpy().astype(np.float32))
    target_key = str(bundle.get("target_key", "progress") or "progress")
    chunk_len, action_dim = bundle["action_shape"]
    hist_actions_np, hist_mask_np = _build_dynamic_coupling_history_actions(recent_actions, int(bundle["history_len"]), int(chunk_len), int(action_dim))
    state_t = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).reshape(1, -1).to(model.device)
    base_t = torch.from_numpy(np.asarray(current_summary_exec, dtype=np.float32)).reshape(1, -1).to(model.device)
    target_t = torch.from_numpy(np.asarray(target_summary_exec, dtype=np.float32)).reshape(1, -1).to(model.device)
    hist_actions_t = torch.from_numpy(hist_actions_np).unsqueeze(0).to(model.device)
    hist_mask_t = torch.from_numpy(hist_mask_np).reshape(1, -1).to(model.device)
    task_vec_t = torch.from_numpy(_dynamic_coupling_task_vec(task, int(bundle["task_dim"]))).reshape(1, -1).to(model.device)
    sub_t = torch.tensor([[float(subtask_index or 0) / 5.0]], dtype=torch.float32, device=model.device)
    with torch.no_grad():
        out = bundle["model"](state_t, base_t, target_t, hist_actions_t, hist_mask_t, task_vec_t, sub_t)
        committed = out["committed"].reshape(-1)
        min_committed = float(getattr(cfg, "dynamic_coupling_min_committed", 0.0))
        need_gain = float(getattr(cfg, "dynamic_coupling_need_gain", 0.35))
        min_scale = float(getattr(cfg, "dynamic_coupling_min_scale", 0.5))
        max_scale = float(getattr(cfg, "dynamic_coupling_max_scale", 1.25))
        baseline_gap_norm = float(torch.norm((slow_target_future - current_future).reshape(-1), p=2).item())
        hist_mask_list = hist_mask_np.astype(float).tolist()
        alpha_list = out["alpha"].reshape(-1).detach().cpu().numpy().astype(float).tolist()
        committed_norm = float(torch.norm(committed, p=2).item()) if committed.numel() > 0 else 0.0

        vector_mode = int(bundle.get("progress_dim", 1)) == int(base_t.shape[-1])
        corrected_future = slow_target_future
        corrected_summary_exec = None
        future_mix = float(getattr(cfg, "dynamic_coupling_future_mix", 1.0))
        future_mix = float(np.clip(future_mix, 0.0, 1.0))
        auto_shrink = bool(getattr(cfg, "dynamic_coupling_auto_shrink", False))
        max_shift_norm = float(getattr(cfg, "dynamic_coupling_max_future_shift_norm", 1e9))
        max_shift_ratio = float(getattr(cfg, "dynamic_coupling_max_future_shift_ratio", 1e9))

        def _maybe_shrink(raw_corrected_future: torch.Tensor):
            proposed = slow_target_future + future_mix * (raw_corrected_future - slow_target_future)
            raw_shift_norm = float(torch.norm((raw_corrected_future - slow_target_future).reshape(-1), p=2).item())
            shift_norm = float(torch.norm((proposed - slow_target_future).reshape(-1), p=2).item())
            shift_ratio = shift_norm / max(baseline_gap_norm, 1e-6)
            if shift_norm <= max_shift_norm and shift_ratio <= max_shift_ratio:
                return proposed, {
                    "dynamic_coupling_shrunk": False,
                    "dynamic_coupling_shrink_scale": 1.0,
                    "dynamic_coupling_raw_shift_norm_vs_slow": raw_shift_norm,
                    "dynamic_coupling_shift_norm_vs_slow": shift_norm,
                    "dynamic_coupling_shift_ratio_vs_slow": shift_ratio,
                }, None
            if not auto_shrink:
                reason = []
                if shift_norm > max_shift_norm:
                    reason.append(f"shift_norm>{max_shift_norm:g}")
                if shift_ratio > max_shift_ratio:
                    reason.append(f"shift_ratio>{max_shift_ratio:g}")
                return None, {
                    "dynamic_coupling_shrunk": False,
                    "dynamic_coupling_shrink_scale": 1.0,
                    "dynamic_coupling_raw_shift_norm_vs_slow": raw_shift_norm,
                    "dynamic_coupling_shift_norm_vs_slow": shift_norm,
                    "dynamic_coupling_shift_ratio_vs_slow": shift_ratio,
                }, ";".join(reason)
            shrink_scale = 1.0
            if shift_norm > max_shift_norm and shift_norm > 1e-8:
                shrink_scale = min(shrink_scale, max_shift_norm / shift_norm)
            if shift_ratio > max_shift_ratio and shift_ratio > 1e-8:
                shrink_scale = min(shrink_scale, max_shift_ratio / shift_ratio)
            shrink_scale = float(np.clip(shrink_scale, 0.0, 1.0))
            shrunk = slow_target_future + shrink_scale * (proposed - slow_target_future)
            shrunk_shift_norm = float(torch.norm((shrunk - slow_target_future).reshape(-1), p=2).item())
            shrunk_shift_ratio = shrunk_shift_norm / max(baseline_gap_norm, 1e-6)
            return shrunk, {
                "dynamic_coupling_shrunk": True,
                "dynamic_coupling_shrink_scale": shrink_scale,
                "dynamic_coupling_raw_shift_norm_vs_slow": raw_shift_norm,
                "dynamic_coupling_shift_norm_vs_slow": shrunk_shift_norm,
                "dynamic_coupling_shift_ratio_vs_slow": shrunk_shift_ratio,
            }, None

        if vector_mode:
            dt = (target_t - base_t).reshape(-1)
            d_committed, delta_z_f = _coupled_future_delta_from_committed(target_key, dt, committed)
            corrected_summary_t = base_t.reshape(-1) + delta_z_f
            corrected_summary_exec = corrected_summary_t.detach().cpu().numpy().astype(np.float32)
            if committed_norm < min_committed:
                return slow_target_future, None, {
                    "dynamic_coupling_used": True,
                    "dynamic_coupling_applied": False,
                    "dynamic_coupling_reason": f"committed_norm<{min_committed:g}",
                    "dynamic_coupling_vector_mode": True,
                    "dynamic_coupling_internal_bias": "delta_z_f=f_theta(d_target-d_committed)",
                    "dynamic_coupling_target_key": target_key,
                    "dynamic_coupling_committed_norm": committed_norm,
                    "dynamic_coupling_dt_norm": float(torch.norm(dt, p=2).item()),
                    "dynamic_coupling_dneed_norm": float(torch.norm(delta_z_f, p=2).item()),
                    "dynamic_coupling_hist_mask": hist_mask_list,
                    "dynamic_coupling_alpha_mean": alpha_list,
                    "dynamic_coupling_current_to_target_norm": baseline_gap_norm,
                }
            adapter_bundle = load_summary_future_adapter(cfg, model.device)
            if adapter_bundle is None:
                return slow_target_future, None, {
                    "dynamic_coupling_used": True,
                    "dynamic_coupling_applied": False,
                    "dynamic_coupling_reason": "missing_summary_future_adapter",
                    "dynamic_coupling_vector_mode": True,
                    "dynamic_coupling_internal_bias": "delta_z_f=f_theta(d_target-d_committed)",
                    "dynamic_coupling_target_key": target_key,
                    "dynamic_coupling_committed_norm": committed_norm,
                    "dynamic_coupling_hist_mask": hist_mask_list,
                    "dynamic_coupling_alpha_mean": alpha_list,
                }
            try:
                corrected_future_t, adapter_aux = adapter_bundle["adapter"](
                    current_future.unsqueeze(0).to(dtype=slow_target_future.dtype),
                    base_t.to(dtype=slow_target_future.dtype),
                    corrected_summary_t.unsqueeze(0).to(dtype=slow_target_future.dtype),
                )
                raw_corrected_future = corrected_future_t.squeeze(0).detach()
                corrected_future, shrink_info, reject_reason = _maybe_shrink(raw_corrected_future)
                if corrected_future is None:
                    return slow_target_future, None, {
                        "dynamic_coupling_used": True,
                        "dynamic_coupling_applied": False,
                        "dynamic_coupling_reason": str(reject_reason),
                        "dynamic_coupling_vector_mode": True,
                        "dynamic_coupling_internal_bias": "delta_z_f=f_theta(d_target-d_committed)",
                        "dynamic_coupling_target_key": target_key,
                        "dynamic_coupling_committed_norm": committed_norm,
                        "dynamic_coupling_dt_norm": float(torch.norm(dt, p=2).item()),
                        "dynamic_coupling_dneed_norm": float(torch.norm(delta_z_f, p=2).item()),
                        "dynamic_coupling_hist_mask": hist_mask_list,
                        "dynamic_coupling_alpha_mean": alpha_list,
                        "dynamic_coupling_current_to_target_norm": baseline_gap_norm,
                        "dynamic_coupling_future_mix": future_mix,
                        **shrink_info,
                    }
                info = {
                    "dynamic_coupling_used": True,
                    "dynamic_coupling_applied": True,
                    "dynamic_coupling_vector_mode": True,
                    "dynamic_coupling_internal_bias": "delta_z_f=f_theta(d_target-d_committed)",
                    "dynamic_coupling_target_key": target_key,
                    "dynamic_coupling_committed_norm": committed_norm,
                    "dynamic_coupling_dt_norm": float(torch.norm(dt, p=2).item()),
                    "dynamic_coupling_dneed_norm": float(torch.norm(delta_z_f, p=2).item()),
                    "dynamic_coupling_hist_mask": hist_mask_list,
                    "dynamic_coupling_alpha_mean": alpha_list,
                    "dynamic_coupling_current_to_target_norm": baseline_gap_norm,
                    "dynamic_coupling_future_mix": future_mix,
                    "dynamic_coupling_final_shift_norm": float(torch.norm((corrected_future - current_future).reshape(-1), p=2).item()),
                    "dynamic_coupling_adapter_gate_mean": float(adapter_aux["gate"].mean().item()),
                    "dynamic_coupling_adapter_delta_norm": float(torch.norm(adapter_aux["delta"].reshape(-1), p=2).item()),
                    "dynamic_coupling_trigger_probability": None,
                    **shrink_info,
                }
            except Exception as exc:
                return slow_target_future, None, {
                    "dynamic_coupling_used": True,
                    "dynamic_coupling_applied": False,
                    "dynamic_coupling_reason": f"summary_future_adapter_error:{exc}",
                    "dynamic_coupling_vector_mode": True,
                    "dynamic_coupling_internal_bias": "delta_z_f=f_theta(d_target-d_committed)",
                    "dynamic_coupling_target_key": target_key,
                    "dynamic_coupling_committed_norm": committed_norm,
                    "dynamic_coupling_hist_mask": hist_mask_list,
                    "dynamic_coupling_alpha_mean": alpha_list,
                }
        else:
            committed_scalar = float(committed[0].item()) if committed.numel() > 0 else 0.0
            if abs(committed_scalar) < min_committed:
                return slow_target_future, None, {
                    "dynamic_coupling_used": True,
                    "dynamic_coupling_applied": False,
                    "dynamic_coupling_reason": f"abs_committed<{min_committed:g}",
                    "dynamic_coupling_vector_mode": False,
                    "dynamic_coupling_committed_scalar": committed_scalar,
                    "dynamic_coupling_hist_mask": hist_mask_list,
                    "dynamic_coupling_alpha_mean": alpha_list,
                    "dynamic_coupling_current_to_target_norm": baseline_gap_norm,
                }
            need_scale = float(np.clip(1.0 - need_gain * np.tanh(committed_scalar), min_scale, max_scale))
            raw_corrected_future = current_future + need_scale * (slow_target_future - current_future)
            corrected_future, shrink_info, reject_reason = _maybe_shrink(raw_corrected_future)
            if corrected_future is None:
                return slow_target_future, None, {
                    "dynamic_coupling_used": True,
                    "dynamic_coupling_applied": False,
                    "dynamic_coupling_reason": str(reject_reason),
                    "dynamic_coupling_vector_mode": False,
                    "dynamic_coupling_committed_scalar": committed_scalar,
                    "dynamic_coupling_need_scale": need_scale,
                    "dynamic_coupling_hist_mask": hist_mask_list,
                    "dynamic_coupling_alpha_mean": alpha_list,
                    "dynamic_coupling_current_to_target_norm": baseline_gap_norm,
                    "dynamic_coupling_future_mix": future_mix,
                    **shrink_info,
                }
            corrected_summary_exec = _joint_pair_future_summary_exec(corrected_future.detach().float().cpu().numpy().astype(np.float32))
            info = {
                "dynamic_coupling_used": True,
                "dynamic_coupling_applied": True,
                "dynamic_coupling_vector_mode": False,
                "dynamic_coupling_committed_scalar": committed_scalar,
                "dynamic_coupling_need_scale": need_scale,
                "dynamic_coupling_hist_mask": hist_mask_list,
                "dynamic_coupling_alpha_mean": alpha_list,
                "dynamic_coupling_current_to_target_norm": baseline_gap_norm,
                "dynamic_coupling_future_mix": future_mix,
                "dynamic_coupling_final_shift_norm": float(torch.norm((corrected_future - current_future).reshape(-1), p=2).item()),
                "dynamic_coupling_trigger_probability": None,
                **shrink_info,
            }
    if corrected_summary_exec is None:
        corrected_summary_exec = _joint_pair_future_summary_exec(corrected_future.detach().float().cpu().numpy().astype(np.float32))
    action_intent = None
    action_info = {"dynamic_coupling_action_generated": False}
    if bool(getattr(cfg, "dynamic_coupling_generate_action_intent", True)):
        try:
            action_gen_bundle = load_joint_action_generator(cfg, model.device)
            if action_gen_bundle is None:
                action_gen_bundle = load_joint_pair_selector(cfg, model.device)
            if action_gen_bundle is not None:
                task_dim = None
                if "action_generator" in action_gen_bundle:
                    task_dim = int(getattr(action_gen_bundle["action_generator"], "task_dim", 128))
                if task_dim is None:
                    task_dim = int(action_gen_bundle.get("task_dim", 128))
                task_vec = np.asarray(q_hashed_task(task, task_dim), dtype=np.float32)
                sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)
                target_delta_exec = np.asarray(corrected_summary_exec - current_summary_exec, dtype=np.float32)
                generated_action, action_info = _joint_pair_generate_action_init(
                    action_gen_bundle,
                    base_summary_exec=np.asarray(current_summary_exec, dtype=np.float32),
                    target_summary_exec=np.asarray(corrected_summary_exec, dtype=np.float32),
                    target_delta_exec=target_delta_exec,
                    task_vec=task_vec,
                    sub=sub,
                    current_state=np.asarray(current_state, dtype=np.float32),
                    device=model.device,
                )
                if generated_action is not None:
                    action_intent = generated_action.to(device=model.device, dtype=slow_target_future.dtype)
                    action_info = dict(action_info)
                    action_info["dynamic_coupling_action_generated"] = True
                    action_info["dynamic_coupling_action_source"] = "corrected_future_summary"
        except Exception as exc:
            action_info = {
                "dynamic_coupling_action_generated": False,
                "dynamic_coupling_action_error": str(exc)[:300],
            }
    info.update(action_info)
    return corrected_future.detach(), action_intent, info


def apply_dynamic_coupling_candidate_prior(
    cfg,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    base_future: torch.Tensor,
    slow_target_future: torch.Tensor,
    candidate_action_intent: torch.Tensor | None,
):
    if current_state is None or candidate_action_intent is None:
        return slow_target_future, candidate_action_intent, {
            "joint_pair_coupling_prior_used": False,
            "joint_pair_coupling_prior_reason": "missing_state_or_action",
        }
    bundle = load_dynamic_coupling_operator(cfg, base_future.device)
    if bundle is None:
        return slow_target_future, candidate_action_intent, {
            "joint_pair_coupling_prior_used": False,
            "joint_pair_coupling_prior_reason": "missing_ckpt",
        }
    try:
        current_summary_exec = _joint_pair_future_summary_exec(
            base_future.detach().float().cpu().numpy().astype(np.float32)
        )
        target_summary_exec = _joint_pair_future_summary_exec(
            slow_target_future.detach().float().cpu().numpy().astype(np.float32)
        )
        target_key = str(bundle.get("target_key", "progress") or "progress")
        action_np = candidate_action_intent.detach().float().cpu().numpy().astype(np.float32)
        pseudo_recent_actions = [row.copy() for row in action_np]
        chunk_len, action_dim = bundle["action_shape"]
        hist_actions_np, hist_mask_np = _build_dynamic_coupling_history_actions(
            pseudo_recent_actions,
            int(bundle["history_len"]),
            int(chunk_len),
            int(action_dim),
        )
        state_t = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).reshape(1, -1).to(base_future.device)
        base_t = torch.from_numpy(np.asarray(current_summary_exec, dtype=np.float32)).reshape(1, -1).to(base_future.device)
        target_t = torch.from_numpy(np.asarray(target_summary_exec, dtype=np.float32)).reshape(1, -1).to(base_future.device)
        hist_actions_t = torch.from_numpy(hist_actions_np).unsqueeze(0).to(base_future.device)
        hist_mask_t = torch.from_numpy(hist_mask_np).reshape(1, -1).to(base_future.device)
        task_vec_t = torch.from_numpy(_dynamic_coupling_task_vec(task, int(bundle["task_dim"]))).reshape(1, -1).to(base_future.device)
        sub_t = torch.tensor([[float(subtask_index or 0) / 5.0]], dtype=torch.float32, device=base_future.device)
        with torch.no_grad():
            out = bundle["model"](state_t, base_t, target_t, hist_actions_t, hist_mask_t, task_vec_t, sub_t)
            committed = out["committed"].reshape(-1)
            min_committed = float(getattr(cfg, "dynamic_coupling_min_committed", 0.0))
            committed_norm = float(torch.norm(committed, p=2).item()) if committed.numel() > 0 else 0.0
            if committed_norm < min_committed:
                return slow_target_future, candidate_action_intent, {
                    "joint_pair_coupling_prior_used": True,
                    "joint_pair_coupling_prior_applied": False,
                    "joint_pair_coupling_prior_reason": f"committed_norm<{min_committed:g}",
                    "joint_pair_coupling_prior_committed_norm": committed_norm,
                }
            vector_mode = int(bundle.get("progress_dim", 1)) == int(base_t.shape[-1])
            if vector_mode:
                dt = (target_t - base_t).reshape(-1)
                if target_key in {"pending_effect", "residual_to_target"}:
                    d_need = committed
                elif target_key == "observed_effect":
                    d_need = dt - committed
                elif target_key == "residual_before":
                    d_need = committed
                else:
                    d_need = dt - committed
                corrected_summary_t = base_t.reshape(-1) + d_need
                adapter_bundle = load_summary_future_adapter(cfg, base_future.device)
                if adapter_bundle is None:
                    return slow_target_future, candidate_action_intent, {
                        "joint_pair_coupling_prior_used": True,
                        "joint_pair_coupling_prior_applied": False,
                        "joint_pair_coupling_prior_reason": "missing_summary_future_adapter",
                        "joint_pair_coupling_prior_committed_norm": committed_norm,
                        "joint_pair_coupling_prior_vector_mode": True,
                        "joint_pair_coupling_prior_target_key": target_key,
                    }
                corrected_future_t, _ = adapter_bundle["adapter"](
                    base_future.unsqueeze(0).to(dtype=slow_target_future.dtype),
                    base_t.to(dtype=slow_target_future.dtype),
                    corrected_summary_t.unsqueeze(0).to(dtype=slow_target_future.dtype),
                )
                corrected_future = corrected_future_t.squeeze(0).detach()
            else:
                committed_scalar = float(committed[0].item()) if committed.numel() > 0 else 0.0
                min_scale = float(getattr(cfg, "dynamic_coupling_min_scale", 0.5))
                max_scale = float(getattr(cfg, "dynamic_coupling_max_scale", 1.25))
                need_gain = float(getattr(cfg, "dynamic_coupling_need_gain", 0.35))
                need_scale = float(np.clip(1.0 - need_gain * np.tanh(committed_scalar), min_scale, max_scale))
                corrected_future = base_future + need_scale * (slow_target_future - base_future)
            action_gen_bundle = load_joint_action_generator(cfg, base_future.device)
            corrected_action = candidate_action_intent
            action_generated = False
            if action_gen_bundle is not None:
                task_dim = int(getattr(action_gen_bundle["action_generator"], "task_dim", 128))
                task_vec = np.asarray(q_hashed_task(task, task_dim), dtype=np.float32)
                sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)
                corrected_summary_exec = _joint_pair_future_summary_exec(
                    corrected_future.detach().float().cpu().numpy().astype(np.float32)
                )
                generated_action, _ = _joint_pair_generate_action_init(
                    action_gen_bundle,
                    base_summary_exec=np.asarray(current_summary_exec, dtype=np.float32),
                    target_summary_exec=np.asarray(corrected_summary_exec, dtype=np.float32),
                    target_delta_exec=np.asarray(corrected_summary_exec - current_summary_exec, dtype=np.float32),
                    task_vec=task_vec,
                    sub=sub,
                    current_state=np.asarray(current_state, dtype=np.float32),
                    device=base_future.device,
                )
                if generated_action is not None:
                    corrected_action = generated_action.to(device=base_future.device, dtype=candidate_action_intent.dtype)
                    action_generated = True
        return corrected_future, corrected_action, {
            "joint_pair_coupling_prior_used": True,
            "joint_pair_coupling_prior_applied": True,
            "joint_pair_coupling_prior_committed_norm": committed_norm,
            "joint_pair_coupling_prior_vector_mode": bool(vector_mode),
            "joint_pair_coupling_prior_target_key": target_key,
            "joint_pair_coupling_prior_future_shift_norm": float(torch.norm((corrected_future - slow_target_future).reshape(-1), p=2).item()),
            "joint_pair_coupling_prior_action_generated": bool(action_generated),
            "joint_pair_coupling_prior_action_shift_norm": float(torch.norm((corrected_action - candidate_action_intent).reshape(-1), p=2).item()),
        }
    except Exception as exc:
        return slow_target_future, candidate_action_intent, {
            "joint_pair_coupling_prior_used": True,
            "joint_pair_coupling_prior_applied": False,
            "joint_pair_coupling_prior_reason": f"error:{exc}",
        }


def apply_coupled_correction(
    cfg,
    *,
    mode: str,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None = None,
    current_future: torch.Tensor | None = None,
    current_action: torch.Tensor | None = None,
    pending_effect: torch.Tensor | np.ndarray | None = None,
    base_future_for_score: torch.Tensor | None = None,
    target_transition: np.ndarray | None = None,
    env=None,
    model=None,
    obs=None,
    lang_text: str | None = None,
    slow_target_future: torch.Tensor | None = None,
    recent_actions=None,
):
    """
    Unified coupled correction interface.

    - slow: one-shot larger correction on h=(z_f, z_a) before rollout / retry.
    - fast: small online correction during rollout using recent action history.
    """
    if mode == "slow":
        future_t, action_t, info = apply_joint_repair_advantage_gate(
            cfg,
            task=task,
            subtask_index=int(subtask_index or 0),
            current_state=current_state,
            current_future=current_future,
            current_action=current_action,
            pending_effect=pending_effect,
            base_future_for_score=base_future_for_score,
            target_transition=target_transition,
        )
        return future_t, action_t, {
            "coupled_correction_used": True,
            "coupled_correction_mode": "slow",
            "coupled_correction_history_len": 0,
            **info,
        }
    if mode == "fast":
        future_t, action_t, info = apply_dynamic_coupling_future(
            cfg,
            env,
            model,
            obs,
            str(lang_text or ""),
            task,
            int(subtask_index or 0),
            slow_target_future,
            recent_actions or [],
        )
        return future_t, action_t, {
            "coupled_correction_used": True,
            "coupled_correction_mode": "fast",
            "coupled_correction_history_len": int(len(recent_actions or [])),
            **info,
        }
    return current_future, current_action, {
        "coupled_correction_used": False,
        "coupled_correction_mode": str(mode),
        "coupled_correction_reason": "unknown_mode",
    }


def maybe_apply_initial_slow_correction(
    cfg,
    *,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    base_future: torch.Tensor | None,
    initial_future: torch.Tensor | None,
    initial_action: torch.Tensor | None,
    target_transition: np.ndarray | None = None,
):
    """
    Run the one-shot slow coupled correction on the initial joint hypothesis
    h0=(z_f^0, z_a^0) before rollout starts.
    """
    if initial_future is None or initial_action is None:
        return initial_future, initial_action, {
            "coupled_correction_used": False,
            "coupled_correction_mode": "slow",
            "coupled_correction_reason": "missing_initial_hypothesis",
        }
    if not bool(getattr(cfg, "joint_pair_repair_ckpt", "") or ""):
        return initial_future, initial_action, {
            "coupled_correction_used": False,
            "coupled_correction_mode": "slow",
            "coupled_correction_reason": "missing_slow_repair_ckpt",
        }
    return apply_coupled_correction(
        cfg,
        mode="slow",
        task=task,
        subtask_index=int(subtask_index or 0),
        current_state=current_state,
        current_future=initial_future,
        current_action=initial_action,
        pending_effect=None,
        base_future_for_score=base_future,
        target_transition=target_transition,
    )


class RelationFutureProbeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(RELATION_PROBE_TARGETS)),
        )

    def forward(self, x):
        return self.net(x)


logger = logging.getLogger(__name__)


RULE_MEMORY = {
    "default": "Prefer conservative future calibration and preserve verified progress.",
}


MISMATCH_TO_FACTOR = {
    "contact_not_realized": "contact",
    "object_displacement_overestimated": "object_displacement",
    "object_displacement_underestimated": "object_displacement",
    "wrong_object_identity": "object_identity",
    "drawer_progress_hallucinated": "drawer_slider_progress",
    "future_outcome_mismatch": "goal_completion",
    "goal_progress_hallucinated": "goal_completion",
}

FACTOR_ORDER = [
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
]

RISK_FACTOR_ORDER = ["contact", "drawer_slider_progress", "goal_completion", "object_displacement", "object_identity", "none", "unknown"]

_FACTOR_MASK_CACHE = {}
_RISK_DETECTOR_CACHE = {}
_COUNTERFACTUAL_VERIFIER_CACHE = {}
_IMMUNE_REPULSION_CACHE = {}
_TOKEN_IMMUNE_ADAPTER_CACHE = {}
_LOCAL_REPAIR_FIELD_SCORER_CACHE = {}
_IMMUNE_Q_FILTER_CACHE = {}
_CAUSAL_INTERVENTION_ADAPTER_CACHE = {}
_FUTURE_MANIFOLD_NAVIGATOR_CACHE = {}
_FUTURE_SUCCESS_VERIFIER_CACHE = {}
_FUTURE_ENERGY_CACHE = {}
_FUTURE_ENERGY_MASK_CACHE = {}
_PROGRAM_CONDITIONED_MASK_CACHE = {}
_JOINT_FUTURE_EDITOR_CACHE = {}
_JOINT_EDITOR_TRIGGER_CACHE = {}
_CHANNEL_CAUSAL_PATCH_CACHE = {}
_CHANNEL_GROUP_SELECTOR_CACHE = {}
_CHANNEL_PATCH_EDITOR_CACHE = {}
_CHANNEL_PATCH_Q_CACHE = {}
_RELATION_FUTURE_PROBE_CACHE = {}


class JointHypothesisRepairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, future_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_head = nn.Linear(hidden_dim, future_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        pred_future = self.future_head(h)
        pred_action = self.action_head(h)
        return {
            "pred_future": pred_future,
            "pred_action": pred_action,
            "pred": torch.cat([pred_future, pred_action], dim=-1),
        }


class JointHypothesisValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class JointHypothesisManifoldMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.success_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {
            "z": z,
            "success_logit": self.success_head(z).squeeze(-1),
        }


class CrossStageConnectivityMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = latent_dim * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cur: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(cur - nxt)
        x = torch.cat([cur, nxt, delta], dim=-1)
        return self.net(x).squeeze(-1)


def _read_jsonl_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_collection_trace(row, trace_dir, idx):
    raw = row.get("collection_trace_path") or row.get("target_proxy_trace_path") or row.get("trace_path") or row.get("trace")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
        candidate = Path(trace_dir) / path.name
        if candidate.exists():
            return candidate
    candidate = Path(trace_dir) / f"future_trace_{idx:04d}.npz"
    return candidate if candidate.exists() else None


def _load_future_npz(path, key):
    try:
        with np.load(path) as data:
            if key in data:
                arr = data[key]
            elif "base_future" in data:
                arr = data["base_future"]
            elif "original_base_future" in data:
                arr = data["original_base_future"]
            else:
                return None
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    return arr if arr.ndim == 2 else None


def load_channel_causal_patch_memory(cfg):
    key = (
        str(getattr(cfg, "channel_patch_memory_jsonl", "") or ""),
        str(getattr(cfg, "channel_patch_trace_dir", "") or ""),
        str(getattr(cfg, "channel_patch_future_key", "base_future") or "base_future"),
    )
    if key in _CHANNEL_CAUSAL_PATCH_CACHE:
        return _CHANNEL_CAUSAL_PATCH_CACHE[key]
    jsonl, trace_dir, future_key = key
    if not jsonl or not trace_dir:
        _CHANNEL_CAUSAL_PATCH_CACHE[key] = None
        return None
    rows_raw = _read_jsonl_rows(jsonl)
    rows = []
    futures = []
    for idx, row in enumerate(rows_raw):
        trace = _resolve_collection_trace(row, trace_dir, idx)
        if trace is None:
            continue
        future = _load_future_npz(trace, future_key)
        if future is None:
            continue
        rows.append(dict(row, _trace_path=str(trace), _row_index=idx))
        futures.append(future)
    if not futures:
        _CHANNEL_CAUSAL_PATCH_CACHE[key] = None
        return None
    futures_np = np.stack(futures, axis=0).astype(np.float32)
    pooled = futures_np.mean(axis=1)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    bundle = {
        "rows": rows,
        "futures": futures_np,
        "pooled": pooled,
        "tasks": [str(row.get("task", "unknown")) for row in rows],
        "success_ids": [idx for idx, row in enumerate(rows) if bool(row.get("success"))],
        "failure_ids": [idx for idx, row in enumerate(rows) if not bool(row.get("success"))],
    }
    print(
        "[INFO] Loaded channel causal patch memory: "
        f"rows={len(rows)} success={len(bundle['success_ids'])} failure={len(bundle['failure_ids'])}"
    )
    _CHANNEL_CAUSAL_PATCH_CACHE[key] = bundle
    return bundle


def canonical_energy_factor(factor):
    factor = str(factor or "").strip()
    aliases = {
        "object_pose": "object",
        "object_displacement": "object",
        "object_identity": "object",
        "drawer_slider_progress": "motion",
        "progress": "motion",
        "goal_completion": "goal",
        "future_outcome_mismatch": "goal",
        "none": "goal",
        "unknown": "goal",
    }
    return aliases.get(factor, factor if factor in {"contact", "object", "motion", "goal"} else "goal")


def infer_energy_factor_from_task(task):
    task = str(task or "")
    if any(key in task for key in ("drawer", "slider", "push_")):
        return "motion"
    if any(key in task for key in ("lift_", "stack", "unstack", "place_in")):
        return "contact"
    if any(key in task for key in ("rotate_", "block")):
        return "object"
    if any(key in task for key in ("led", "lightbulb")):
        return "goal"
    return "goal"


def infer_diagnosis_factor_from_task(task):
    task = str(task or "")
    if any(key in task for key in ("open_drawer", "close_drawer", "move_slider", "place_in_slider")):
        return "drawer_slider_progress"
    if "push_" in task or "push_into_drawer" in task:
        return "object_displacement"
    if any(key in task for key in ("lift_", "stack", "unstack", "place_in_drawer")):
        return "contact"
    if any(key in task for key in ("rotate_",)):
        return "object_identity"
    if any(key in task for key in ("led", "lightbulb")):
        return "goal_completion"
    return "goal_completion"


def make_rule_based_reflection_decision(task: str, success: bool, steps: int, ep_len: int, rule_text: str):
    from policy_evaluation.oracle_hypothesis_rollout import MemoryReflectionDecision

    mismatch_type = infer_mismatch_type(task, success, steps, ep_len)
    failure_factor = infer_diagnosis_factor_from_task(task)
    recoverability = "not_needed" if success else "repairable"
    trust_score = 0.85 if success else 0.5
    explanation = rule_text or RULE_MEMORY.get(task, RULE_MEMORY["default"])
    raw_text = (
        f"trust: {'high' if success else 'medium'} | "
        f"mismatch_type: {mismatch_type} | "
        f"failure_factor: {failure_factor} | "
        f"recoverability: {recoverability}"
    )
    return MemoryReflectionDecision(
        trust_score=trust_score,
        mismatch_type=mismatch_type,
        correction_direction="stabilize contact" if not success else "continue",
        recoverability=recoverability,
        explanation=explanation,
        raw_text=raw_text,
        hypothetical_failure="" if success else explanation,
        counterfactual_future="",
        failure_factor=failure_factor if not success else "none",
        intervention="stabilize contact" if not success else "",
    )


def is_push_sensitive_task(task):
    task = str(task or "")
    return any(
        key in task
        for key in (
            "push_into_drawer",
            "push_blue_block_right",
            "push_red_block_right",
            "push_pink_block_right",
            "push_pink_block_left",
            "push_blue_block_left",
            "push_red_block_left",
        )
    )


def _channel_patch_cosine(a, b):
    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(af, bf) / denom)


def filter_eval_sequences_by_task(eval_sequences, task_filter_csv):
    task_filter = parse_csv_set(task_filter_csv)
    if not task_filter:
        return eval_sequences
    filtered = []
    for initial_state, sequence in eval_sequences:
        if any(str(task) in task_filter for task in sequence):
            filtered.append((initial_state, sequence))
    return filtered


class ChannelPatchSelectorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ChannelPatchTokenEditor(nn.Module):
    def __init__(self, future_dim, cond_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.future_proj = nn.Linear(future_dim * 2 + 1, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, future, mask, cond):
        t = future.shape[1]
        pos = torch.linspace(0.0, 1.0, t, device=future.device, dtype=future.dtype)[None, :, None].expand(future.shape[0], t, 1)
        h = self.future_proj(torch.cat([future, mask, pos], dim=-1)) + self.cond_proj(cond)[:, None, :]
        return self.net(h)


def _channel_patch_hashed_task(task, dim):
    vec = np.zeros(int(dim), dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % int(dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def _channel_patch_future_summary(future):
    return np.concatenate([future.mean(axis=0), future.std(axis=0), future[-1] - future[0]], axis=0).astype(np.float32)


def _channel_patch_selector_feature(future, task, subtask_index, task_dim):
    sub = float(subtask_index or 0.0) / 5.0
    return np.concatenate(
        [
            _channel_patch_future_summary(future),
            _channel_patch_hashed_task(task, int(task_dim)),
            np.asarray([sub], dtype=np.float32),
        ]
    ).astype(np.float32)


def _channel_patch_context(task, subtask_index, task_dim):
    sub = float(subtask_index or 0.0) / 5.0
    return np.concatenate(
        [_channel_patch_hashed_task(task, int(task_dim)), np.asarray([sub], dtype=np.float32)]
    ).astype(np.float32)


def _channel_patch_group_mask(group_indices, groups, future_shape):
    mask = np.zeros(future_shape, dtype=np.float32)
    for gi in group_indices:
        if 0 <= int(gi) < len(groups):
            start, end = groups[int(gi)]
            mask[:, int(start) : int(end)] = 1.0
    return mask


def load_channel_group_selector(cfg, device):
    path = str(getattr(cfg, "channel_patch_selector_ckpt", "") or "")
    if not path:
        return None
    key = (path, str(device))
    if key in _CHANNEL_GROUP_SELECTOR_CACHE:
        return _CHANNEL_GROUP_SELECTOR_CACHE[key]
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"[WARN] channel group selector ckpt not found: {path}", flush=True)
        _CHANNEL_GROUP_SELECTOR_CACHE[key] = None
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = ChannelPatchSelectorMLP(
        int(checkpoint["input_dim"]),
        int(checkpoint["num_groups"]),
        int(checkpoint.get("hidden_dim", 512)),
        float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    bundle = {
        "model": model,
        "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["std"], dtype=np.float32),
        "task_dim": int(checkpoint.get("task_dim", 128)),
        "groups": [tuple(map(int, g)) for g in checkpoint.get("groups", [])],
        "path": path,
    }
    print(f"[INFO] Loaded channel group selector: {path}", flush=True)
    _CHANNEL_GROUP_SELECTOR_CACHE[key] = bundle
    return bundle


def load_channel_patch_editor(cfg, device):
    path = str(getattr(cfg, "channel_patch_editor_ckpt", "") or "")
    if not path:
        return None
    key = (path, str(device))
    if key in _CHANNEL_PATCH_EDITOR_CACHE:
        return _CHANNEL_PATCH_EDITOR_CACHE[key]
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"[WARN] channel patch editor ckpt not found: {path}", flush=True)
        _CHANNEL_PATCH_EDITOR_CACHE[key] = None
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = ChannelPatchTokenEditor(
        int(checkpoint["future_dim"]),
        int(checkpoint.get("cond_dim", 129)),
        int(checkpoint.get("hidden_dim", 512)),
        float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    bundle = {
        "model": model,
        "future_dim": int(checkpoint["future_dim"]),
        "cond_dim": int(checkpoint.get("cond_dim", 129)),
        "path": path,
    }
    print(f"[INFO] Loaded channel patch editor: {path}", flush=True)
    _CHANNEL_PATCH_EDITOR_CACHE[key] = bundle
    return bundle


def load_channel_patch_q(cfg, device):
    path = str(getattr(cfg, "channel_patch_q_critic_ckpt", "") or "")
    if not path:
        return None
    key = (path, str(device))
    if key in _CHANNEL_PATCH_Q_CACHE:
        return _CHANNEL_PATCH_Q_CACHE[key]
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"[WARN] channel patch Q critic ckpt not found: {path}", flush=True)
        _CHANNEL_PATCH_Q_CACHE[key] = None
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = ChannelPatchSelectorMLP(
        int(checkpoint["input_dim"]),
        1,
        int(checkpoint.get("hidden_dim", 512)),
        float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    bundle = {
        "model": model,
        "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["std"], dtype=np.float32),
        "task_dim": int(checkpoint.get("task_dim", 128)),
        "path": path,
    }
    print(f"[INFO] Loaded channel patch Q critic: {path}", flush=True)
    _CHANNEL_PATCH_Q_CACHE[key] = bundle
    return bundle


def predict_channel_patch_success(cfg, base_np, task, subtask_index, device):
    q_bundle = load_channel_patch_q(cfg, device)
    if q_bundle is None:
        return None
    feat = _channel_patch_selector_feature(base_np, task, subtask_index, q_bundle["task_dim"])
    x = (feat[None, :] - q_bundle["mean"]) / np.maximum(q_bundle["std"], 1e-6)
    with torch.no_grad():
        logit = q_bundle["model"](torch.from_numpy(x.astype(np.float32)).to(device))
        prob = torch.sigmoid(logit).reshape(-1)[0]
    return float(prob.detach().cpu().item())


def predict_channel_group(cfg, base_np, task, subtask_index, device):
    selector = load_channel_group_selector(cfg, device)
    if selector is None:
        return None
    feat = _channel_patch_selector_feature(base_np, task, subtask_index, selector["task_dim"])
    x = (feat[None, :] - selector["mean"]) / np.maximum(selector["std"], 1e-6)
    with torch.no_grad():
        logits = selector["model"](torch.from_numpy(x.astype(np.float32)).to(device))
        prob = torch.softmax(logits, dim=-1)[0]
        group_idx = int(torch.argmax(prob).item())
        conf = float(prob[group_idx].item())
        topk_n = max(1, int(getattr(cfg, "channel_patch_topk_groups", 1)))
        topk = torch.topk(prob, k=min(max(5, topk_n), prob.numel()))
    return {
        "group_index": group_idx,
        "confidence": conf,
        "top_indices": [int(i) for i in topk.indices.detach().cpu().tolist()],
        "top_probs": [float(v) for v in topk.values.detach().cpu().tolist()],
        "groups": selector["groups"],
        "path": selector["path"],
    }


def _nearest_patch_ids(query, bank, candidates, k):
    if not candidates:
        return [], np.zeros((0,), dtype=np.float32)
    sims = query[None, :] @ bank[candidates].T
    order = np.argsort(-sims.reshape(-1))[: max(1, min(int(k), len(candidates)))]
    ids = [candidates[int(i)] for i in order]
    weights = np.maximum(sims.reshape(-1)[order], 0.0).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones(len(ids), dtype=np.float32) / max(len(ids), 1)
    else:
        weights /= weights.sum()
    return ids, weights


def apply_channel_causal_patch(cfg, base_future, task, subtask_index=0):
    if not bool(getattr(cfg, "channel_causal_patch", False)):
        return base_future, {"channel_causal_patch": False, "channel_causal_patch_used": False}
    bundle = load_channel_causal_patch_memory(cfg)
    if bundle is None:
        return base_future, {
            "channel_causal_patch": True,
            "channel_causal_patch_used": False,
            "channel_causal_patch_reason": "missing_memory",
        }
    base_np = base_future.detach().float().cpu().numpy().astype(np.float32)
    if base_np.ndim != 2:
        return base_future, {
            "channel_causal_patch": True,
            "channel_causal_patch_used": False,
            "channel_causal_patch_reason": "bad_future_shape",
        }
    query = base_np.mean(axis=0)
    query = query / max(float(np.linalg.norm(query)), 1e-8)
    same_task = bool(getattr(cfg, "channel_patch_same_task_only", False))
    tasks = bundle["tasks"]
    success_ids = [idx for idx in bundle["success_ids"] if (not same_task or tasks[idx] == task)]
    failure_ids = [idx for idx in bundle["failure_ids"] if (not same_task or tasks[idx] == task)]
    if not success_ids:
        success_ids = bundle["success_ids"]
    if not failure_ids:
        failure_ids = bundle["failure_ids"]
    if not success_ids or not failure_ids:
        return base_future, {
            "channel_causal_patch": True,
            "channel_causal_patch_used": False,
            "channel_causal_patch_reason": "no_candidates",
        }
    succ_ids, succ_w = _nearest_patch_ids(query, bundle["pooled"], success_ids, int(getattr(cfg, "channel_patch_k_success", 5)))
    fail_ids, fail_w = _nearest_patch_ids(query, bundle["pooled"], failure_ids, int(getattr(cfg, "channel_patch_k_failure", 5)))
    succ_center = np.tensordot(succ_w, bundle["futures"][succ_ids], axes=(0, 0)).astype(np.float32)
    fail_center = np.tensordot(fail_w, bundle["futures"][fail_ids], axes=(0, 0)).astype(np.float32)
    base_success_cos = _channel_patch_cosine(base_np, succ_center)
    base_failure_cos = _channel_patch_cosine(base_np, fail_center)
    group_size = max(1, int(getattr(cfg, "channel_patch_group_size", 32)))
    groups = [(start, min(start + group_size, base_np.shape[-1])) for start in range(0, base_np.shape[-1], group_size)]
    rng = np.random.default_rng(int(getattr(cfg, "channel_patch_random_seed", 0)) + int(getattr(cfg, "_channel_patch_count", 0)))
    mode = str(getattr(cfg, "channel_patch_mode", "best") or "best")
    q_success = predict_channel_patch_success(cfg, base_np, task, subtask_index, base_future.device)
    q_risk = None if q_success is None else float(1.0 - q_success)
    q_threshold = float(getattr(cfg, "channel_patch_risk_threshold", -1.0))
    if q_risk is not None and q_threshold >= 0.0 and q_risk < q_threshold:
        cfg._channel_patch_count = int(getattr(cfg, "_channel_patch_count", 0)) + 1
        return base_future, {
            "channel_causal_patch": True,
            "channel_causal_patch_used": False,
            "channel_causal_patch_reason": "below_q_risk_threshold",
            "channel_causal_patch_mode": mode,
            "channel_causal_patch_q_critic_ckpt": str(getattr(cfg, "channel_patch_q_critic_ckpt", "") or ""),
            "channel_causal_patch_q_success": float(q_success),
            "channel_causal_patch_q_risk": float(q_risk),
            "channel_causal_patch_q_risk_threshold": float(q_threshold),
            "channel_causal_patch_base_success_cos": float(base_success_cos),
            "channel_causal_patch_base_failure_cos": float(base_failure_cos),
        }
    selector_pred = None
    selector_group = None
    selector_groups = []
    if mode in {"selector", "complement"} and str(getattr(cfg, "channel_patch_selector_ckpt", "") or ""):
        selector_pred = predict_channel_group(cfg, base_np, task, subtask_index, base_future.device)
        if selector_pred is not None and selector_pred["groups"]:
            gi = max(0, min(int(selector_pred["group_index"]), len(selector_pred["groups"]) - 1))
            selector_group = selector_pred["groups"][gi]
            topk_groups = max(1, int(getattr(cfg, "channel_patch_topk_groups", 1)))
            selector_groups = [
                max(0, min(int(idx), len(selector_pred["groups"]) - 1))
                for idx in selector_pred["top_indices"][:topk_groups]
            ]
    candidates = []
    for start, end in groups:
        patched = base_np.copy()
        patched[:, start:end] = succ_center[:, start:end]
        success_gain = _channel_patch_cosine(patched, succ_center) - base_success_cos
        failure_repulsion = base_failure_cos - _channel_patch_cosine(patched, fail_center)
        margin_gain = success_gain + failure_repulsion
        candidates.append((margin_gain, success_gain, failure_repulsion, start, end, patched))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if mode == "best":
        margin_gain, success_gain, failure_repulsion, start, end, patched_np = candidates[0]
    elif mode == "selector":
        if selector_group is None or not selector_groups:
            return base_future, {
                "channel_causal_patch": True,
                "channel_causal_patch_used": False,
                "channel_causal_patch_reason": "missing_selector",
            }
        mask_np = _channel_patch_group_mask(selector_groups, selector_pred["groups"], base_np.shape)
        editor = load_channel_patch_editor(cfg, base_future.device)
        if editor is not None:
            cond_dim = int(editor["cond_dim"])
            cond_np = _channel_patch_context(task, subtask_index, max(1, cond_dim - 1))
            with torch.no_grad():
                future_t = torch.from_numpy(base_np[None]).to(device=base_future.device, dtype=torch.float32)
                mask_t = torch.from_numpy(mask_np[None]).to(device=base_future.device, dtype=torch.float32)
                cond_t = torch.from_numpy(cond_np[None]).to(device=base_future.device, dtype=torch.float32)
                delta_t = editor["model"](future_t, mask_t, cond_t) * mask_t
                patched_np = (future_t + delta_t).detach().cpu().numpy()[0].astype(np.float32)
        else:
            patched_np = base_np.copy()
            patched_np[mask_np > 0.0] = succ_center[mask_np > 0.0]
        start = int(min(selector_pred["groups"][gi][0] for gi in selector_groups))
        end = int(max(selector_pred["groups"][gi][1] for gi in selector_groups))
        success_gain = _channel_patch_cosine(patched_np, succ_center) - base_success_cos
        failure_repulsion = base_failure_cos - _channel_patch_cosine(patched_np, fail_center)
        margin_gain = success_gain + failure_repulsion
    elif mode == "random":
        margin_gain, success_gain, failure_repulsion, start, end, patched_np = candidates[int(rng.integers(0, len(candidates)))]
    elif mode == "global":
        patched_np = succ_center
        start, end = 0, base_np.shape[-1]
        success_gain = _channel_patch_cosine(patched_np, succ_center) - base_success_cos
        failure_repulsion = base_failure_cos - _channel_patch_cosine(patched_np, fail_center)
        margin_gain = success_gain + failure_repulsion
    elif mode == "complement":
        if selector_group is not None:
            best_start, best_end = selector_group
        else:
            _, _, _, best_start, best_end, _ = candidates[0]
        patched_np = base_np.copy()
        patched_np[:, :best_start] = succ_center[:, :best_start]
        patched_np[:, best_end:] = succ_center[:, best_end:]
        start, end = best_start, best_end
        success_gain = _channel_patch_cosine(patched_np, succ_center) - base_success_cos
        failure_repulsion = base_failure_cos - _channel_patch_cosine(patched_np, fail_center)
        margin_gain = success_gain + failure_repulsion
    else:
        return base_future, {
            "channel_causal_patch": True,
            "channel_causal_patch_used": False,
            "channel_causal_patch_reason": f"unknown_mode:{mode}",
        }
    gate = float(getattr(cfg, "channel_patch_gate", 1.0))
    edited_np = base_np + gate * (patched_np - base_np)
    edited = torch.from_numpy(edited_np).to(device=base_future.device, dtype=base_future.dtype)
    cfg._channel_patch_count = int(getattr(cfg, "_channel_patch_count", 0)) + 1
    return edited, {
        "channel_causal_patch": True,
        "channel_causal_patch_used": True,
        "channel_causal_patch_mode": mode,
        "channel_causal_patch_selector_ckpt": str(getattr(cfg, "channel_patch_selector_ckpt", "") or ""),
        "channel_causal_patch_editor_ckpt": str(getattr(cfg, "channel_patch_editor_ckpt", "") or ""),
        "channel_causal_patch_q_critic_ckpt": str(getattr(cfg, "channel_patch_q_critic_ckpt", "") or ""),
        "channel_causal_patch_q_success": None if q_success is None else float(q_success),
        "channel_causal_patch_q_risk": None if q_risk is None else float(q_risk),
        "channel_causal_patch_q_risk_threshold": float(q_threshold),
        "channel_causal_patch_topk_groups": int(getattr(cfg, "channel_patch_topk_groups", 1)),
        "channel_causal_patch_selector_group_index": None if selector_pred is None else int(selector_pred["group_index"]),
        "channel_causal_patch_selector_confidence": None if selector_pred is None else float(selector_pred["confidence"]),
        "channel_causal_patch_selector_top_indices": [] if selector_pred is None else selector_pred["top_indices"],
        "channel_causal_patch_selector_top_probs": [] if selector_pred is None else selector_pred["top_probs"],
        "channel_causal_patch_group_start": int(start),
        "channel_causal_patch_group_end": int(end),
        "channel_causal_patch_gate": float(gate),
        "channel_causal_patch_same_task": bool(same_task),
        "channel_causal_patch_base_success_cos": float(base_success_cos),
        "channel_causal_patch_base_failure_cos": float(base_failure_cos),
        "channel_causal_patch_success_gain_proxy": float(success_gain),
        "channel_causal_patch_failure_repulsion_proxy": float(failure_repulsion),
        "channel_causal_patch_margin_gain_proxy": float(margin_gain),
        "channel_causal_patch_nearest_success_tasks": [tasks[idx] for idx in succ_ids],
        "channel_causal_patch_nearest_failure_tasks": [tasks[idx] for idx in fail_ids],
        "channel_causal_patch_shift_norm": float(np.linalg.norm((edited_np - base_np).reshape(-1))),
    }


def count_success_upto(results, max_steps):
    count = Counter(results)
    step_success = []
    for i in range(1, max_steps + 1):
        n_success = sum(count[j] for j in range(i, max_steps + 1))
        step_success.append(n_success / max(1, len(results)))
    return step_success


def print_and_save_variable_horizon(total_results, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)
    sequences = cfg.eval_sequences if "eval_sequences" in cfg and cfg.eval_sequences is not None else get_sequences(cfg.num_sequences)
    horizon = max((len(sequence) for _, sequence in sequences), default=5)
    current_data = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem
        avg_seq_len = np.mean(results)
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success_upto(results, horizon))}
        cnt_success = Counter()
        cnt_fail = Counter()
        for result, (_, sequence) in zip(results, sequences):
            for successful_task in sequence[:result]:
                cnt_success[successful_task] += 1
            if result < len(sequence):
                cnt_fail[sequence[result]] += 1
        total = cnt_success + cnt_fail
        task_info = {task: {"success": cnt_success[task], "total": total[task]} for task in total}
        print(f"Results for Epoch {epoch}:")
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")
        current_data[epoch] = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    with open(Path(log_dir) / "results.json", "w") as handle:
        json.dump(current_data, handle, indent=2)


def snapshot_env_raw(env):
    return env.env.get_obs()


def _strong_gt_to_jsonable(value, max_items=256):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size > max_items:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "mean": float(np.nanmean(flat)) if flat.size else 0.0,
                "std": float(np.nanstd(flat)) if flat.size else 0.0,
                "min": float(np.nanmin(flat)) if flat.size else 0.0,
                "max": float(np.nanmax(flat)) if flat.size else 0.0,
                "head": flat[: min(32, flat.size)].astype(float).tolist(),
            }
        return value.astype(float).tolist() if np.issubdtype(value.dtype, np.number) else value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _strong_gt_to_jsonable(v, max_items=max_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strong_gt_to_jsonable(v, max_items=max_items) for v in list(value)[:max_items]]
    return str(value)


def _strong_gt_array_summary(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"dim": 0}
    return {
        "dim": int(arr.size),
        "norm": float(np.linalg.norm(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _strong_gt_delta_summary(start, end):
    try:
        lhs = np.asarray(start, dtype=np.float32).reshape(-1)
        rhs = np.asarray(end, dtype=np.float32).reshape(-1)
        if lhs.shape != rhs.shape:
            return {"available": False, "reason": f"shape_mismatch:{lhs.shape}!={rhs.shape}"}
        delta = rhs - lhs
        return {
            "available": True,
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_mean": float(np.mean(delta)),
            "delta_std": float(np.std(delta)),
            "delta_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _strong_gt_pose_from_obj(obj):
    if not isinstance(obj, dict):
        return {}
    out = {}
    for src_key, dst_key in [
        ("current_pos", "pos"),
        ("current_orn", "orn"),
        ("current_lin_vel", "lin_vel"),
        ("current_ang_vel", "ang_vel"),
    ]:
        if src_key in obj:
            out[dst_key] = _strong_gt_to_jsonable(obj.get(src_key))
    return out


def _strong_gt_object_poses(info):
    scene = info.get("scene_info", {}) if isinstance(info, dict) else {}
    out = {}
    for group in ["movable_objects", "doors", "buttons", "switches", "lights", "fixed_objects"]:
        group_items = scene.get(group, {})
        if isinstance(group_items, dict):
            out[group] = {name: _strong_gt_pose_from_obj(obj) for name, obj in group_items.items()}
    return out


def _strong_gt_gripper_pose(raw, info):
    payload = {"available": False}
    robot_obs = raw.get("robot_obs") if isinstance(raw, dict) else None
    if robot_obs is not None:
        arr = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
        payload.update(
            {
                "available": True,
                "source": "robot_obs",
                "robot_obs_dim": int(arr.size),
                "pos": arr[:3].astype(float).tolist() if arr.size >= 3 else [],
                "orn_or_euler": arr[3:6].astype(float).tolist() if arr.size >= 6 else [],
                "gripper_width_or_action": float(arr[-1]) if arr.size >= 1 else None,
                "raw": arr.astype(float).tolist(),
            }
        )
    robot_info = info.get("robot_info", {}) if isinstance(info, dict) else {}
    if robot_info:
        payload["robot_info"] = _strong_gt_to_jsonable(robot_info)
    return payload


def _strong_gt_progress_fields(info):
    scene = info.get("scene_info", {}) if isinstance(info, dict) else {}
    out = {"available": False, "doors": {}, "buttons": {}, "switches": {}, "lights": {}}
    for group in ["doors", "buttons", "switches", "lights"]:
        items = scene.get(group, {})
        if not isinstance(items, dict):
            continue
        for name, obj in items.items():
            if not isinstance(obj, dict):
                continue
            vals = {}
            for key, value in obj.items():
                key_l = str(key).lower()
                if any(tok in key_l for tok in ["state", "joint", "pos", "open", "button", "light", "switch"]):
                    vals[str(key)] = _strong_gt_to_jsonable(value)
            if vals:
                out[group][name] = vals
                out["available"] = True
    return out


def _strong_gt_progress_delta(start_progress, end_progress):
    deltas = {}
    for group in ["doors", "buttons", "switches", "lights"]:
        deltas[group] = {}
        names = set((start_progress or {}).get(group, {})) | set((end_progress or {}).get(group, {}))
        for name in names:
            svals = ((start_progress or {}).get(group, {}) or {}).get(name, {})
            evals = ((end_progress or {}).get(group, {}) or {}).get(name, {})
            item = {}
            for key in set(svals) | set(evals):
                sv = svals.get(key)
                ev = evals.get(key)
                try:
                    item[key] = {
                        "start": sv,
                        "end": ev,
                        "delta": float(np.asarray(ev, dtype=np.float32).reshape(-1)[0] - np.asarray(sv, dtype=np.float32).reshape(-1)[0]),
                    }
                except Exception:
                    item[key] = {"start": sv, "end": ev}
            if item:
                deltas[group][name] = item
    return deltas


def _strong_gt_contact_state(start_info, end_info, task):
    if strong_gt_observed_contact_sets is None:
        return {"available": False, "reason": "failure_warning_rollout_import_failed"}
    try:
        payload = {
            "available": True,
            "start_contact_sets": strong_gt_observed_contact_sets(start_info),
            "end_contact_sets": strong_gt_observed_contact_sets(end_info),
            "start_contacts": strong_gt_observed_movable_contacts(start_info),
            "end_contacts": strong_gt_observed_movable_contacts(end_info),
        }
        if strong_gt_subtask_diagnostics is not None:
            payload["task_diagnostics"] = strong_gt_subtask_diagnostics(task, start_info, end_info)
        return _strong_gt_to_jsonable(payload)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _strong_gt_goal_predicate(task_info, success):
    return {
        "success": bool(success),
        "oracle_completed_tasks": _strong_gt_to_jsonable(task_info),
        "num_completed": len(task_info) if hasattr(task_info, "__len__") else None,
    }


def build_strong_gt_payload(env, task_oracle, subtask, start_raw, start_info, end_info, success, steps):
    end_raw = snapshot_env_raw(env)
    try:
        task_info = task_oracle.get_task_info_for_set(start_info, end_info, {subtask})
    except Exception as exc:
        task_info = {"error": str(exc)}
    start_object_poses = _strong_gt_object_poses(start_info)
    end_object_poses = _strong_gt_object_poses(end_info)
    start_progress = _strong_gt_progress_fields(start_info)
    end_progress = _strong_gt_progress_fields(end_info)
    object_motion = (
        _strong_gt_to_jsonable(strong_gt_all_object_motion(start_info, end_info))
        if strong_gt_all_object_motion is not None
        else {}
    )
    robot_start = start_raw.get("robot_obs") if isinstance(start_raw, dict) else None
    robot_end = end_raw.get("robot_obs") if isinstance(end_raw, dict) else None
    scene_start = start_raw.get("scene_obs") if isinstance(start_raw, dict) else None
    scene_end = end_raw.get("scene_obs") if isinstance(end_raw, dict) else None
    return {
        "strong_gt_available": True,
        "strong_gt_success": bool(success),
        "strong_gt_steps": int(steps),
        "strong_gt_task_oracle_info": _strong_gt_to_jsonable(task_info),
        "strong_gt_start_info": _strong_gt_to_jsonable(start_info),
        "strong_gt_end_info": _strong_gt_to_jsonable(end_info),
        "object_pose_start": _strong_gt_to_jsonable(start_object_poses),
        "object_pose_end": _strong_gt_to_jsonable(end_object_poses),
        "object_motion": object_motion,
        "block_displacement": object_motion,
        "gripper_pose_start": _strong_gt_gripper_pose(start_raw, start_info),
        "gripper_pose_end": _strong_gt_gripper_pose(end_raw, end_info),
        "contact_state": _strong_gt_contact_state(start_info, end_info, subtask),
        "drawer_slider_progress_start": _strong_gt_to_jsonable(start_progress),
        "drawer_slider_progress_end": _strong_gt_to_jsonable(end_progress),
        "drawer_slider_progress_delta": _strong_gt_progress_delta(start_progress, end_progress),
        "goal_predicate": _strong_gt_goal_predicate(task_info, success),
        "strong_gt_robot_obs_start": _strong_gt_to_jsonable(robot_start),
        "strong_gt_robot_obs_end": _strong_gt_to_jsonable(robot_end),
        "strong_gt_scene_obs_start": _strong_gt_to_jsonable(scene_start),
        "strong_gt_scene_obs_end": _strong_gt_to_jsonable(scene_end),
        "strong_gt_robot_obs_start_summary": _strong_gt_array_summary(robot_start) if robot_start is not None else {},
        "strong_gt_robot_obs_end_summary": _strong_gt_array_summary(robot_end) if robot_end is not None else {},
        "strong_gt_scene_obs_start_summary": _strong_gt_array_summary(scene_start) if scene_start is not None else {},
        "strong_gt_scene_obs_end_summary": _strong_gt_array_summary(scene_end) if scene_end is not None else {},
        "strong_gt_robot_obs_delta": _strong_gt_delta_summary(robot_start, robot_end) if robot_start is not None and robot_end is not None else {},
        "strong_gt_scene_obs_delta": _strong_gt_delta_summary(scene_start, scene_end) if scene_start is not None and scene_end is not None else {},
    }


def reset_env_raw(env, raw):
    env.reset(robot_obs=raw["robot_obs"], scene_obs=raw["scene_obs"])


def make_state_memory_row(memory_id, seq_idx, sub_idx, task, success, steps, source):
    return {
        "memory_id": memory_id,
        "memory_type": source,
        "sequence_index": int(seq_idx),
        "subtask_index": int(sub_idx),
        "task": task,
        "success": bool(success),
        "steps": int(steps),
        "state_semantics": {
            "completed_actions": [task] if success else [],
            "object_state_summary": "state_verified" if success else "state_unverified",
            "relation_summary": "relation_verified" if success else "relation_unverified",
        },
    }


def tensor_cosine(lhs, rhs):
    lhs = lhs.detach().float().reshape(-1)
    rhs = rhs.detach().float().reshape(-1)
    denom = torch.norm(lhs, p=2) * torch.norm(rhs, p=2)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(lhs, rhs).div(denom).item())


def get_memory_target_future(adapter_bundle, memory_ids, memory_sims, device):
    if not memory_ids:
        return None
    memory_arrays = adapter_bundle["memory_arrays"]
    row_ids = [int(item) for item in memory_arrays["row_ids"].tolist()]
    row_to_index = {row_id: idx for idx, row_id in enumerate(row_ids)}
    selected = []
    weights = []
    for out_idx, memory_id in enumerate(memory_ids):
        try:
            row_id = int(str(memory_id).replace("KEY_", ""))
        except ValueError:
            continue
        bank_idx = row_to_index.get(row_id)
        if bank_idx is None:
            continue
        selected.append(memory_arrays["repair_targets"][bank_idx].astype(np.float32))
        weights.append(max(float(memory_sims[out_idx]), 0.0))
    if not selected:
        return None
    weights_np = np.asarray(weights, dtype=np.float32)
    if float(weights_np.sum()) <= 1e-8:
        weights_np = np.ones_like(weights_np) / float(len(weights_np))
    else:
        weights_np = weights_np / weights_np.sum()
    target = np.tensordot(weights_np, np.stack(selected, axis=0), axes=(0, 0)).astype(np.float32)
    return torch.from_numpy(target).to(device)


def _joint_pair_future_summary(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[-1] - arr[0]], axis=0).astype(np.float32)


def _joint_pair_future_summary_exec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def _joint_pair_future_summary_exec_torch(arr: torch.Tensor) -> torch.Tensor:
    arr = arr.float()
    return torch.cat(
        [
            arr.mean(dim=0),
            arr.std(dim=0, unbiased=False),
            arr[0],
            arr[-1] - arr[0],
        ],
        dim=0,
    )


def _joint_pair_dynamics_loss(
    bundle,
    current_state: np.ndarray | None,
    target_transition: np.ndarray | None,
    action_chunk: torch.Tensor,
    device: torch.device,
):
    if current_state is None or target_transition is None:
        return None
    dyn_model = bundle.get("dynamics_model")
    dyn_mean = bundle.get("dynamics_transition_mean")
    dyn_std = bundle.get("dynamics_transition_std")
    if dyn_model is None or dyn_mean is None or dyn_std is None:
        return None
    state_t = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).unsqueeze(0).to(device)
    target_t = torch.from_numpy(np.asarray(target_transition, dtype=np.float32)).unsqueeze(0).to(device)
    out = dyn_model(state_t, action_chunk.unsqueeze(0))
    pred = out["pred_transition"]
    pred_norm = (pred - dyn_mean) / dyn_std
    target_norm = (target_t - dyn_mean) / dyn_std
    cos = F.cosine_similarity(pred_norm, target_norm, dim=-1).mean()
    l1 = F.smooth_l1_loss(pred_norm, target_norm)
    # Minimize low energy + low dynamics mismatch.
    loss = l1 - cos
    return {
        "loss": loss,
        "cos": cos,
        "l1": l1,
    }


def _joint_pair_hypothesis_parts(
    cfg,
    bundle,
    base_summary_exec: torch.Tensor,
    cur_future: torch.Tensor,
    cur_action: torch.Tensor,
    task_vec: torch.Tensor,
    sub: torch.Tensor,
    init_future: torch.Tensor,
    init_action: torch.Tensor,
    current_state: np.ndarray | None,
    target_transition: np.ndarray | None,
    term_scales: dict[str, torch.Tensor] | None = None,
):
    summary = _joint_pair_future_summary_exec_torch(cur_future).unsqueeze(0)
    delta = summary - base_summary_exec
    scorer_type = str(bundle.get("scorer_type", "energy"))

    exp_energy = torch.zeros((), device=base_summary_exec.device)
    latent_energy = torch.zeros((), device=base_summary_exec.device)
    physical_energy = torch.zeros((), device=base_summary_exec.device)
    physical_dyn_cos = torch.zeros((), device=base_summary_exec.device)
    physical_dyn_l1 = torch.zeros((), device=base_summary_exec.device)
    robotics_task_energy = torch.zeros((), device=base_summary_exec.device)
    robotics_future_dyn_energy = torch.zeros((), device=base_summary_exec.device)
    robotics_state_dyn_energy = torch.zeros((), device=base_summary_exec.device)
    robotics_contact_energy = torch.zeros((), device=base_summary_exec.device)

    bundle["_current_state"] = current_state
    bundle["_target_state_transition"] = target_transition

    if scorer_type == "robotics_energy":
        parts = _joint_pair_robotics_energy_parts(
            bundle,
            base_summary_exec,
            summary,
            delta,
            cur_action.unsqueeze(0),
            task_vec,
            sub,
            current_state=current_state,
            target_state_transition=target_transition,
        )
        robotics_task_energy = parts["task_energy"].mean()
        robotics_future_dyn_energy = parts["future_dyn_energy"].mean()
        robotics_state_dyn_energy = parts["state_dyn_energy"].mean()
        robotics_contact_energy = parts["contact_energy"].mean()
        exp_energy = robotics_task_energy
        latent_energy = latent_energy + robotics_future_dyn_energy
        physical_energy = physical_energy + robotics_state_dyn_energy + robotics_contact_energy
    else:
        exp_energy = _joint_pair_model_energy(
            bundle,
            base_summary_exec,
            summary,
            delta,
            cur_action.unsqueeze(0),
            task_vec,
            sub,
        ).mean()

    dyn_parts = _joint_pair_dynamics_loss(
        bundle,
        current_state=current_state,
        target_transition=target_transition,
        action_chunk=cur_action,
        device=base_summary_exec.device,
    )
    if dyn_parts is not None:
        physical_energy = physical_energy + float(getattr(cfg, "joint_pair_refine_lambda_dynamics", 0.0)) * dyn_parts["loss"]
        physical_dyn_cos = dyn_parts["cos"]
        physical_dyn_l1 = dyn_parts["l1"]

    future_anchor = F.mse_loss(cur_future, init_future)
    action_anchor = F.mse_loss(cur_action, init_action)
    latent_energy = latent_energy + (
        float(getattr(cfg, "joint_pair_refine_lambda_future_anchor", 0.1)) * future_anchor
        + float(getattr(cfg, "joint_pair_refine_lambda_action_anchor", 0.1)) * action_anchor
    )

    exp_scale = term_scales["exp"] if term_scales is not None else torch.ones((), device=base_summary_exec.device)
    latent_scale = term_scales["latent"] if term_scales is not None else torch.ones((), device=base_summary_exec.device)
    physical_scale = term_scales["physical"] if term_scales is not None else torch.ones((), device=base_summary_exec.device)
    exp_energy_norm = exp_energy / exp_scale
    latent_energy_norm = latent_energy / latent_scale
    physical_energy_norm = physical_energy / physical_scale
    total = (
        float(getattr(cfg, "joint_pair_refine_lambda_exp", 1.0)) * exp_energy_norm
        + float(getattr(cfg, "joint_pair_refine_lambda_latent", 1.0)) * latent_energy_norm
        + float(getattr(cfg, "joint_pair_refine_lambda_physical", 1.0)) * physical_energy_norm
    )
    return {
        "loss": total,
        "summary": summary,
        "delta": delta,
        "c_exp": exp_energy,
        "c_latent": latent_energy,
        "c_physical": physical_energy,
        "c_exp_norm": exp_energy_norm,
        "c_latent_norm": latent_energy_norm,
        "c_physical_norm": physical_energy_norm,
        "scale_exp": exp_scale,
        "scale_latent": latent_scale,
        "scale_physical": physical_scale,
        "future_anchor": future_anchor,
        "action_anchor": action_anchor,
        "dyn_cos": physical_dyn_cos,
        "dyn_l1": physical_dyn_l1,
        "robotics_task_energy": robotics_task_energy,
        "robotics_future_dyn_energy": robotics_future_dyn_energy,
        "robotics_state_dyn_energy": robotics_state_dyn_energy,
        "robotics_contact_energy": robotics_contact_energy,
    }


def refine_joint_pair_energy(
    cfg,
    bundle,
    base_future: torch.Tensor,
    target_future: torch.Tensor,
    action_chunk: torch.Tensor,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None = None,
    target_transition: np.ndarray | None = None,
):
    steps = int(getattr(cfg, "joint_pair_refine_steps", 0))
    if steps <= 0:
        return target_future, action_chunk, None

    model = bundle["model"]
    device = base_future.device
    task_vec = torch.from_numpy(
        np.asarray(q_hashed_task(task, bundle["task_dim"]), dtype=np.float32)
    ).unsqueeze(0).to(device)
    sub = torch.tensor([[float(subtask_index or 0) / 5.0]], dtype=torch.float32, device=device)
    base_summary_exec = torch.from_numpy(
        _joint_pair_future_summary_exec(base_future.detach().float().cpu().numpy().astype(np.float32))
    ).unsqueeze(0).to(device)

    future_var = target_future.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    action_var = action_chunk.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    future_lr = float(getattr(cfg, "joint_pair_refine_future_lr", 1e-2))
    action_lr = float(getattr(cfg, "joint_pair_refine_action_lr", 1e-2))
    clip = float(getattr(cfg, "joint_pair_refine_grad_clip", 1.0))
    future_max_delta = float(getattr(cfg, "joint_pair_refine_future_max_delta", 5.0))
    action_max_delta = float(getattr(cfg, "joint_pair_refine_action_max_delta", 5.0))
    backtrack_factor = float(getattr(cfg, "joint_pair_refine_backtrack_factor", 0.5))
    min_step_scale = float(getattr(cfg, "joint_pair_refine_min_step_scale", 0.03125))
    max_backtracks = int(getattr(cfg, "joint_pair_refine_max_backtracks", 5))
    early_stop_patience = int(getattr(cfg, "joint_pair_refine_early_stop_patience", 2))
    early_stop_min_improve = float(getattr(cfg, "joint_pair_refine_early_stop_min_improve", 1e-3))
    early_stop_grad_norm = float(getattr(cfg, "joint_pair_refine_early_stop_grad_norm", 1e-3))
    init_future = future_var.detach().clone()
    init_action = action_var.detach().clone()
    no_improve_steps = 0
    stopped_early = False
    stop_reason = ""
    history = []
    use_term_norm = bool(getattr(cfg, "joint_pair_refine_normalize_terms", False))
    term_scales = None

    with torch.enable_grad():
        def objective_parts(cur_future, cur_action):
            return _joint_pair_hypothesis_parts(
                cfg,
                bundle,
                base_summary_exec,
                cur_future,
                cur_action,
                task_vec,
                sub,
                init_future,
                init_action,
                current_state=current_state,
                target_transition=target_transition,
                term_scales=term_scales,
            )

        init_parts = objective_parts(future_var, action_var)
        if use_term_norm:
            term_scales = {
                "exp": init_parts["c_exp"].detach().abs().clamp_min(1e-4),
                "latent": init_parts["c_latent"].detach().abs().clamp_min(1e-4),
                "physical": init_parts["c_physical"].detach().abs().clamp_min(1e-4),
            }
            init_parts = objective_parts(future_var, action_var)
        init_energy = init_parts["loss"]
        best_objective = float(init_parts["loss"].detach().item())

        for step_idx in range(1, steps + 1):
            parts = objective_parts(future_var, action_var)
            loss = parts["loss"]
            grad_future, grad_action = torch.autograd.grad(loss, [future_var, action_var], allow_unused=False)
            grad_future_norm = float(torch.norm(grad_future).item())
            grad_action_norm = float(torch.norm(grad_action).item())
            grad_norm = float((grad_future_norm**2 + grad_action_norm**2) ** 0.5)
            if clip > 0:
                grad_future = grad_future / max(grad_future_norm / clip, 1.0)
                grad_action = grad_action / max(grad_action_norm / clip, 1.0)

            current_objective = float(loss.detach().item())
            accepted = False
            step_scale = 1.0
            accepted_parts = parts
            accepted_future = future_var.detach().clone()
            accepted_action = action_var.detach().clone()
            for _ in range(max_backtracks):
                with torch.no_grad():
                    cand_future = future_var - step_scale * future_lr * grad_future
                    cand_action = action_var - step_scale * action_lr * grad_action
                    future_delta = cand_future - init_future
                    action_delta = cand_action - init_action
                    future_norm = float(torch.norm(future_delta.reshape(-1), p=2).item())
                    action_norm = float(torch.norm(action_delta.reshape(-1), p=2).item())
                    if future_max_delta > 0.0 and future_norm > future_max_delta:
                        cand_future = init_future + future_delta * (future_max_delta / max(future_norm, 1e-8))
                    if action_max_delta > 0.0 and action_norm > action_max_delta:
                        cand_action = init_action + action_delta * (action_max_delta / max(action_norm, 1e-8))
                cand_parts = objective_parts(cand_future, cand_action)
                cand_objective = float(cand_parts["loss"].detach().item())
                if cand_objective + 1e-8 < current_objective:
                    accepted = True
                    accepted_parts = cand_parts
                    accepted_future = cand_future.detach().clone()
                    accepted_action = cand_action.detach().clone()
                    break
                step_scale *= backtrack_factor
                if step_scale < min_step_scale:
                    break

            improve = 0.0
            if accepted:
                with torch.no_grad():
                    future_var.copy_(accepted_future)
                    action_var.copy_(accepted_action)
                improve = current_objective - float(accepted_parts["loss"].detach().item())
                if float(accepted_parts["loss"].detach().item()) < best_objective - early_stop_min_improve:
                    best_objective = float(accepted_parts["loss"].detach().item())
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1

            if step_idx == 1 or step_idx == steps or step_idx % max(1, steps // 5) == 0:
                history.append(
                    {
                        "step": int(step_idx),
                        "energy": float(accepted_parts["loss"].detach().item()),
                        "c_exp": float(accepted_parts["c_exp"].detach().item()),
                        "c_latent": float(accepted_parts["c_latent"].detach().item()),
                        "c_physical": float(accepted_parts["c_physical"].detach().item()),
                        "c_exp_norm": float(accepted_parts["c_exp_norm"].detach().item()),
                        "c_latent_norm": float(accepted_parts["c_latent_norm"].detach().item()),
                        "c_physical_norm": float(accepted_parts["c_physical_norm"].detach().item()),
                        "dyn_loss": float(accepted_parts["c_physical"].detach().item()),
                        "dyn_cos": float(accepted_parts["dyn_cos"].detach().item()),
                        "dyn_l1": float(accepted_parts["dyn_l1"].detach().item()),
                        "robotics_task_energy": float(accepted_parts["robotics_task_energy"].detach().item()),
                        "robotics_future_dyn_energy": float(accepted_parts["robotics_future_dyn_energy"].detach().item()),
                        "robotics_state_dyn_energy": float(accepted_parts["robotics_state_dyn_energy"].detach().item()),
                        "robotics_contact_energy": float(accepted_parts["robotics_contact_energy"].detach().item()),
                        "future_shift": float(torch.norm((future_var.detach() - init_future).reshape(-1), p=2).item()),
                        "action_shift": float(torch.norm((action_var.detach() - init_action).reshape(-1), p=2).item()),
                        "grad_norm": float(grad_norm),
                        "step_scale": float(step_scale),
                        "accepted": bool(accepted),
                        "objective_improve": float(improve),
                    }
                )
            if grad_norm <= early_stop_grad_norm:
                stopped_early = True
                stop_reason = "grad_norm"
                break
            if improve <= early_stop_min_improve and no_improve_steps >= early_stop_patience:
                stopped_early = True
                stop_reason = "no_improve"
                break

        final_parts = objective_parts(future_var, action_var)
        final_energy = final_parts["loss"]

    info = {
        "joint_pair_refine_used": True,
        "joint_pair_refine_steps": steps,
        "joint_pair_refine_init_energy": float(init_energy.detach().item()),
        "joint_pair_refine_final_energy": float(final_energy.detach().item()),
        "joint_pair_refine_delta_energy": float((init_energy - final_energy).detach().item()),
        "joint_pair_refine_init_c_exp": float(init_parts["c_exp"].detach().item()),
        "joint_pair_refine_init_c_latent": float(init_parts["c_latent"].detach().item()),
        "joint_pair_refine_init_c_physical": float(init_parts["c_physical"].detach().item()),
        "joint_pair_refine_init_c_exp_norm": float(init_parts["c_exp_norm"].detach().item()),
        "joint_pair_refine_init_c_latent_norm": float(init_parts["c_latent_norm"].detach().item()),
        "joint_pair_refine_init_c_physical_norm": float(init_parts["c_physical_norm"].detach().item()),
        "joint_pair_refine_future_shift_norm": float(torch.norm((future_var.detach() - init_future).reshape(-1), p=2).item()),
        "joint_pair_refine_action_shift_norm": float(torch.norm((action_var.detach() - init_action).reshape(-1), p=2).item()),
        "joint_pair_refine_c_exp": float(final_parts["c_exp"].detach().item()),
        "joint_pair_refine_c_latent": float(final_parts["c_latent"].detach().item()),
        "joint_pair_refine_c_physical": float(final_parts["c_physical"].detach().item()),
        "joint_pair_refine_c_exp_norm": float(final_parts["c_exp_norm"].detach().item()),
        "joint_pair_refine_c_latent_norm": float(final_parts["c_latent_norm"].detach().item()),
        "joint_pair_refine_c_physical_norm": float(final_parts["c_physical_norm"].detach().item()),
        "joint_pair_refine_dyn_loss": float(final_parts["c_physical"].detach().item()),
        "joint_pair_refine_dyn_cos": float(final_parts["dyn_cos"].detach().item()),
        "joint_pair_refine_dyn_l1": float(final_parts["dyn_l1"].detach().item()),
        "joint_pair_refine_normalize_terms": bool(use_term_norm),
        "joint_pair_refine_scale_exp": float(final_parts["scale_exp"].detach().item()),
        "joint_pair_refine_scale_latent": float(final_parts["scale_latent"].detach().item()),
        "joint_pair_refine_scale_physical": float(final_parts["scale_physical"].detach().item()),
        "joint_pair_refine_robotics_task_energy": float(final_parts["robotics_task_energy"].detach().item()),
        "joint_pair_refine_robotics_future_dyn_energy": float(final_parts["robotics_future_dyn_energy"].detach().item()),
        "joint_pair_refine_robotics_state_dyn_energy": float(final_parts["robotics_state_dyn_energy"].detach().item()),
        "joint_pair_refine_robotics_contact_energy": float(final_parts["robotics_contact_energy"].detach().item()),
        "joint_pair_refine_lambda_exp": float(getattr(cfg, "joint_pair_refine_lambda_exp", 1.0)),
        "joint_pair_refine_lambda_latent": float(getattr(cfg, "joint_pair_refine_lambda_latent", 1.0)),
        "joint_pair_refine_lambda_physical": float(getattr(cfg, "joint_pair_refine_lambda_physical", 1.0)),
        "joint_pair_refine_lambda_dynamics": float(getattr(cfg, "joint_pair_refine_lambda_dynamics", 0.0)),
        "joint_pair_refine_stopped_early": bool(stopped_early),
        "joint_pair_refine_stop_reason": stop_reason,
        "joint_pair_refine_history": history,
    }
    return future_var.detach().to(dtype=target_future.dtype), action_var.detach().to(dtype=action_chunk.dtype), info


def verify_joint_pair_hypothesis(
    cfg,
    bundle,
    base_future: torch.Tensor,
    target_future: torch.Tensor,
    action_chunk: torch.Tensor,
    task: str,
    subtask_index: int,
    init_future: torch.Tensor,
    init_action: torch.Tensor,
    current_state: np.ndarray | None = None,
    target_transition: np.ndarray | None = None,
):
    device = base_future.device
    task_vec = torch.from_numpy(
        np.asarray(q_hashed_task(task, bundle["task_dim"]), dtype=np.float32)
    ).unsqueeze(0).to(device)
    sub = torch.tensor([[float(subtask_index or 0) / 5.0]], dtype=torch.float32, device=device)
    base_summary_exec = torch.from_numpy(
        _joint_pair_future_summary_exec(base_future.detach().float().cpu().numpy().astype(np.float32))
    ).unsqueeze(0).to(device)
    init_future_t = init_future.detach().to(device=device, dtype=torch.float32)
    init_action_t = init_action.detach().to(device=device, dtype=torch.float32)
    final_future_t = target_future.detach().to(device=device, dtype=torch.float32)
    final_action_t = action_chunk.detach().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        init_parts = _joint_pair_hypothesis_parts(
            cfg,
            bundle,
            base_summary_exec,
            init_future_t,
            init_action_t,
            task_vec,
            sub,
            init_future_t,
            init_action_t,
            current_state=current_state,
            target_transition=target_transition,
            term_scales=None,
        )
        final_parts = _joint_pair_hypothesis_parts(
            cfg,
            bundle,
            base_summary_exec,
            final_future_t,
            final_action_t,
            task_vec,
            sub,
            init_future_t,
            init_action_t,
            current_state=current_state,
            target_transition=target_transition,
            term_scales=None,
        )

    c_exp = float(final_parts["c_exp"].detach().item())
    c_latent = float(final_parts["c_latent"].detach().item())
    c_physical = float(final_parts["c_physical"].detach().item())
    init_exp = float(init_parts["c_exp"].detach().item())
    init_latent = float(init_parts["c_latent"].detach().item())
    init_physical = float(init_parts["c_physical"].detach().item())
    delta_exp = init_exp - c_exp
    delta_latent = init_latent - c_latent
    delta_physical = init_physical - c_physical

    max_exp = float(getattr(cfg, "joint_pair_verify_exp_max", 1e9))
    max_latent = float(getattr(cfg, "joint_pair_verify_latent_max", 1e9))
    max_physical = float(getattr(cfg, "joint_pair_verify_physical_max", 1e9))
    require_improve = bool(getattr(cfg, "joint_pair_verify_require_improve", False))
    min_improve = float(getattr(cfg, "joint_pair_verify_min_improve", 0.0))

    reasons = []
    if c_exp > max_exp:
        reasons.append(f"c_exp>{max_exp:g}")
    if c_latent > max_latent:
        reasons.append(f"c_latent>{max_latent:g}")
    if c_physical > max_physical:
        reasons.append(f"c_physical>{max_physical:g}")
    if require_improve:
        if delta_exp < min_improve:
            reasons.append(f"delta_exp<{min_improve:g}")
        if delta_latent < min_improve:
            reasons.append(f"delta_latent<{min_improve:g}")
        if delta_physical < min_improve:
            reasons.append(f"delta_physical<{min_improve:g}")

    return {
        "joint_pair_verify_used": True,
        "joint_pair_verify_passed": len(reasons) == 0,
        "joint_pair_verify_reason": "pass" if not reasons else ";".join(reasons),
        "joint_pair_verify_c_exp": c_exp,
        "joint_pair_verify_c_latent": c_latent,
        "joint_pair_verify_c_physical": c_physical,
        "joint_pair_verify_init_c_exp": init_exp,
        "joint_pair_verify_init_c_latent": init_latent,
        "joint_pair_verify_init_c_physical": init_physical,
        "joint_pair_verify_delta_c_exp": delta_exp,
        "joint_pair_verify_delta_c_latent": delta_latent,
        "joint_pair_verify_delta_c_physical": delta_physical,
        "joint_pair_verify_dyn_cos": float(final_parts["dyn_cos"].detach().item()),
        "joint_pair_verify_dyn_l1": float(final_parts["dyn_l1"].detach().item()),
        "joint_pair_verify_robotics_task_energy": float(final_parts["robotics_task_energy"].detach().item()),
        "joint_pair_verify_robotics_future_dyn_energy": float(final_parts["robotics_future_dyn_energy"].detach().item()),
        "joint_pair_verify_robotics_state_dyn_energy": float(final_parts["robotics_state_dyn_energy"].detach().item()),
        "joint_pair_verify_robotics_contact_energy": float(final_parts["robotics_contact_energy"].detach().item()),
    }


def score_joint_hypothesis_candidate(
    cfg,
    base_future: torch.Tensor | None,
    target_future: torch.Tensor | None,
    action_chunk: torch.Tensor | None,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None = None,
    target_transition: np.ndarray | None = None,
    prefix: str = "joint_pair_score",
):
    if base_future is None or target_future is None or action_chunk is None:
        return {
            f"{prefix}_used": False,
            f"{prefix}_reason": "missing_base_or_target_or_action",
        }
    bundle = load_joint_pair_selector(cfg, base_future.device)
    if bundle is None:
        return {
            f"{prefix}_used": False,
            f"{prefix}_reason": "missing_selector_bundle",
        }
    device = base_future.device
    task_vec = torch.from_numpy(
        np.asarray(q_hashed_task(task, bundle["task_dim"]), dtype=np.float32)
    ).unsqueeze(0).to(device)
    sub = torch.tensor([[float(subtask_index or 0) / 5.0]], dtype=torch.float32, device=device)
    base_summary_exec = torch.from_numpy(
        _joint_pair_future_summary_exec(base_future.detach().float().cpu().numpy().astype(np.float32))
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        parts = _joint_pair_hypothesis_parts(
            cfg,
            bundle,
            base_summary_exec,
            target_future.detach().to(device=device, dtype=torch.float32),
            action_chunk.detach().to(device=device, dtype=torch.float32),
            task_vec,
            sub,
            target_future.detach().to(device=device, dtype=torch.float32),
            action_chunk.detach().to(device=device, dtype=torch.float32),
            current_state=current_state,
            target_transition=target_transition,
            term_scales=None,
        )
    return {
        f"{prefix}_used": True,
        f"{prefix}_energy": float(parts["loss"].detach().item()),
        f"{prefix}_c_exp": float(parts["c_exp"].detach().item()),
        f"{prefix}_c_latent": float(parts["c_latent"].detach().item()),
        f"{prefix}_c_physical": float(parts["c_physical"].detach().item()),
        f"{prefix}_dyn_cos": float(parts["dyn_cos"].detach().item()),
        f"{prefix}_dyn_l1": float(parts["dyn_l1"].detach().item()),
        f"{prefix}_robotics_task_energy": float(parts["robotics_task_energy"].detach().item()),
        f"{prefix}_robotics_future_dyn_energy": float(parts["robotics_future_dyn_energy"].detach().item()),
        f"{prefix}_robotics_state_dyn_energy": float(parts["robotics_state_dyn_energy"].detach().item()),
        f"{prefix}_robotics_contact_energy": float(parts["robotics_contact_energy"].detach().item()),
    }


def _joint_pair_selector_constraint_terms(
    cfg,
    bundle,
    *,
    base_summary_exec_np: np.ndarray,
    target_summary_exec_np: np.ndarray,
    target_delta_exec_np: np.ndarray,
    action_chunk_np: np.ndarray,
    task_vec_np: np.ndarray,
    sub_np: np.ndarray,
    current_state: np.ndarray | None = None,
    target_transition: np.ndarray | None = None,
    device: torch.device,
):
    base_summary_t = torch.from_numpy(np.asarray(base_summary_exec_np, dtype=np.float32)).unsqueeze(0).to(device)
    target_summary_t = torch.from_numpy(np.asarray(target_summary_exec_np, dtype=np.float32)).unsqueeze(0).to(device)
    target_delta_t = torch.from_numpy(np.asarray(target_delta_exec_np, dtype=np.float32)).unsqueeze(0).to(device)
    action_chunk_t = torch.from_numpy(np.asarray(action_chunk_np, dtype=np.float32)).unsqueeze(0).to(device)
    task_vec_t = torch.from_numpy(np.asarray(task_vec_np, dtype=np.float32)).unsqueeze(0).to(device)
    sub_t = torch.from_numpy(np.asarray(sub_np, dtype=np.float32)).to(device)

    exp_energy = 0.0
    latent_energy = 0.0
    physical_energy = 0.0
    dyn_cos = 0.0
    dyn_l1 = 0.0
    robotics_task_energy = 0.0
    robotics_future_dyn_energy = 0.0
    robotics_state_dyn_energy = 0.0
    robotics_contact_energy = 0.0
    scorer_type = str(bundle.get("scorer_type", "energy"))
    extra = {}

    with torch.no_grad():
        if scorer_type == "robotics_energy":
            parts = _joint_pair_robotics_energy_parts(
                bundle,
                base_summary_t,
                target_summary_t,
                target_delta_t,
                action_chunk_t,
                task_vec_t,
                sub_t,
                current_state=current_state,
                target_state_transition=target_transition,
            )
            robotics_task_energy = float(parts["task_energy"].mean().item())
            robotics_future_dyn_energy = float(parts["future_dyn_energy"].mean().item())
            robotics_state_dyn_energy = float(parts["state_dyn_energy"].mean().item())
            robotics_contact_energy = float(parts["contact_energy"].mean().item())
            exp_energy = robotics_task_energy
            latent_energy = robotics_future_dyn_energy
            physical_energy = robotics_state_dyn_energy + robotics_contact_energy
            extra = {
                "robotics_contact_bad_prob": float(torch.sigmoid(parts["contact_bad_logit"]).mean().item()),
                "robotics_success_prob": float(torch.sigmoid(parts["success_logit"]).mean().item()),
            }
        else:
            exp_energy = float(
                _joint_pair_model_energy(
                    bundle,
                    base_summary_t,
                    target_summary_t,
                    target_delta_t,
                    action_chunk_t,
                    task_vec_t,
                    sub_t,
                ).mean().item()
            )

        dyn_parts = _joint_pair_dynamics_loss(
            bundle,
            current_state=current_state,
            target_transition=target_transition,
            action_chunk=action_chunk_t.squeeze(0),
            device=device,
        )
        if dyn_parts is not None:
            dyn_cos = float(dyn_parts["cos"].detach().item())
            dyn_l1 = float(dyn_parts["l1"].detach().item())
            physical_energy = physical_energy + float(getattr(cfg, "joint_pair_lambda_dynamics", 0.0)) * float(
                dyn_parts["loss"].detach().item()
            )

    return {
        "c_exp": float(exp_energy),
        "c_latent": float(latent_energy),
        "c_physical": float(physical_energy),
        "dyn_cos": float(dyn_cos),
        "dyn_l1": float(dyn_l1),
        "robotics_task_energy": float(robotics_task_energy),
        "robotics_future_dyn_energy": float(robotics_future_dyn_energy),
        "robotics_state_dyn_energy": float(robotics_state_dyn_energy),
        "robotics_contact_energy": float(robotics_contact_energy),
        **extra,
    }


def _joint_pair_model_energy(bundle, base_summary_exec, target_summary_exec, target_delta_exec, action_chunk, task_vec, sub):
    model = bundle["model"]
    scorer_type = str(bundle.get("scorer_type", "energy"))
    if scorer_type == "robotics_energy":
        out = _joint_pair_robotics_energy_parts(
            bundle,
            base_summary_exec,
            target_summary_exec,
            target_delta_exec,
            action_chunk,
            task_vec,
            sub,
            current_state=bundle.get("_current_state"),
            target_state_transition=bundle.get("_target_state_transition"),
        )
        return out["energy"].squeeze(-1) if out["energy"].ndim > 0 else out["energy"]
    if scorer_type == "unified_policy":
        out = model(
            base_summary_exec,
            target_summary_exec,
            target_delta_exec,
            action_chunk,
            task_vec,
            sub,
        )
        return out["energy"].squeeze(-1) if out["energy"].ndim > 0 else out["energy"]
    return model(
        base_summary_exec,
        target_summary_exec,
        target_delta_exec,
        action_chunk,
        task_vec,
        sub,
    ).squeeze(-1)


def _load_future_npz_array(path: str | Path, key: str) -> np.ndarray | None:
    try:
        with np.load(str(path)) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "target_proxy_future" and "final_future" in data:
                arr = np.asarray(data["final_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def _joint_pair_state_from_raw(raw) -> np.ndarray | None:
    if not isinstance(raw, dict):
        return None
    robot = raw.get("robot_obs")
    scene = raw.get("scene_obs")
    if robot is None or scene is None:
        return None
    try:
        robot_arr = np.asarray(robot, dtype=np.float32).reshape(-1)
        scene_arr = np.asarray(scene, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if robot_arr.size == 0 or scene_arr.size == 0:
        return None
    return np.concatenate([robot_arr, scene_arr], axis=0).astype(np.float32)


def _joint_pair_robotics_energy_parts(
    bundle,
    base_summary_exec: torch.Tensor,
    target_summary_exec: torch.Tensor,
    target_delta_exec: torch.Tensor,
    action_chunk: torch.Tensor,
    task_vec: torch.Tensor,
    sub: torch.Tensor,
    current_state: np.ndarray | torch.Tensor | None = None,
    target_state_transition: np.ndarray | torch.Tensor | None = None,
):
    if current_state is None:
        raise ValueError("robotics_energy scorer requires current_state")
    device = base_summary_exec.device
    if not torch.is_tensor(current_state):
        current_state_t = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).unsqueeze(0).to(device)
    else:
        current_state_t = current_state.to(device=device, dtype=base_summary_exec.dtype)
        if current_state_t.dim() == 1:
            current_state_t = current_state_t.unsqueeze(0)
    target_state_t = None
    if target_state_transition is not None:
        if not torch.is_tensor(target_state_transition):
            target_state_t = torch.from_numpy(np.asarray(target_state_transition, dtype=np.float32)).unsqueeze(0).to(device)
        else:
            target_state_t = target_state_transition.to(device=device, dtype=base_summary_exec.dtype)
            if target_state_t.dim() == 1:
                target_state_t = target_state_t.unsqueeze(0)
    model = bundle["model"]
    out = model(
        base_summary_exec,
        target_summary_exec,
        target_delta_exec,
        action_chunk,
        task_vec,
        sub,
        current_state_t,
        target_state_t,
        bundle.get("robotics_state_transition_mean"),
        bundle.get("robotics_state_transition_std"),
        float(bundle.get("robotics_lambda_future_dyn", 1.0)),
        float(bundle.get("robotics_lambda_state_dyn", 0.5)),
        float(bundle.get("robotics_beta_contact", 0.25)),
    )
    return out


def _joint_pair_generate_action_init(
    bundle,
    base_summary_exec: np.ndarray,
    target_summary_exec: np.ndarray,
    target_delta_exec: np.ndarray,
    task_vec: np.ndarray,
    sub: np.ndarray,
    current_state: np.ndarray | None,
    device: torch.device,
):
    gen = bundle.get("action_generator")
    if gen is None or current_state is None:
        return None, {}
    with torch.no_grad():
        state_t = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).reshape(1, -1).to(device)
        base_t = torch.from_numpy(np.asarray(base_summary_exec, dtype=np.float32)).reshape(1, -1).to(device)
        target_t = torch.from_numpy(np.asarray(target_summary_exec, dtype=np.float32)).reshape(1, -1).to(device)
        delta_t = torch.from_numpy(np.asarray(target_delta_exec, dtype=np.float32)).reshape(1, -1).to(device)
        task_t = torch.from_numpy(np.asarray(task_vec, dtype=np.float32)).reshape(1, -1).to(device)
        sub_t = torch.from_numpy(np.asarray(sub, dtype=np.float32)).reshape(1, -1).to(device)
        pred = gen(state_t, base_t, target_t, delta_t, task_t, sub_t).squeeze(0)
    info = {
        "joint_pair_action_generator_used": True,
        "joint_pair_action_generator_norm": float(torch.norm(pred.reshape(-1), p=2).item()),
    }
    return pred, info


def _joint_pair_safe_cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _joint_pair_build_initial_hypothesis(
    cfg,
    bundle,
    base_future: torch.Tensor,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    initial_action_intent: torch.Tensor | None = None,
):
    base_np = base_future.detach().float().cpu().numpy().astype(np.float32)
    base_summary = _joint_pair_future_summary(base_np)
    base_summary_exec = _joint_pair_future_summary_exec(base_np)
    info = {
        "joint_hypothesis_init_used": False,
        "joint_hypothesis_init_reason": "missing_action_init",
        "joint_hypothesis_init_stage": int(subtask_index or 0),
        "joint_hypothesis_init_future_norm": float(np.linalg.norm(base_summary_exec.reshape(-1))),
    }
    if initial_action_intent is not None:
        action_t = initial_action_intent.detach().to(device=base_future.device, dtype=base_future.dtype)
        info.update(
            {
                "joint_hypothesis_init_used": True,
                "joint_hypothesis_init_reason": "provided",
                "joint_hypothesis_init_action_norm": float(torch.norm(action_t.reshape(-1), p=2).item()),
            }
        )
        return action_t, base_summary, base_summary_exec, info
    if bundle is None or current_state is None:
        return None, base_summary, base_summary_exec, info
    task_dim = int(bundle.get("task_dim", 128))
    task_vec = np.asarray(q_hashed_task(task, task_dim), dtype=np.float32)
    sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)
    zero_delta = np.zeros_like(base_summary_exec, dtype=np.float32)
    generated_action, action_info = _joint_pair_generate_action_init(
        bundle,
        base_summary_exec=base_summary_exec,
        target_summary_exec=base_summary_exec,
        target_delta_exec=zero_delta,
        task_vec=task_vec,
        sub=sub,
        current_state=np.asarray(current_state, dtype=np.float32),
        device=base_future.device,
    )
    if generated_action is None:
        info.update(action_info)
        info["joint_hypothesis_init_reason"] = "generator_returned_none"
        return None, base_summary, base_summary_exec, info
    info.update(action_info)
    info.update(
        {
            "joint_hypothesis_init_used": True,
            "joint_hypothesis_init_reason": "generated_from_base_future",
            "joint_hypothesis_init_action_norm": float(torch.norm(generated_action.reshape(-1), p=2).item()),
        }
    )
    return generated_action.to(device=base_future.device, dtype=base_future.dtype), base_summary, base_summary_exec, info


def _joint_pair_joint_query_similarity(
    memory,
    mem_idx: int,
    *,
    current_state: np.ndarray | None,
    base_summary: np.ndarray,
    init_action_np: np.ndarray | None,
    subtask_index: int,
):
    sim_future = _joint_pair_safe_cos(base_summary, np.asarray(memory["base_summary"][mem_idx], dtype=np.float32))
    sim_action = 0.0
    if init_action_np is not None and "action_chunk" in memory:
        sim_action = _joint_pair_safe_cos(init_action_np, np.asarray(memory["action_chunk"][mem_idx], dtype=np.float32))
    sim_state = 0.0
    if current_state is not None and "state_start" in memory:
        mem_state = np.asarray(memory["state_start"][mem_idx], dtype=np.float32).reshape(-1)
        cur_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if mem_state.shape == cur_state.shape:
            sim_state = _joint_pair_safe_cos(cur_state, mem_state)
    stage_sim = 1.0
    if "subtask_index" in memory:
        mem_stage = int(memory["subtask_index"][mem_idx])
        stage_gap = abs(mem_stage - int(subtask_index or 0))
        stage_sim = max(0.0, 1.0 - 0.25 * float(stage_gap))
    total = 0.55 * sim_future + 0.25 * sim_action + 0.15 * sim_state + 0.05 * stage_sim
    return {
        "joint_query_similarity": float(total),
        "joint_query_similarity_future": float(sim_future),
        "joint_query_similarity_action": float(sim_action),
        "joint_query_similarity_state": float(sim_state),
        "joint_query_similarity_stage": float(stage_sim),
    }


def generate_initial_joint_action_intent(
    cfg,
    env,
    model,
    obs,
    lang_text: str,
    task: str,
    subtask_index: int,
    target_future: torch.Tensor | None,
):
    if target_future is None:
        return None, {"joint_hypothesis_init_used": False, "joint_hypothesis_init_reason": "missing_target_future"}
    current_state = _joint_pair_state_from_raw(snapshot_env_raw(env))
    if current_state is None:
        return None, {"joint_hypothesis_init_used": False, "joint_hypothesis_init_reason": "missing_state"}
    from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

    current_future = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
    current_summary_exec = _joint_pair_future_summary_exec(current_future.detach().float().cpu().numpy().astype(np.float32))
    target_summary_exec = _joint_pair_future_summary_exec(target_future.detach().float().cpu().numpy().astype(np.float32))
    target_delta_exec = np.asarray(target_summary_exec - current_summary_exec, dtype=np.float32)

    action_gen_bundle = load_joint_action_generator(cfg, model.device)
    if action_gen_bundle is None:
        action_gen_bundle = load_joint_pair_selector(cfg, model.device)
    if action_gen_bundle is None:
        return None, {"joint_hypothesis_init_used": False, "joint_hypothesis_init_reason": "missing_action_generator"}

    task_dim = None
    if isinstance(action_gen_bundle, dict) and "action_generator" in action_gen_bundle:
        task_dim = int(getattr(action_gen_bundle["action_generator"], "task_dim", 128))
    if task_dim is None:
        task_dim = int(action_gen_bundle.get("task_dim", 128))
    task_vec = np.asarray(q_hashed_task(task, task_dim), dtype=np.float32)
    sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)

    generated_action, action_info = _joint_pair_generate_action_init(
        action_gen_bundle,
        base_summary_exec=np.asarray(current_summary_exec, dtype=np.float32),
        target_summary_exec=np.asarray(target_summary_exec, dtype=np.float32),
        target_delta_exec=target_delta_exec,
        task_vec=task_vec,
        sub=sub,
        current_state=np.asarray(current_state, dtype=np.float32),
        device=model.device,
    )
    if generated_action is None:
        info = {"joint_hypothesis_init_used": False, "joint_hypothesis_init_reason": "generator_returned_none"}
        info.update(action_info)
        return None, info
    if not _tensor_is_finite(generated_action):
        info = {"joint_hypothesis_init_used": False, "joint_hypothesis_init_reason": "generator_non_finite"}
        info.update(action_info)
        return None, info
    info = {
        "joint_hypothesis_init_used": True,
        "joint_hypothesis_init_reason": "generated",
        "joint_hypothesis_init_action_norm": float(torch.norm(generated_action.reshape(-1), p=2).item()),
        "joint_hypothesis_init_delta_norm": float(np.linalg.norm(target_delta_exec.reshape(-1))),
    }
    info.update(action_info)
    return generated_action.to(device=model.device, dtype=target_future.dtype), info


def compute_shadow_joint_hypothesis(
    cfg,
    env,
    model,
    obs,
    lang_text: str,
    task: str,
    subtask_index: int,
):
    """Compute h=(z_f,z_a) for logging without writing policy overrides."""
    info = {"joint_belief_shadow_used": False}
    try:
        from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

        future = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
        future = _finite_tensor_or_none(future, name="joint_belief_shadow_future", info=info)
        if future is None:
            return None, None, info
        action_intent, action_info = generate_initial_joint_action_intent(
            cfg,
            env,
            model,
            obs,
            lang_text,
            task,
            int(subtask_index or 0),
            future,
        )
        info.update(action_info)
        info["joint_belief_shadow_used"] = True
        return future, action_intent, info
    except Exception as exc:
        info["joint_belief_shadow_error"] = str(exc)
        return None, None, info


def joint_pair_memory_free_mode(cfg) -> bool:
    return bool(getattr(cfg, "joint_pair_no_memory", False) or getattr(cfg, "joint_pair_direct_generate_only", False))


def load_joint_action_generator(cfg, device: torch.device):
    path = str(getattr(cfg, "joint_pair_action_generator_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _JOINT_ACTION_GENERATOR_CACHE:
        return _JOINT_ACTION_GENERATOR_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    gen = JointActionGeneratorMLP(
        state_dim=int(ckpt["state_dim"]),
        future_dim=int(ckpt["future_dim"]),
        action_dim=int(ckpt["action_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    gen.load_state_dict(ckpt["model_state"])
    gen.eval()
    bundle = {"action_generator": gen, "action_generator_ckpt": path}
    _JOINT_ACTION_GENERATOR_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded standalone action generator: {path}", flush=True)
    return bundle


def load_delta_action_repair(cfg, device: torch.device):
    path = str(getattr(cfg, "dynamic_coupling_delta_action_repair_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _DELTA_ACTION_REPAIR_CACHE:
        return _DELTA_ACTION_REPAIR_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    model = DeltaActionRepairMLP(
        state_dim=int(ckpt["state_dim"]),
        future_dim=int(ckpt["future_dim"]),
        action_dim=int(ckpt["action_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    load_result = model.load_state_dict(ckpt["model_state"], strict=False)
    if load_result.missing_keys:
        print(f"[INFO] DirectionalSensitivityTCN missing keys: {load_result.missing_keys}", flush=True)
    if load_result.unexpected_keys:
        print(f"[INFO] DirectionalSensitivityTCN unexpected keys: {load_result.unexpected_keys}", flush=True)
    model.eval()
    bundle = {
        "model": model,
        "state_dim": int(ckpt["state_dim"]),
        "future_dim": int(ckpt["future_dim"]),
        "chunk_len": int(ckpt["chunk_len"]),
        "action_dim": int(ckpt["action_dim"]),
        "task_dim": int(ckpt["task_dim"]),
        "path": path,
    }
    _DELTA_ACTION_REPAIR_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded delta action repair model: {path}", flush=True)
    return bundle


def load_joint_execution_update(cfg, device: torch.device):
    path = str(getattr(cfg, "joint_belief_transition_ckpt", "") or getattr(cfg, "joint_execution_update_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _JOINT_EXECUTION_UPDATE_CACHE:
        return _JOINT_EXECUTION_UPDATE_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    if str(ckpt.get("runtime_type", "")) == "factored_belief_action_transition":
        t_model = BeliefTransitionMLP(
            state_dim=int(ckpt["state_dim"]),
            summary_dim=int(ckpt["summary_dim"]),
            action_dim=int(ckpt["action_dim"]),
            task_dim=int(ckpt["task_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            dropout=float(ckpt.get("dropout", 0.1)),
        ).to(device)
        adaptation_input = str(ckpt.get("adaptation_input", "ht_hnext"))
        d_model = ActionAdaptationMLP(
            summary_dim=int(ckpt["summary_dim"]),
            action_dim=int(ckpt["action_dim"]),
            chunk_len=int(ckpt["chunk_len"]),
            task_dim=int(ckpt["task_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            dropout=float(ckpt.get("dropout", 0.1)),
            adaptation_input=adaptation_input,
        ).to(device)
        f_model = None
        if ckpt.get("f_model_state") is not None:
            f_model = ExpectedTransitionMLP(
                state_dim=int(ckpt["state_dim"]),
                summary_dim=int(ckpt["summary_dim"]),
                action_dim=int(ckpt["action_dim"]),
                task_dim=int(ckpt["task_dim"]),
                hidden_dim=int(ckpt["hidden_dim"]),
                dropout=float(ckpt.get("dropout", 0.1)),
            ).to(device)
            f_model.load_state_dict(ckpt["f_model_state"])
            f_model.eval()
        t_model.load_state_dict(ckpt["t_model_state"])
        d_model.load_state_dict(ckpt["d_model_state"])
        t_model.eval()
        d_model.eval()
        bundle = {
            "runtime_type": "factored_belief_action_transition",
            "t_model": t_model,
            "d_model": d_model,
            "f_model": f_model,
            "has_expected_transition": f_model is not None,
            "adaptation_input": adaptation_input,
            "state_dim": int(ckpt["state_dim"]),
            "summary_dim": int(ckpt["summary_dim"]),
            "action_dim": int(ckpt["action_dim"]),
            "chunk_len": int(ckpt["chunk_len"]),
            "task_dim": int(ckpt["task_dim"]),
            "has_learned_gate": True,
            "path": path,
        }
        _JOINT_EXECUTION_UPDATE_CACHE[cache_key] = bundle
        print(f"[INFO] Loaded factored belief/action transition model: {path}", flush=True)
        return bundle
    if str(ckpt.get("runtime_type", "")) in {"rssm_plan_compatibility", "full_rssm_plan_compatibility"}:
        if str(ckpt.get("runtime_type", "")) == "full_rssm_plan_compatibility":
            rssm_model = FullRSSMTransitionMLP(
                state_dim=int(ckpt["state_dim"]),
                summary_dim=int(ckpt["summary_dim"]),
                action_dim=int(ckpt["action_dim"]),
                task_dim=int(ckpt["task_dim"]),
                hidden_dim=int(ckpt["hidden_dim"]),
                dropout=float(ckpt.get("dropout", 0.1)),
                deter_dim=int(ckpt.get("rssm_deter_dim", 512)),
                stoch_dim=int(ckpt.get("rssm_stoch_dim", 256)),
            ).to(device)
        else:
            rssm_model = RSSMTransitionMLP(
                state_dim=int(ckpt["state_dim"]),
                summary_dim=int(ckpt["summary_dim"]),
                action_dim=int(ckpt["action_dim"]),
                task_dim=int(ckpt["task_dim"]),
                hidden_dim=int(ckpt["hidden_dim"]),
                dropout=float(ckpt.get("dropout", 0.1)),
            ).to(device)
        compat_head = None
        if ckpt.get("compat_head_state") is not None:
            # Kept only for old checkpoints/diagnostics. Directional sensitivity
            # is the primary keep/patch/replan decision path.
            compat_head = PlanCompatibilityHead(
                summary_dim=int(ckpt["summary_dim"]),
                action_dim=int(ckpt["action_dim"]),
                chunk_len=int(ckpt["chunk_len"]),
                task_dim=int(ckpt["task_dim"]),
                hidden_dim=int(ckpt["hidden_dim"]),
                dropout=float(ckpt.get("dropout", 0.1)),
            ).to(device)
            compat_head.load_state_dict(ckpt["compat_head_state"])
            compat_head.eval()
        rssm_model.load_state_dict(ckpt["rssm_model_state"])
        rssm_model.eval()
        bundle = {
            "runtime_type": str(ckpt.get("runtime_type", "rssm_plan_compatibility")),
            "rssm_model": rssm_model,
            "compat_head": compat_head,
            "state_dim": int(ckpt["state_dim"]),
            "summary_dim": int(ckpt["summary_dim"]),
            "action_dim": int(ckpt["action_dim"]),
            "chunk_len": int(ckpt["chunk_len"]),
            "task_dim": int(ckpt["task_dim"]),
            "path": path,
        }
        _JOINT_EXECUTION_UPDATE_CACHE[cache_key] = bundle
        print(f"[INFO] Loaded {bundle['runtime_type']} model: {path}", flush=True)
        return bundle
    model = JointExecutionUpdateMLP(
        state_dim=int(ckpt["state_dim"]),
        summary_dim=int(ckpt["summary_dim"]),
        action_dim=int(ckpt["action_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    bundle = {
        "model": model,
        "state_dim": int(ckpt["state_dim"]),
        "summary_dim": int(ckpt["summary_dim"]),
        "action_dim": int(ckpt["action_dim"]),
        "chunk_len": int(ckpt["chunk_len"]),
        "task_dim": int(ckpt["task_dim"]),
        "has_learned_gate": bool(ckpt.get("has_learned_gate", not any(str(k).startswith("gate_head.") for k in missing))),
        "path": path,
    }
    _JOINT_EXECUTION_UPDATE_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded joint belief transition model: {path}", flush=True)
    if missing or unexpected:
        print(f"[INFO] Joint belief transition load missing={missing} unexpected={unexpected}", flush=True)
    return bundle


def load_directional_sensitivity_tcn(cfg, device: torch.device):
    path = str(getattr(cfg, "directional_sensitivity_tcn_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _DIRECTIONAL_SENSITIVITY_TCN_CACHE:
        return _DIRECTIONAL_SENSITIVITY_TCN_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    model = DirectionalSensitivityTCN(
        state_dim=int(ckpt["state_dim"]),
        summary_dim=int(ckpt["summary_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_dim=int(ckpt["task_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        cond_dim=int(ckpt.get("cond_dim", 256)),
        num_blocks=int(ckpt.get("num_blocks", 3)),
        kernel_size=int(ckpt.get("kernel_size", 3)),
        dropout=float(ckpt.get("dropout", 0.05)),
        output_scale=float(ckpt.get("output_scale", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def _norm(name: str, dim: int, default: float) -> torch.Tensor:
        value = ckpt.get(name)
        if value is None:
            value = np.zeros((dim,), dtype=np.float32) if default == 0.0 else np.ones((dim,), dtype=np.float32)
        return torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(1, -1)).to(device)

    state_dim = int(ckpt["state_dim"])
    summary_dim = int(ckpt["summary_dim"])
    bundle = {
        "model": model,
        "path": path,
        "has_validity_head": bool(ckpt.get("has_validity_head", False)),
        "state_dim": state_dim,
        "summary_dim": summary_dim,
        "action_dim": int(ckpt["action_dim"]),
        "task_dim": int(ckpt["task_dim"]),
        "chunk_len": int(ckpt["chunk_len"]),
        "state_mean": _norm("state_mean", state_dim, 0.0),
        "state_std": torch.clamp(_norm("state_std", state_dim, 1.0), min=1e-6),
        "h_mean": _norm("h_mean", summary_dim, 0.0),
        "h_std": torch.clamp(_norm("h_std", summary_dim, 1.0), min=1e-6),
        "delta_h_mean": _norm("delta_h_mean", summary_dim, 0.0),
        "delta_h_std": torch.clamp(_norm("delta_h_std", summary_dim, 1.0), min=1e-6),
    }
    _DIRECTIONAL_SENSITIVITY_TCN_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded directional sensitivity TCN: {path}", flush=True)
    return bundle


def load_direct_action_correction_tcn(cfg, device: torch.device):
    path = str(getattr(cfg, "direct_action_correction_tcn_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _DIRECT_ACTION_CORRECTION_TCN_CACHE:
        return _DIRECT_ACTION_CORRECTION_TCN_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    model = DirectActionCorrectionTCN(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_dim=int(ckpt["task_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        cond_dim=int(ckpt.get("cond_dim", 256)),
        num_blocks=int(ckpt.get("num_blocks", 3)),
        kernel_size=int(ckpt.get("kernel_size", 3)),
        dropout=float(ckpt.get("dropout", 0.05)),
        output_scale=float(ckpt.get("output_scale", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def _norm(name: str, dim: int, default: float) -> torch.Tensor:
        value = ckpt.get(name)
        if value is None:
            value = np.zeros((dim,), dtype=np.float32) if default == 0.0 else np.ones((dim,), dtype=np.float32)
        return torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(1, -1)).to(device)

    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    bundle = {
        "model": model,
        "path": path,
        "has_validity_head": bool(ckpt.get("has_validity_head", False)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "task_dim": int(ckpt["task_dim"]),
        "chunk_len": int(ckpt["chunk_len"]),
        "state_start_mean": _norm("state_start_mean", state_dim, 0.0),
        "state_start_std": torch.clamp(_norm("state_start_std", state_dim, 1.0), min=1e-6),
        "state_end_mean": _norm("state_end_mean", state_dim, 0.0),
        "state_end_std": torch.clamp(_norm("state_end_std", state_dim, 1.0), min=1e-6),
        "executed_action_mean": _norm("executed_action_mean", action_dim, 0.0),
        "executed_action_std": torch.clamp(_norm("executed_action_std", action_dim, 1.0), min=1e-6),
    }
    _DIRECT_ACTION_CORRECTION_TCN_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded direct action correction TCN: {path}", flush=True)
    return bundle


def _smooth_action_delta(
    delta: torch.Tensor,
    smooth_weight: float,
    mpc_iters: int,
    affected_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if delta.dim() != 3 or delta.shape[1] < 3 or float(smooth_weight) <= 0.0 or int(mpc_iters) <= 0:
        return delta
    out = delta
    w = float(np.clip(float(smooth_weight), 0.0, 1.0))
    for _ in range(int(mpc_iters)):
        proposal = out.clone()
        proposal[:, 1:-1] = 0.5 * (out[:, :-2] + out[:, 2:])
        if affected_mask is not None:
            proposal = torch.where(affected_mask.unsqueeze(-1), proposal, out)
        out = (1.0 - w) * out + w * proposal
    return out


def predict_direct_action_correction_tcn(
    cfg,
    model,
    *,
    task: str,
    subtask_index: int,
    state_before: np.ndarray | None,
    state_after: np.ndarray | None,
    executed_action: np.ndarray | torch.Tensor | None,
    remaining_action_chunk: torch.Tensor | None,
):
    bundle = load_direct_action_correction_tcn(cfg, model.device)
    if bundle is None:
        return None, {"direct_action_correction_tcn_used": False, "reason": "missing_ckpt"}
    if state_before is None or state_after is None or executed_action is None or remaining_action_chunk is None:
        return None, {
            "direct_action_correction_tcn_used": True,
            "directional_sensitivity_tcn_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": "direct_missing_inputs",
        }
    try:
        state_start_np = _coerce_runtime_vector_np(state_before, int(bundle["state_dim"])).reshape(1, -1)
        state_end_np = _coerce_runtime_vector_np(state_after, int(bundle["state_dim"])).reshape(1, -1)
        exec_np = _coerce_runtime_vector_np(executed_action, int(bundle["action_dim"])).reshape(1, -1)
        action_t = _coerce_runtime_action_chunk(
            remaining_action_chunk,
            int(bundle["chunk_len"]),
            int(bundle["action_dim"]),
            model.device,
        )
        task_vec = np.asarray(q_hashed_task(task, int(bundle["task_dim"])), dtype=np.float32).reshape(1, -1)
        meta = _chunk_effect_runtime_meta(int(subtask_index or 0), int(action_t.shape[1]), int(bundle["chunk_len"]))
        with torch.no_grad():
            start_t = (torch.from_numpy(state_start_np).to(model.device) - bundle["state_start_mean"]) / bundle["state_start_std"]
            end_t = (torch.from_numpy(state_end_np).to(model.device) - bundle["state_end_mean"]) / bundle["state_end_std"]
            exec_t = (torch.from_numpy(exec_np).to(model.device) - bundle["executed_action_mean"]) / bundle["executed_action_std"]
            task_t = torch.from_numpy(task_vec).to(model.device)
            meta_t = torch.from_numpy(meta).to(model.device)
            use_structured = bool(
                getattr(cfg, "directional_sensitivity_tcn_structured_mask", False)
                and bool(bundle.get("has_validity_head", False))
                and hasattr(bundle["model"], "forward_with_heads")
            )
            if use_structured:
                heads = bundle["model"].forward_with_heads(start_t, end_t, exec_t, action_t, task_t, meta_t)
                pred = heads["delta"]
                invalidity_score = torch.sigmoid(heads["invalidity_logit"])
                uncertainty_score = torch.sigmoid(heads["uncertainty_logit"])
            else:
                pred = bundle["model"](start_t, end_t, exec_t, action_t, task_t, meta_t)
                invalidity_score = None
                uncertainty_score = None
        max_abs_delta = float(getattr(cfg, "directional_sensitivity_tcn_max_abs_delta", 0.015))
        if max_abs_delta > 0:
            pred = torch.clamp(pred, -max_abs_delta, max_abs_delta)
        raw_step_norm = torch.norm(pred.detach().float(), dim=-1)
        if invalidity_score is not None:
            validity_threshold = float(getattr(cfg, "directional_sensitivity_tcn_validity_threshold", 0.5))
            affected_mask = _structured_binary_mask_from_scores(
                invalidity_score,
                patch_cost=float(getattr(cfg, "directional_sensitivity_tcn_selection_lambda", validity_threshold)),
                switch_cost=float(getattr(cfg, "directional_sensitivity_tcn_selection_beta", 0.15)),
            )
            if validity_threshold > 0.0:
                affected_mask = affected_mask & (invalidity_score >= validity_threshold)
            uncertainty_max = float(torch.max(uncertainty_score).item()) if uncertainty_score is not None else 0.0
            invalidity_mean = float(torch.mean(invalidity_score).item())
            invalidity_max = float(torch.max(invalidity_score).item())
        else:
            step_threshold = float(getattr(cfg, "directional_sensitivity_tcn_step_threshold", 0.0))
            affected_mask = raw_step_norm >= step_threshold if step_threshold > 0.0 else torch.ones_like(raw_step_norm, dtype=torch.bool)
            uncertainty_max = 0.0
            invalidity_mean = 0.0
            invalidity_max = 0.0
        pred = torch.where(affected_mask.unsqueeze(-1), pred, torch.zeros_like(pred))
        pred = _smooth_action_delta(
            pred,
            smooth_weight=float(getattr(cfg, "directional_sensitivity_tcn_smooth_weight", 0.0)),
            mpc_iters=int(getattr(cfg, "directional_sensitivity_tcn_mpc_iters", 1)),
            affected_mask=affected_mask,
        )
        pred = torch.where(affected_mask.unsqueeze(-1), pred, torch.zeros_like(pred))
        pred_norm = float(torch.norm(pred.reshape(-1), p=2).item())
        affected_count = int(affected_mask.reshape(-1).sum().item())
        affected_fraction = float(affected_count) / float(max(1, int(affected_mask.numel())))
        affected_np = affected_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        affected_segments = 0
        in_segment = False
        for is_affected in affected_np:
            if is_affected and not in_segment:
                affected_segments += 1
                in_segment = True
            elif not is_affected:
                in_segment = False
        max_patch_fraction = float(getattr(cfg, "directional_sensitivity_tcn_max_patch_fraction", 1.0))
        max_affected_segments = int(getattr(cfg, "directional_sensitivity_tcn_max_affected_segments", 0))
        uncertainty_replan_threshold = float(getattr(cfg, "directional_sensitivity_tcn_uncertainty_replan_threshold", 0.0))
        min_norm = float(getattr(cfg, "directional_sensitivity_tcn_min_delta_norm", 0.0))
        if invalidity_score is not None and uncertainty_replan_threshold > 0.0 and uncertainty_max >= uncertainty_replan_threshold:
            decision_hint = "replan"
            reason = "direct_structured_uncertainty_too_high"
        elif affected_count <= 0 or pred_norm < min_norm:
            decision_hint = "keep"
            reason = "direct_no_affected_actions"
        elif (
            (max_patch_fraction > 0.0 and affected_fraction > max_patch_fraction)
            or (max_affected_segments > 0 and affected_segments > max_affected_segments)
        ):
            decision_hint = "replan"
            reason = "direct_affected_fraction_or_spread_too_large"
        else:
            decision_hint = "repair_suffix"
            reason = "direct_selective_affected_actions"
        if pred_norm < min_norm:
            pred_out = None
        else:
            pred_out = pred.detach()
        return pred_out, {
            "direct_action_correction_tcn_used": True,
            "direct_action_correction_tcn_applied": decision_hint == "repair_suffix",
            "direct_action_correction_tcn_path": str(bundle["path"]),
            # Reuse the existing clean-controller decision path and logging keys.
            "directional_sensitivity_tcn_used": True,
            "directional_sensitivity_tcn_applied": decision_hint == "repair_suffix",
            "directional_sensitivity_tcn_decision_hint": decision_hint,
            "directional_sensitivity_tcn_reason": reason,
            "directional_sensitivity_tcn_delta_h_norm": 0.0,
            "directional_sensitivity_tcn_affected_count": affected_count,
            "directional_sensitivity_tcn_affected_fraction": affected_fraction,
            "directional_sensitivity_tcn_affected_segments": affected_segments,
            "directional_sensitivity_tcn_structured_mask": bool(invalidity_score is not None),
            "directional_sensitivity_tcn_invalidity_mean": invalidity_mean,
            "directional_sensitivity_tcn_invalidity_max": invalidity_max,
            "directional_sensitivity_tcn_uncertainty_max": uncertainty_max,
            "directional_sensitivity_tcn_delta_norm": pred_norm,
            "directional_sensitivity_tcn_delta_abs_max": float(torch.max(torch.abs(pred)).item()),
            "directional_sensitivity_tcn_delta_abs_mean": float(torch.mean(torch.abs(pred)).item()),
            "directional_sensitivity_tcn_step_norm_mean": float(torch.mean(raw_step_norm).item()),
            "directional_sensitivity_tcn_step_norm_max": float(torch.max(raw_step_norm).item()),
            "directional_sensitivity_tcn_max_abs_delta": max_abs_delta,
        }
    except Exception as exc:
        return None, {"direct_action_correction_tcn_used": True, "direct_action_correction_tcn_error": str(exc)}


def predict_directional_sensitivity_tcn(
    cfg,
    model,
    *,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    belief_before: np.ndarray | torch.Tensor | None,
    belief_after: np.ndarray | torch.Tensor | None,
    innovation: torch.Tensor | None,
    remaining_action_chunk: torch.Tensor | None,
):
    bundle = load_directional_sensitivity_tcn(cfg, model.device)
    if bundle is None:
        return None, {"directional_sensitivity_tcn_used": False, "reason": "missing_ckpt"}
    if current_state is None or remaining_action_chunk is None:
        return None, {"directional_sensitivity_tcn_used": True, "reason": "missing_state_or_action"}
    try:
        state_np = _coerce_runtime_vector_np(current_state, int(bundle["state_dim"])).reshape(1, -1)
        action_t = _coerce_runtime_action_chunk(
            remaining_action_chunk,
            int(bundle["chunk_len"]),
            int(bundle["action_dim"]),
            model.device,
        )

        if belief_before is None:
            h_np = np.zeros((1, int(bundle["summary_dim"])), dtype=np.float32)
        elif torch.is_tensor(belief_before):
            h_np = _pool_exec_summary_np(belief_before.detach().float().cpu().numpy(), int(bundle["summary_dim"])).reshape(1, -1)
        else:
            h_np = _pool_exec_summary_np(belief_before, int(bundle["summary_dim"])).reshape(1, -1)

        if innovation is not None and torch.is_tensor(innovation):
            delta_h_np = _pool_exec_summary_np(innovation.detach().float().cpu().numpy(), int(bundle["summary_dim"])).reshape(1, -1)
        elif belief_after is not None and belief_before is not None:
            after_np = (
                _pool_exec_summary_np(belief_after.detach().float().cpu().numpy(), int(bundle["summary_dim"]))
                if torch.is_tensor(belief_after)
                else _pool_exec_summary_np(belief_after, int(bundle["summary_dim"]))
            )
            before_np = (
                _pool_exec_summary_np(belief_before.detach().float().cpu().numpy(), int(bundle["summary_dim"]))
                if torch.is_tensor(belief_before)
                else _pool_exec_summary_np(belief_before, int(bundle["summary_dim"]))
            )
            delta_h_np = (after_np - before_np).reshape(1, -1).astype(np.float32)
        else:
            delta_h_np = np.zeros((1, int(bundle["summary_dim"])), dtype=np.float32)

        task_vec = np.asarray(q_hashed_task(task, int(bundle["task_dim"])), dtype=np.float32).reshape(1, -1)
        meta = _chunk_effect_runtime_meta(int(subtask_index or 0), int(action_t.shape[1]), int(bundle["chunk_len"]))
        with torch.no_grad():
            state_t = (torch.from_numpy(state_np).to(model.device) - bundle["state_mean"]) / bundle["state_std"]
            h_t = (torch.from_numpy(h_np).to(model.device) - bundle["h_mean"]) / bundle["h_std"]
            delta_h_t = (torch.from_numpy(delta_h_np).to(model.device) - bundle["delta_h_mean"]) / bundle["delta_h_std"]
            use_structured = bool(
                getattr(cfg, "directional_sensitivity_tcn_structured_mask", False)
                and bool(bundle.get("has_validity_head", False))
                and hasattr(bundle["model"], "forward_with_heads")
            )
            if use_structured:
                heads = bundle["model"].forward_with_heads(
                    state_t,
                    h_t,
                    action_t,
                    delta_h_t,
                    torch.from_numpy(task_vec).to(model.device),
                    torch.from_numpy(meta).to(model.device),
                )
                pred = heads["delta"]
                invalidity_score = torch.sigmoid(heads["invalidity_logit"])
                uncertainty_score = torch.sigmoid(heads["uncertainty_logit"])
            else:
                pred = bundle["model"](
                    state_t,
                    h_t,
                    action_t,
                    delta_h_t,
                    torch.from_numpy(task_vec).to(model.device),
                    torch.from_numpy(meta).to(model.device),
                )
                invalidity_score = None
                uncertainty_score = None
        max_abs_delta = float(getattr(cfg, "directional_sensitivity_tcn_max_abs_delta", 0.015))
        if max_abs_delta > 0:
            pred = torch.clamp(pred, -max_abs_delta, max_abs_delta)
        raw_step_norm = torch.norm(pred.detach().float(), dim=-1)
        step_threshold = float(getattr(cfg, "directional_sensitivity_tcn_step_threshold", 0.0))
        top_k = int(getattr(cfg, "directional_sensitivity_tcn_top_k", 0))
        top_k_floor = float(getattr(cfg, "directional_sensitivity_tcn_top_k_floor", 0.0))
        top_k_min_margin = float(getattr(cfg, "directional_sensitivity_tcn_top_k_min_margin", 0.0))
        force_top_k_patch = bool(getattr(cfg, "directional_sensitivity_tcn_force_top_k_patch", False))
        if invalidity_score is not None:
            validity_threshold = float(getattr(cfg, "directional_sensitivity_tcn_validity_threshold", 0.5))
            structured_lambda = float(getattr(cfg, "directional_sensitivity_tcn_selection_lambda", validity_threshold))
            structured_beta = float(getattr(cfg, "directional_sensitivity_tcn_selection_beta", 0.15))
            affected_mask = _structured_binary_mask_from_scores(
                invalidity_score,
                patch_cost=structured_lambda,
                switch_cost=structured_beta,
            )
            if validity_threshold > 0.0:
                affected_mask = affected_mask & (invalidity_score >= validity_threshold)
            uncertainty_max = float(torch.max(uncertainty_score).item()) if uncertainty_score is not None else 0.0
            invalidity_mean = float(torch.mean(invalidity_score).item())
            invalidity_max = float(torch.max(invalidity_score).item())
        elif top_k > 0:
            flat_scores = raw_step_norm.reshape(-1)
            affected_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
            top_k_selected = False
            if flat_scores.numel() > 0:
                max_score = float(torch.max(flat_scores).item())
                median_score = float(torch.median(flat_scores).item())
                if max_score >= top_k_floor and (max_score - median_score) >= top_k_min_margin:
                    k = min(top_k, int(flat_scores.numel()))
                    _, top_idx = torch.topk(flat_scores, k=k, largest=True)
                    affected_mask[top_idx] = True
                    top_k_selected = True
            affected_mask = affected_mask.reshape_as(raw_step_norm)
            uncertainty_max = 0.0
            invalidity_mean = 0.0
            invalidity_max = 0.0
        else:
            affected_mask = (
                raw_step_norm >= step_threshold
                if step_threshold > 0.0
                else torch.ones_like(raw_step_norm, dtype=torch.bool)
            )
            uncertainty_max = 0.0
            invalidity_mean = 0.0
            invalidity_max = 0.0
        pred = torch.where(affected_mask.unsqueeze(-1), pred, torch.zeros_like(pred))
        pred = _smooth_action_delta(
            pred,
            smooth_weight=float(getattr(cfg, "directional_sensitivity_tcn_smooth_weight", 0.0)),
            mpc_iters=int(getattr(cfg, "directional_sensitivity_tcn_mpc_iters", 1)),
            affected_mask=affected_mask,
        )
        pred = torch.where(affected_mask.unsqueeze(-1), pred, torch.zeros_like(pred))
        min_norm = float(getattr(cfg, "directional_sensitivity_tcn_min_delta_norm", 0.0))
        pred_norm = float(torch.norm(pred.reshape(-1), p=2).item())
        affected_count = int(affected_mask.reshape(-1).sum().item())
        affected_fraction = float(affected_count) / float(max(1, int(affected_mask.numel())))
        affected_np = affected_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        affected_segments = 0
        in_segment = False
        for is_affected in affected_np:
            if is_affected and not in_segment:
                affected_segments += 1
                in_segment = True
            elif not is_affected:
                in_segment = False
        delta_h_norm = float(np.linalg.norm(delta_h_np.reshape(-1), ord=2))
        max_patch_fraction = float(getattr(cfg, "directional_sensitivity_tcn_max_patch_fraction", 1.0))
        max_affected_segments = int(getattr(cfg, "directional_sensitivity_tcn_max_affected_segments", 0))
        replan_delta_h_norm = float(getattr(cfg, "directional_sensitivity_tcn_replan_delta_h_norm", 0.0))
        uncertainty_replan_threshold = float(getattr(cfg, "directional_sensitivity_tcn_uncertainty_replan_threshold", 0.0))
        replan_disabled = bool(getattr(cfg, "dynamic_coupling_disable_replan", False))
        top_k_forced_patch = bool(force_top_k_patch and top_k > 0 and affected_count > 0)
        if (
            invalidity_score is not None
            and uncertainty_replan_threshold > 0.0
            and uncertainty_max >= uncertainty_replan_threshold
            and not (replan_disabled and top_k_forced_patch)
        ):
            decision_hint = "replan"
            reason = "structured_uncertainty_too_high"
        elif (
            replan_delta_h_norm > 0.0
            and delta_h_norm > replan_delta_h_norm
            and not (replan_disabled and top_k_forced_patch)
        ):
            decision_hint = "replan"
            reason = "delta_h_too_large"
        elif affected_count <= 0 or (pred_norm < min_norm and not (force_top_k_patch and top_k > 0)):
            decision_hint = "keep"
            reason = "no_affected_actions"
        elif (
            (max_patch_fraction > 0.0 and affected_fraction > max_patch_fraction)
            or (max_affected_segments > 0 and affected_segments > max_affected_segments)
        ):
            decision_hint = "replan"
            reason = "affected_fraction_or_spread_too_large"
        else:
            decision_hint = "repair_suffix"
            reason = "selective_affected_actions"
        if pred_norm < min_norm and not (force_top_k_patch and top_k > 0 and affected_count > 0):
            return None, {
                "directional_sensitivity_tcn_used": True,
                "directional_sensitivity_tcn_applied": False,
                "directional_sensitivity_tcn_reason": reason,
                "directional_sensitivity_tcn_decision_hint": decision_hint,
                "directional_sensitivity_tcn_delta_norm": pred_norm,
                "directional_sensitivity_tcn_delta_h_norm": delta_h_norm,
                "directional_sensitivity_tcn_affected_count": affected_count,
                "directional_sensitivity_tcn_affected_fraction": affected_fraction,
                "directional_sensitivity_tcn_affected_segments": affected_segments,
            }
        return pred.detach(), {
            "directional_sensitivity_tcn_used": True,
            "directional_sensitivity_tcn_applied": decision_hint == "repair_suffix",
            "directional_sensitivity_tcn_decision_hint": decision_hint,
            "directional_sensitivity_tcn_reason": reason,
            "directional_sensitivity_tcn_delta_h_norm": delta_h_norm,
            "directional_sensitivity_tcn_affected_count": affected_count,
            "directional_sensitivity_tcn_affected_fraction": affected_fraction,
            "directional_sensitivity_tcn_affected_segments": affected_segments,
            "directional_sensitivity_tcn_step_threshold": step_threshold,
            "directional_sensitivity_tcn_top_k": top_k,
            "directional_sensitivity_tcn_top_k_floor": top_k_floor,
            "directional_sensitivity_tcn_top_k_min_margin": top_k_min_margin,
            "directional_sensitivity_tcn_force_top_k_patch": force_top_k_patch,
            "directional_sensitivity_tcn_max_patch_fraction": max_patch_fraction,
            "directional_sensitivity_tcn_max_affected_segments": max_affected_segments,
            "directional_sensitivity_tcn_replan_delta_h_norm": replan_delta_h_norm,
            "directional_sensitivity_tcn_structured_mask": bool(invalidity_score is not None),
            "directional_sensitivity_tcn_invalidity_mean": invalidity_mean,
            "directional_sensitivity_tcn_invalidity_max": invalidity_max,
            "directional_sensitivity_tcn_uncertainty_max": uncertainty_max,
            "directional_sensitivity_tcn_delta_norm": pred_norm,
            "directional_sensitivity_tcn_delta_abs_max": float(torch.max(torch.abs(pred)).item()),
            "directional_sensitivity_tcn_delta_abs_mean": float(torch.mean(torch.abs(pred)).item()),
            "directional_sensitivity_tcn_step_norm_mean": float(torch.mean(raw_step_norm).item()),
            "directional_sensitivity_tcn_step_norm_max": float(torch.max(raw_step_norm).item()),
            "directional_sensitivity_tcn_max_abs_delta": max_abs_delta,
        }
    except Exception as exc:
        return None, {"directional_sensitivity_tcn_used": True, "directional_sensitivity_tcn_error": str(exc)}


def _suffix_control_ablation_delta_from_teacher(
    cfg,
    teacher_info,
    *,
    device: torch.device,
    sequence_index: int,
    subtask_index: int,
    step: int,
):
    mode = str(getattr(cfg, "suffix_mask_ablation_mode", "off") or "off")
    if mode in {"off", "tcn_selective"}:
        return None, {"suffix_mask_ablation_mode": mode, "suffix_mask_ablation_used": False}
    if mode == "full_replan":
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "replan",
            "directional_sensitivity_tcn_reason": "ablation_full_replan",
            "dynamic_coupling_full_replan_requested": True,
        }
    if mode == "no_patch":
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": "ablation_no_patch",
        }
    if not isinstance(teacher_info, dict) or not bool(teacher_info.get("available", False)):
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": "ablation_teacher_unavailable",
        }

    delta = teacher_info.get("suffix_delta_action", None)
    if delta is None:
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": "ablation_missing_teacher_delta",
        }
    if not torch.is_tensor(delta):
        delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
    else:
        delta_t = delta.detach().to(device=device, dtype=torch.float32)
    if delta_t.dim() == 2:
        delta_t = delta_t.unsqueeze(0)
    if delta_t.numel() == 0:
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": "ablation_empty_teacher_delta",
        }
    max_abs_delta = float(getattr(cfg, "suffix_mask_ablation_max_abs_delta", 0.0))
    if max_abs_delta > 0:
        delta_t = torch.clamp(delta_t, -max_abs_delta, max_abs_delta)

    scores = teacher_info.get("step_scores", None)
    if scores is None:
        score_t = torch.norm(delta_t.detach().float(), dim=-1)
    elif torch.is_tensor(scores):
        score_t = scores.detach().to(device=device, dtype=torch.float32)
    else:
        score_t = torch.as_tensor(scores, dtype=torch.float32, device=device)
    if score_t.dim() == 1:
        score_t = score_t.unsqueeze(0)
    threshold = float(getattr(cfg, "suffix_mask_ablation_teacher_step_threshold", 0.0))
    teacher_mask = score_t >= threshold if threshold > 0 else score_t > 0
    if mode == "gt_selective":
        mask = teacher_mask
    elif mode == "whole_suffix":
        mask = torch.ones_like(teacher_mask, dtype=torch.bool)
    elif mode == "complement_mask":
        mask = ~teacher_mask
    elif mode == "random_mask":
        mask_np = np.zeros(tuple(teacher_mask.shape), dtype=bool)
        teacher_np = teacher_mask.detach().cpu().numpy().astype(bool)
        seed = int(getattr(cfg, "suffix_mask_ablation_seed", 0))
        seed = seed + int(sequence_index) * 1000003 + int(subtask_index) * 9176 + int(step) * 37
        rng = np.random.default_rng(seed)
        for row_id, row in enumerate(teacher_np):
            count = int(row.sum())
            if count <= 0:
                continue
            idx = rng.choice(row.size, size=min(count, row.size), replace=False)
            mask_np[row_id, idx] = True
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
    else:
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": f"ablation_unknown_mode_{mode}",
        }

    patched = torch.where(mask.unsqueeze(-1), delta_t, torch.zeros_like(delta_t))
    pred_norm = float(torch.norm(patched.reshape(-1), p=2).item())
    affected_count = int(mask.reshape(-1).sum().item())
    affected_fraction = float(affected_count) / float(max(1, int(mask.numel())))
    min_fraction = float(getattr(cfg, "suffix_mask_ablation_min_affected_fraction", 0.0))
    max_fraction = float(getattr(cfg, "suffix_mask_ablation_max_affected_fraction", 1.0))
    if affected_fraction < min_fraction or affected_fraction > max_fraction:
        return None, {
            "suffix_mask_ablation_mode": mode,
            "suffix_mask_ablation_used": True,
            "suffix_mask_ablation_teacher_threshold": threshold,
            "suffix_mask_ablation_affected_count": affected_count,
            "suffix_mask_ablation_affected_fraction": affected_fraction,
            "suffix_mask_ablation_delta_norm": pred_norm,
            "directional_sensitivity_tcn_used": True,
            "directional_sensitivity_tcn_applied": False,
            "directional_sensitivity_tcn_decision_hint": "keep",
            "directional_sensitivity_tcn_reason": "ablation_affected_fraction_out_of_range",
            "directional_sensitivity_tcn_delta_norm": pred_norm,
            "directional_sensitivity_tcn_affected_count": affected_count,
            "directional_sensitivity_tcn_affected_fraction": affected_fraction,
            "suffix_mask_ablation_min_affected_fraction": min_fraction,
            "suffix_mask_ablation_max_affected_fraction": max_fraction,
        }
    decision_hint = "repair_suffix" if affected_count > 0 and pred_norm > 0 else "keep"
    return patched if decision_hint == "repair_suffix" else None, {
        "suffix_mask_ablation_mode": mode,
        "suffix_mask_ablation_used": True,
        "suffix_mask_ablation_teacher_threshold": threshold,
        "suffix_mask_ablation_affected_count": affected_count,
        "suffix_mask_ablation_affected_fraction": affected_fraction,
        "suffix_mask_ablation_delta_norm": pred_norm,
        "directional_sensitivity_tcn_used": True,
        "directional_sensitivity_tcn_applied": decision_hint == "repair_suffix",
        "directional_sensitivity_tcn_decision_hint": decision_hint,
        "directional_sensitivity_tcn_reason": f"ablation_{mode}",
        "directional_sensitivity_tcn_delta_norm": pred_norm,
        "directional_sensitivity_tcn_affected_count": affected_count,
        "directional_sensitivity_tcn_affected_fraction": affected_fraction,
    }


def _suffix_mask_schedule_allows(cfg, *, sequence_index: int, subtask_index: int, step: int) -> bool:
    if not SUFFIX_MASK_ABLATION_SCHEDULE_SET:
        return True
    key = (int(sequence_index), int(subtask_index), int(step))
    return key in SUFFIX_MASK_ABLATION_SCHEDULE_SET


def load_suffix_trigger_runtime(cfg, device: torch.device):
    path = str(getattr(cfg, "suffix_trigger_mlp_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _SUFFIX_TRIGGER_CACHE:
        return _SUFFIX_TRIGGER_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    model = SuffixTriggerMLP(
        feature_dim=int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bundle = {
        "model": model,
        "feature_dim": int(ckpt["feature_dim"]),
        "task_dim": int(ckpt.get("task_dim", 32)),
        "path": path,
    }
    _SUFFIX_TRIGGER_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded suffix trigger model: {path}", flush=True)
    return bundle


def predict_suffix_trigger_runtime(
    bundle,
    *,
    task: str,
    subtask_index: int,
    chunk_decision: dict,
    coupling_info: dict,
) -> dict:
    if bundle is None:
        return {"used": False}
    try:
        feature = build_suffix_trigger_feature(
            task=task,
            subtask_index=int(subtask_index or 0),
            residual_norm=float(chunk_decision.get("residual_norm", 0.0) or 0.0),
            compat=float(chunk_decision.get("compat", 0.0) or 0.0),
            success_prob=float(chunk_decision.get("success_prob", 0.0) or 0.0),
            current_to_target_norm=float(coupling_info.get("dynamic_coupling_current_to_target_norm", 0.0) or 0.0),
            shift_norm_vs_slow=float(coupling_info.get("dynamic_coupling_shift_norm_vs_slow", 0.0) or 0.0),
            committed_scalar=float(coupling_info.get("dynamic_coupling_committed_scalar", 0.0) or 0.0),
            learned_gate=float(chunk_decision.get("joint_execution_learned_gate_raw", 0.0) or 0.0),
            joint_gate=float(chunk_decision.get("joint_execution_gate", 0.0) or 0.0),
            joint_error=float(chunk_decision.get("joint_execution_error", 0.0) or 0.0),
            task_dim=int(bundle["task_dim"]),
        )
        x = torch.from_numpy(feature).to(next(bundle["model"].parameters()).device).unsqueeze(0)
        with torch.no_grad():
            prob = float(torch.sigmoid(bundle["model"](x))[0].item())
        return {"used": True, "prob": prob}
    except Exception as exc:
        return {"used": True, "error": str(exc)}


def load_future_action_dynamics_runtime(cfg, device: torch.device):
    path = str(getattr(cfg, "dynamic_coupling_chunk_effect_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _FUTURE_ACTION_DYNAMICS_CACHE:
        return _FUTURE_ACTION_DYNAMICS_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    if "summary_dim" in ckpt and "future_dim" not in ckpt:
        model = ChunkEffectPhiMLP(
            state_dim=int(ckpt["state_dim"]),
            summary_dim=int(ckpt["summary_dim"]),
            action_dim=int(ckpt["action_dim"]),
            chunk_len=int(ckpt["chunk_len"]),
            task_dim=int(ckpt["task_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            dropout=float(ckpt.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        bundle = {
            "model": model,
            "runtime_type": "chunk_effect_phi_mlp",
            "task_dim": int(ckpt["task_dim"]),
            "summary_dim": int(ckpt["summary_dim"]),
            "state_dim": int(ckpt["state_dim"]),
            "chunk_len": int(ckpt["chunk_len"]),
            "action_dim": int(ckpt["action_dim"]),
            "path": path,
        }
        _FUTURE_ACTION_DYNAMICS_CACHE[cache_key] = bundle
        print(f"[INFO] Loaded chunk-effect phi model: {path}", flush=True)
        return bundle
    model = FutureActionDynamicsProbe(
        action_dim=int(ckpt["action_dim"]),
        future_dim=int(ckpt["future_dim"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        z_dim=int(ckpt["z_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    delta_mean = torch.from_numpy(np.asarray(ckpt["delta_mean"], dtype=np.float32)).to(device)
    delta_std = torch.from_numpy(np.asarray(ckpt["delta_std"], dtype=np.float32)).to(device)
    bundle = {
        "model": model,
        "runtime_type": "future_action_dynamics_probe",
        "task_dim": int(ckpt["task_dim"]),
        "future_dim": int(ckpt["future_dim"]),
        "action_dim": int(ckpt["action_dim"]),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "path": path,
    }
    _FUTURE_ACTION_DYNAMICS_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded future action dynamics probe: {path}", flush=True)
    return bundle


def load_summary_future_adapter(cfg, device: torch.device):
    path = str(getattr(cfg, "joint_pair_summary_future_adapter_ckpt", "") or "")
    if not path:
        return None
    cache_key = (path, str(device))
    if cache_key in _SUMMARY_FUTURE_ADAPTER_CACHE:
        return _SUMMARY_FUTURE_ADAPTER_CACHE[cache_key]
    ckpt = torch.load(path, map_location="cpu")
    adapter = SummaryConditionedFutureAdapter(
        future_dim=int(ckpt["future_dim"]),
        summary_dim=int(ckpt["summary_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    ).to(device)
    adapter.load_state_dict(ckpt["model_state"])
    adapter.eval()
    bundle = {
        "adapter": adapter,
        "summary_dim": int(ckpt["summary_dim"]),
        "future_token_shape": tuple(int(x) for x in ckpt.get("future_token_shape", ())),
        "path": path,
    }
    _SUMMARY_FUTURE_ADAPTER_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded standalone summary future adapter: {path}", flush=True)
    return bundle


def apply_summary_adapter_to_runtime_future(
    cfg,
    *,
    base_future: torch.Tensor | None,
    joint_update_payload: dict | None,
    gate: float,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "summary_future_adapter_runtime_used": False,
        "summary_future_adapter_runtime_applied": False,
    }
    if not bool(getattr(cfg, "joint_belief_summary_future_adapter_update", False)):
        info["summary_future_adapter_runtime_reason"] = "disabled"
        return None, info
    if base_future is None or not torch.is_tensor(base_future):
        info["summary_future_adapter_runtime_reason"] = "missing_base_future"
        return None, info
    if joint_update_payload is None:
        info["summary_future_adapter_runtime_reason"] = "missing_joint_update"
        return None, info
    if not np.isfinite(float(gate)) or float(gate) <= 0.0:
        info["summary_future_adapter_runtime_reason"] = "gate_zero"
        return None, info
    adapter_bundle = load_summary_future_adapter(cfg, base_future.device)
    if adapter_bundle is None:
        info["summary_future_adapter_runtime_reason"] = "missing_adapter_ckpt"
        return None, info

    delta_key = str(getattr(cfg, "joint_belief_summary_future_adapter_delta_key", "future_delta") or "future_delta")
    delta_value = joint_update_payload.get(delta_key)
    if delta_value is None and delta_key != "innovation":
        delta_value = joint_update_payload.get("innovation")
        delta_key = "innovation"
    if delta_value is None:
        info["summary_future_adapter_runtime_reason"] = "missing_summary_delta"
        return None, info
    try:
        summary_dim = int(adapter_bundle["summary_dim"])
        base_np = base_future.detach().float().cpu().numpy()
        if base_np.ndim != 2:
            info["summary_future_adapter_runtime_reason"] = f"bad_base_future_ndim:{base_np.ndim}"
            return None, info
        source_summary_np = _pool_exec_summary_np(_joint_pair_future_summary_exec(base_np), summary_dim)
        if torch.is_tensor(delta_value):
            delta_np = delta_value.detach().float().cpu().numpy()
        else:
            delta_np = np.asarray(delta_value, dtype=np.float32)
        delta_summary_np = _pool_exec_summary_np(delta_np, summary_dim)
        delta_scale = float(getattr(cfg, "joint_belief_summary_future_adapter_delta_scale", 1.0))
        target_summary_np = source_summary_np + delta_scale * delta_summary_np
        source_summary = torch.from_numpy(source_summary_np.reshape(1, -1)).to(
            device=base_future.device,
            dtype=base_future.dtype,
        )
        target_summary = torch.from_numpy(target_summary_np.reshape(1, -1)).to(
            device=base_future.device,
            dtype=base_future.dtype,
        )
        with torch.no_grad():
            corrected, aux = adapter_bundle["adapter"](
                base_future.unsqueeze(0).to(dtype=base_future.dtype),
                source_summary,
                target_summary,
            )
        corrected_future = corrected.squeeze(0).detach().to(dtype=base_future.dtype)
        if corrected_future.shape != base_future.shape:
            info["summary_future_adapter_runtime_reason"] = (
                f"shape_mismatch:{tuple(corrected_future.shape)}!={tuple(base_future.shape)}"
            )
            return None, info
        if not torch.isfinite(corrected_future).all():
            info["summary_future_adapter_runtime_reason"] = "non_finite_output"
            return None, info
        raw_shift = corrected_future - base_future
        raw_shift_norm = float(torch.norm(raw_shift.reshape(-1), p=2).item())
        max_shift = float(getattr(cfg, "joint_belief_summary_future_adapter_max_shift_norm", 0.0))
        shrink = 1.0
        if max_shift > 0.0 and raw_shift_norm > max_shift:
            shrink = max_shift / max(raw_shift_norm, 1e-8)
            corrected_future = base_future + shrink * raw_shift
        mix = float(np.clip(getattr(cfg, "joint_belief_summary_future_adapter_mix", 1.0), 0.0, 1.0))
        corrected_future = base_future + mix * (corrected_future - base_future)
        final_shift_norm = float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item())
        info.update(
            {
                "summary_future_adapter_runtime_used": True,
                "summary_future_adapter_runtime_applied": True,
                "summary_future_adapter_runtime_reason": "applied",
                "summary_future_adapter_runtime_delta_key": delta_key,
                "summary_future_adapter_runtime_delta_scale": delta_scale,
                "summary_future_adapter_runtime_mix": mix,
                "summary_future_adapter_runtime_raw_shift_norm": raw_shift_norm,
                "summary_future_adapter_runtime_final_shift_norm": final_shift_norm,
                "summary_future_adapter_runtime_shrink": shrink,
                "summary_future_adapter_runtime_gate_mean": float(aux["gate"].mean().item()),
                "summary_future_adapter_runtime_delta_norm": float(torch.norm(aux["delta"].reshape(-1), p=2).item()),
            }
        )
        return corrected_future, info
    except Exception as exc:
        info["summary_future_adapter_runtime_reason"] = f"error:{exc}"
        return None, info


def _pool_exec_summary_np(summary: np.ndarray, out_dim: int) -> np.ndarray:
    summary = np.asarray(summary, dtype=np.float32).reshape(-1)
    if summary.size == int(out_dim):
        return summary.astype(np.float32)
    if summary.size < int(out_dim):
        out = np.zeros((int(out_dim),), dtype=np.float32)
        out[: summary.size] = summary
        return out
    trim = (summary.size // int(out_dim)) * int(out_dim)
    return summary[:trim].reshape(int(out_dim), trim // int(out_dim)).mean(axis=1).astype(np.float32)


def _coerce_runtime_vector_np(values: np.ndarray, out_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    out_dim = int(out_dim)
    if values.size == out_dim:
        return values.astype(np.float32)
    if values.size < out_dim:
        out = np.zeros((out_dim,), dtype=np.float32)
        out[: values.size] = values
        return out
    return values[:out_dim].astype(np.float32)


def _coerce_runtime_action_chunk(action_chunk: torch.Tensor, chunk_len: int, action_dim: int, device: torch.device) -> torch.Tensor:
    action_t = action_chunk.detach().to(device=device, dtype=torch.float32)
    if action_t.dim() == 2:
        action_t = action_t.unsqueeze(0)
    if action_t.shape[-1] != int(action_dim):
        raise ValueError(f"delta action repair expected action_dim={action_dim}, got {action_t.shape[-1]}")
    if action_t.shape[1] == int(chunk_len):
        return action_t
    fixed = torch.zeros((action_t.shape[0], int(chunk_len), int(action_dim)), device=device, dtype=torch.float32)
    n = min(int(chunk_len), int(action_t.shape[1]))
    fixed[:, :n] = action_t[:, :n]
    return fixed


def _tensor_is_finite(value) -> bool:
    return torch.is_tensor(value) and (not value.is_floating_point() or bool(torch.isfinite(value).all().item()))


def _finite_tensor_or_none(value, *, name: str, info: dict | None = None):
    if value is None:
        return None
    if not torch.is_tensor(value):
        if info is not None:
            info[f"{name}_finite"] = False
            info[f"{name}_reject_reason"] = "not_tensor"
        return None
    if _tensor_is_finite(value):
        if info is not None:
            info[f"{name}_finite"] = True
        return value
    if info is not None:
        detached = value.detach()
        finite_values = detached[torch.isfinite(detached)] if detached.is_floating_point() else detached.reshape(-1)
        info[f"{name}_finite"] = False
        info[f"{name}_reject_reason"] = "non_finite"
        info[f"{name}_finite_count"] = int(finite_values.numel())
    return None


def _chunk_effect_runtime_meta(subtask_index: int, action_len: int, full_len: int) -> np.ndarray:
    full_len = max(int(full_len), 1)
    action_len = max(int(action_len), 1)
    cut_idx = max(0, full_len - action_len)
    return np.asarray(
        [
            float(subtask_index or 0) / 5.0,
            float(cut_idx) / float(max(full_len - 1, 1)),
            float(action_len) / float(full_len),
        ],
        dtype=np.float32,
    ).reshape(1, -1)


def predict_joint_execution_update(
    cfg,
    model,
    *,
    task: str,
    subtask_index: int,
    state_start: np.ndarray | None,
    state_end: np.ndarray | None,
    base_summary_exec: np.ndarray,
    target_summary_exec: np.ndarray,
    remaining_action_chunk: torch.Tensor | None,
    executed_action: np.ndarray | None = None,
):
    bundle = load_joint_execution_update(cfg, model.device)
    if bundle is None:
        return None, {"joint_execution_update_used": False, "joint_execution_update_reason": "missing_ckpt"}
    if state_start is None or state_end is None or remaining_action_chunk is None:
        return None, {"joint_execution_update_used": False, "joint_execution_update_reason": "missing_state_or_action"}
    try:
        state_dim = int(bundle["state_dim"])
        summary_dim = int(bundle["summary_dim"])
        action_dim = int(bundle["action_dim"])
        chunk_len = int(bundle["chunk_len"])
        action_t = _coerce_runtime_action_chunk(remaining_action_chunk, chunk_len, action_dim, model.device)
        state_start_np = _coerce_runtime_vector_np(state_start, state_dim).reshape(1, -1)
        state_end_np = _coerce_runtime_vector_np(state_end, state_dim).reshape(1, -1)
        base_np = _pool_exec_summary_np(base_summary_exec, summary_dim).reshape(1, -1)
        target_np = _pool_exec_summary_np(target_summary_exec, summary_dim).reshape(1, -1)
        task_vec = np.asarray(q_hashed_task(task, int(bundle["task_dim"])), dtype=np.float32).reshape(1, -1)
        meta = _chunk_effect_runtime_meta(int(subtask_index or 0), int(action_t.shape[1]), chunk_len)
        if str(bundle.get("runtime_type", "")) in {"rssm_plan_compatibility", "full_rssm_plan_compatibility"}:
            exec_action_np = _coerce_runtime_vector_np(
                np.zeros((action_dim,), dtype=np.float32) if executed_action is None else executed_action,
                action_dim,
            ).reshape(1, -1)
            if str(getattr(cfg, "joint_execution_update_ablation_mode", "rssm") or "rssm") == "instant":
                with torch.no_grad():
                    base_t = torch.from_numpy(base_np).to(model.device)
                    target_t = torch.from_numpy(target_np).to(model.device)
                    innovation = target_t - base_t
                    zero_delta = torch.zeros_like(innovation)
                    error = float(torch.norm(innovation.reshape(-1), p=2).item())
                    zero_action_delta = torch.zeros_like(action_t)
                return {
                    "future_delta": innovation.detach(),
                    "expected_future_delta": zero_delta.detach(),
                    "innovation": innovation.detach(),
                    "action_delta": zero_action_delta,
                    "error": error,
                    "learned_gate": 1.0,
                    "success_prob": float("nan"),
                    "mode": "patch",
                    "mode_prob": None,
                    "confidence": float("nan"),
                    "action_delta_norm": 0.0,
                    "h_prior": base_t.detach(),
                    "h_posterior": target_t.detach(),
                }, {
                    "joint_execution_update_used": True,
                    "joint_execution_update_runtime_type": "instant_observation_revision",
                    "joint_execution_has_expected_transition": False,
                    "joint_execution_adaptation_input": "instant_target_minus_current",
                    "joint_execution_error": error,
                    "joint_execution_innovation_norm": error,
                    "joint_execution_innovation_l1": float(torch.mean(torch.abs(innovation)).item()),
                    "joint_execution_expected_delta_norm": 0.0,
                    "joint_execution_learned_gate": 1.0,
                    "joint_execution_has_learned_gate": False,
                    "joint_execution_success_prob": float("nan"),
                    "joint_execution_mode": "patch",
                    "joint_execution_mode_keep_prob": float("nan"),
                    "joint_execution_mode_patch_prob": float("nan"),
                    "joint_execution_mode_replan_prob": float("nan"),
                    "joint_execution_confidence": float("nan"),
                    "joint_execution_future_delta_norm": error,
                    "joint_execution_action_delta_norm": 0.0,
                    "joint_execution_runtime_state_dim": state_dim,
                    "joint_execution_runtime_summary_dim": summary_dim,
                    "joint_execution_runtime_chunk_len": chunk_len,
                }
            with torch.no_grad():
                state_end_t = torch.from_numpy(state_end_np).to(model.device)
                base_t = torch.from_numpy(base_np).to(model.device)
                task_t = torch.from_numpy(task_vec).to(model.device)
                meta_t = torch.from_numpy(meta).to(model.device)
                rssm_out = bundle["rssm_model"](
                    base_t,
                    torch.from_numpy(exec_action_np).to(model.device),
                    state_end_t,
                    task_t,
                    meta_t,
                )
                if bundle.get("compat_head") is not None:
                    d_out = bundle["compat_head"](
                        action_t,
                        rssm_out["h_prior"],
                        rssm_out["h_posterior"],
                        task_t,
                        meta_t,
                    )
                else:
                    d_out = None
            if d_out is not None:
                mode_logits = d_out["mode_logits"].detach()
                mode_prob = torch.softmax(mode_logits, dim=-1)
                mode_idx = int(mode_prob.argmax(dim=-1).reshape(-1)[0].item())
                mode_name = ["keep", "patch", "replan"][max(0, min(2, mode_idx))]
                confidence = float(torch.sigmoid(d_out["confidence_logit"]).reshape(-1)[0].item())
                patch_prob = float(mode_prob[:, 1].reshape(-1)[0].item())
                keep_prob = float(mode_prob[:, 0].reshape(-1)[0].item())
                replan_prob = float(mode_prob[:, 2].reshape(-1)[0].item())
            else:
                mode_prob = None
                mode_name = ""
                confidence = float("nan")
                patch_prob = float("nan")
                keep_prob = float("nan")
                replan_prob = float("nan")
            innovation = rssm_out["innovation"].detach()
            posterior_delta = rssm_out["posterior_delta"].detach()
            prior_delta = rssm_out["prior_delta"].detach()
            error = float(torch.norm(innovation.reshape(-1), p=2).item())
            zero_action_delta = torch.zeros_like(action_t)
            return {
                "future_delta": posterior_delta,
                "expected_future_delta": prior_delta,
                "innovation": innovation,
                "action_delta": zero_action_delta,
                "error": error,
                "learned_gate": patch_prob,
                "success_prob": confidence,
                "mode": mode_name,
                "mode_prob": mode_prob.detach() if mode_prob is not None else None,
                "confidence": confidence,
                "action_delta_norm": 0.0,
                "h_prior": rssm_out["h_prior"].detach(),
                "h_posterior": rssm_out["h_posterior"].detach(),
            }, {
                "joint_execution_update_used": True,
                "joint_execution_update_runtime_type": str(bundle.get("runtime_type", "rssm_plan_compatibility")),
                "joint_execution_has_expected_transition": True,
                "joint_execution_adaptation_input": "rssm_prior_posterior",
                "joint_execution_error": error,
                "joint_execution_innovation_norm": error,
                "joint_execution_innovation_l1": float(torch.mean(torch.abs(innovation)).item()),
                "joint_execution_expected_delta_norm": float(torch.norm(prior_delta.reshape(-1), p=2).item()),
                "joint_execution_learned_gate": patch_prob,
                "joint_execution_has_learned_gate": d_out is not None,
                "joint_execution_success_prob": confidence,
                "joint_execution_mode": mode_name,
                "joint_execution_mode_keep_prob": keep_prob,
                "joint_execution_mode_patch_prob": patch_prob,
                "joint_execution_mode_replan_prob": replan_prob,
                "joint_execution_confidence": confidence,
                "joint_execution_future_delta_norm": float(torch.norm(posterior_delta.reshape(-1), p=2).item()),
                "joint_execution_action_delta_norm": 0.0,
                "joint_execution_runtime_state_dim": state_dim,
                "joint_execution_runtime_summary_dim": summary_dim,
                "joint_execution_runtime_chunk_len": chunk_len,
            }
        if str(bundle.get("runtime_type", "")) == "factored_belief_action_transition":
            exec_action_np = _coerce_runtime_vector_np(
                np.zeros((action_dim,), dtype=np.float32) if executed_action is None else executed_action,
                action_dim,
            ).reshape(1, -1)
            with torch.no_grad():
                state_start_t = torch.from_numpy(state_start_np).to(model.device)
                state_end_t = torch.from_numpy(state_end_np).to(model.device)
                base_t = torch.from_numpy(base_np).to(model.device)
                task_t = torch.from_numpy(task_vec).to(model.device)
                meta_t = torch.from_numpy(meta).to(model.device)
                pred_expected_h = None
                pred_expected_delta_h = None
                if bundle.get("f_model") is not None:
                    pred_expected_delta_h = bundle["f_model"](
                        base_t,
                        state_start_t,
                        torch.from_numpy(exec_action_np).to(model.device),
                        task_t,
                        meta_t,
                    )
                    pred_expected_h = base_t + pred_expected_delta_h
                pred_delta_h = bundle["t_model"](
                    base_t,
                    state_start_t,
                    torch.from_numpy(exec_action_np).to(model.device),
                    state_end_t,
                    task_t,
                    meta_t,
                )
                pred_next_h = base_t + pred_delta_h
                d_out = bundle["d_model"](action_t, base_t, pred_next_h, task_t, meta_t, h_expected=pred_expected_h)
            action_delta = d_out["action_delta"].detach()
            mode_logits = d_out["mode_logits"].detach()
            mode_prob = torch.softmax(mode_logits, dim=-1)
            mode_idx = int(mode_prob.argmax(dim=-1).reshape(-1)[0].item())
            mode_name = ["keep", "patch", "replan"][max(0, min(2, mode_idx))]
            confidence = float(torch.sigmoid(d_out["confidence_logit"]).reshape(-1)[0].item())
            if pred_expected_delta_h is not None:
                innovation = pred_delta_h - pred_expected_delta_h
                error = float(torch.norm(innovation.reshape(-1), p=2).item())
                expected_delta_norm = float(torch.norm(pred_expected_delta_h.reshape(-1), p=2).item())
                innovation_l1 = float(torch.mean(torch.abs(innovation)).item())
            else:
                innovation = pred_delta_h
                error = float(torch.norm(pred_delta_h.reshape(-1), p=2).item())
                expected_delta_norm = float("nan")
                innovation_l1 = float("nan")
            learned_gate = float(mode_prob[:, 1].reshape(-1)[0].item())
            success_prob = confidence
            max_delta_norm = float(torch.norm(action_delta.reshape(-1), p=2).item())
            return {
                "future_delta": pred_delta_h.detach(),
                "expected_future_delta": pred_expected_delta_h.detach() if pred_expected_delta_h is not None else None,
                "innovation": innovation.detach(),
                "action_delta": action_delta,
                "error": error,
                "learned_gate": learned_gate,
                "success_prob": success_prob,
                "mode": mode_name,
                "mode_prob": mode_prob.detach(),
                "confidence": confidence,
                "action_delta_norm": max_delta_norm,
            }, {
                "joint_execution_update_used": True,
                "joint_execution_update_runtime_type": "factored_belief_action_transition",
                "joint_execution_has_expected_transition": bool(pred_expected_delta_h is not None),
                "joint_execution_adaptation_input": str(bundle.get("adaptation_input", "ht_hnext")),
                "joint_execution_error": error,
                "joint_execution_innovation_norm": error,
                "joint_execution_innovation_l1": innovation_l1,
                "joint_execution_expected_delta_norm": expected_delta_norm,
                "joint_execution_learned_gate": learned_gate,
                "joint_execution_has_learned_gate": True,
                "joint_execution_success_prob": success_prob,
                "joint_execution_mode": mode_name,
                "joint_execution_mode_keep_prob": float(mode_prob[:, 0].reshape(-1)[0].item()),
                "joint_execution_mode_patch_prob": float(mode_prob[:, 1].reshape(-1)[0].item()),
                "joint_execution_mode_replan_prob": float(mode_prob[:, 2].reshape(-1)[0].item()),
                "joint_execution_confidence": confidence,
                "joint_execution_future_delta_norm": float(torch.norm(pred_delta_h.reshape(-1), p=2).item()),
                "joint_execution_action_delta_norm": max_delta_norm,
                "joint_execution_runtime_state_dim": state_dim,
                "joint_execution_runtime_summary_dim": summary_dim,
                "joint_execution_runtime_chunk_len": chunk_len,
            }
        with torch.no_grad():
            out = bundle["model"](
                torch.from_numpy(state_start_np).to(model.device),
                torch.from_numpy(state_end_np).to(model.device),
                torch.from_numpy(base_np).to(model.device),
                torch.from_numpy(target_np).to(model.device),
                action_t,
                torch.from_numpy(task_vec).to(model.device),
                torch.from_numpy(meta).to(model.device),
            )
        error = float(out["error"].reshape(-1)[0].item())
        learned_gate = float(torch.sigmoid(out["gate_logit"]).reshape(-1)[0].item()) if "gate_logit" in out else float("nan")
        success_prob = float(torch.sigmoid(out["success_logit"]).reshape(-1)[0].item())
        future_delta = out["future_delta"].detach()
        action_delta = out["action_delta"].detach()
        return {
            "future_delta": future_delta,
            "action_delta": action_delta,
            "error": error,
            "learned_gate": learned_gate,
            "success_prob": success_prob,
        }, {
            "joint_execution_update_used": True,
            "joint_execution_update_runtime_type": "joint_execution_update_mlp",
            "joint_execution_error": error,
            "joint_execution_learned_gate": learned_gate,
            "joint_execution_has_learned_gate": bool(bundle.get("has_learned_gate", False)),
            "joint_execution_success_prob": success_prob,
            "joint_execution_future_delta_norm": float(torch.norm(future_delta.reshape(-1), p=2).item()),
            "joint_execution_action_delta_norm": float(torch.norm(action_delta.reshape(-1), p=2).item()),
            "joint_execution_runtime_state_dim": state_dim,
            "joint_execution_runtime_summary_dim": summary_dim,
            "joint_execution_runtime_chunk_len": chunk_len,
        }
    except Exception as exc:
        return None, {"joint_execution_update_used": True, "joint_execution_update_reason": f"error:{exc}"}


def predict_delta_action_repair(
    cfg,
    model,
    *,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    current_action_chunk: torch.Tensor | None,
    base_summary_exec: np.ndarray,
    target_summary_exec: np.ndarray,
):
    if current_state is None or current_action_chunk is None:
        return None, {"delta_action_repair_used": False, "delta_action_repair_reason": "missing_state_or_action"}
    bundle = load_delta_action_repair(cfg, model.device)
    if bundle is None:
        return None, {"delta_action_repair_used": False, "delta_action_repair_reason": "missing_ckpt"}
    try:
        state_dim = int(bundle["state_dim"])
        future_dim = int(bundle["future_dim"])
        action_dim = int(bundle["action_dim"])
        chunk_len = int(bundle["chunk_len"])
        action_t = _coerce_runtime_action_chunk(current_action_chunk, chunk_len, action_dim, model.device)
        task_vec = np.asarray(q_hashed_task(task, int(bundle["task_dim"])), dtype=np.float32)
        sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)
        state_start = _coerce_runtime_vector_np(current_state, state_dim).reshape(1, -1)
        base_summary_exec = _pool_exec_summary_np(base_summary_exec, future_dim).reshape(1, -1)
        target_summary_exec = _pool_exec_summary_np(target_summary_exec, future_dim).reshape(1, -1)
        target_delta_exec = (target_summary_exec - base_summary_exec).astype(np.float32)
        with torch.no_grad():
            delta = bundle["model"](
                torch.from_numpy(state_start).to(model.device),
                torch.from_numpy(base_summary_exec).to(model.device),
                torch.from_numpy(target_summary_exec).to(model.device),
                torch.from_numpy(target_delta_exec).to(model.device),
                action_t,
                torch.from_numpy(task_vec.reshape(1, -1)).to(model.device),
                torch.from_numpy(sub).to(model.device),
            )
        info = {
            "delta_action_repair_used": True,
            "delta_action_repair_applied": True,
            "delta_action_repair_delta_norm": float(torch.norm(delta.reshape(-1), p=2).item()),
            "delta_action_repair_input_action_norm": float(torch.norm(action_t.reshape(-1), p=2).item()),
            "delta_action_repair_runtime_state_dim": state_dim,
            "delta_action_repair_runtime_future_dim": future_dim,
            "delta_action_repair_runtime_chunk_len": chunk_len,
        }
        return delta.detach(), info
    except Exception as exc:
        return None, {"delta_action_repair_used": True, "delta_action_repair_applied": False, "delta_action_repair_reason": f"error:{exc}"}


def predict_remaining_chunk_effect(
    cfg,
    model,
    *,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    base_summary_exec: np.ndarray,
    target_summary_exec: np.ndarray,
    remaining_action_chunk: torch.Tensor | None,
):
    if remaining_action_chunk is None:
        return None, {
            "chunk_effect_used": False,
            "chunk_effect_reason": "missing_action_chunk",
        }
    bundle = load_future_action_dynamics_runtime(cfg, model.device)
    if bundle is None:
        return None, {
            "chunk_effect_used": False,
            "chunk_effect_reason": "missing_ckpt",
        }
    try:
        action_t = remaining_action_chunk.detach().to(device=model.device, dtype=torch.float32)
        if action_t.dim() == 2:
            action_t = action_t.unsqueeze(0)
        if str(bundle.get("runtime_type", "")) == "chunk_effect_phi_mlp":
            if current_state is None:
                return None, {
                    "chunk_effect_used": True,
                    "chunk_effect_reason": "missing_state_for_phi_mlp",
                }
            full_len = int(bundle["chunk_len"])
            if action_t.shape[1] < full_len:
                padded = torch.zeros((action_t.shape[0], full_len, action_t.shape[-1]), dtype=action_t.dtype, device=action_t.device)
                padded[:, : action_t.shape[1]] = action_t
                action_t = padded
            elif action_t.shape[1] > full_len:
                action_t = action_t[:, :full_len]
            task_vec = np.asarray(q_hashed_task(task, int(bundle["task_dim"])), dtype=np.float32).reshape(1, -1)
            base_summary_exec = _pool_exec_summary_np(base_summary_exec, int(bundle["summary_dim"])).reshape(1, -1)
            target_summary_exec = _pool_exec_summary_np(target_summary_exec, int(bundle["summary_dim"])).reshape(1, -1)
            target_delta_exec = (target_summary_exec - base_summary_exec).astype(np.float32)
            meta = _chunk_effect_runtime_meta(
                subtask_index=int(subtask_index or 0),
                action_len=int(min(remaining_action_chunk.shape[-2], full_len)),
                full_len=full_len,
            )
            with torch.no_grad():
                out = bundle["model"](
                    torch.from_numpy(np.asarray(current_state, dtype=np.float32).reshape(1, -1)).to(model.device),
                    torch.from_numpy(base_summary_exec).to(model.device),
                    action_t,
                    torch.from_numpy(task_vec).to(model.device),
                    torch.from_numpy(meta).to(model.device),
                )
            pred_delta = out["delta"].detach()
            target_delta_t = torch.from_numpy(target_delta_exec).to(model.device)
            residual = target_delta_t - pred_delta
            compat = F.cosine_similarity(pred_delta, target_delta_t, dim=-1).mean()
            success_prob = torch.sigmoid(out["success_logit"]).mean()
            info = {
                "chunk_effect_used": True,
                "chunk_effect_runtime_type": "chunk_effect_phi_mlp",
                "chunk_effect_pred_norm": float(torch.norm(pred_delta.reshape(-1), p=2).item()),
                "chunk_effect_target_norm": float(torch.norm(target_delta_t.reshape(-1), p=2).item()),
                "chunk_effect_residual_norm": float(torch.norm(residual.reshape(-1), p=2).item()),
                "chunk_effect_transition_cos": float(compat.item()),
                "chunk_effect_success_prob": float(success_prob.item()),
                "chunk_effect_remaining_steps": int(action_t.shape[1]),
            }
            return {
                "pred_delta": pred_delta,
                "target_delta": target_delta_t,
                "residual": residual,
                "compat": compat,
                "success_prob": success_prob,
            }, info
        task_vec = np.asarray(q_hashed_task(task, int(bundle["task_dim"])), dtype=np.float32).reshape(1, -1)
        sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)
        base_summary_exec = np.asarray(base_summary_exec, dtype=np.float32).reshape(1, -1)
        target_summary_exec = np.asarray(target_summary_exec, dtype=np.float32).reshape(1, -1)
        target_delta_exec = (target_summary_exec - base_summary_exec).astype(np.float32)
        with torch.no_grad():
            out = bundle["model"](
                torch.from_numpy(base_summary_exec).to(model.device),
                action_t,
                torch.from_numpy(task_vec).to(model.device),
                torch.from_numpy(sub).to(model.device),
                torch.from_numpy(target_delta_exec).to(model.device),
            )
        pred_delta = out["pred_delta"].detach()
        target_delta_t = torch.from_numpy(target_delta_exec).to(model.device)
        residual = target_delta_t - pred_delta
        pred_delta_norm = (pred_delta - bundle["delta_mean"].unsqueeze(0)) / bundle["delta_std"].unsqueeze(0)
        target_delta_norm = (target_delta_t - bundle["delta_mean"].unsqueeze(0)) / bundle["delta_std"].unsqueeze(0)
        compat = F.cosine_similarity(pred_delta_norm, target_delta_norm, dim=-1).mean()
        success_prob = torch.sigmoid(out["success_logit"]).mean()
        info = {
            "chunk_effect_used": True,
            "chunk_effect_pred_norm": float(torch.norm(pred_delta.reshape(-1), p=2).item()),
            "chunk_effect_target_norm": float(torch.norm(target_delta_t.reshape(-1), p=2).item()),
            "chunk_effect_residual_norm": float(torch.norm(residual.reshape(-1), p=2).item()),
            "chunk_effect_transition_cos": float(compat.item()),
            "chunk_effect_success_prob": float(success_prob.item()),
            "chunk_effect_remaining_steps": int(action_t.shape[1]),
        }
        return {
            "pred_delta": pred_delta,
            "target_delta": target_delta_t,
            "residual": residual,
            "compat": compat,
            "success_prob": success_prob,
        }, info
    except Exception as exc:
        return None, {
            "chunk_effect_used": True,
            "chunk_effect_reason": f"error:{exc}",
        }


def decide_chunk_intervention(cfg, effect_payload, delta_action_t):
    keep_threshold = float(getattr(cfg, "dynamic_coupling_chunk_keep_threshold", 0.35))
    replan_threshold = float(getattr(cfg, "dynamic_coupling_chunk_replan_threshold", 0.75))
    success_threshold = float(getattr(cfg, "dynamic_coupling_chunk_success_threshold", 0.5))
    low_compat_threshold = float(getattr(cfg, "dynamic_coupling_chunk_low_compat_threshold", 0.0))
    disable_replan = bool(getattr(cfg, "dynamic_coupling_disable_replan", False))
    suffix_success_below = float(getattr(cfg, "dynamic_coupling_suffix_trigger_success_below", 1.0))
    suffix_residual_above = float(getattr(cfg, "dynamic_coupling_suffix_trigger_residual_above", 0.0))
    suffix_compat_below = float(getattr(cfg, "dynamic_coupling_suffix_trigger_compat_below", 1.0))
    if effect_payload is None:
        return {
            "decision": "fallback",
            "decision_reason": "missing_effect_payload",
            "residual_norm": float("nan"),
            "compat": float("nan"),
            "success_prob": float("nan"),
            "keep_threshold": keep_threshold,
            "replan_threshold": replan_threshold,
            "success_threshold": success_threshold,
            "low_compat_threshold": low_compat_threshold,
        }
    residual_norm = float(torch.norm(effect_payload["residual"].reshape(-1), p=2).item())
    compat = float(effect_payload["compat"].item())
    success_prob = float(effect_payload["success_prob"].item())
    suffix_triggered = (
        delta_action_t is not None
        and (
            success_prob <= suffix_success_below
            or residual_norm >= suffix_residual_above
            or compat <= suffix_compat_below
        )
    )
    if residual_norm <= keep_threshold and compat >= low_compat_threshold and success_prob >= success_threshold:
        decision = "keep"
        reason = "residual_small_and_chunk_still_compatible"
    elif residual_norm >= replan_threshold or compat < low_compat_threshold:
        if disable_replan and suffix_triggered:
            decision = "repair_suffix"
            reason = "suffix_trigger_patch_large_residual_or_low_score"
        elif disable_replan:
            decision = "keep"
            reason = "suffix_trigger_rejected_keep_no_replan"
        else:
            decision = "replan"
            reason = "residual_large_or_direction_incompatible"
    elif suffix_triggered:
        decision = "repair_suffix"
        reason = "suffix_trigger_patch_mid_residual_or_low_score"
    elif disable_replan:
        decision = "keep"
        reason = "suffix_trigger_rejected_keep_no_replan"
    else:
        decision = "replan"
        reason = "mid_residual_but_no_suffix_repair_model"
    return {
        "decision": decision,
        "decision_reason": reason,
        "residual_norm": residual_norm,
        "compat": compat,
        "success_prob": success_prob,
        "keep_threshold": keep_threshold,
        "replan_threshold": replan_threshold,
        "success_threshold": success_threshold,
        "low_compat_threshold": low_compat_threshold,
        "disable_replan": disable_replan,
        "suffix_triggered": bool(suffix_triggered),
        "suffix_success_below": suffix_success_below,
        "suffix_residual_above": suffix_residual_above,
        "suffix_compat_below": suffix_compat_below,
    }


def joint_execution_update_gain(cfg, effect_payload):
    """Compute a scalar execution mismatch e_t and update gate g(e_t)."""
    if effect_payload is None:
        return {
            "joint_execution_update_used": True,
            "joint_execution_error": 0.0,
            "joint_execution_gate": 0.0,
            "joint_execution_reason": "missing_effect_payload",
        }
    residual_norm = float(torch.norm(effect_payload["residual"].reshape(-1), p=2).item())
    compat = float(effect_payload["compat"].item())
    success_prob = float(effect_payload["success_prob"].item())
    residual_scale = max(float(getattr(cfg, "joint_execution_error_residual_scale", 30.0)), 1e-6)
    success_target = float(getattr(cfg, "joint_execution_error_success_target", 0.05))
    compat_target = float(getattr(cfg, "joint_execution_error_compat_target", -0.1))
    success_weight = float(getattr(cfg, "joint_execution_error_success_weight", 2.0))
    compat_weight = float(getattr(cfg, "joint_execution_error_compat_weight", 1.0))
    error = (
        residual_norm / residual_scale
        + success_weight * max(0.0, success_target - success_prob)
        + compat_weight * max(0.0, compat_target - compat)
    )
    threshold = float(getattr(cfg, "joint_execution_update_threshold", 1.0))
    gain = float(getattr(cfg, "joint_execution_update_gain", 1.0))
    min_gate = float(getattr(cfg, "joint_execution_update_min_gate", 0.0))
    max_gate = float(getattr(cfg, "joint_execution_update_max_gate", 0.3))
    gate = float(np.clip(max(0.0, error - threshold) * gain, min_gate, max_gate))
    return {
        "joint_execution_update_used": True,
        "joint_execution_error": float(error),
        "joint_execution_gate": float(gate),
        "joint_execution_residual_norm": residual_norm,
        "joint_execution_success_prob": success_prob,
        "joint_execution_compat": compat,
        "joint_execution_error_threshold": threshold,
        "joint_execution_error_residual_scale": residual_scale,
        "joint_execution_error_success_target": success_target,
        "joint_execution_error_compat_target": compat_target,
    }


def decide_joint_execution_suffix(cfg, effect_payload, delta_action_t):
    update_info = joint_execution_update_gain(cfg, effect_payload)
    threshold = float(getattr(cfg, "joint_execution_suffix_error_threshold", 1.0))
    if delta_action_t is not None and float(update_info["joint_execution_error"]) >= threshold:
        decision = "repair_suffix"
        reason = "joint_execution_error_above_suffix_threshold"
    else:
        decision = "keep"
        reason = "joint_execution_error_below_suffix_threshold"
    return {
        "decision": decision,
        "decision_reason": reason,
        "residual_norm": float(update_info["joint_execution_residual_norm"]) if "joint_execution_residual_norm" in update_info else float("nan"),
        "compat": float(update_info["joint_execution_compat"]) if "joint_execution_compat" in update_info else float("nan"),
        "success_prob": float(update_info["joint_execution_success_prob"]) if "joint_execution_success_prob" in update_info else float("nan"),
        "keep_threshold": float(getattr(cfg, "dynamic_coupling_chunk_keep_threshold", 0.35)),
        "replan_threshold": float(getattr(cfg, "dynamic_coupling_chunk_replan_threshold", 0.75)),
        "success_threshold": float(getattr(cfg, "dynamic_coupling_chunk_success_threshold", 0.5)),
        "low_compat_threshold": float(getattr(cfg, "dynamic_coupling_chunk_low_compat_threshold", 0.0)),
        "disable_replan": True,
        **update_info,
    }


def load_joint_repair_runtime(cfg, device: torch.device):
    repair_ckpt_path = str(getattr(cfg, "joint_pair_repair_ckpt", "") or "")
    critic_ckpt_path = str(getattr(cfg, "joint_pair_repair_critic_ckpt", "") or "")
    summary_future_adapter_ckpt = str(getattr(cfg, "joint_pair_summary_future_adapter_ckpt", "") or "")
    if not repair_ckpt_path:
        return None
    cache_key = (repair_ckpt_path, critic_ckpt_path, summary_future_adapter_ckpt, str(device))
    if cache_key in _JOINT_REPAIR_RUNTIME_CACHE:
        return _JOINT_REPAIR_RUNTIME_CACHE[cache_key]

    repair_ckpt = torch.load(repair_ckpt_path, map_location="cpu")
    repair = JointHypothesisRepairMLP(
        input_dim=int(repair_ckpt["input_dim"]),
        hidden_dim=int(repair_ckpt["hidden_dim"]),
        future_dim=int(repair_ckpt["future_dim"]),
        action_dim=int(repair_ckpt["action_dim"]),
        dropout=float(repair_ckpt.get("dropout", 0.1)),
    ).to(device)
    repair.load_state_dict(repair_ckpt["model_state"])
    repair.eval()

    repair_task_names = [str(x) for x in repair_ckpt.get("task_names", [])]
    bundle = {
        "repair_model": repair,
        "repair_num_tasks": int(repair_ckpt.get("num_tasks", len(repair_task_names))),
        "repair_task_to_id": {task: idx for idx, task in enumerate(repair_task_names)},
        "repair_future_mix": float(getattr(cfg, "joint_pair_repair_future_mix", 0.5)),
        "repair_action_mix": float(getattr(cfg, "joint_pair_repair_action_mix", 0.5)),
    }
    if critic_ckpt_path:
        critic_ckpt = torch.load(critic_ckpt_path, map_location="cpu")
        critic = JointRepairCriticMLP(
            input_dim=int(critic_ckpt["input_dim"]),
            hidden_dim=int(critic_ckpt["hidden_dim"]),
            dropout=float(critic_ckpt.get("dropout", 0.1)),
        ).to(device)
        critic.load_state_dict(critic_ckpt["model_state"])
        critic.eval()
        critic_task_names = [str(x) for x in critic_ckpt.get("task_names", [])]
        bundle["repair_critic_model"] = critic
        bundle["repair_critic_input_dim"] = int(critic_ckpt["input_dim"])
        bundle["repair_critic_advantage_scale"] = float(critic_ckpt.get("advantage_scale", 1.0))
        bundle["repair_critic_task_names"] = critic_task_names
        bundle["repair_critic_task_to_id"] = {task: idx for idx, task in enumerate(critic_task_names)}
    if summary_future_adapter_ckpt:
        try:
            adapter_bundle = load_summary_future_adapter(cfg, device)
            if adapter_bundle is not None:
                bundle["summary_future_adapter"] = adapter_bundle["adapter"]
        except Exception as exc:
            print(f"[WARN] Failed to load repair runtime summary future adapter: {summary_future_adapter_ckpt} | {exc}", flush=True)
    _JOINT_REPAIR_RUNTIME_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded joint repair runtime: repair={repair_ckpt_path} critic={critic_ckpt_path}", flush=True)
    return bundle


def load_joint_value_runtime(cfg, device: torch.device):
    value_ckpt_path = str(getattr(cfg, "joint_pair_value_ckpt", "") or "")
    if not value_ckpt_path:
        return None
    cache_key = (value_ckpt_path, str(device))
    if cache_key in _JOINT_VALUE_RUNTIME_CACHE:
        return _JOINT_VALUE_RUNTIME_CACHE[cache_key]
    value_ckpt = torch.load(value_ckpt_path, map_location="cpu")
    value_model = JointHypothesisValueMLP(
        input_dim=int(value_ckpt["input_dim"]),
        hidden_dim=int(value_ckpt["hidden_dim"]),
        dropout=float(value_ckpt.get("dropout", 0.1)),
    ).to(device)
    value_model.load_state_dict(value_ckpt["model_state"])
    value_model.eval()
    task_names = [str(x) for x in value_ckpt.get("task_names", [])]
    bundle = {
        "value_model": value_model,
        "value_input_dim": int(value_ckpt["input_dim"]),
        "value_num_tasks": int(value_ckpt.get("num_tasks", len(task_names))),
        "value_task_names": task_names,
        "value_task_to_id": {task: idx for idx, task in enumerate(task_names)},
        "value_device": device,
    }
    _JOINT_VALUE_RUNTIME_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded joint value runtime: value={value_ckpt_path}", flush=True)
    return bundle


def _joint_value_score_standalone(
    bundle,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    target_future: torch.Tensor,
    target_action: torch.Tensor,
):
    if bundle is None or current_state is None:
        return None
    value_model = bundle.get("value_model")
    if value_model is None:
        return None
    state_np = np.asarray(current_state, dtype=np.float32).reshape(-1)
    future_arr = np.asarray(target_future.detach().float().cpu().numpy(), dtype=np.float32)
    if future_arr.ndim == 2:
        future_np = _joint_pair_future_summary_exec(future_arr).reshape(-1)
    else:
        future_np = future_arr.reshape(-1)
    action_np = np.asarray(target_action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
    task_to_id = bundle.get("value_task_to_id", {})
    num_tasks = int(bundle.get("value_num_tasks", 0))
    if task not in task_to_id or num_tasks <= 0:
        return None
    task_onehot = np.zeros((num_tasks,), dtype=np.float32)
    task_onehot[int(task_to_id[task])] = 1.0
    stage_np = np.asarray([float(subtask_index or 0) / 5.0], dtype=np.float32)
    feat = np.concatenate([state_np, future_np, action_np, task_onehot, stage_np], axis=0).astype(np.float32)
    expected_dim = int(bundle.get("value_input_dim", feat.size))
    if feat.size != expected_dim:
        return None
    with torch.no_grad():
        score = value_model(torch.from_numpy(feat).unsqueeze(0).to(bundle["value_device"])).reshape(-1)[0]
    return float(score.detach().cpu().item())


def _repair_pending_to_summary_exec(pending_effect: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if pending_effect is None:
        return None
    if torch.is_tensor(pending_effect):
        arr = pending_effect.detach().float().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(pending_effect, dtype=np.float32)
    if arr.ndim == 2:
        return _joint_pair_future_summary_exec(arr).reshape(-1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _joint_score_gate_info(
    cfg,
    *,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    base_future: torch.Tensor | None,
    target_future: torch.Tensor | None,
    target_action: torch.Tensor | None,
    target_transition: np.ndarray | None = None,
    prefix: str = "joint_repair_score",
):
    score_info = score_joint_hypothesis_candidate(
        cfg,
        base_future=base_future,
        target_future=target_future,
        action_chunk=target_action,
        task=task,
        subtask_index=int(subtask_index or 0),
        current_state=current_state,
        target_transition=target_transition,
        prefix=prefix,
    )
    if not bool(score_info.get(f"{prefix}_used", False)):
        return False, {
            f"{prefix}_gate_used": False,
            f"{prefix}_gate_reason": str(score_info.get(f"{prefix}_reason", "score_unavailable")),
            **score_info,
        }
    c_exp = float(score_info.get(f"{prefix}_c_exp", 0.0))
    c_latent = float(score_info.get(f"{prefix}_c_latent", 0.0))
    c_physical = float(score_info.get(f"{prefix}_c_physical", 0.0))
    max_exp = float(getattr(cfg, "joint_pair_verify_exp_max", 1e9))
    max_latent = float(getattr(cfg, "joint_pair_verify_latent_max", 1e9))
    max_physical = float(getattr(cfg, "joint_pair_verify_physical_max", 1e9))
    reasons = []
    if c_exp > max_exp:
        reasons.append(f"c_exp>{max_exp:g}")
    if c_latent > max_latent:
        reasons.append(f"c_latent>{max_latent:g}")
    if c_physical > max_physical:
        reasons.append(f"c_physical>{max_physical:g}")
    passed = len(reasons) == 0
    return passed, {
        f"{prefix}_gate_used": True,
        f"{prefix}_gate_passed": bool(passed),
        f"{prefix}_gate_reason": "pass" if passed else ";".join(reasons),
        f"{prefix}_gate_exp_max": float(max_exp),
        f"{prefix}_gate_latent_max": float(max_latent),
        f"{prefix}_gate_physical_max": float(max_physical),
        **score_info,
    }


def apply_joint_repair_advantage_gate(
    cfg,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    current_future: torch.Tensor | None,
    current_action: torch.Tensor | None,
    pending_effect: torch.Tensor | np.ndarray | None = None,
    base_future_for_score: torch.Tensor | None = None,
    target_transition: np.ndarray | None = None,
):
    if current_state is None or current_future is None or current_action is None:
        return current_future, current_action, {"joint_repair_advantage_used": False, "joint_repair_advantage_reason": "missing_inputs"}
    score_base_future = base_future_for_score if base_future_for_score is not None else current_future
    score_passed, score_gate_info = _joint_score_gate_info(
        cfg,
        task=task,
        subtask_index=int(subtask_index or 0),
        current_state=current_state,
        base_future=score_base_future,
        target_future=current_future,
        target_action=current_action,
        target_transition=target_transition,
        prefix="joint_repair_score_before",
    )
    if bool(score_gate_info.get("joint_repair_score_before_gate_used", False)) and score_passed:
        return current_future, current_action, {
            "joint_repair_advantage_used": True,
            "joint_repair_advantage_mode": "joint_score_pass",
            "joint_repair_advantage_reason": "joint_score_pass",
            "joint_repair_advantage_pred": 0.0,
            "joint_repair_advantage_passed": True,
            "joint_repair_advantage_future_delta_norm": 0.0,
            "joint_repair_advantage_action_delta_norm": 0.0,
            "joint_repair_advantage_pending_norm": 0.0,
            "joint_pair_repair_used": False,
            **score_gate_info,
        }
    bundle = load_joint_repair_runtime(cfg, current_future.device)
    if bundle is None:
        return current_future, current_action, {
            "joint_repair_advantage_used": False,
            "joint_repair_advantage_reason": "missing_repair_runtime",
            **score_gate_info,
        }

    repaired_future, repaired_action, repair_info = _joint_pair_apply_repair(
        bundle,
        task=task,
        subtask_index=int(subtask_index or 0),
        current_state=current_state,
        target_future=current_future,
        target_action=current_action,
        target_transition=target_transition,
    )
    if not bool(repair_info.get("joint_pair_repair_used", False)):
        return current_future, current_action, {
            "joint_repair_advantage_used": False,
            "joint_repair_advantage_reason": "repair_not_used",
            **score_gate_info,
            **repair_info,
        }

    state_np = np.asarray(current_state, dtype=np.float32).reshape(-1)
    future_arr = np.asarray(current_future.detach().float().cpu().numpy(), dtype=np.float32)
    future_np = _joint_pair_future_summary_exec(future_arr).reshape(-1) if future_arr.ndim == 2 else future_arr.reshape(-1)
    action_np = np.asarray(current_action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
    repaired_future_arr = np.asarray(repaired_future.detach().float().cpu().numpy(), dtype=np.float32)
    repaired_future_np = _joint_pair_future_summary_exec(repaired_future_arr).reshape(-1) if repaired_future_arr.ndim == 2 else repaired_future_arr.reshape(-1)
    repaired_action_np = np.asarray(repaired_action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
    delta_future_np = (repaired_future_np - future_np).astype(np.float32)
    delta_action_np = (repaired_action_np - action_np).astype(np.float32)
    pending_np = _repair_pending_to_summary_exec(pending_effect)
    if pending_np is None or pending_np.size != delta_future_np.size:
        pending_np = delta_future_np.copy()
    value_bundle = load_joint_value_runtime(cfg, current_future.device)
    v_before = None
    v_after = None
    adv = 0.0
    if value_bundle is not None:
        v_before = _joint_value_score_standalone(
            value_bundle,
            task=task,
            subtask_index=int(subtask_index or 0),
            current_state=current_state,
            target_future=current_future,
            target_action=current_action,
        )
        v_after = _joint_value_score_standalone(
            value_bundle,
            task=task,
            subtask_index=int(subtask_index or 0),
            current_state=current_state,
            target_future=repaired_future,
            target_action=repaired_action,
        )
        if v_before is not None and v_after is not None:
            adv = float(v_after - v_before)
    _, score_after_info = _joint_score_gate_info(
        cfg,
        task=task,
        subtask_index=int(subtask_index or 0),
        current_state=current_state,
        base_future=score_base_future,
        target_future=repaired_future,
        target_action=repaired_action,
        target_transition=target_transition,
        prefix="joint_repair_score_after",
    )
    info = {
        "joint_repair_advantage_used": True,
        "joint_repair_advantage_mode": "joint_score_repair",
        "joint_repair_advantage_reason": "joint_score_repair",
        "joint_repair_advantage_pred": float(adv),
        "joint_repair_advantage_value_before": None if v_before is None else float(v_before),
        "joint_repair_advantage_value_after": None if v_after is None else float(v_after),
        "joint_repair_advantage_passed": True,
        "joint_repair_advantage_future_delta_norm": float(np.linalg.norm(delta_future_np)),
        "joint_repair_advantage_action_delta_norm": float(np.linalg.norm(delta_action_np)),
        "joint_repair_advantage_pending_norm": float(np.linalg.norm(pending_np)),
        **score_gate_info,
        **score_after_info,
        **repair_info,
    }
    return repaired_future, repaired_action, info


def _joint_pair_apply_repair(
    bundle,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    target_future: torch.Tensor,
    target_action: torch.Tensor,
    target_transition: np.ndarray | None = None,
):
    repair = bundle.get("repair_model")
    if repair is None or current_state is None:
        return target_future, target_action, {}
    task_to_id = bundle.get("repair_task_to_id", {})
    num_tasks = int(bundle.get("repair_num_tasks", 0))
    if task not in task_to_id or num_tasks <= 0:
        return target_future, target_action, {
            "joint_pair_repair_used": False,
            "joint_pair_repair_reason": "task_missing",
        }
    state_np = np.asarray(current_state, dtype=np.float32).reshape(-1)
    future_arr = np.asarray(target_future.detach().float().cpu().numpy(), dtype=np.float32)
    if future_arr.ndim == 2:
        future_np = _joint_pair_future_summary_exec(future_arr).reshape(-1)
    else:
        future_np = future_arr.reshape(-1)
    action_np = np.asarray(target_action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
    task_onehot = np.zeros((num_tasks,), dtype=np.float32)
    task_onehot[int(task_to_id[task])] = 1.0
    stage_np = np.asarray([float(subtask_index or 0) / 5.0], dtype=np.float32)
    feat = np.concatenate([state_np, future_np, action_np, task_onehot, stage_np], axis=0).astype(np.float32)
    expected_dim = None
    try:
        expected_dim = int(repair.backbone[0].normalized_shape[0])
    except Exception:
        expected_dim = None
    if expected_dim is not None and feat.size != expected_dim:
        return target_future, target_action, {
            "joint_pair_repair_used": False,
            "joint_pair_repair_reason": f"dim_mismatch:{feat.size}!={expected_dim}",
        }
    device = target_future.device
    with torch.no_grad():
        out = repair(torch.from_numpy(feat).unsqueeze(0).to(device))
        delta_future = out["pred_future"].squeeze(0)
        delta_action = out["pred_action"].squeeze(0)
    future_mix = float(bundle.get("repair_future_mix", 0.5))
    action_mix = float(bundle.get("repair_action_mix", 0.5))
    future_repair_applied = False
    future_repair_reason = "not_attempted"
    future_adapter_gate_mean = 0.0
    future_adapter_delta_norm = 0.0
    if delta_future.numel() == target_future.numel():
        repaired_future = target_future + future_mix * delta_future.reshape_as(target_future).to(dtype=target_future.dtype)
        future_repair_applied = True
        future_repair_reason = "full_future_delta"
    else:
        repaired_future = target_future
        future_repair_reason = f"future_dim_mismatch:{delta_future.numel()}!={target_future.numel()}"
        adapter = bundle.get("summary_future_adapter")
        if adapter is not None and future_arr.ndim == 2 and delta_future.numel() == future_np.size:
            try:
                source_summary_t = torch.from_numpy(future_np).reshape(1, -1).to(device=device, dtype=target_future.dtype)
                target_summary_t = source_summary_t + future_mix * delta_future.reshape(1, -1).to(dtype=target_future.dtype)
                corrected_future, adapter_aux = adapter(
                    target_future.unsqueeze(0).to(dtype=target_future.dtype),
                    source_summary_t,
                    target_summary_t,
                )
                repaired_future = corrected_future.squeeze(0).to(dtype=target_future.dtype)
                future_repair_applied = True
                future_repair_reason = "summary_future_adapter"
                future_adapter_gate_mean = float(adapter_aux["gate"].mean().item())
                future_adapter_delta_norm = float(torch.norm(adapter_aux["delta"].reshape(-1), p=2).item())
            except Exception as exc:
                future_repair_reason = f"summary_future_adapter_error:{exc}"
    repaired_action = target_action + action_mix * delta_action.reshape_as(target_action).to(dtype=target_action.dtype)
    manifold_used = False
    manifold_score_before = 0.0
    manifold_score_after = 0.0
    manifold_score_best = 0.0
    manifold_mix_final = float(action_mix)
    manifold_reason = "not_used"
    if (
        current_state is not None
        and target_transition is not None
        and "dynamics_model" in bundle
    ):
        try:
            dynamics_before = score_state_action_pair(
                bundle["dynamics_model"],
                bundle["dynamics_transition_mean"],
                bundle["dynamics_transition_std"],
                np.asarray(current_state, dtype=np.float32),
                np.asarray(target_action.detach().float().cpu().numpy(), dtype=np.float32),
                np.asarray(target_transition, dtype=np.float32),
                target_future.device,
            )
            manifold_score_before = float(dynamics_before["dynamics_score"])
            manifold_score_best = manifold_score_before
            best_action = target_action
            best_mix = 0.0
            candidate_mixes = [
                float(action_mix),
                float(action_mix) * 0.5,
                float(action_mix) * 0.25,
                0.0,
            ]
            min_improve = float(bundle.get("repair_manifold_min_improve", 0.0))
            for cand_mix in candidate_mixes:
                cand_action = target_action + float(cand_mix) * delta_action.reshape_as(target_action).to(dtype=target_action.dtype)
                dynamics_after = score_state_action_pair(
                    bundle["dynamics_model"],
                    bundle["dynamics_transition_mean"],
                    bundle["dynamics_transition_std"],
                    np.asarray(current_state, dtype=np.float32),
                    np.asarray(cand_action.detach().float().cpu().numpy(), dtype=np.float32),
                    np.asarray(target_transition, dtype=np.float32),
                    target_future.device,
                )
                cand_score = float(dynamics_after["dynamics_score"])
                if cand_mix == float(action_mix):
                    manifold_score_after = cand_score
                if cand_score > manifold_score_best + min_improve:
                    manifold_score_best = cand_score
                    best_action = cand_action
                    best_mix = float(cand_mix)
            repaired_action = best_action
            manifold_mix_final = float(best_mix)
            manifold_used = True
            manifold_reason = "accepted_best_mix" if best_mix > 0.0 else "fallback_to_init_action"
        except Exception as exc:
            manifold_reason = f"error:{exc}"
    manifold_h_used = False
    manifold_h_reason = "not_used"
    manifold_h_dist = 0.0
    manifold_h_success_before = 0.0
    manifold_h_success_after = 0.0
    manifold_h_success_best = 0.0
    manifold_h_mix_final = None
    if current_state is not None and "manifold_model" in bundle:
        try:
            init_eval = _joint_pair_manifold_eval(
                bundle, task, subtask_index, current_state, target_future, target_action
            )
            if init_eval is not None:
                manifold_h_success_before = float(init_eval["success_prob"])
                best_future = target_future
                best_action = target_action
                best_score = manifold_h_success_before
                best_dist = 0.0
                best_mix = 0.0
                candidate_mixes = [1.0, 0.5, 0.25, 0.0]
                max_dist = float(bundle.get("repair_manifold_max_dist", 1e9))
                min_gain = float(bundle.get("repair_manifold_min_success_gain", 0.0))
                for alpha in candidate_mixes:
                    cand_future = target_future + float(alpha) * (repaired_future - target_future)
                    cand_action = target_action + float(alpha) * (repaired_action - target_action)
                    cand_eval = _joint_pair_manifold_eval(
                        bundle, task, subtask_index, current_state, cand_future, cand_action
                    )
                    if cand_eval is None:
                        continue
                    cand_dist = float(torch.norm(cand_eval["z"] - init_eval["z"], p=2).item())
                    cand_score = float(cand_eval["success_prob"])
                    if alpha == 1.0:
                        manifold_h_success_after = cand_score
                    if cand_dist <= max_dist and cand_score >= best_score + min_gain:
                        best_future = cand_future
                        best_action = cand_action
                        best_score = cand_score
                        best_dist = cand_dist
                        best_mix = float(alpha)
                repaired_future = best_future
                repaired_action = best_action
                manifold_h_used = True
                manifold_h_reason = "accepted_best_mix" if best_mix > 0.0 else "fallback_to_init_pair"
                manifold_h_dist = float(best_dist)
                manifold_h_success_best = float(best_score)
                manifold_h_mix_final = float(best_mix)
        except Exception as exc:
            manifold_h_reason = f"error:{exc}"
    return repaired_future, repaired_action, {
        "joint_pair_repair_used": True,
        "joint_pair_repair_future_mix": float(future_mix),
        "joint_pair_repair_action_mix": float(action_mix),
        "joint_pair_repair_action_mix_final": float(manifold_mix_final),
        "joint_pair_repair_future_applied": bool(future_repair_applied),
        "joint_pair_repair_future_reason": str(future_repair_reason),
        "joint_pair_repair_future_delta_norm": float(torch.norm(delta_future.reshape(-1), p=2).item()),
        "joint_pair_repair_action_delta_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
        "joint_pair_repair_future_shift_norm": float(torch.norm((repaired_future - target_future).reshape(-1), p=2).item()),
        "joint_pair_repair_action_shift_norm": float(torch.norm((repaired_action - target_action).reshape(-1), p=2).item()),
        "joint_pair_repair_future_adapter_used": bool(future_repair_reason == "summary_future_adapter"),
        "joint_pair_repair_future_adapter_gate_mean": float(future_adapter_gate_mean),
        "joint_pair_repair_future_adapter_delta_norm": float(future_adapter_delta_norm),
        "joint_pair_repair_manifold_used": bool(manifold_used),
        "joint_pair_repair_manifold_reason": str(manifold_reason),
        "joint_pair_repair_manifold_score_before": float(manifold_score_before),
        "joint_pair_repair_manifold_score_after": float(manifold_score_after),
        "joint_pair_repair_manifold_score_best": float(manifold_score_best),
        "joint_pair_repair_hmanifold_used": bool(manifold_h_used),
        "joint_pair_repair_hmanifold_reason": str(manifold_h_reason),
        "joint_pair_repair_hmanifold_dist": float(manifold_h_dist),
        "joint_pair_repair_hmanifold_success_before": float(manifold_h_success_before),
        "joint_pair_repair_hmanifold_success_after": float(manifold_h_success_after),
        "joint_pair_repair_hmanifold_success_best": float(manifold_h_success_best),
        "joint_pair_repair_hmanifold_mix_final": None if manifold_h_mix_final is None else float(manifold_h_mix_final),
    }


def _joint_pair_value_score(
    bundle,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    target_summary_exec: np.ndarray,
    action_chunk: np.ndarray,
):
    value_model = bundle.get("value_model")
    if value_model is None or current_state is None:
        return None
    state_np = np.asarray(current_state, dtype=np.float32).reshape(-1)
    future_arr = np.asarray(target_summary_exec, dtype=np.float32)
    if future_arr.ndim == 2:
        future_np = _joint_pair_future_summary_exec(future_arr).reshape(-1)
    else:
        future_np = future_arr.reshape(-1)
    action_np = np.asarray(action_chunk, dtype=np.float32).reshape(-1)
    task_names = bundle.get("value_task_names", [])
    num_tasks = int(bundle.get("value_num_tasks", 0))
    task_to_id = bundle.get("value_task_to_id", {})
    inferred_num_tasks = int(bundle.get("value_input_dim", 0)) - state_np.size - future_np.size - action_np.size - 1
    if num_tasks <= 0:
        num_tasks = max(inferred_num_tasks, 0)
    if not task_names and "memory" in bundle:
        task_names = sorted({str(t) for t in bundle["memory"]["task"].tolist()})
        bundle["value_task_names"] = task_names
        bundle["value_task_to_id"] = {name: idx for idx, name in enumerate(task_names)}
        task_to_id = bundle["value_task_to_id"]
        num_tasks = len(task_names)
        bundle["value_num_tasks"] = num_tasks
    if task not in task_to_id or num_tasks <= 0:
        return None
    task_onehot = np.zeros((num_tasks,), dtype=np.float32)
    task_onehot[int(task_to_id[task])] = 1.0
    stage_np = np.asarray([float(subtask_index or 0) / 5.0], dtype=np.float32)
    feat = np.concatenate([state_np, future_np, action_np, task_onehot, stage_np], axis=0).astype(np.float32)
    if feat.size != int(bundle.get("value_input_dim", feat.size)):
        return None
    device = bundle["value_device"]
    with torch.no_grad():
        score = value_model(torch.from_numpy(feat).unsqueeze(0).to(device)).reshape(-1)[0]
    return float(score.detach().cpu().item())


def _joint_pair_manifold_feature(
    bundle,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    target_future: torch.Tensor,
    target_action: torch.Tensor,
):
    manifold = bundle.get("manifold_model")
    if manifold is None or current_state is None:
        return None
    state_np = np.asarray(current_state, dtype=np.float32).reshape(-1)
    future_arr = np.asarray(target_future.detach().float().cpu().numpy(), dtype=np.float32)
    if future_arr.ndim == 2:
        future_np = _joint_pair_future_summary_exec(future_arr).reshape(-1)
    else:
        future_np = future_arr.reshape(-1)
    action_np = np.asarray(target_action.detach().float().cpu().numpy(), dtype=np.float32).reshape(-1)
    task_to_id = bundle.get("manifold_task_to_id", {})
    num_tasks = int(bundle.get("manifold_num_tasks", 0))
    if task not in task_to_id or num_tasks <= 0:
        return None
    task_onehot = np.zeros((num_tasks,), dtype=np.float32)
    task_onehot[int(task_to_id[task])] = 1.0
    stage_np = np.asarray([float(subtask_index or 0) / 5.0], dtype=np.float32)
    feat = np.concatenate([state_np, future_np, action_np, task_onehot, stage_np], axis=0).astype(np.float32)
    expected_dim = int(bundle.get("manifold_input_dim", feat.size))
    if feat.size != expected_dim:
        return None
    return feat


def _joint_pair_manifold_eval(
    bundle,
    task: str,
    subtask_index: int,
    current_state: np.ndarray | None,
    target_future: torch.Tensor,
    target_action: torch.Tensor,
):
    feat = _joint_pair_manifold_feature(bundle, task, subtask_index, current_state, target_future, target_action)
    if feat is None:
        return None
    model = bundle.get("manifold_model")
    device = target_future.device
    with torch.no_grad():
        out = model(torch.from_numpy(feat).unsqueeze(0).to(device))
    return {
        "z": out["z"].squeeze(0),
        "success_logit": out["success_logit"].reshape(-1)[0],
        "success_prob": float(torch.sigmoid(out["success_logit"]).reshape(-1)[0].item()),
    }


def _joint_pair_cross_stage_score(
    bundle,
    *,
    task: str,
    next_task: str | None,
    subtask_index: int,
    current_state: np.ndarray | None,
    target_future: torch.Tensor,
    target_action: torch.Tensor,
):
    if not next_task:
        return None
    model = bundle.get("cross_stage_model")
    manifold = bundle.get("manifold_model")
    memory = bundle.get("memory")
    if model is None or manifold is None or memory is None or current_state is None:
        return None
    current_eval = _joint_pair_manifold_eval(
        bundle,
        task=task,
        subtask_index=subtask_index,
        current_state=current_state,
        target_future=target_future,
        target_action=target_action,
    )
    if current_eval is None:
        return None
    cur_z = current_eval["z"].unsqueeze(0)
    next_stage = int(subtask_index or 0) + 1
    task_arr = np.asarray(memory["task"])
    stage_arr = np.asarray(memory["subtask_index"], dtype=np.int32)
    success_arr = np.asarray(memory["success"], dtype=np.float32)
    mask = (task_arr == str(next_task)) & (success_arr > 0.5)
    stage_mask = mask & (stage_arr == next_stage)
    candidate_ids = np.where(stage_mask)[0]
    if candidate_ids.size == 0:
        candidate_ids = np.where(mask)[0]
    if candidate_ids.size == 0:
        return None
    state_np = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if "state_start" in memory:
        dists = []
        for idx in candidate_ids.tolist():
            cand_state = np.asarray(memory["state_start"][idx], dtype=np.float32).reshape(-1)
            if cand_state.shape != state_np.shape:
                continue
            dists.append((float(np.linalg.norm(cand_state - state_np)), idx))
        if dists:
            dists.sort(key=lambda x: x[0])
            candidate_ids = np.asarray([idx for _, idx in dists[: max(1, int(bundle.get("cross_stage_topk", 4)))]], dtype=np.int32)
    best_score = None
    best_idx = None
    with torch.no_grad():
        for idx in candidate_ids.tolist():
            next_future_np = np.asarray(memory["target_summary_exec"][idx], dtype=np.float32)
            next_action_np = np.asarray(memory["action_chunk"][idx], dtype=np.float32)
            next_future_t = torch.from_numpy(next_future_np).to(device=target_future.device, dtype=target_future.dtype)
            next_action_t = torch.from_numpy(next_action_np).to(device=target_action.device, dtype=target_action.dtype)
            nxt_eval = _joint_pair_manifold_eval(
                bundle,
                task=str(next_task),
                subtask_index=next_stage,
                current_state=current_state,
                target_future=next_future_t,
                target_action=next_action_t,
            )
            if nxt_eval is None:
                continue
            nxt_z = nxt_eval["z"].unsqueeze(0)
            score = float(model(cur_z, nxt_z).reshape(-1)[0].item())
            if best_score is None or score > best_score:
                best_score = score
                best_idx = int(idx)
    if best_score is None:
        return None
    return {
        "cross_stage_score": float(best_score),
        "cross_stage_memory_index": int(best_idx),
        "cross_stage_next_task": str(next_task),
    }


def load_joint_pair_selector(cfg, device):
    memory_path = str(getattr(cfg, "joint_pair_memory_npz", "") or "")
    ckpt_path = str(getattr(cfg, "joint_energy_rerank_ckpt", "") or "")
    if not memory_path or not ckpt_path:
        return None
    dyn_ckpt_path = str(getattr(cfg, "joint_pair_dynamics_probe_ckpt", "") or "")
    repair_ckpt_path = str(getattr(cfg, "joint_pair_repair_ckpt", "") or "")
    value_ckpt_path = str(getattr(cfg, "joint_pair_value_ckpt", "") or "")
    cross_stage_ckpt_path = str(getattr(cfg, "joint_pair_cross_stage_ckpt", "") or "")
    cache_key = (memory_path, ckpt_path, dyn_ckpt_path, repair_ckpt_path, value_ckpt_path, cross_stage_ckpt_path, str(device))
    cached = _JOINT_PAIR_SELECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with np.load(memory_path, allow_pickle=True) as data:
        memory = {k: data[k] for k in data.files}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    scorer_type = "energy"
    if str(ckpt.get("model_type", "")) == "robotics_energy":
        scorer_type = "robotics_energy"
        model = JointFutureActionRoboticsEnergy(
            future_dim=int(ckpt["future_dim"]),
            action_dim=int(ckpt["action_dim"]),
            state_dim=int(ckpt["state_dim"]),
            task_dim=int(ckpt["task_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            z_dim=int(ckpt["z_dim"]),
            dropout=float(ckpt.get("dropout", 0.1)),
        ).to(device)
    elif "z_dim" in ckpt:
        scorer_type = "unified_policy"
        model = JointFutureActionPolicy(
            future_dim=int(ckpt["future_dim"]),
            action_dim=int(ckpt["action_dim"]),
            task_dim=int(ckpt["task_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            z_dim=int(ckpt["z_dim"]),
            dropout=float(ckpt.get("dropout", 0.1)),
        ).to(device)
    else:
        model = JointFutureActionEnergy(
            action_dim=int(ckpt["action_dim"]),
            future_dim=int(ckpt["future_dim"]),
            task_dim=int(ckpt["task_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            dropout=float(ckpt.get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bundle = {
        "memory": memory,
        "model": model,
        "task_dim": int(ckpt["task_dim"]),
        "scorer_type": scorer_type,
    }
    action_gen_ckpt = str(getattr(cfg, "joint_pair_action_generator_ckpt", "") or "")
    if action_gen_ckpt:
        try:
            gen_ckpt = torch.load(action_gen_ckpt, map_location="cpu")
            gen = JointActionGeneratorMLP(
                state_dim=int(gen_ckpt["state_dim"]),
                future_dim=int(gen_ckpt["future_dim"]),
                action_dim=int(gen_ckpt["action_dim"]),
                chunk_len=int(gen_ckpt["chunk_len"]),
                task_dim=int(gen_ckpt["task_dim"]),
                hidden_dim=int(gen_ckpt["hidden_dim"]),
                dropout=float(gen_ckpt.get("dropout", 0.1)),
            ).to(device)
            gen.load_state_dict(gen_ckpt["model_state"])
            gen.eval()
            bundle["action_generator"] = gen
            bundle["action_generator_ckpt"] = action_gen_ckpt
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair action generator: {action_gen_ckpt} | {exc}", flush=True)
    if scorer_type == "robotics_energy":
        bundle["robotics_state_transition_mean"] = torch.from_numpy(np.asarray(ckpt["state_transition_mean"], dtype=np.float32)).to(device)
        bundle["robotics_state_transition_std"] = torch.from_numpy(np.asarray(ckpt["state_transition_std"], dtype=np.float32)).to(device)
        bundle["robotics_lambda_future_dyn"] = float(ckpt.get("lambda_future_dyn", 1.0))
        bundle["robotics_lambda_state_dyn"] = float(ckpt.get("lambda_state_dyn", 0.5))
        bundle["robotics_beta_contact"] = float(ckpt.get("beta_contact", 0.25))
    if dyn_ckpt_path:
        try:
            dyn_model, dyn_mean, dyn_std = load_state_action_dynamics_probe(Path(dyn_ckpt_path), device)
            bundle["dynamics_model"] = dyn_model
            bundle["dynamics_transition_mean"] = dyn_mean
            bundle["dynamics_transition_std"] = dyn_std
            bundle["dynamics_ckpt"] = dyn_ckpt_path
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair dynamics probe: {dyn_ckpt_path} | {exc}", flush=True)
    if repair_ckpt_path:
        try:
            repair_ckpt = torch.load(repair_ckpt_path, map_location="cpu")
            repair = JointHypothesisRepairMLP(
                input_dim=int(repair_ckpt["input_dim"]),
                hidden_dim=int(repair_ckpt["hidden_dim"]),
                future_dim=int(repair_ckpt["future_dim"]),
                action_dim=int(repair_ckpt["action_dim"]),
                dropout=float(repair_ckpt.get("dropout", 0.1)),
            ).to(device)
            repair.load_state_dict(repair_ckpt["model_state"])
            repair.eval()
            task_names = [str(x) for x in repair_ckpt.get("task_names", [])]
            bundle["repair_model"] = repair
            bundle["repair_num_tasks"] = int(repair_ckpt.get("num_tasks", len(task_names)))
            bundle["repair_task_to_id"] = {task: idx for idx, task in enumerate(task_names)}
            bundle["repair_future_mix"] = float(getattr(cfg, "joint_pair_repair_future_mix", 0.5))
            bundle["repair_action_mix"] = float(getattr(cfg, "joint_pair_repair_action_mix", 0.5))
            bundle["repair_manifold_min_improve"] = float(getattr(cfg, "joint_pair_repair_manifold_min_improve", 0.0))
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair repair model: {repair_ckpt_path} | {exc}", flush=True)
    summary_future_adapter_ckpt = str(getattr(cfg, "joint_pair_summary_future_adapter_ckpt", "") or "")
    if summary_future_adapter_ckpt:
        try:
            adapter_ckpt = torch.load(summary_future_adapter_ckpt, map_location="cpu")
            adapter = SummaryConditionedFutureAdapter(
                future_dim=int(adapter_ckpt["future_dim"]),
                summary_dim=int(adapter_ckpt["summary_dim"]),
                hidden_dim=int(adapter_ckpt["hidden_dim"]),
            ).to(device)
            adapter.load_state_dict(adapter_ckpt["model_state"])
            adapter.eval()
            bundle["summary_future_adapter"] = adapter
            bundle["summary_future_adapter_ckpt"] = summary_future_adapter_ckpt
            bundle["summary_future_adapter_summary_dim"] = int(adapter_ckpt["summary_dim"])
            bundle["summary_future_adapter_future_token_shape"] = tuple(int(x) for x in adapter_ckpt.get("future_token_shape", ()))
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair summary future adapter: {summary_future_adapter_ckpt} | {exc}", flush=True)
    if value_ckpt_path:
        try:
            value_ckpt = torch.load(value_ckpt_path, map_location="cpu")
            value_model = JointHypothesisValueMLP(
                input_dim=int(value_ckpt["input_dim"]),
                hidden_dim=int(value_ckpt["hidden_dim"]),
                dropout=float(value_ckpt.get("dropout", 0.1)),
            ).to(device)
            value_model.load_state_dict(value_ckpt["model_state"])
            value_model.eval()
            bundle["value_model"] = value_model
            bundle["value_input_dim"] = int(value_ckpt["input_dim"])
            bundle["value_num_tasks"] = int(value_ckpt.get("num_tasks", 0))
            bundle["value_task_names"] = [str(x) for x in value_ckpt.get("task_names", [])]
            bundle["value_task_to_id"] = {task: idx for idx, task in enumerate(bundle["value_task_names"])}
            bundle["value_device"] = device
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair value model: {value_ckpt_path} | {exc}", flush=True)
    manifold_ckpt_path = str(getattr(cfg, "joint_pair_manifold_ckpt", "") or "")
    if manifold_ckpt_path:
        try:
            manifold_ckpt = torch.load(manifold_ckpt_path, map_location="cpu")
            manifold = JointHypothesisManifoldMLP(
                input_dim=int(manifold_ckpt["input_dim"]),
                hidden_dim=int(manifold_ckpt["hidden_dim"]),
                latent_dim=int(manifold_ckpt["latent_dim"]),
                dropout=float(manifold_ckpt.get("dropout", 0.1)),
            ).to(device)
            manifold.load_state_dict(manifold_ckpt["model_state"])
            manifold.eval()
            bundle["manifold_model"] = manifold
            task_names = [str(x) for x in manifold_ckpt.get("task_names", [])]
            bundle["manifold_num_tasks"] = int(manifold_ckpt.get("num_tasks", len(task_names)))
            bundle["manifold_task_to_id"] = {task: idx for idx, task in enumerate(task_names)}
            bundle["manifold_input_dim"] = int(manifold_ckpt["input_dim"])
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair manifold model: {manifold_ckpt_path} | {exc}", flush=True)
    if cross_stage_ckpt_path:
        try:
            cross_ckpt = torch.load(cross_stage_ckpt_path, map_location="cpu")
            cross_model = CrossStageConnectivityMLP(
                latent_dim=int(cross_ckpt["latent_dim"]),
                hidden_dim=int(cross_ckpt["hidden_dim"]),
                dropout=float(cross_ckpt.get("dropout", 0.1)),
            ).to(device)
            cross_model.load_state_dict(cross_ckpt["model_state"])
            cross_model.eval()
            bundle["cross_stage_model"] = cross_model
            bundle["cross_stage_topk"] = int(getattr(cfg, "joint_pair_cross_stage_topk", 4))
        except Exception as exc:
            print(f"[WARN] Failed to load joint pair cross-stage model: {cross_stage_ckpt_path} | {exc}", flush=True)
    _JOINT_PAIR_SELECTOR_CACHE[cache_key] = bundle
    print(f"[INFO] Loaded joint pair selector: {memory_path} | {ckpt_path} | scorer={scorer_type}", flush=True)
    return bundle


def select_joint_pair_target_future(
    cfg,
    base_future,
    task,
    subtask_index,
    current_state=None,
    next_task=None,
    initial_action_intent: torch.Tensor | None = None,
):
    bundle = load_joint_pair_selector(cfg, base_future.device)
    if bundle is None:
        return None, None, {"joint_pair_selector_used": False, "joint_pair_selector_reason": "missing_ckpt_or_memory"}
    memory = bundle["memory"]
    init_action_t, base_summary, base_summary_exec, init_info = _joint_pair_build_initial_hypothesis(
        cfg,
        bundle,
        base_future,
        task,
        int(subtask_index or 0),
        current_state,
        initial_action_intent=initial_action_intent,
    )
    init_action_np = None
    if init_action_t is not None:
        init_action_np = init_action_t.detach().float().cpu().numpy().astype(np.float32)
    same_task = bool(getattr(cfg, "joint_pair_same_task_only", True))
    mask = np.ones(len(memory["task"]), dtype=bool)
    if same_task:
        mask = np.asarray(memory["task"] == task)
    candidate_ids = np.where(mask)[0]
    if candidate_ids.size == 0:
        return None, None, {
            "joint_pair_selector_used": False,
            "joint_pair_selector_reason": "no_candidates",
            **init_info,
        }
    sim_details = [
        _joint_pair_joint_query_similarity(
            memory,
            int(idx),
            current_state=current_state,
            base_summary=base_summary,
            init_action_np=init_action_np,
            subtask_index=int(subtask_index or 0),
        )
        for idx in candidate_ids.tolist()
    ]
    sims = np.asarray([float(item["joint_query_similarity"]) for item in sim_details], dtype=np.float32)
    top_k = max(1, min(int(getattr(cfg, "joint_pair_topk", 8)), len(candidate_ids)))
    order = np.argsort(-sims)[:top_k]
    chosen_ids = candidate_ids[order]
    task_vec = np.asarray(q_hashed_task(task, bundle["task_dim"]), dtype=np.float32)
    sub = np.asarray([[float(subtask_index or 0) / 5.0]], dtype=np.float32)
    lambda_dynamics = float(getattr(cfg, "joint_pair_lambda_dynamics", 0.0))
    use_dynamics = (
        current_state is not None
        and lambda_dynamics > 0.0
        and "state_transition" in memory
        and "dynamics_model" in bundle
    )
    lambda_value = float(getattr(cfg, "joint_pair_lambda_value", 0.0))
    lambda_cross_stage = float(getattr(cfg, "joint_pair_lambda_cross_stage", 0.0))
    lambda_exp = float(getattr(cfg, "joint_pair_score_lambda_exp", 1.0))
    lambda_latent = float(getattr(cfg, "joint_pair_score_lambda_latent", 1.0))
    lambda_physical = float(getattr(cfg, "joint_pair_score_lambda_physical", 1.0))
    best = None
    scored = []
    with torch.no_grad():
        for rank, mem_idx in enumerate(chosen_ids.tolist(), start=1):
            sim_meta = sim_details[int(order[rank - 1])]
            target_summary_exec = np.asarray(memory["target_summary_exec"][mem_idx], dtype=np.float32)
            target_delta_exec = np.asarray(memory["target_delta_exec"][mem_idx], dtype=np.float32)
            action_chunk = np.asarray(memory["action_chunk"][mem_idx], dtype=np.float32)
            target_state_transition = np.asarray(memory["state_transition"][mem_idx], dtype=np.float32) if "state_transition" in memory else None
            constraint_info = _joint_pair_selector_constraint_terms(
                cfg,
                bundle,
                base_summary_exec_np=base_summary_exec,
                target_summary_exec_np=target_summary_exec,
                target_delta_exec_np=target_delta_exec,
                action_chunk_np=action_chunk,
                task_vec_np=task_vec,
                sub_np=sub,
                current_state=np.asarray(current_state, dtype=np.float32) if current_state is not None else None,
                target_transition=target_state_transition,
                device=base_future.device,
            )
            energy = float(
                lambda_exp * float(constraint_info["c_exp"])
                + lambda_latent * float(constraint_info["c_latent"])
                + lambda_physical * float(constraint_info["c_physical"])
            )
            fused_score = float(-energy)
            value_score = None
            if current_state is not None and lambda_value > 0.0:
                value_score = _joint_pair_value_score(
                    bundle,
                    task=task,
                    subtask_index=int(subtask_index or 0),
                    current_state=np.asarray(current_state, dtype=np.float32),
                    target_summary_exec=target_summary_exec,
                    action_chunk=action_chunk,
                )
                if value_score is not None:
                    fused_score = float(fused_score + lambda_value * float(value_score))
            cross_stage_info = {}
            cross_stage_score = None
            if current_state is not None and lambda_cross_stage > 0.0 and next_task:
                cross_stage_info = _joint_pair_cross_stage_score(
                    bundle,
                    task=task,
                    next_task=next_task,
                    subtask_index=int(subtask_index or 0),
                    current_state=np.asarray(current_state, dtype=np.float32),
                    target_future=torch.from_numpy(target_summary_exec).to(base_future.device, dtype=base_future.dtype),
                    target_action=action_chunk_t.squeeze(0),
                )
                if cross_stage_info is not None:
                    cross_stage_score = float(cross_stage_info["cross_stage_score"])
                    fused_score = float(fused_score + lambda_cross_stage * cross_stage_score)
            item = {
                "rank": rank,
                "memory_index": int(mem_idx),
                "energy": float(energy),
                "fused_score": float(fused_score),
                "selector_score": float(fused_score),
                "selector_energy_term": float(-energy),
                "selector_c_exp_term": float(-lambda_exp * float(constraint_info["c_exp"])),
                "selector_c_latent_term": float(-lambda_latent * float(constraint_info["c_latent"])),
                "selector_c_physical_term": float(-lambda_physical * float(constraint_info["c_physical"])),
                "selector_dynamics_term": 0.0,
                "selector_value_term": 0.0 if value_score is None else float(lambda_value * float(value_score)),
                "selector_cross_stage_term": 0.0 if cross_stage_score is None else float(lambda_cross_stage * cross_stage_score),
                "value_score": None if value_score is None else float(value_score),
                "cross_stage_score": None if cross_stage_score is None else float(cross_stage_score),
                "similarity": float(sims[order[rank - 1]]),
                **sim_meta,
                "trace_path": str(memory["trace_path"][mem_idx]),
                "outcome": str(memory["outcome"][mem_idx]),
                "factor": str(memory["factor"][mem_idx]),
                "action_chunk": np.asarray(memory["action_chunk"][mem_idx], dtype=np.float32).copy(),
                "state_transition": np.asarray(memory["state_transition"][mem_idx], dtype=np.float32).copy()
                if "state_transition" in memory
                else None,
                **constraint_info,
                **(cross_stage_info or {}),
            }
            scored.append(item)
            if best is None or fused_score > float(best["fused_score"]):
                best = item
    if best is None:
        return None, None, {"joint_pair_selector_used": False, "joint_pair_selector_reason": "energy_failed"}
    scored.sort(key=lambda x: float(x["fused_score"]), reverse=True)
    second_fused = float(scored[1]["fused_score"]) if len(scored) > 1 else float(best["fused_score"])
    fused_margin = float(float(best["fused_score"]) - second_fused)
    second_energy = float(scored[1]["energy"]) if len(scored) > 1 else float(best["energy"])
    energy_margin = float(second_energy - float(best["energy"]))
    min_margin = float(getattr(cfg, "joint_pair_min_energy_margin", 0.0))
    max_energy = float(getattr(cfg, "joint_pair_max_energy", 1e9))
    min_similarity = float(getattr(cfg, "joint_pair_min_similarity", -1.0))
    gate_reasons = []
    if (fused_margin if use_dynamics else energy_margin) < min_margin:
        gate_reasons.append(f"margin<{min_margin:g}")
    if float(best["energy"]) > max_energy:
        gate_reasons.append(f"energy>{max_energy:g}")
    if float(best["similarity"]) < min_similarity:
        gate_reasons.append(f"sim<{min_similarity:g}")
    if gate_reasons:
        return None, None, {
            "joint_pair_selector_used": True,
            "joint_pair_selector_reason": "conservative_gate_block",
            "joint_hypothesis_init_used": bool(init_info.get("joint_hypothesis_init_used", False)),
            "joint_pair_selector_memory_index": int(best["memory_index"]),
            "joint_pair_selector_rank": int(best["rank"]),
            "joint_pair_selector_energy": float(best["energy"]),
            "joint_pair_selector_second_energy": second_energy,
            "joint_pair_selector_energy_margin": energy_margin,
            "joint_pair_selector_fused_score": float(best["fused_score"]),
            "joint_pair_selector_fused_margin": float(fused_margin),
            "joint_pair_selector_value_score": 0.0 if best.get("value_score") is None else float(best["value_score"]),
            "joint_pair_selector_used_dynamics": bool(use_dynamics),
            "joint_pair_selector_similarity": float(best["similarity"]),
            "joint_pair_selector_joint_score": float(best["selector_score"]),
            "joint_pair_selector_query_sim_future": float(best.get("joint_query_similarity_future", 0.0)),
            "joint_pair_selector_query_sim_action": float(best.get("joint_query_similarity_action", 0.0)),
            "joint_pair_selector_query_sim_state": float(best.get("joint_query_similarity_state", 0.0)),
            "joint_pair_selector_query_sim_stage": float(best.get("joint_query_similarity_stage", 0.0)),
            "joint_pair_selector_gate_passed": False,
            "joint_pair_selector_gate_reason": ";".join(gate_reasons),
            **init_info,
            **{
                k: best[k]
                for k in [
                    "c_exp",
                    "c_latent",
                    "c_physical",
                    "dynamics_cos",
                    "dynamics_l1",
                    "robotics_success_prob",
                    "robotics_contact_bad_prob",
                    "selector_energy_term",
                    "selector_c_exp_term",
                    "selector_c_latent_term",
                    "selector_c_physical_term",
                    "selector_dynamics_term",
                    "selector_value_term",
                    "selector_cross_stage_term",
                ]
                if k in best
            },
        }
    target_key = str(getattr(cfg, "joint_pair_target_future_key", "target_proxy_future") or "target_proxy_future")
    target_np = _load_future_npz_array(best["trace_path"], target_key)
    if target_np is None:
        return None, None, {"joint_pair_selector_used": False, "joint_pair_selector_reason": "missing_target_future"}
    target_future = torch.from_numpy(target_np).to(device=base_future.device, dtype=base_future.dtype)
    target_action = torch.from_numpy(best["action_chunk"]).to(device=base_future.device, dtype=base_future.dtype)
    action_gen_info = {}
    generated_action, action_gen_info = _joint_pair_generate_action_init(
        bundle,
        base_summary_exec=base_summary_exec,
        target_summary_exec=np.asarray(memory["target_summary_exec"][int(best["memory_index"])], dtype=np.float32),
        target_delta_exec=np.asarray(memory["target_delta_exec"][int(best["memory_index"])], dtype=np.float32),
        task_vec=task_vec,
        sub=sub,
        current_state=current_state,
        device=base_future.device,
    )
    if generated_action is not None:
        gen_mix = float(getattr(cfg, "joint_pair_action_generator_mix", 0.5))
        with torch.no_grad():
            target_action = (1.0 - gen_mix) * target_action + gen_mix * generated_action.to(dtype=target_action.dtype, device=target_action.device)
        action_gen_info["joint_pair_action_generator_mix"] = float(gen_mix)
    repair_info = {}
    target_future, target_action, repair_info = apply_coupled_correction(
        cfg,
        mode="slow",
        task=task,
        subtask_index=int(subtask_index or 0),
        current_state=current_state,
        current_future=target_future,
        current_action=target_action,
        pending_effect=None,
        base_future_for_score=base_future,
        target_transition=best.get("state_transition"),
    )
    coupling_prior_info = {}
    if bool(getattr(cfg, "joint_pair_repair_with_coupling_prior", False)):
        target_future, target_action, coupling_prior_info = apply_dynamic_coupling_candidate_prior(
            cfg,
            task=task,
            subtask_index=int(subtask_index or 0),
            current_state=current_state,
            base_future=base_future,
            slow_target_future=target_future,
            candidate_action_intent=target_action,
        )
    init_future_for_verify = target_future.detach().clone()
    init_action = target_action.detach().clone()
    refine_info = None
    verify_info = {}
    verify_enabled = False
    refine_enabled = False
    refine_steps = int(getattr(cfg, "joint_pair_refine_steps", 0))
    if verify_enabled:
        verify_info = verify_joint_pair_hypothesis(
            cfg,
            bundle,
            base_future,
            target_future,
            target_action,
            task,
            int(subtask_index or 0),
            init_future_for_verify,
            init_action,
            current_state=current_state,
            target_transition=best.get("state_transition"),
        )
    if (
        verify_enabled
        and not bool(verify_info.get("joint_pair_verify_passed", False))
        and refine_enabled
        and refine_steps > 0
    ):
        target_future, target_action, refine_info = refine_joint_pair_energy(
            cfg,
            bundle,
            base_future,
            target_future,
            target_action,
            task,
            int(subtask_index or 0),
            current_state=current_state,
            target_transition=best.get("state_transition"),
        )
        refine_info = dict(refine_info)
        refine_info["joint_pair_refine_fallback_used"] = True
        refine_info["joint_pair_refine_fallback_trigger_reason"] = str(
            verify_info.get("joint_pair_verify_reason", "verification_block")
        )
        delta_e = float(refine_info["joint_pair_refine_delta_energy"])
        refine_tau = float(getattr(cfg, "joint_pair_refine_min_delta_energy", 0.0))
        if delta_e <= refine_tau:
            return None, None, {
                "joint_pair_selector_used": True,
                "joint_pair_selector_reason": "refine_fallback_delta_energy_block",
                "joint_hypothesis_init_used": bool(init_info.get("joint_hypothesis_init_used", False)),
                "joint_pair_selector_memory_index": int(best["memory_index"]),
                "joint_pair_selector_rank": int(best["rank"]),
                "joint_pair_selector_energy": float(best["energy"]),
                "joint_pair_selector_second_energy": second_energy,
                "joint_pair_selector_energy_margin": energy_margin,
                "joint_pair_selector_fused_score": float(best["fused_score"]),
                "joint_pair_selector_fused_margin": float(fused_margin),
                "joint_pair_selector_value_score": 0.0 if best.get("value_score") is None else float(best["value_score"]),
                "joint_pair_selector_used_dynamics": bool(use_dynamics),
                "joint_pair_selector_similarity": float(best["similarity"]),
                "joint_pair_selector_joint_score": float(best["selector_score"]),
                "joint_pair_selector_gate_passed": False,
                "joint_pair_selector_gate_reason": f"fallback_deltaE<={refine_tau:g}",
                **init_info,
                **{
                    k: best[k]
                    for k in [
                        "c_exp",
                        "c_latent",
                        "c_physical",
                        "dynamics_cos",
                        "dynamics_l1",
                        "robotics_success_prob",
                        "robotics_contact_bad_prob",
                        "selector_energy_term",
                        "selector_c_exp_term",
                        "selector_c_latent_term",
                        "selector_c_physical_term",
                        "selector_dynamics_term",
                        "selector_value_term",
                        "selector_cross_stage_term",
                    ]
                    if k in best
                },
                **action_gen_info,
                **repair_info,
                **coupling_prior_info,
                **verify_info,
                **refine_info,
            }
        verify_info = verify_joint_pair_hypothesis(
            cfg,
            bundle,
            base_future,
            target_future,
            target_action,
            task,
            int(subtask_index or 0),
            init_future_for_verify,
            init_action,
            current_state=current_state,
            target_transition=best.get("state_transition"),
        )
    elif refine_enabled and refine_steps > 0:
        refine_info = {
            "joint_pair_refine_used": False,
            "joint_pair_refine_fallback_used": False,
            "joint_pair_refine_skip_reason": "verification_passed_or_disabled",
        }
    future_shift_norm = float(repair_info.get("joint_pair_repair_future_shift_norm", 0.0))
    max_future_shift = float(getattr(cfg, "joint_pair_max_future_shift", 0.0))
    if max_future_shift > 0.0 and future_shift_norm > max_future_shift:
        return None, None, {
            "joint_pair_selector_used": True,
            "joint_pair_selector_reason": "future_shift_block",
            "joint_hypothesis_init_used": bool(init_info.get("joint_hypothesis_init_used", False)),
            "joint_pair_selector_memory_index": int(best["memory_index"]),
            "joint_pair_selector_rank": int(best["rank"]),
            "joint_pair_selector_energy": float(best["energy"]),
            "joint_pair_selector_second_energy": second_energy,
            "joint_pair_selector_energy_margin": energy_margin,
            "joint_pair_selector_fused_score": float(best["fused_score"]),
            "joint_pair_selector_fused_margin": float(fused_margin),
            "joint_pair_selector_value_score": 0.0 if best.get("value_score") is None else float(best["value_score"]),
            "joint_pair_selector_used_dynamics": bool(use_dynamics),
            "joint_pair_selector_similarity": float(best["similarity"]),
            "joint_pair_selector_joint_score": float(best["selector_score"]),
            "joint_pair_selector_gate_passed": False,
            "joint_pair_selector_gate_reason": f"future_shift>{max_future_shift:g}",
            **init_info,
            **{
                k: best[k]
                for k in [
                    "c_exp",
                    "c_latent",
                    "c_physical",
                    "dynamics_cos",
                    "dynamics_l1",
                    "robotics_success_prob",
                    "robotics_contact_bad_prob",
                    "selector_energy_term",
                    "selector_c_exp_term",
                    "selector_c_latent_term",
                    "selector_c_physical_term",
                    "selector_dynamics_term",
                    "selector_value_term",
                    "selector_cross_stage_term",
                ]
                if k in best
            },
            **action_gen_info,
            **repair_info,
            **coupling_prior_info,
            **(refine_info or {}),
            **verify_info,
        }
    push_sensitive = is_push_sensitive_task(task)
    if refine_info is not None and bool(refine_info.get("joint_pair_refine_used", False)) and push_sensitive:
        push_max_action_shift = float(
            getattr(
                cfg,
                "joint_pair_push_sensitive_max_action_shift",
                getattr(cfg, "joint_pair_refine_action_max_delta", 5.0),
            )
        )
        refine_action_shift = float(refine_info.get("joint_pair_refine_action_shift_norm", 0.0))
        if push_max_action_shift > 0.0 and refine_action_shift > push_max_action_shift:
            return None, None, {
                "joint_pair_selector_used": True,
                "joint_pair_selector_reason": "push_sensitive_action_shift_block",
                "joint_hypothesis_init_used": bool(init_info.get("joint_hypothesis_init_used", False)),
                "joint_pair_selector_memory_index": int(best["memory_index"]),
                "joint_pair_selector_rank": int(best["rank"]),
                "joint_pair_selector_energy": float(best["energy"]),
                "joint_pair_selector_second_energy": second_energy,
                "joint_pair_selector_energy_margin": energy_margin,
                "joint_pair_selector_fused_score": float(best["fused_score"]),
                "joint_pair_selector_fused_margin": float(fused_margin),
                "joint_pair_selector_value_score": 0.0 if best.get("value_score") is None else float(best["value_score"]),
                "joint_pair_selector_used_dynamics": bool(use_dynamics),
                "joint_pair_selector_similarity": float(best["similarity"]),
                "joint_pair_selector_joint_score": float(best["selector_score"]),
                "joint_pair_selector_gate_passed": False,
                "joint_pair_selector_gate_reason": f"push_action_shift>{push_max_action_shift:g}",
                **init_info,
                **{
                    k: best[k]
                    for k in [
                        "c_exp",
                        "c_latent",
                        "c_physical",
                        "dynamics_cos",
                        "dynamics_l1",
                        "robotics_success_prob",
                        "robotics_contact_bad_prob",
                        "selector_energy_term",
                        "selector_c_exp_term",
                        "selector_c_latent_term",
                        "selector_c_physical_term",
                        "selector_dynamics_term",
                        "selector_value_term",
                        "selector_cross_stage_term",
                    ]
                    if k in best
                },
                **action_gen_info,
                **repair_info,
                **coupling_prior_info,
                **refine_info,
            }
        action_mix = float(getattr(cfg, "joint_pair_refine_action_mix", 1.0))
        push_action_mix = float(getattr(cfg, "joint_pair_push_sensitive_action_mix", action_mix))
        if push_action_mix < 1.0:
            with torch.no_grad():
                target_action = init_action + push_action_mix * (target_action - init_action)
            mixed_shift = float(torch.norm((target_action - init_action).reshape(-1), p=2).item())
            refine_info = dict(refine_info)
            refine_info["joint_pair_refine_action_mix"] = float(push_action_mix)
            refine_info["joint_pair_refine_action_shift_norm_mixed"] = mixed_shift
            refine_info["joint_pair_refine_push_sensitive"] = True
    elif refine_info is not None and bool(refine_info.get("joint_pair_refine_used", False)):
        action_mix = float(getattr(cfg, "joint_pair_refine_action_mix", 1.0))
        if action_mix < 1.0:
            with torch.no_grad():
                target_action = init_action + action_mix * (target_action - init_action)
            mixed_shift = float(torch.norm((target_action - init_action).reshape(-1), p=2).item())
            refine_info = dict(refine_info)
            refine_info["joint_pair_refine_action_mix"] = float(action_mix)
            refine_info["joint_pair_refine_action_shift_norm_mixed"] = mixed_shift
    if verify_enabled:
        if not bool(verify_info.get("joint_pair_verify_passed", False)):
            return None, None, {
                "joint_pair_selector_used": True,
                "joint_pair_selector_reason": "verification_block",
                "joint_hypothesis_init_used": bool(init_info.get("joint_hypothesis_init_used", False)),
                "joint_pair_selector_memory_index": int(best["memory_index"]),
                "joint_pair_selector_rank": int(best["rank"]),
                "joint_pair_selector_energy": float(best["energy"]),
                "joint_pair_selector_second_energy": second_energy,
                "joint_pair_selector_energy_margin": energy_margin,
                "joint_pair_selector_fused_score": float(best["fused_score"]),
                "joint_pair_selector_fused_margin": float(fused_margin),
                "joint_pair_selector_used_dynamics": bool(use_dynamics),
                "joint_pair_selector_similarity": float(best["similarity"]),
                "joint_pair_selector_joint_score": float(best["selector_score"]),
                "joint_pair_selector_gate_passed": False,
                "joint_pair_selector_gate_reason": str(verify_info.get("joint_pair_verify_reason", "verification_block")),
                **init_info,
                **{
                    k: best[k]
                    for k in [
                        "c_exp",
                        "c_latent",
                        "c_physical",
                        "dynamics_cos",
                        "dynamics_l1",
                        "robotics_success_prob",
                        "robotics_contact_bad_prob",
                        "selector_energy_term",
                        "selector_c_exp_term",
                        "selector_c_latent_term",
                        "selector_c_physical_term",
                        "selector_dynamics_term",
                        "selector_value_term",
                        "selector_cross_stage_term",
                    ]
                    if k in best
                },
                **action_gen_info,
                **repair_info,
                **coupling_prior_info,
                **(refine_info or {}),
                **verify_info,
            }
    return (
        target_future,
        target_action,
        {
        "joint_pair_selector_used": True,
        "joint_pair_selector_reason": "selected",
        "joint_pair_selector_rank": int(best["rank"]),
        "joint_pair_selector_memory_index": int(best["memory_index"]),
        "joint_pair_selector_energy": float(best["energy"]),
        "joint_pair_selector_second_energy": second_energy,
        "joint_pair_selector_energy_margin": energy_margin,
        "joint_pair_selector_fused_score": float(best["fused_score"]),
        "joint_pair_selector_joint_score": float(best["selector_score"]),
        "joint_pair_selector_fused_margin": float(fused_margin),
        "joint_pair_selector_value_score": 0.0 if best.get("value_score") is None else float(best["value_score"]),
        "joint_pair_selector_used_dynamics": bool(use_dynamics),
        "joint_pair_selector_similarity": float(best["similarity"]),
        "joint_pair_selector_query_sim_future": float(best.get("joint_query_similarity_future", 0.0)),
        "joint_pair_selector_query_sim_action": float(best.get("joint_query_similarity_action", 0.0)),
        "joint_pair_selector_query_sim_state": float(best.get("joint_query_similarity_state", 0.0)),
        "joint_pair_selector_query_sim_stage": float(best.get("joint_query_similarity_stage", 0.0)),
        "joint_pair_selector_outcome": best["outcome"],
        "joint_pair_selector_factor": best["factor"],
        "joint_pair_selector_trace_path": best["trace_path"],
        "joint_pair_selector_action_shape": list(best["action_chunk"].shape),
        "joint_pair_selector_gate_passed": True,
        "joint_pair_selector_gate_reason": "pass",
        **{
            k: best[k]
                    for k in [
                        "c_exp",
                        "c_latent",
                        "c_physical",
                        "dynamics_cos",
                        "dynamics_l1",
                        "robotics_success_prob",
                        "robotics_contact_bad_prob",
                        "selector_energy_term",
                        "selector_c_exp_term",
                        "selector_c_latent_term",
                        "selector_c_physical_term",
                        "selector_dynamics_term",
                        "selector_value_term",
                        "selector_cross_stage_term",
                    ]
            if k in best
        },
        **init_info,
        **action_gen_info,
        **repair_info,
        **coupling_prior_info,
        **(refine_info or {}),
        **verify_info,
    },
    )


def load_immune_repulsion_memory(cfg):
    path = str(getattr(cfg, "immune_memory_npz", "") or "")
    if not path:
        return None
    cached = _IMMUNE_REPULSION_CACHE.get(path)
    if cached is not None:
        return cached
    data = np.load(path, allow_pickle=True)
    failure = np.asarray(data["failure_features"], dtype=np.float32)
    norms = np.maximum(np.linalg.norm(failure, axis=1, keepdims=True), 1e-8)
    bundle = {
        "path": path,
        "failure_features": failure,
        "success_features": np.asarray(data["success_features"], dtype=np.float32),
        "attract_deltas": np.asarray(data["attract_deltas"], dtype=np.float32),
        "tasks": [str(x) for x in data["tasks"].tolist()],
        "failure_norm": failure / norms,
    }
    if "failure_factors" in data.files:
        bundle["failure_factors"] = [str(x) for x in data["failure_factors"].tolist()]
    if "failure_token_features" in data.files and "success_token_features" in data.files:
        bundle["failure_token_features"] = np.asarray(data["failure_token_features"], dtype=np.float32)
        bundle["success_token_features"] = np.asarray(data["success_token_features"], dtype=np.float32)
    _IMMUNE_REPULSION_CACHE[path] = bundle
    return bundle


def load_token_immune_adapter(cfg, device):
    path = str(getattr(cfg, "token_immune_adapter_ckpt", "") or "")
    if not path:
        return None
    cached = _TOKEN_IMMUNE_ADAPTER_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    adapter = TokenImmuneAdapter(
        future_dim=int(ckpt["future_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
    )
    adapter.load_state_dict(ckpt["model_state"], strict=False)
    adapter = adapter.to(device).eval()
    _TOKEN_IMMUNE_ADAPTER_CACHE[path] = adapter
    print(f"[INFO] Loaded token immune adapter: {path}")
    return adapter


def load_local_repair_field_scorer(cfg, device):
    path = str(getattr(cfg, "immune_local_repair_field_ckpt", "") or "")
    if not path:
        return None
    cached = _LOCAL_REPAIR_FIELD_SCORER_CACHE.get((path, str(device)))
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    model = LocalRepairFieldScorer(
        feature_dim=int(ckpt.get("feature_dim", 8)),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device).eval()
    bundle = {
        "path": path,
        "model": model,
        "temperature": float(ckpt.get("temperature", 1.0)),
        "topk": int(ckpt.get("topk", 0)),
    }
    _LOCAL_REPAIR_FIELD_SCORER_CACHE[(path, str(device))] = bundle
    print(f"[INFO] Loaded local repair field scorer: {path}")
    return bundle


def load_immune_q_filter(cfg, device):
    path = str(getattr(cfg, "immune_q_filter_ckpt", "") or "")
    if not path:
        return None
    cached = _IMMUNE_Q_FILTER_CACHE.get((path, str(device)))
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    model = FutureQCritic(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device).eval()
    bundle = {
        "path": path,
        "model": model,
        "task_dim": int(ckpt.get("task_dim", 128)),
        "mean": torch.from_numpy(np.asarray(ckpt["mean"], dtype=np.float32).reshape(1, -1)).to(device),
        "std": torch.from_numpy(np.asarray(ckpt["std"], dtype=np.float32).reshape(1, -1)).to(device),
    }
    _IMMUNE_Q_FILTER_CACHE[(path, str(device))] = bundle
    print(f"[INFO] Loaded immune Q filter: {path}")
    return bundle


def immune_q_success_prob(q_bundle, futures, task, subtask_index):
    if q_bundle is None:
        return None
    if futures.ndim == 2:
        futures = futures.unsqueeze(0)
    task_vec = torch.from_numpy(q_hashed_task(str(task), int(q_bundle["task_dim"]))).to(
        device=futures.device, dtype=futures.dtype
    )
    task_vec = task_vec.unsqueeze(0).expand(futures.shape[0], -1)
    sub = torch.full(
        (futures.shape[0], 1),
        float(subtask_index if subtask_index is not None else 0.0) / 5.0,
        device=futures.device,
        dtype=futures.dtype,
    )
    features = torch.cat([q_future_summary_torch(futures.float()).to(dtype=futures.dtype), task_vec, sub], dim=-1)
    features = (features - q_bundle["mean"].to(dtype=futures.dtype)) / torch.clamp(q_bundle["std"].to(dtype=futures.dtype), min=1e-6)
    with torch.no_grad():
        return torch.sigmoid(q_bundle["model"](features)).detach()


def load_causal_intervention_adapter(cfg, device):
    path = str(getattr(cfg, "causal_intervention_adapter_ckpt", "") or "")
    if not path:
        return None
    cached = _CAUSAL_INTERVENTION_ADAPTER_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    future_dim = int(ckpt.get("future_dim", 0) or (ckpt.get("future_shape", [0, 0])[-1]))
    adapter = CausalInterventionAdapter(
        future_dim=future_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
    )
    adapter.load_state_dict(ckpt["model_state"], strict=False)
    adapter = adapter.to(device).eval()
    bundle = {
        "path": path,
        "adapter": adapter,
        "factors": [str(x) for x in ckpt.get("factors", CAUSAL_FACTORS)],
        "segments": [str(x) for x in ckpt.get("segments", CAUSAL_SEGMENTS)],
    }
    _CAUSAL_INTERVENTION_ADAPTER_CACHE[path] = bundle
    print(f"[INFO] Loaded causal intervention adapter: {path}")
    return bundle


def load_future_manifold_navigator(cfg, device):
    path = str(getattr(cfg, "future_manifold_navigator_ckpt", "") or "")
    if not path:
        return None
    cached = _FUTURE_MANIFOLD_NAVIGATOR_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("model_type") == "learned_future_geometry_field":
        model = LearnedFutureGeometryField(
            future_dim=int(ckpt.get("future_dim", 0) or ckpt.get("future_shape", [0, 0])[-1]),
            task_dim=int(ckpt.get("task_dim", 128)),
            z_dim=int(ckpt.get("z_dim", 256)),
            hidden_dim=int(ckpt.get("hidden_dim", 512)),
        )
        model_type = "learned_future_geometry_field"
    elif "z_dim" in ckpt:
        model = TopTaskFutureManifold(
            future_dim=int(ckpt.get("future_dim", 0) or ckpt.get("future_shape", [0, 0])[-1]),
            task_dim=int(ckpt.get("task_dim", 128)),
            z_dim=int(ckpt.get("z_dim", 256)),
            hidden_dim=int(ckpt.get("hidden_dim", 512)),
        )
        model_type = "top_task_future_manifold"
    else:
        model = FutureManifoldNavigator(
            future_dim=int(ckpt.get("future_dim", 0) or ckpt.get("future_shape", [0, 0])[-1]),
            task_dim=int(ckpt.get("task_dim", 128)),
            hidden_dim=int(ckpt.get("hidden_dim", 512)),
        )
        model_type = "future_manifold_navigator"
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device).eval()
    bundle = {
        "path": path,
        "model": model,
        "model_type": model_type,
        "task_dim": int(ckpt.get("task_dim", 128)),
        "tasks": [str(x) for x in ckpt.get("tasks", [])],
    }
    _FUTURE_MANIFOLD_NAVIGATOR_CACHE[path] = bundle
    print(f"[INFO] Loaded future manifold navigator: {path} type={model_type}")
    return bundle


def infer_causal_factor_segment_from_task(task):
    text = str(task).lower()
    if any(key in text for key in ("push", "slider", "move_slider", "rotate")):
        return "motion", "mid"
    if any(key in text for key in ("lift", "stack", "unstack", "grasp")):
        return "contact", "early"
    if any(key in text for key in ("drawer", "place_in")):
        return "object", "mid"
    if any(key in text for key in ("led", "lightbulb", "turn_on", "turn_off")):
        return "goal", "late"
    return "motion", "mid"


def load_future_success_verifier(cfg, device):
    path = str(getattr(cfg, "future_success_verifier_ckpt", "") or "")
    if not path:
        return None
    cached = _FUTURE_SUCCESS_VERIFIER_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    model = FutureSuccessVerifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    bundle = {
        "path": path,
        "model": model,
        "future_key": str(ckpt.get("future_key", "base_future")),
        "task_dim": int(ckpt.get("task_dim", 128)),
        "mean": np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1),
        "std": np.asarray(ckpt["std"], dtype=np.float32).reshape(-1),
    }
    _FUTURE_SUCCESS_VERIFIER_CACHE[path] = bundle
    print(f"[INFO] Loaded future success verifier: {path}")
    return bundle


def predict_future_success_risk(cfg, base_future, task, subtask_index=None):
    bundle = load_future_success_verifier(cfg, base_future.device)
    if bundle is None:
        return None
    future_np = base_future.detach().float().cpu().numpy().astype(np.float32)
    future_sig = summarize_future(future_np)
    task_sig = hashed_text_features(str(task), int(bundle["task_dim"]))
    sub_idx = float(subtask_index if subtask_index is not None else 0.0) / 5.0
    features = np.concatenate([future_sig, task_sig, np.asarray([sub_idx], dtype=np.float32)], axis=0).astype(np.float32)
    features = (features - bundle["mean"]) / np.maximum(bundle["std"], 1e-6)
    tensor = torch.from_numpy(features).unsqueeze(0).to(base_future.device)
    with torch.no_grad():
        prob_success = torch.sigmoid(bundle["model"](tensor)).item()
    risk = 1.0 - float(prob_success)
    return {
        "future_success_verifier_prob_success": float(prob_success),
        "future_success_verifier_risk": float(risk),
        "future_success_verifier_ckpt": str(bundle["path"]),
    }


def load_relation_future_probe(cfg, device):
    path = str(getattr(cfg, "relation_probe_ckpt", "") or "")
    if not path:
        return None
    cached = _RELATION_FUTURE_PROBE_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    targets = [str(x) for x in ckpt.get("targets", RELATION_PROBE_TARGETS)]
    if targets != RELATION_PROBE_TARGETS:
        raise ValueError(f"relation probe targets mismatch: expected {RELATION_PROBE_TARGETS}, got {targets}")
    model = RelationFutureProbeMLP(
        in_dim=int(ckpt.get("input_dim", ckpt.get("in_dim"))),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        dropout=0.0,
    )
    state = ckpt.get("model_state") or ckpt.get("model")
    model.load_state_dict(state)
    model = model.to(device).eval()
    bundle = {
        "path": path,
        "model": model,
        "task_dim": int(ckpt.get("task_dim", 128)),
        "future_shape": tuple(ckpt.get("future_shape", ())),
        "targets": targets,
    }
    _RELATION_FUTURE_PROBE_CACHE[path] = bundle
    print(f"[INFO] Loaded relation future probe: {path}")
    return bundle


def predict_relation_future_probe(cfg, base_future, task):
    bundle = load_relation_future_probe(cfg, base_future.device)
    if bundle is None:
        return None
    future = base_future.detach().float()
    if future.ndim < 2:
        future = future.reshape(1, -1)
    pooled = future.mean(dim=0)
    std = future.std(dim=0, unbiased=False)
    task_vec_np = task_hash(str(task), int(bundle["task_dim"]))
    task_vec = torch.from_numpy(task_vec_np).to(device=base_future.device, dtype=pooled.dtype)
    features = torch.cat([pooled, std, task_vec], dim=0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(bundle["model"](features)).detach().cpu().numpy().reshape(-1)
    info = {
        "relation_probe": True,
        "relation_probe_ckpt": str(bundle["path"]),
    }
    for name, prob in zip(RELATION_PROBE_TARGETS, probs.tolist()):
        info[f"relation_{name}_prob"] = float(prob)
    return info


def relation_probe_should_trigger(cfg, relation_info, task):
    if relation_info is None:
        return True, "missing_relation_probe"
    risk_prob = float(relation_info.get("relation_risk_prob", 0.0))
    progress_prob = float(relation_info.get("relation_progress_prob", 0.0))
    risk_threshold = float(getattr(cfg, "relation_risk_threshold", 0.65))
    progress_threshold = float(getattr(cfg, "relation_progress_threshold", 0.6))
    task_text = str(task).lower()
    progress_task = any(key in task_text for key in ("drawer", "slider"))
    if risk_prob >= risk_threshold:
        return True, "risk_above_threshold"
    if progress_task and progress_prob >= progress_threshold:
        return True, "progress_above_threshold"
    return False, "below_relation_threshold"


def load_future_energy_model(cfg, device):
    path = str(getattr(cfg, "future_energy_ckpt", "") or "")
    if not path:
        return None
    cached = _FUTURE_ENERGY_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    model = TaskConditionedFutureEnergy(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    bundle = {
        "path": path,
        "model": model,
        "task_dim": int(ckpt.get("task_dim", 128)),
        "mean": torch.from_numpy(np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1)).to(device),
        "std": torch.from_numpy(np.asarray(ckpt["std"], dtype=np.float32).reshape(-1)).to(device),
    }
    _FUTURE_ENERGY_CACHE[path] = bundle
    print(f"[INFO] Loaded task-conditioned future energy: {path}")
    return bundle


def future_energy_score(bundle, future, task, subtask_index):
    features = build_energy_features_torch(
        future,
        str(task),
        float(subtask_index if subtask_index is not None else 0.0),
        int(bundle["task_dim"]),
        future.device,
    )
    features = (features - bundle["mean"]) / torch.clamp(bundle["std"], min=1e-6)
    return bundle["model"](features.unsqueeze(0)).squeeze(0)


def load_future_energy_factor_masks(cfg, device, dtype):
    path = str(getattr(cfg, "future_energy_factor_mask_npz", "") or "")
    if not path:
        return None
    cached = _FUTURE_ENERGY_MASK_CACHE.get((path, str(device), str(dtype)))
    if cached is not None:
        return cached
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f"[WARN] failed to load future energy factor masks: {path}: {exc}")
        return None

    def read_mask(*names):
        masks = []
        for name in names:
            if name in data.files:
                masks.append(np.asarray(data[name], dtype=np.float32))
            prefixed = f"mask_{name}"
            if prefixed in data.files:
                masks.append(np.asarray(data[prefixed], dtype=np.float32))
        if not masks:
            return None
        out = np.maximum.reduce(masks)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    masks = {
        "contact": read_mask("contact"),
        "object": read_mask("object", "object_displacement", "object_identity"),
        "motion": read_mask("motion", "drawer_slider_progress", "object_displacement"),
        "goal": read_mask("goal", "goal_completion"),
    }
    masks = {key: value for key, value in masks.items() if value is not None}
    if not masks:
        return None
    _FUTURE_ENERGY_MASK_CACHE[(path, str(device), str(dtype))] = masks
    print(f"[INFO] Loaded future energy factor masks: {path} factors={sorted(masks)}")
    return masks


def select_future_energy_factor(cfg, task):
    mode = str(getattr(cfg, "future_energy_factor", "auto") or "auto")
    if mode == "auto":
        return infer_energy_factor_from_task(task)
    return canonical_energy_factor(mode)


def apply_future_energy_descent(cfg, base_future, task, subtask_index=None):
    if not bool(getattr(cfg, "future_energy_descent", False)):
        return base_future, {"future_energy_descent": False, "future_energy_applied": False}
    bundle = load_future_energy_model(cfg, base_future.device)
    if bundle is None:
        return base_future, {
            "future_energy_descent": True,
            "future_energy_applied": False,
            "future_energy_reason": "missing_ckpt",
        }
    original = base_future.detach()
    future_var = original.clone().detach().requires_grad_(True)
    energy = future_energy_score(bundle, future_var, task, subtask_index)
    threshold = float(getattr(cfg, "future_energy_threshold", 0.0))
    if float(energy.detach().cpu()) <= threshold:
        return original, {
            "future_energy_descent": True,
            "future_energy_applied": False,
            "future_energy_reason": "below_threshold",
            "future_energy_before": float(energy.detach().cpu()),
            "future_energy_threshold": threshold,
        }
    grad = torch.autograd.grad(energy, future_var, retain_graph=False, create_graph=False)[0]
    mask_mode = str(getattr(cfg, "future_energy_mask_mode", "topk") or "topk")
    selected_factor = select_future_energy_factor(cfg, task)
    density = float(getattr(cfg, "future_energy_mask_density", 0.01))
    density = min(max(density, 0.0), 1.0)
    mask_source = "topk_gradient"
    masks = None
    if mask_mode == "factor":
        masks = load_future_energy_factor_masks(cfg, grad.device, grad.dtype)
    if mask_mode == "factor" and masks is not None and selected_factor in masks:
        factor_mask = masks[selected_factor]
        while factor_mask.ndim < grad.ndim:
            factor_mask = factor_mask.unsqueeze(0)
        factor_mask = factor_mask.expand_as(grad)
        if density > 0.0 and density < 1.0:
            factor_grad = grad.detach().abs() * factor_mask
            active = factor_grad.reshape(-1)
            positive = active[active > 0]
            if positive.numel() > 0:
                k = max(1, int(round(positive.numel() * density)))
                threshold_grad = torch.topk(positive, min(k, positive.numel())).values[-1]
                mask = ((factor_grad >= threshold_grad) & (factor_mask > 0)).to(dtype=grad.dtype)
            else:
                mask = factor_mask.to(dtype=grad.dtype)
        else:
            mask = factor_mask.to(dtype=grad.dtype)
        mask_source = f"factor:{selected_factor}"
    else:
        if density <= 0.0:
            mask = torch.ones_like(grad)
        elif density >= 1.0:
            mask = torch.ones_like(grad)
        else:
            flat = grad.detach().abs().reshape(-1)
            k = max(1, int(round(flat.numel() * density)))
            threshold_grad = torch.topk(flat, k).values[-1]
            mask = (grad.detach().abs() >= threshold_grad).to(dtype=grad.dtype)
    step_size = float(getattr(cfg, "future_energy_step_size", 0.001))
    risk_gate = float(getattr(cfg, "future_energy_risk_gate", 1.0))
    max_shift = float(getattr(cfg, "future_energy_max_shift_norm", 1.0))
    direction = mask * grad.detach()
    direction_norm = torch.norm(direction.reshape(-1), p=2)
    normalize_grad = bool(getattr(cfg, "future_energy_normalize_grad", False))
    if normalize_grad and float(direction_norm.item()) > 1e-8:
        raw_delta = -step_size * risk_gate * direction / torch.clamp(direction_norm, min=1e-8)
    else:
        raw_delta = -step_size * risk_gate * direction
    raw_norm = torch.norm(raw_delta.reshape(-1), p=2)
    scale = 1.0
    if max_shift > 0.0 and float(raw_norm.item()) > max_shift:
        scale = max_shift / max(float(raw_norm.item()), 1e-8)
    delta = raw_delta * scale
    edited = original + delta.to(dtype=original.dtype)
    with torch.no_grad():
        energy_after = future_energy_score(bundle, edited, task, subtask_index)
    return edited.detach(), {
        "future_energy_descent": True,
        "future_energy_applied": True,
        "future_energy_reason": "applied",
        "future_energy_ckpt": str(bundle["path"]),
        "future_energy_before": float(energy.detach().cpu()),
        "future_energy_after": float(energy_after.detach().cpu()),
        "future_energy_delta": float((energy_after - energy).detach().cpu()),
        "future_energy_threshold": threshold,
        "future_energy_step_size": step_size,
        "future_energy_risk_gate": risk_gate,
        "future_energy_normalize_grad": bool(normalize_grad),
        "future_energy_mask_density": density,
        "future_energy_mask_mode": mask_mode,
        "future_energy_mask_source": mask_source,
        "future_energy_selected_factor": selected_factor,
        "future_energy_grad_norm": float(torch.norm(grad.detach().reshape(-1), p=2).cpu()),
        "future_energy_direction_norm": float(direction_norm.detach().cpu()),
        "future_energy_raw_shift_norm": float(raw_norm.detach().cpu()),
        "future_energy_shift_norm": float(torch.norm(delta.reshape(-1), p=2).detach().cpu()),
        "future_energy_shift_scale": float(scale),
        "future_energy_mask_active": int(mask.detach().sum().cpu()),
    }


def apply_immune_repulsion(cfg, base_future, task, subtask_index=None):
    if not bool(getattr(cfg, "immune_repulsion", False)):
        return base_future, {"immune_repulsion": False, "immune_applied": False, "immune_reason": "disabled"}
    bundle = load_immune_repulsion_memory(cfg)
    if bundle is None:
        return base_future, {"immune_repulsion": True, "immune_applied": False, "immune_reason": "missing_memory"}
    pooled = base_future.detach().float().mean(dim=0).cpu().numpy().astype(np.float32)
    pooled_norm = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
    task = str(task)
    same_task = bool(getattr(cfg, "immune_same_task_only", True))
    candidates = [idx for idx, memory_task in enumerate(bundle["tasks"]) if (not same_task or memory_task == task)]
    if not candidates:
        candidates = list(range(len(bundle["tasks"])))
    if not candidates:
        return base_future, {"immune_repulsion": True, "immune_applied": False, "immune_reason": "no_candidates"}
    sims = bundle["failure_norm"][candidates] @ pooled_norm
    k = max(1, min(int(getattr(cfg, "immune_topk_failure", 5)), len(candidates)))
    order = np.argsort(-sims)[:k]
    selected = [candidates[int(i)] for i in order]
    raw_sims = sims[order].astype(np.float32)
    local_repair_field = bool(getattr(cfg, "immune_local_repair_field", False))
    learned_repair_bundle = load_local_repair_field_scorer(cfg, base_future.device)
    if learned_repair_bundle is not None:
        same = np.asarray(
            [1.0 if bundle["tasks"][idx] == task else max(float(getattr(cfg, "immune_cross_task_weight", 0.25)), 0.0) for idx in selected],
            dtype=np.float32,
        )
        pair_features = build_pair_features(
            pooled,
            bundle["failure_features"][selected].astype(np.float32),
            bundle["success_features"][selected].astype(np.float32),
            bundle["attract_deltas"][selected].astype(np.float32),
            same,
        )
        with torch.no_grad():
            logits = learned_repair_bundle["model"](torch.from_numpy(pair_features).to(base_future.device))
            logits = logits / max(float(getattr(cfg, "immune_learned_weight_temperature", learned_repair_bundle["temperature"])), 1e-6)
            weights_t = torch.softmax(logits, dim=0)
        weights = weights_t.detach().float().cpu().numpy().astype(np.float32)
        locality = pair_features[:, 0].astype(np.float32)
        task_compat = same.astype(np.float32)
        consistency = pair_features[:, 3].astype(np.float32)
        repair_consistency = consistency.astype(np.float32)
        local_repair_field = True
        learned_local_repair_field = True
    elif local_repair_field:
        locality_temp = max(float(getattr(cfg, "immune_locality_temperature", 0.05)), 1e-6)
        consistency_temp = max(float(getattr(cfg, "immune_consistency_temperature", 0.25)), 1e-6)
        cross_task_weight = max(float(getattr(cfg, "immune_cross_task_weight", 0.25)), 0.0)
        locality = np.exp((raw_sims - float(np.max(raw_sims))) / locality_temp).astype(np.float32)
        task_compat = np.asarray(
            [1.0 if bundle["tasks"][idx] == task else cross_task_weight for idx in selected],
            dtype=np.float32,
        )
        selected_deltas = bundle["attract_deltas"][selected].astype(np.float32)
        delta_norm = np.maximum(np.linalg.norm(selected_deltas, axis=1, keepdims=True), 1e-8)
        delta_unit = selected_deltas / delta_norm
        seed_weights = locality * np.maximum(task_compat, 1e-8)
        if float(seed_weights.sum()) <= 1e-8:
            seed_weights = np.ones(len(selected), dtype=np.float32) / float(len(selected))
        else:
            seed_weights = seed_weights / seed_weights.sum()
        mean_direction = np.tensordot(seed_weights, delta_unit, axes=(0, 0)).astype(np.float32)
        mean_direction = mean_direction / max(float(np.linalg.norm(mean_direction)), 1e-8)
        consistency = np.maximum(delta_unit @ mean_direction, 0.0).astype(np.float32)
        repair_consistency = np.exp((consistency - float(np.max(consistency))) / consistency_temp).astype(np.float32)
        weights = locality * task_compat * repair_consistency
        if float(weights.sum()) <= 1e-8:
            weights = np.ones(len(selected), dtype=np.float32) / float(len(selected))
        else:
            weights = weights / weights.sum()
        learned_local_repair_field = False
    else:
        locality = np.maximum(raw_sims, 0.0).astype(np.float32)
        task_compat = np.asarray(
            [1.0 if bundle["tasks"][idx] == task else 0.0 for idx in selected],
            dtype=np.float32,
        )
        consistency = np.ones(len(selected), dtype=np.float32)
        repair_consistency = np.ones(len(selected), dtype=np.float32)
        weights = locality
        if float(weights.sum()) <= 1e-8:
            weights = np.ones(len(selected), dtype=np.float32) / float(len(selected))
        else:
            weights = weights / weights.sum()
        learned_local_repair_field = False
    q_filter_info = {}
    q_filter_bundle = load_immune_q_filter(cfg, base_future.device)
    if q_filter_bundle is not None and len(selected) > 0:
        selected_deltas_np = bundle["attract_deltas"][selected].astype(np.float32)
        delta_candidates = torch.from_numpy(selected_deltas_np).to(device=base_future.device, dtype=base_future.dtype)
        while delta_candidates.ndim < base_future.ndim + 1:
            delta_candidates = delta_candidates.unsqueeze(1)
        delta_candidates = delta_candidates.expand((len(selected),) + tuple(base_future.shape))
        scale = float(getattr(cfg, "immune_q_filter_candidate_scale", 1.0))
        candidate_futures = base_future.unsqueeze(0) + scale * delta_candidates
        q_base = immune_q_success_prob(q_filter_bundle, base_future, task, subtask_index)
        q_candidates = immune_q_success_prob(q_filter_bundle, candidate_futures, task, subtask_index)
        if q_base is not None and q_candidates is not None:
            q_base_value = float(q_base.reshape(-1)[0].item())
            q_values = q_candidates.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
            min_gain = float(getattr(cfg, "immune_q_filter_min_gain", 0.0))
            gain = np.maximum(q_values - q_base_value - min_gain, 0.0).astype(np.float32)
            mode = str(getattr(cfg, "immune_q_filter_mode", "soft") or "soft")
            if mode == "hard":
                q_factor = (gain > 0.0).astype(np.float32)
            else:
                temp = max(float(getattr(cfg, "immune_q_filter_temperature", 0.1)), 1e-6)
                q_factor = np.exp((gain - float(np.max(gain))) / temp).astype(np.float32)
                if float(gain.max()) <= 0.0:
                    q_factor = np.zeros_like(q_factor, dtype=np.float32)
            if float(q_factor.sum()) > 1e-8:
                weights = (np.asarray(weights, dtype=np.float32) * q_factor).astype(np.float32)
                weights = weights / max(float(weights.sum()), 1e-8)
                q_reason = "applied"
            elif bool(getattr(cfg, "immune_q_filter_allow_fallback", True)):
                q_reason = "fallback_no_positive_gain"
            else:
                weights = np.zeros_like(weights, dtype=np.float32)
                q_reason = "zeroed_no_positive_gain"
            q_filter_info = {
                "immune_q_filter": True,
                "immune_q_filter_ckpt": str(getattr(cfg, "immune_q_filter_ckpt", "") or ""),
                "immune_q_filter_reason": q_reason,
                "immune_q_filter_mode": mode,
                "immune_q_filter_min_gain": float(min_gain),
                "immune_q_filter_candidate_scale": float(scale),
                "immune_q_base": float(q_base_value),
                "immune_q_candidates": [float(x) for x in q_values.tolist()],
                "immune_q_gains": [float(x) for x in gain.tolist()],
                "immune_q_factor": [float(x) for x in q_factor.tolist()],
            }
    if float(np.asarray(weights, dtype=np.float32).sum()) <= 1e-8:
        weights = np.ones(len(selected), dtype=np.float32) / float(len(selected))
    failure_center = np.tensordot(weights, bundle["failure_features"][selected], axes=(0, 0)).astype(np.float32)
    success_center = np.tensordot(weights, bundle["success_features"][selected], axes=(0, 0)).astype(np.float32)
    attract_center = np.tensordot(weights, bundle["attract_deltas"][selected], axes=(0, 0)).astype(np.float32)
    repair_pair_only = bool(getattr(cfg, "immune_repair_pair_only", False))
    alpha = 0.0 if repair_pair_only else float(getattr(cfg, "immune_alpha_repel", 1.0))
    beta = 1.0 if repair_pair_only else float(getattr(cfg, "immune_beta_attract", 1.0))
    base_gate = float(getattr(cfg, "immune_gate", 0.02))
    gate = base_gate
    relation_info = predict_relation_future_probe(cfg, base_future, task)
    relation_gate_forced_zero = False
    if relation_info is not None and bool(getattr(cfg, "relation_probe_trigger", False)):
        relation_allowed, relation_reason = relation_probe_should_trigger(cfg, relation_info, task)
        relation_info["relation_trigger_allowed"] = bool(relation_allowed)
        relation_info["relation_trigger_reason"] = str(relation_reason)
        relation_info["relation_risk_threshold"] = float(getattr(cfg, "relation_risk_threshold", 0.65))
        relation_info["relation_progress_threshold"] = float(getattr(cfg, "relation_progress_threshold", 0.6))
        if not relation_allowed:
            gate = 0.0
            relation_gate_forced_zero = True
    verifier_info = predict_future_success_risk(cfg, base_future, task, subtask_index)
    if verifier_info is not None:
        risk = float(verifier_info["future_success_verifier_risk"])
        min_risk = float(getattr(cfg, "future_success_verifier_min_risk", 0.0))
        max_gate = float(getattr(cfg, "future_success_verifier_max_gate", base_gate))
        min_gate = float(getattr(cfg, "future_success_verifier_min_gate", 0.0))
        if risk < min_risk:
            gate = 0.0
            verifier_info["future_success_verifier_gate_reason"] = "below_min_risk"
        else:
            denom = max(1.0 - min_risk, 1e-6)
            scaled = (risk - min_risk) / denom
            gate = min(max_gate, max(min_gate, base_gate * scaled))
            verifier_info["future_success_verifier_gate_reason"] = "scaled_by_risk"
        verifier_info["future_success_verifier_base_gate"] = float(base_gate)
        verifier_info["future_success_verifier_min_risk"] = float(min_risk)
        verifier_info["future_success_verifier_min_gate"] = float(min_gate)
        verifier_info["future_success_verifier_max_gate"] = float(max_gate)
        verifier_info["future_success_verifier_scaled_gate"] = float(gate)
    geometry_info = {}
    if bool(getattr(cfg, "immune_geometry_gate", False)):
        pooled_unit = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
        failure_unit = failure_center / max(float(np.linalg.norm(failure_center)), 1e-8)
        success_unit = success_center / max(float(np.linalg.norm(success_center)), 1e-8)
        sim_failure = float(np.dot(pooled_unit, failure_unit))
        sim_success = float(np.dot(pooled_unit, success_unit))
        risk_margin = sim_failure - sim_success
        field_confidence = float(np.sum(np.asarray(weights, dtype=np.float32) * np.asarray(consistency, dtype=np.float32)))
        min_margin = float(getattr(cfg, "immune_geometry_min_margin", 0.0))
        max_margin = float(getattr(cfg, "immune_geometry_max_margin", 0.02))
        min_gate = float(getattr(cfg, "immune_geometry_min_gate", 0.0))
        max_gate = float(getattr(cfg, "immune_geometry_max_gate", base_gate))
        if bool(getattr(cfg, "immune_field_confidence_gate", False)):
            alpha = float(getattr(cfg, "immune_field_gate_risk_alpha", 1.0))
            beta = float(getattr(cfg, "immune_field_gate_consistency_beta", 1.0))
            min_conf = float(getattr(cfg, "immune_field_gate_min_consistency", 0.25))
            temperature = max(float(getattr(cfg, "immune_field_gate_temperature", 0.25)), 1e-6)
            risk_score = (risk_margin - min_margin) / max(max_margin - min_margin, 1e-8)
            confidence_score = field_confidence - min_conf
            gate_score = (alpha * risk_score + beta * confidence_score) / temperature
            scaled = float(1.0 / (1.0 + np.exp(-np.clip(gate_score, -60.0, 60.0))))
            if field_confidence < min_conf and risk_margin <= min_margin:
                gate = 0.0
                reason = "below_margin_and_consistency"
            else:
                gate = min(max_gate, max(min_gate, max_gate * scaled))
                reason = "scaled_by_risk_and_repair_consistency"
        else:
            if risk_margin <= min_margin:
                gate = 0.0
                reason = "below_min_margin"
            else:
                denom = max(max_margin - min_margin, 1e-8)
                scaled = max(0.0, min(1.0, (risk_margin - min_margin) / denom))
                gate = min(max_gate, max(min_gate, max_gate * scaled))
                reason = "scaled_by_geometry_margin"
        geometry_info = {
            "immune_geometry_gate": True,
            "immune_geometry_sim_failure": sim_failure,
            "immune_geometry_sim_success": sim_success,
            "immune_geometry_risk_margin": float(risk_margin),
            "immune_geometry_field_confidence": float(field_confidence),
            "immune_field_confidence_gate": bool(getattr(cfg, "immune_field_confidence_gate", False)),
            "immune_field_gate_risk_alpha": float(getattr(cfg, "immune_field_gate_risk_alpha", 1.0)),
            "immune_field_gate_consistency_beta": float(getattr(cfg, "immune_field_gate_consistency_beta", 1.0)),
            "immune_field_gate_min_consistency": float(getattr(cfg, "immune_field_gate_min_consistency", 0.25)),
            "immune_field_gate_temperature": float(getattr(cfg, "immune_field_gate_temperature", 0.25)),
            "immune_geometry_min_margin": float(min_margin),
            "immune_geometry_max_margin": float(max_margin),
            "immune_geometry_min_gate": float(min_gate),
            "immune_geometry_max_gate": float(max_gate),
            "immune_geometry_scaled_gate": float(gate),
            "immune_geometry_gate_reason": reason,
        }
    repel = pooled - failure_center
    delta = alpha * repel + beta * attract_center
    delta_t = torch.from_numpy(delta).to(device=base_future.device, dtype=base_future.dtype)
    while delta_t.ndim < base_future.ndim:
        delta_t = delta_t.unsqueeze(0)
    manifold_nav_used = False
    manifold_nav_delta_norm = 0.0
    manifold_nav_gate_mean = 0.0
    manifold_nav_risk = None
    manifold_nav_boundary_margin = None
    manifold_nav_gate_reason = "disabled"
    manifold_nav_energy_base = None
    manifold_nav_energy_edited = None
    manifold_bundle = load_future_manifold_navigator(cfg, base_future.device)
    if manifold_bundle is not None:
        nav = manifold_bundle["model"]
        failure_t = torch.from_numpy(failure_center).to(device=base_future.device, dtype=base_future.dtype)
        success_t = torch.from_numpy(success_center).to(device=base_future.device, dtype=base_future.dtype)
        while failure_t.ndim < base_future.ndim:
            failure_t = failure_t.unsqueeze(0)
            success_t = success_t.unsqueeze(0)
        if failure_t.shape != base_future.shape:
            failure_t = failure_t.expand_as(base_future)
            success_t = success_t.expand_as(base_future)
        task_vec_np = task_hash(task, int(manifold_bundle["task_dim"]))
        task_vec = torch.from_numpy(task_vec_np).unsqueeze(0).to(device=base_future.device, dtype=base_future.dtype)
        sub_idx = torch.tensor([float(subtask_index if subtask_index is not None else 0.0) / 5.0], dtype=base_future.dtype, device=base_future.device)
        with torch.no_grad():
            nav_inputs = (
                base_future.unsqueeze(0),
                success_t.unsqueeze(0),
                failure_t.unsqueeze(0),
                task_vec,
                sub_idx,
            )
            if hasattr(nav, "navigate"):
                nav_delta, nav_aux = nav.navigate(*nav_inputs)
            else:
                nav_delta, nav_aux = nav(*nav_inputs)
            nav_delta = nav_delta.squeeze(0).to(dtype=base_future.dtype)
            nav_weight = float(getattr(cfg, "future_manifold_nav_weight", 1.0))
            delta_t = nav_weight * nav_delta
            if hasattr(nav, "energy"):
                manifold_nav_energy_base = float(nav.energy(base_future.unsqueeze(0), task_vec, sub_idx).item())
                manifold_nav_energy_edited = float(nav.energy((base_future + gate * delta_t).unsqueeze(0), task_vec, sub_idx).item())
        manifold_nav_used = True
        manifold_nav_delta_norm = float(torch.norm(nav_delta.reshape(-1), p=2).item())
        manifold_nav_gate_mean = float(nav_aux.get("gate", torch.zeros(1, device=base_future.device)).mean().item())
        if "risk" in nav_aux:
            manifold_nav_risk = float(nav_aux["risk"].detach().reshape(-1)[0].item())
        if "boundary_margin" in nav_aux:
            manifold_nav_boundary_margin = float(nav_aux["boundary_margin"].detach().reshape(-1)[0].item())
        if bool(getattr(cfg, "future_manifold_risk_gate", False)):
            threshold = float(getattr(cfg, "future_manifold_risk_threshold", 0.0))
            score = manifold_nav_risk
            if score is None:
                score = manifold_nav_boundary_margin
            if score is None:
                gate = 0.0
                manifold_nav_gate_reason = "missing_risk"
            elif score <= threshold:
                gate = 0.0
                manifold_nav_gate_reason = "below_risk_threshold"
            else:
                manifold_nav_gate_reason = "above_risk_threshold"
    if relation_gate_forced_zero:
        gate = 0.0
        if relation_info is not None:
            relation_info["relation_final_gate_override"] = "zeroed_by_relation_probe"
    causal_adapter_delta_norm = 0.0
    causal_adapter_gate_mean = 0.0
    causal_adapter_used = False
    causal_adapter_factor = "none"
    causal_adapter_segment = "none"
    causal_bundle = None if manifold_nav_used else load_causal_intervention_adapter(cfg, base_future.device)
    if causal_bundle is not None:
        causal_adapter = causal_bundle["adapter"]
        factors = causal_bundle["factors"]
        segments = causal_bundle["segments"]
        factor_name, segment_name = infer_causal_factor_segment_from_task(task)
        if factor_name not in factors:
            factor_name = factors[0]
        if segment_name not in segments:
            segment_name = segments[min(1, len(segments) - 1)]
        factor_idx = torch.tensor([factors.index(factor_name)], dtype=torch.long, device=base_future.device)
        segment_idx = torch.tensor([segments.index(segment_name)], dtype=torch.long, device=base_future.device)
        node_score = torch.tensor([max(float(sims[order[0]]) if len(order) else 0.0, 0.0)], dtype=torch.float32, device=base_future.device)
        failure_t = torch.from_numpy(failure_center).to(device=base_future.device, dtype=base_future.dtype)
        success_t = torch.from_numpy(success_center).to(device=base_future.device, dtype=base_future.dtype)
        while failure_t.ndim < base_future.ndim:
            failure_t = failure_t.unsqueeze(0)
            success_t = success_t.unsqueeze(0)
        if failure_t.shape != base_future.shape:
            failure_t = failure_t.expand_as(base_future)
            success_t = success_t.expand_as(base_future)
        with torch.no_grad():
            adapter_delta, adapter_gate = causal_adapter(
                base_future.unsqueeze(0),
                success_t.unsqueeze(0),
                failure_t.unsqueeze(0),
                factor_idx,
                segment_idx,
                node_score,
            )
        adapter_delta = adapter_delta.squeeze(0).to(dtype=base_future.dtype)
        causal_weight = float(getattr(cfg, "causal_intervention_gate", 1.0))
        delta_t = adapter_delta * causal_weight
        causal_adapter_delta_norm = float(torch.norm(adapter_delta.reshape(-1), p=2).item())
        causal_adapter_gate_mean = float(adapter_gate.mean().item())
        causal_adapter_used = True
        causal_adapter_factor = factor_name
        causal_adapter_segment = segment_name
    token_adapter_delta_norm = 0.0
    token_adapter_gate_mean = 0.0
    token_adapter_used = False
    token_adapter = load_token_immune_adapter(cfg, base_future.device)
    if token_adapter is not None:
        if "failure_token_features" in bundle and "success_token_features" in bundle:
            failure_token_center = np.tensordot(weights, bundle["failure_token_features"][selected], axes=(0, 0)).astype(np.float32)
            success_token_center = np.tensordot(weights, bundle["success_token_features"][selected], axes=(0, 0)).astype(np.float32)
            failure_t = torch.from_numpy(failure_token_center).to(device=base_future.device, dtype=base_future.dtype)
            success_t = torch.from_numpy(success_token_center).to(device=base_future.device, dtype=base_future.dtype)
        else:
            failure_t = torch.from_numpy(failure_center).to(device=base_future.device, dtype=base_future.dtype)
            success_t = torch.from_numpy(failure_center + attract_center).to(device=base_future.device, dtype=base_future.dtype)
            while failure_t.ndim < base_future.ndim:
                failure_t = failure_t.unsqueeze(0)
                success_t = success_t.unsqueeze(0)
            failure_t = failure_t.expand_as(base_future)
            success_t = success_t.expand_as(base_future)
        if failure_t.ndim == 2 and base_future.ndim == 2:
            pass
        elif failure_t.ndim < base_future.ndim:
            failure_t = failure_t.unsqueeze(0)
            success_t = success_t.unsqueeze(0)
        with torch.no_grad():
            adapter_delta, adapter_aux = token_adapter(base_future.unsqueeze(0), failure_t.unsqueeze(0), success_t.unsqueeze(0))
        adapter_delta = adapter_delta.squeeze(0).to(dtype=base_future.dtype)
        token_weight = float(getattr(cfg, "token_immune_adapter_weight", 1.0))
        delta_t = delta_t + token_weight * adapter_delta
        token_adapter_delta_norm = float(torch.norm(adapter_delta.reshape(-1), p=2).item())
        token_adapter_gate_mean = float(adapter_aux.get("gate", torch.zeros(1, device=base_future.device)).mean().item())
        token_adapter_used = True
    mask_mode = str(getattr(cfg, "future_manifold_mask_mode", "none") or "none")
    mask_density = float(getattr(cfg, "future_manifold_mask_density", 1.0))
    mask_info = {
        "future_manifold_mask_mode": mask_mode,
        "future_manifold_mask_density": float(mask_density),
        "future_manifold_mask_applied": False,
        "future_manifold_mask_active_fraction": 1.0,
        "future_manifold_mask_delta_norm_before": float(torch.norm(delta_t.reshape(-1), p=2).item()),
        "future_manifold_mask_delta_norm_after": float(torch.norm(delta_t.reshape(-1), p=2).item()),
    }
    if mask_mode != "none" and delta_t.numel() > 0:
        density = max(0.0, min(1.0, mask_density))
        if density <= 0.0:
            mask = torch.zeros_like(delta_t)
        elif density >= 1.0:
            mask = torch.ones_like(delta_t)
        elif mask_mode == "topk_element":
            flat = delta_t.detach().abs().reshape(-1)
            k_elem = max(1, int(round(float(flat.numel()) * density)))
            threshold = torch.topk(flat, min(k_elem, flat.numel())).values[-1]
            mask = (delta_t.detach().abs() >= threshold).to(dtype=delta_t.dtype)
        elif mask_mode == "topk_channel":
            score = delta_t.detach().abs().mean(dim=0)
            k_chan = max(1, int(round(float(score.numel()) * density)))
            threshold = torch.topk(score.reshape(-1), min(k_chan, score.numel())).values[-1]
            channel_mask = (score >= threshold).to(dtype=delta_t.dtype)
            mask = channel_mask.unsqueeze(0).expand_as(delta_t)
        elif mask_mode == "topk_time":
            score = delta_t.detach().abs().mean(dim=-1)
            k_time = max(1, int(round(float(score.numel()) * density)))
            threshold = torch.topk(score.reshape(-1), min(k_time, score.numel())).values[-1]
            time_mask = (score >= threshold).to(dtype=delta_t.dtype)
            mask = time_mask.unsqueeze(-1).expand_as(delta_t)
        else:
            mask = torch.ones_like(delta_t)
            mask_mode = f"unknown:{mask_mode}"
        delta_before = delta_t
        delta_t = delta_t * mask
        mask_info.update(
            {
                "future_manifold_mask_mode": mask_mode,
                "future_manifold_mask_applied": bool(mask_mode != "none"),
                "future_manifold_mask_active_fraction": float(mask.detach().float().mean().item()),
                "future_manifold_mask_delta_norm_before": float(torch.norm(delta_before.reshape(-1), p=2).item()),
                "future_manifold_mask_delta_norm_after": float(torch.norm(delta_t.reshape(-1), p=2).item()),
            }
        )
    edited = base_future + gate * delta_t
    return edited, {
        "immune_repulsion": True,
        "immune_applied": True,
        "immune_reason": "applied",
        "immune_gate": gate,
        "immune_base_gate": base_gate,
        "immune_alpha_repel": alpha,
        "immune_beta_attract": beta,
        "immune_repair_pair_only": bool(repair_pair_only),
        "immune_topk_failure": k,
        "immune_same_task_only": same_task,
        "immune_selected_tasks": [bundle["tasks"][idx] for idx in selected],
        "immune_selected_sims": [float(x) for x in sims[order].tolist()],
        "immune_local_repair_field": bool(local_repair_field),
        "immune_learned_local_repair_field": bool(learned_local_repair_field),
        "immune_local_repair_field_ckpt": str(getattr(cfg, "immune_local_repair_field_ckpt", "") or ""),
        "immune_locality_scores": [float(x) for x in np.asarray(locality).tolist()],
        "immune_task_compatibility": [float(x) for x in np.asarray(task_compat).tolist()],
        "immune_repair_consistency": [float(x) for x in np.asarray(repair_consistency).tolist()],
        "immune_direction_consistency": [float(x) for x in np.asarray(consistency).tolist()],
        "immune_local_field_weights": [float(x) for x in np.asarray(weights).tolist()],
        "immune_selected_factors": [
            bundle.get("failure_factors", ["unknown"] * len(bundle["tasks"]))[idx] for idx in selected
        ],
        **q_filter_info,
        "immune_delta_norm": float(np.linalg.norm(delta)),
        "immune_repel_norm": float(np.linalg.norm(repel)),
        "immune_attract_norm": float(np.linalg.norm(attract_center)),
        "future_manifold_nav_used": bool(manifold_nav_used),
        "future_manifold_nav_delta_norm": float(manifold_nav_delta_norm),
        "future_manifold_nav_gate_mean": float(manifold_nav_gate_mean),
        "future_manifold_nav_weight": float(getattr(cfg, "future_manifold_nav_weight", 1.0)),
        "future_manifold_nav_risk": manifold_nav_risk,
        "future_manifold_nav_boundary_margin": manifold_nav_boundary_margin,
        "future_manifold_risk_gate": bool(getattr(cfg, "future_manifold_risk_gate", False)),
        "future_manifold_risk_threshold": float(getattr(cfg, "future_manifold_risk_threshold", 0.0)),
        "future_manifold_gate_reason": manifold_nav_gate_reason,
        "future_manifold_nav_energy_base": manifold_nav_energy_base,
        "future_manifold_nav_energy_edited": manifold_nav_energy_edited,
        "token_immune_adapter_used": bool(token_adapter_used),
        "token_immune_adapter_delta_norm": float(token_adapter_delta_norm),
        "token_immune_adapter_gate_mean": float(token_adapter_gate_mean),
        "token_immune_adapter_weight": float(getattr(cfg, "token_immune_adapter_weight", 1.0)),
        "causal_intervention_adapter_used": bool(causal_adapter_used),
        "causal_intervention_adapter_factor": causal_adapter_factor,
        "causal_intervention_adapter_segment": causal_adapter_segment,
        "causal_intervention_adapter_delta_norm": float(causal_adapter_delta_norm),
        "causal_intervention_adapter_gate_mean": float(causal_adapter_gate_mean),
        "causal_intervention_gate": float(getattr(cfg, "causal_intervention_gate", 1.0)),
        **mask_info,
        **(verifier_info or {}),
        **(relation_info or {}),
        **geometry_info,
    }


def maybe_save_future_trace(cfg, trace_payload):
    limit = int(getattr(cfg, "future_trace_limit", 0))
    trace_dir = str(getattr(cfg, "future_trace_dir", "") or "")
    if limit <= 0 or not trace_dir:
        return None
    current = int(getattr(cfg, "_future_trace_count", 0))
    if current >= limit:
        return None
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_path = out_dir / f"future_trace_{current:04d}.npz"
    serializable = {}
    for key, value in trace_payload.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            serializable[key] = value.detach().cpu().numpy().astype(np.float32)
        elif isinstance(value, np.ndarray):
            serializable[key] = value
        elif np.isscalar(value):
            serializable[key] = np.asarray(value)
    np.savez_compressed(rel_path, **serializable)
    cfg._future_trace_count = current + 1
    return str(rel_path)


def observation_image_payload(obs, prefix="obs", max_images=8):
    """Extract image-like arrays from a CALVIN observation for trace visualization."""
    payload = {}

    def visit(value, name):
        if len(payload) >= max_images:
            return
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(v, f"{name}_{k}")
            return
        else:
            return
        arr = np.asarray(arr)
        if arr.ndim == 3 and (arr.shape[-1] in (1, 3, 4) or arr.shape[0] in (1, 3, 4)):
            safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
            payload[safe_name] = arr
        elif arr.ndim == 4 and (arr.shape[-1] in (1, 3, 4) or arr.shape[1] in (1, 3, 4)):
            safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
            payload[safe_name] = arr[:1]

    visit(obs, prefix)
    return payload


def raw_rgb_payload(raw, prefix="raw", max_images=8):
    payload = {}

    def visit(value, name):
        if len(payload) >= max_images:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, f"{name}_{k}")
            return
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            return
        arr = np.asarray(arr)
        key_l = name.lower()
        image_like = arr.ndim == 3 and (arr.shape[-1] in (1, 3, 4) or arr.shape[0] in (1, 3, 4))
        if not image_like:
            return
        # Keep RGB-like camera frames; avoid saving depth/state tensors.
        if not any(tag in key_l for tag in ["rgb", "image", "static", "gripper"]):
            return
        safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        payload[safe_name] = arr

    visit(raw, prefix)
    return payload


def _flatten_action_for_probe(action):
    if isinstance(action, torch.Tensor):
        return action.detach().float().cpu().numpy().reshape(-1)
    if isinstance(action, np.ndarray):
        return action.astype(np.float32).reshape(-1)
    if isinstance(action, (list, tuple)):
        parts = [_flatten_action_for_probe(x) for x in action]
        parts = [x for x in parts if x.size > 0]
        return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
    if isinstance(action, dict):
        parts = [_flatten_action_for_probe(action[k]) for k in sorted(action)]
        parts = [x for x in parts if x.size > 0]
        return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
    try:
        return np.asarray(action, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=np.float32)


def maybe_save_action_trace(cfg, actions):
    if not bool(getattr(cfg, "collect_action_trace", False)):
        return None
    trace_dir = str(getattr(cfg, "action_trace_dir", "") or "")
    if not trace_dir:
        return None
    current = int(getattr(cfg, "_action_trace_count", 0))
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"action_trace_{current:04d}.npz"
    arr = np.stack(actions).astype(np.float32) if actions else np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(path, actions=arr)
    cfg._action_trace_count = current + 1
    return str(path)


def _runtime_tensor_to_np(value, fallback_shape=(0,)):
    if value is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    try:
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return np.zeros(fallback_shape, dtype=np.float32)


def maybe_save_joint_belief_transition_row(
    cfg,
    *,
    sequence_index,
    subtask_index,
    task,
    lang_text,
    step,
    state_before,
    state_after,
    action,
    h_future_before,
    h_action_before,
    h_future_after,
    h_action_after,
    action_remain_before,
    action_remain_after,
    info,
):
    if not bool(getattr(cfg, "collect_joint_belief_transition", False)):
        return None
    trace_dir = str(getattr(cfg, "joint_belief_transition_trace_dir", "") or "")
    rows_path = str(getattr(cfg, "joint_belief_transition_rows_path", "") or "")
    if not trace_dir or not rows_path:
        raise RuntimeError(
            "collect_joint_belief_transition=True but rows/trace paths are not initialized"
        )
    current = int(getattr(cfg, "_joint_belief_transition_count", 0))
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_rel = f"joint_belief_transition/row_{current:07d}.npz"
    feature_path = out_dir / f"row_{current:07d}.npz"
    np.savez_compressed(
        feature_path,
        state_before=np.asarray(state_before, dtype=np.float32).reshape(-1),
        state_after=np.asarray(state_after, dtype=np.float32).reshape(-1),
        action=np.asarray(action, dtype=np.float32).reshape(-1),
        h_future_before=_runtime_tensor_to_np(h_future_before),
        h_action_before=_runtime_tensor_to_np(h_action_before),
        h_future_after=_runtime_tensor_to_np(h_future_after),
        h_action_after=_runtime_tensor_to_np(h_action_after),
        action_remain_before=_runtime_tensor_to_np(action_remain_before),
        action_remain_after=_runtime_tensor_to_np(action_remain_after),
    )
    row = {
        "row_id": current,
        "sequence_index": int(sequence_index),
        "subtask_index": int(subtask_index or 0),
        "task": str(task),
        "lang_text": str(lang_text),
        "step": int(step),
        "feature_path": feature_rel,
    }
    for key, value in dict(info or {}).items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            row[key] = value
    rows_file = Path(rows_path)
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    with rows_file.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    cfg._joint_belief_transition_count = current + 1
    return str(feature_path)


def maybe_save_sensitivity_teacher_row(
    cfg,
    *,
    sequence_index,
    subtask_index,
    task,
    lang_text,
    step,
    state_before,
    state_after,
    action,
    h_future_before,
    h_action_before,
    h_future_after,
    h_action_after,
    action_remain_before,
    teacher,
    info,
):
    if not bool(getattr(cfg, "collect_sensitivity_teacher", False)):
        return None
    if not isinstance(teacher, dict) or not bool(teacher.get("available", False)):
        return None
    trace_dir = str(getattr(cfg, "sensitivity_teacher_trace_dir", "") or "")
    rows_path = str(getattr(cfg, "sensitivity_teacher_rows_path", "") or "")
    if not trace_dir or not rows_path:
        raise RuntimeError("collect_sensitivity_teacher=True but rows/trace paths are not initialized")
    current = int(getattr(cfg, "_sensitivity_teacher_count", 0))
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_rel = f"sensitivity_teacher/row_{current:07d}.npz"
    feature_path = out_dir / f"row_{current:07d}.npz"
    np.savez_compressed(
        feature_path,
        state_before=np.asarray(state_before, dtype=np.float32).reshape(-1),
        state_after=np.asarray(state_after, dtype=np.float32).reshape(-1),
        executed_action=np.asarray(action, dtype=np.float32).reshape(-1),
        h_future_before=_runtime_tensor_to_np(h_future_before),
        h_action_before=_runtime_tensor_to_np(h_action_before),
        h_future_after=_runtime_tensor_to_np(h_future_after),
        h_action_after=_runtime_tensor_to_np(h_action_after),
        action_remain_before=_runtime_tensor_to_np(action_remain_before),
        delta_h=_runtime_tensor_to_np(teacher.get("delta_h")),
        teacher_delta_action=_runtime_tensor_to_np(teacher.get("delta_action")),
        teacher_suffix_delta_action=_runtime_tensor_to_np(teacher.get("suffix_delta_action")),
        teacher_suffix_action=_runtime_tensor_to_np(teacher.get("suffix_action")),
        teacher_step_scores=_runtime_tensor_to_np(teacher.get("step_scores")),
    )
    row = {
        "row_id": current,
        "sequence_index": int(sequence_index),
        "subtask_index": int(subtask_index or 0),
        "task": str(task),
        "lang_text": str(lang_text),
        "step": int(step),
        "feature_path": feature_rel,
        "teacher_rank": int(teacher.get("rank", 0)),
        "teacher_eps": float(teacher.get("eps", 0.0)),
        "teacher_h_dim": int(teacher.get("h_dim", 0)),
        "teacher_start_idx": int(teacher.get("start_idx", 0)),
        "teacher_delta_h_norm": float(teacher.get("delta_h_norm", 0.0)),
        "teacher_delta_action_norm": float(teacher.get("delta_action_norm", 0.0)),
        "teacher_suffix_delta_norm": float(teacher.get("suffix_delta_norm", 0.0)),
        "teacher_step_score_max": float(teacher.get("step_score_max", 0.0)),
        "teacher_step_score_mean": float(teacher.get("step_score_mean", 0.0)),
        "teacher_future_delta_mode": str(teacher.get("future_delta_mode", "")),
        "teacher_action_delta_mode": str(teacher.get("action_delta_mode", "")),
        "teacher_mode": str(teacher.get("teacher_mode", "low_rank_jvp")),
    }
    for key, value in dict(info or {}).items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            row[key] = value
    rows_file = Path(rows_path)
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    with rows_file.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    cfg._sensitivity_teacher_count = current + 1
    return str(feature_path)


def action_trace_summary(actions):
    if not actions:
        return {"num_actions": 0, "action_dim": 0}
    arr = np.stack(actions).astype(np.float32)
    return {
        "num_actions": int(arr.shape[0]),
        "action_dim": int(arr.shape[1]) if arr.ndim >= 2 else 1,
        "action_mean": _strong_gt_to_jsonable(arr.mean(axis=0)),
        "action_std": _strong_gt_to_jsonable(arr.std(axis=0)),
        "action_l2_total": float(np.linalg.norm(arr.reshape(-1))),
        "action_l2_step_mean": float(np.linalg.norm(arr, axis=1).mean()) if arr.ndim >= 2 and arr.shape[0] else 0.0,
        "first_action": _strong_gt_to_jsonable(arr[0]) if arr.shape[0] else [],
        "last_action": _strong_gt_to_jsonable(arr[-1]) if arr.shape[0] else [],
    }


def probe_action_sensitivity_to_future(model, obs, goal, base_future, edit_direction, eps_values):
    """Probe whether the frozen action decoder is sensitive to a future-latent direction."""
    old_mode = getattr(model, "future_feature_mode", None)
    old_override = getattr(model, "override_future_feature", None)
    actions = []
    try:
        for eps in eps_values:
            model.reset()
            model.future_feature_mode = "override"
            model.override_future_feature = base_future + float(eps) * edit_direction
            with torch.no_grad():
                action = model.step(obs, goal)
            actions.append(_flatten_action_for_probe(action))
    except Exception as exc:
        return {"action_sensitivity_error": str(exc)[:300]}
    finally:
        model.future_feature_mode = old_mode
        model.override_future_feature = old_override
        model.reset()

    if not actions or actions[0].size == 0:
        return {"action_sensitivity_error": "empty_action"}
    base_action = actions[0]
    out = {
        "action_sensitivity_eps": [float(x) for x in eps_values],
        "action_sensitivity_base_action_norm": float(np.linalg.norm(base_action)),
    }
    deltas = []
    rels = []
    for eps, action in zip(eps_values[1:], actions[1:]):
        if action.shape != base_action.shape:
            out["action_sensitivity_error"] = f"action_shape_mismatch:{base_action.shape}->{action.shape}"
            return out
        delta = float(np.linalg.norm(action - base_action))
        rel = delta / max(float(np.linalg.norm(base_action)), 1e-8)
        out[f"action_delta_eps_{str(eps).replace('.', 'p').replace('-', 'm')}"] = delta
        out[f"action_delta_rel_eps_{str(eps).replace('.', 'p').replace('-', 'm')}"] = rel
        deltas.append(delta)
        rels.append(rel)
    out["action_sensitivity_max_delta"] = float(max(deltas) if deltas else 0.0)
    out["action_sensitivity_max_rel_delta"] = float(max(rels) if rels else 0.0)
    return out


def parse_reflection_json(raw_text):
    clean = str(raw_text or "").strip()
    if not clean:
        return {}
    try:
        return json.loads(clean)
    except Exception:
        pass
    candidates = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for idx, ch in enumerate(clean):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(clean[start : idx + 1])
                    start = None
    preferred_keys = ("wrong_component", "failure_factor", "mismatch_type", "trust_score")
    parsed = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            parsed.append(payload)
    for payload in reversed(parsed):
        if any(key in payload for key in preferred_keys):
            return payload
    return parsed[-1] if parsed else {}


def summarize_future_tensor_for_diagnosis(future):
    try:
        arr = future.detach().float().cpu().numpy()
    except Exception:
        return {}
    if arr.ndim != 2:
        return {"future_shape": list(arr.shape), "future_norm": safe_float(np.linalg.norm(arr))}
    token_energy = np.linalg.norm(arr, axis=1)
    channel_energy = np.mean(np.abs(arr), axis=0)
    return {
        "future_shape": [int(x) for x in arr.shape],
        "future_norm": round(float(np.linalg.norm(arr)), 4),
        "token_energy_mean": round(float(token_energy.mean()), 4),
        "token_energy_max": round(float(token_energy.max()), 4),
        "top_time_tokens": [int(x) for x in np.argsort(-token_energy)[:5].tolist()],
        "top_channels": [int(x) for x in np.argsort(-channel_energy)[:8].tolist()],
    }


def compact_diagnosis_memory(rows, limit=3):
    out = []
    for row in list(rows or [])[:limit]:
        out.append(
            {
                "task": row.get("task"),
                "success": bool(row.get("success", False)),
                "steps": row.get("steps"),
                "memory_type": row.get("memory_type"),
                "mismatch_type": row.get("mismatch_type"),
            }
        )
    return out


def build_qwen_failure_diagnosis_prompt(
    task,
    lang_text,
    seq_idx,
    sub_idx,
    base_future,
    key_rows,
    recent_memory,
    energy_info=None,
    success=None,
    steps=None,
):
    payload = {
        "task": task,
        "language": lang_text,
        "sequence_context": {
            "sequence_index": int(seq_idx),
            "subtask_index": int(sub_idx),
        },
        "future_risk_evidence": {
            "energy": (energy_info or {}).get("future_energy_before"),
            "energy_after": (energy_info or {}).get("future_energy_after"),
            "energy_delta": (energy_info or {}).get("future_energy_delta"),
            "energy_factor": (energy_info or {}).get("future_energy_selected_factor"),
            "energy_reason": (energy_info or {}).get("future_energy_reason"),
        },
        "base_future_summary": summarize_future_tensor_for_diagnosis(base_future),
        "nearest_failure_memory": compact_diagnosis_memory([row for row in key_rows or [] if not bool(row.get("success", False))]),
        "nearest_success_memory": compact_diagnosis_memory([row for row in key_rows or [] if bool(row.get("success", False))]),
        "instruction": "Before rollout, diagnose where and why the predicted future may fail, and state the intended counterfactual edit. Do not assume the rollout outcome is known.",
    }
    if success is not None or steps is not None:
        payload["rollout_outcome"] = {
            "success": bool(success),
            "steps": int(steps or 0),
            "sequence_index": int(seq_idx),
            "subtask_index": int(sub_idx),
        }
    return (
        "You are a robot pre-rollout future-diagnosis module. Return the JSON object first. "
        "Do not write explanations before or after the JSON. Given a CALVIN task, "
        "predicted future evidence, and nearest success/failure memory summaries, diagnose why the predicted "
        "future may fail before executing the rollout. Return exactly one compact JSON object with keys: "
        '{"failure_mechanism": string, "wrong_component": string, "time_window": [number, number], '
        '"object": string, "edit_intent": string}. '
        "wrong_component must be one of contact, object_displacement, object_identity, "
        "drawer_slider_progress, goal_completion, none.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_qwen_failure_diagnosis(raw_text):
    raw = str(raw_text or "")
    payload = parse_reflection_json(raw_text)
    component = str(payload.get("wrong_component", "") or "").strip()
    if not component:
        match = re.search(
            r'"wrong_component"\s*:\s*"([a-zA-Z_]+)',
            raw,
            flags=re.IGNORECASE,
        )
        if match is not None:
            component = match.group(1).strip()
    if not component:
        match = re.search(
            r"wrong\s+component\s+(?:is|:)\s*['\"]?([a-zA-Z_]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if match is not None:
            component = match.group(1).strip()
    if not component and "no nearest failure memor" in raw.lower():
        component = "none"
    if component not in set(FACTOR_ORDER + ["none"]):
        component = "unknown"
    time_window = payload.get("time_window", [0.0, 0.0])
    if (not isinstance(time_window, list) or len(time_window) != 2) and raw:
        match = re.search(
            r"time\s+window\s+(?:is|:)\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]",
            raw,
            flags=re.IGNORECASE,
        )
        if match is not None:
            time_window = [match.group(1), match.group(2)]
    if not isinstance(time_window, list) or len(time_window) != 2:
        time_window = [0.0, 0.0]
    mechanism = str(payload.get("failure_mechanism", "") or "")
    if not mechanism and raw:
        match = re.search(
            r"failure\s+mechanism\s+(?:is|:)\s*([^.\n]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if match is not None:
            mechanism = match.group(1).strip()
    obj = str(payload.get("object", "") or "")
    if not obj and raw:
        match = re.search(r"object\s+(?:is|:)\s*([^.\n]+)", raw, flags=re.IGNORECASE)
        if match is not None:
            obj = match.group(1).strip().strip("'\"")
    edit_intent = str(payload.get("edit_intent", "") or "")
    if not edit_intent and raw:
        match = re.search(r"edit\s+intent\s+(?:is|:)\s*([^.\n]+)", raw, flags=re.IGNORECASE)
        if match is not None:
            edit_intent = match.group(1).strip()
    return {
        "qwen_diagnosis_raw": raw,
        "qwen_diagnosis_parse_ok": bool(payload) or component not in {"", "unknown"},
        "qwen_diagnosis_failure_mechanism": mechanism,
        "qwen_diagnosis_wrong_component": component,
        "qwen_diagnosis_time_start": safe_float(time_window[0], 0.0),
        "qwen_diagnosis_time_end": safe_float(time_window[1], 0.0),
        "qwen_diagnosis_object": obj,
        "qwen_diagnosis_edit_intent": edit_intent,
    }


def failure_factor_from_decision(decision):
    direct_factor = str(getattr(decision, "failure_factor", "") or "").strip()
    if direct_factor in FACTOR_ORDER or direct_factor == "none":
        return direct_factor
    payload = parse_reflection_json(getattr(decision, "raw_text", ""))
    factor = str(payload.get("failure_factor", "")).strip()
    if factor in FACTOR_ORDER or factor == "none":
        return factor
    mismatch = str(getattr(decision, "mismatch_type", "")).strip()
    return MISMATCH_TO_FACTOR.get(mismatch, "goal_completion")


def safe_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def memory_row_id(memory_id):
    text = str(memory_id)
    if text.startswith("KEY_"):
        text = text[4:]
    try:
        return int(text)
    except ValueError:
        return None


class RiskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CounterfactualVerifierMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_memory_factor_by_id(key_memory_rows):
    out = {}
    for idx, row in enumerate(key_memory_rows or []):
        row_id = int(row.get("row_id", idx))
        mismatch = str(row.get("mismatch_type", "") or "")
        out[row_id] = MISMATCH_TO_FACTOR.get(mismatch, "unknown")
    return out


def factor_vote_stats(memory_ids, memory_scores, memory_factor_by_id):
    votes = {factor: 0.0 for factor in RISK_FACTOR_ORDER}
    for memory_id, score in zip(memory_ids or [], memory_scores or []):
        row_id = memory_row_id(memory_id)
        if row_id is None:
            continue
        factor = memory_factor_by_id.get(row_id, "unknown")
        votes[factor] = votes.get(factor, 0.0) + max(safe_float(score), 0.0)
    total = sum(votes.values())
    dist = [votes.get(factor, 0.0) / max(total, 1e-8) for factor in RISK_FACTOR_ORDER]
    entropy = -sum(p * np.log(max(p, 1e-8)) for p in dist)
    confidence = max(dist) if dist else 0.0
    return dist, float(entropy), float(confidence)


def memory_factor_vote_from_rows(rows):
    votes = Counter()
    for row in rows or []:
        mismatch = str(row.get("mismatch_type", "") or "")
        factor = MISMATCH_TO_FACTOR.get(mismatch, "unknown")
        if factor not in {"", "none", "unknown"}:
            votes[factor] += 1
    if not votes:
        return "unknown", 0
    factor, count = votes.most_common(1)[0]
    return factor, int(count)


def parse_csv_set(value):
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def joint_editor_trigger_decision(cfg, task, key_rows, reflection_row, memory_factor_by_id, device):
    if not bool(getattr(cfg, "joint_editor_risk_gate", False)):
        return {
            "joint_editor_triggered": True,
            "joint_editor_trigger_reason": "risk_gate_disabled",
            "joint_editor_force_non_none": False,
            "joint_editor_fallback_factor": "unknown",
            "joint_editor_memory_vote_factor": "unknown",
            "joint_editor_memory_vote_count": 0,
            "joint_editor_risk_probability": None,
            "joint_editor_risk_threshold": None,
        }
    trigger_probability = None
    trigger_model_info = {}
    base_future_for_trigger = reflection_row.get("_base_future_for_trigger")
    if getattr(cfg, "joint_editor_trigger_ckpt", "") and base_future_for_trigger is not None:
        trigger_probability, trigger_model_info = predict_joint_editor_trigger(
            cfg,
            reflection_row,
            base_future_for_trigger,
            task,
        )
        threshold = float(getattr(cfg, "joint_editor_trigger_threshold", 0.5))
        memory_vote_factor, memory_vote_count = memory_factor_vote_from_rows(key_rows)
        failure_tasks = parse_csv_set(getattr(cfg, "joint_editor_failure_tasks", ""))
        task_prior_factor = infer_diagnosis_factor_from_task(task)
        reasons = []
        if trigger_probability is not None and trigger_probability >= threshold:
            reasons.append("learned_trigger")
        if int(memory_vote_count) >= int(getattr(cfg, "joint_editor_min_failure_memory_votes", 1)):
            reasons.append("failure_memory")
        if task in failure_tasks:
            reasons.append("failure_task_prior")
        triggered = bool(reasons)
        fallback_factor = memory_vote_factor if memory_vote_factor not in {"", "none", "unknown"} else task_prior_factor
        return {
            "joint_editor_triggered": bool(triggered),
            "joint_editor_trigger_reason": "+".join(reasons) if reasons else "learned_trigger_low",
            "joint_editor_force_non_none": bool(triggered and getattr(cfg, "joint_editor_force_non_none", False)),
            "joint_editor_fallback_factor": fallback_factor,
            "joint_editor_task_prior_factor": task_prior_factor,
            "joint_editor_memory_vote_factor": memory_vote_factor,
            "joint_editor_memory_vote_count": int(memory_vote_count),
            "joint_editor_risk_probability": trigger_probability,
            "joint_editor_risk_threshold": threshold,
            **trigger_model_info,
        }
    threshold = float(getattr(cfg, "joint_editor_risk_threshold", getattr(cfg, "risk_detector_threshold", 0.5)))
    risk_probability = predict_risk_probability(cfg, reflection_row, memory_factor_by_id, device)
    memory_vote_factor, memory_vote_count = memory_factor_vote_from_rows(key_rows)
    failure_tasks = parse_csv_set(getattr(cfg, "joint_editor_failure_tasks", ""))
    task_prior_factor = infer_diagnosis_factor_from_task(task)
    reasons = []
    if risk_probability is not None and risk_probability >= threshold:
        reasons.append(f"risk_prob>={threshold:g}")
    if int(memory_vote_count) >= int(getattr(cfg, "joint_editor_min_failure_memory_votes", 1)):
        reasons.append("failure_memory")
    if task in failure_tasks:
        reasons.append("failure_task_prior")
    triggered = bool(reasons)
    fallback_factor = memory_vote_factor if memory_vote_factor not in {"", "none", "unknown"} else task_prior_factor
    return {
        "joint_editor_triggered": bool(triggered),
        "joint_editor_trigger_reason": "+".join(reasons) if reasons else "low_risk",
        "joint_editor_force_non_none": bool(triggered and getattr(cfg, "joint_editor_force_non_none", False)),
        "joint_editor_fallback_factor": fallback_factor,
        "joint_editor_task_prior_factor": task_prior_factor,
        "joint_editor_memory_vote_factor": memory_vote_factor,
        "joint_editor_memory_vote_count": int(memory_vote_count),
        "joint_editor_risk_probability": risk_probability,
        "joint_editor_risk_threshold": threshold,
    }


def load_risk_detector(cfg, device):
    path = str(getattr(cfg, "risk_detector_ckpt", "") or "")
    if not path:
        return None
    cache = _RISK_DETECTOR_CACHE.get(path)
    if cache is not None:
        return cache
    checkpoint = torch.load(path, map_location="cpu")
    feature_names = list(checkpoint["feature_names"])
    model = RiskMLP(len(feature_names), int(checkpoint.get("hidden_dim", 128))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    bundle = {
        "path": path,
        "model": model,
        "feature_names": feature_names,
        "feature_mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(-1),
        "feature_std": np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(-1),
    }
    _RISK_DETECTOR_CACHE[path] = bundle
    return bundle


def build_online_risk_features(row, feature_names, memory_factor_by_id):
    memory_ids = row.get("adapter_memory_ids") or []
    memory_scores = row.get("adapter_memory_scores") or []
    vote_dist, vote_entropy, vote_confidence = factor_vote_stats(memory_ids, memory_scores, memory_factor_by_id)
    factor = str(row.get("failure_factor", "") or row.get("selected_failure_factor", "") or "unknown")
    if factor not in RISK_FACTOR_ORDER:
        factor = "unknown"
    values = {
        "subtask_index": safe_float(row.get("subtask_index"), -1.0),
        "counterfactual_gate": safe_float(row.get("counterfactual_gate")),
        "counterfactual_memory_support": safe_float(row.get("counterfactual_memory_support")),
        "counterfactual_alignment": safe_float(row.get("counterfactual_alignment")),
        "counterfactual_residual_norm_log": np.log1p(max(safe_float(row.get("counterfactual_residual_norm")), 0.0)),
        "counterfactual_shift_norm_log": np.log1p(max(safe_float(row.get("counterfactual_shift_norm")), 0.0)),
        "base_future_norm_log": np.log1p(max(safe_float(row.get("base_future_norm")), 0.0)),
        "proposal_future_norm_log": np.log1p(max(safe_float(row.get("proposal_future_norm")), 0.0)),
        "factor_vote_entropy": vote_entropy,
        "factor_vote_confidence": vote_confidence,
    }
    for idx, name in enumerate(RISK_FACTOR_ORDER):
        values[f"factor_vote_{name}"] = vote_dist[idx]
        values[f"qwen_factor_{name}"] = 1.0 if factor == name else 0.0
    task = str(row.get("task", ""))
    for name in feature_names:
        if name.startswith("task_"):
            values[name] = 1.0 if name == f"task_{task}" else 0.0
    return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=np.float32)


def predict_risk_probability(cfg, row, memory_factor_by_id, device):
    bundle = load_risk_detector(cfg, device)
    if bundle is None:
        return None
    features = build_online_risk_features(row, bundle["feature_names"], memory_factor_by_id)
    features = (features - bundle["feature_mean"]) / np.maximum(bundle["feature_std"], 1e-6)
    tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(bundle["model"](tensor)).item()
    return float(prob)


def load_counterfactual_verifier(cfg, device):
    path = str(getattr(cfg, "counterfactual_verifier_ckpt", "") or "")
    if not path:
        return None
    cache = _COUNTERFACTUAL_VERIFIER_CACHE.get(path)
    if cache is not None:
        return cache
    checkpoint = torch.load(path, map_location="cpu")
    feature_names = list(checkpoint["feature_names"])
    model = CounterfactualVerifierMLP(len(feature_names), int(checkpoint.get("hidden_dim", 128))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    bundle = {
        "path": path,
        "model": model,
        "feature_names": feature_names,
        "feature_mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(-1),
        "feature_std": np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(-1),
        "tasks": list(checkpoint.get("tasks", [])),
    }
    _COUNTERFACTUAL_VERIFIER_CACHE[path] = bundle
    return bundle


def maybe_load_policy_action_intent_lora(model, cfg):
    path = str(getattr(cfg, "policy_action_intent_lora_ckpt", "") or "")
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("lora_state", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"policy_action_intent_lora_ckpt at {path} does not contain a state dict")
    model_state = model.state_dict()
    matched = {k: v for k, v in state.items() if k in model_state}
    if not matched:
        raise ValueError(f"policy_action_intent_lora_ckpt at {path} did not match any model parameters")
    missing = [k for k in state.keys() if k not in model_state]
    model.load_state_dict(matched, strict=False)
    print(
        f"[INFO] Loaded policy action-intent LoRA: {path} | matched={len(matched)}"
        + (f" | skipped={len(missing)}" if missing else "")
    )


def tensor_cosine(a, b):
    av = a.reshape(-1).float()
    bv = b.reshape(-1).float()
    denom = torch.norm(av, p=2) * torch.norm(bv, p=2)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return 0.0
    return float((torch.dot(av, bv) / denom).detach().cpu().item())


def tensor_norm(a):
    return float(torch.norm(a.reshape(-1).float(), p=2).detach().cpu().item())


def build_counterfactual_verifier_features(row, base_future, proposal_future, gated_future, final_future, target_future, feature_names):
    base_cos = tensor_cosine(base_future, target_future)
    proposal_cos = tensor_cosine(proposal_future, target_future)
    gated_cos = tensor_cosine(gated_future, target_future)
    final_cos = tensor_cosine(final_future, target_future)
    factor = str(row.get("failure_factor") or row.get("selected_failure_factor") or row.get("applied_factor_mask") or "unknown")
    if factor not in FACTOR_ORDER:
        factor = "unknown"
    task = str(row.get("task", ""))
    values = {
        "subtask_index": safe_float(row.get("subtask_index"), -1.0),
        "base_to_target_cos": base_cos,
        "proposal_to_target_cos": proposal_cos,
        "gated_to_target_cos": gated_cos,
        "final_to_target_cos": final_cos,
        "proposal_cos_gain": proposal_cos - base_cos,
        "gated_cos_gain": gated_cos - base_cos,
        "final_cos_gain": final_cos - base_cos,
        "proposal_to_target_delta_cos": tensor_cosine(proposal_future - base_future, target_future - base_future),
        "gated_to_target_delta_cos": tensor_cosine(gated_future - base_future, target_future - base_future),
        "final_to_target_delta_cos": tensor_cosine(final_future - base_future, target_future - base_future),
        "proposal_delta_norm_log": np.log1p(max(tensor_norm(proposal_future - base_future), 0.0)),
        "gated_delta_norm_log": np.log1p(max(tensor_norm(gated_future - base_future), 0.0)),
        "final_delta_norm_log": np.log1p(max(tensor_norm(final_future - base_future), 0.0)),
        "target_delta_norm_log": np.log1p(max(tensor_norm(target_future - base_future), 0.0)),
        "counterfactual_gate": safe_float(row.get("counterfactual_gate")),
        "counterfactual_memory_support": safe_float(row.get("counterfactual_memory_support")),
        "counterfactual_alignment": safe_float(row.get("counterfactual_alignment")),
        "counterfactual_residual_norm_log": np.log1p(max(safe_float(row.get("counterfactual_residual_norm")), 0.0)),
        "counterfactual_shift_norm_log": np.log1p(max(safe_float(row.get("counterfactual_shift_norm")), 0.0)),
        "adapter_delta_norm_log": np.log1p(max(safe_float(row.get("adapter_delta_norm")), 0.0)),
        "adapter_gate_mean": safe_float(row.get("adapter_gate_mean")),
        "adapter_weight_mean": safe_float(row.get("adapter_weight_mean")),
        "memory_score_mean": float(np.mean([safe_float(x) for x in (row.get("adapter_memory_scores") or [])])) if row.get("adapter_memory_scores") else 0.0,
        "memory_score_max": float(np.max([safe_float(x) for x in (row.get("adapter_memory_scores") or [])])) if row.get("adapter_memory_scores") else 0.0,
        "factor_mask_applied": 1.0 if bool(row.get("factor_mask_applied", False)) else 0.0,
    }
    for name in FACTOR_ORDER:
        values[f"factor_{name}"] = 1.0 if factor == name else 0.0
    for name in feature_names:
        if name.startswith("task_"):
            values[name] = 1.0 if name == f"task_{task}" else 0.0
    return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=np.float32), values


def predict_counterfactual_verifier_probability(cfg, row, base_future, proposal_future, gated_future, final_future, target_future, device):
    bundle = load_counterfactual_verifier(cfg, device)
    if bundle is None or target_future is None:
        return None, {}
    features, debug_values = build_counterfactual_verifier_features(
        row,
        base_future,
        proposal_future,
        gated_future,
        final_future,
        target_future,
        bundle["feature_names"],
    )
    features = (features - bundle["feature_mean"]) / np.maximum(bundle["feature_std"], 1e-6)
    tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = float(bundle["model"](tensor).item())
        temperature = max(float(getattr(cfg, "counterfactual_verifier_temperature", 1.0)), 1e-6)
        prob = float(torch.sigmoid(torch.tensor(logit / temperature)).item())
        min_prob = float(getattr(cfg, "counterfactual_verifier_min_prob", 0.0))
        max_prob = float(getattr(cfg, "counterfactual_verifier_max_prob", 1.0))
        prob = float(np.clip(prob, min_prob, max_prob))
    debug_values["counterfactual_verifier_logit"] = logit
    debug_values["counterfactual_verifier_temperature"] = temperature
    return float(prob), debug_values


def load_factor_masks(cfg):
    path = str(getattr(cfg, "factor_mask_npz", "") or "")
    if not path:
        return None
    cache = _FACTOR_MASK_CACHE.get(path)
    if cache is not None and cache.get("path") == path:
        return cache
    arrays = np.load(path, allow_pickle=True)
    masks = {
        key.replace("mask_", ""): arrays[key].astype(np.float32)
        for key in arrays.files
        if key.startswith("mask_")
    }
    cache = {"path": path, "masks": masks}
    _FACTOR_MASK_CACHE[path] = cache
    return cache


def choose_factor_mask(cfg, factor):
    mode = str(getattr(cfg, "factor_mask_mode", "global") or "global")
    if mode == "global":
        return None, factor, "global"
    bundle = load_factor_masks(cfg)
    masks = {} if bundle is None else bundle["masks"]
    valid = [name for name in FACTOR_ORDER if name in masks]
    if not valid:
        return None, factor, "missing"
    selected = factor if factor in masks else "goal_completion"
    if mode == "wrong":
        selected = valid[(valid.index(selected) + 1) % len(valid)] if selected in valid else valid[0]
    elif mode == "random":
        base = masks[selected]
        rng = np.random.default_rng(int(getattr(cfg, "factor_mask_random_seed", 0)))
        density = float((base > 0).mean())
        random_mask = (rng.random(base.shape) < density).astype(np.float32)
        if bool(getattr(cfg, "factor_mask_soft_random", True)) and float(base.max()) > 0:
            random_mask *= float(base[base > 0].mean())
        return random_mask, factor, "random"
    elif mode != "correct":
        return None, factor, f"unknown_mode:{mode}"
    return masks[selected], factor, selected


def load_program_conditioned_mask_head(cfg, device):
    path = str(getattr(cfg, "program_conditioned_mask_ckpt", "") or "")
    if not path:
        return None
    cached = _PROGRAM_CONDITIONED_MASK_CACHE.get(path)
    if cached is not None:
        return cached
    checkpoint = torch.load(path, map_location="cpu")
    factors = [str(x) for x in checkpoint.get("factors", FACTOR_ORDER)]
    model = ProgramConditionedMaskHead(
        token_dim=int(checkpoint["token_dim"]),
        channel_dim=int(checkpoint["channel_dim"]),
        num_factors=len(factors),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    bundle = {
        "path": path,
        "model": model,
        "factors": factors,
        "factor_to_idx": {factor: idx for idx, factor in enumerate(factors)},
    }
    _PROGRAM_CONDITIONED_MASK_CACHE[path] = bundle
    print(f"[INFO] Loaded program-conditioned mask head: {path}", flush=True)
    return bundle


def program_time_window_for_factor(factor):
    start, end = PROGRAM_MASK_TIME_WINDOWS.get(str(factor), (0.0, 1.0))
    return float(start), float(end)


def compute_program_conditioned_mask(cfg, base_future, factor):
    bundle = load_program_conditioned_mask_head(cfg, base_future.device)
    if bundle is None:
        return None, "missing_program_mask"
    factor = str(factor)
    if factor not in bundle["factor_to_idx"]:
        return None, f"unsupported_factor:{factor}"
    start, end = program_time_window_for_factor(factor)
    with torch.no_grad():
        mask = bundle["model"](
            base_future.detach().float().unsqueeze(0),
            torch.tensor([bundle["factor_to_idx"][factor]], device=base_future.device, dtype=torch.long),
            torch.tensor([[start, end]], device=base_future.device, dtype=torch.float32),
        ).squeeze(0)
    density = float(getattr(cfg, "program_conditioned_mask_density", 0.01))
    density = float(np.clip(density, 1e-6, 1.0))
    flat = mask.reshape(-1)
    k = max(1, int(round(flat.numel() * density)))
    threshold = torch.topk(flat, k).values[-1]
    hard = (mask >= threshold).to(dtype=base_future.dtype)
    return hard, "program_conditioned"


def load_unified_future_encoder_from_ckpt(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = UnifiedFutureEncoder(
        future_dim=int(ckpt["future_dim"]),
        task_dim=int(ckpt["task_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_factors=len(ckpt["factors"]),
        encoder_type=str(ckpt.get("encoder_type", "pooled")),
        token_dim=int(ckpt.get("token_dim", 256)),
        num_layers=int(ckpt.get("num_layers", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_joint_future_editor(cfg, device):
    path = str(getattr(cfg, "joint_future_editor_ckpt", "") or "")
    if not path:
        return None
    cached = _JOINT_FUTURE_EDITOR_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    encoder, enc_ckpt = load_unified_future_encoder_from_ckpt(ckpt["future_encoder_ckpt"], device)
    mask_bundle = {"path": ckpt["mask_head_ckpt"]}
    mask_ckpt = torch.load(ckpt["mask_head_ckpt"], map_location="cpu")
    mask_model = ProgramConditionedMaskHead(
        token_dim=int(mask_ckpt["token_dim"]),
        channel_dim=int(mask_ckpt["channel_dim"]),
        num_factors=len(mask_ckpt["factors"]),
        hidden_dim=int(mask_ckpt.get("hidden_dim", 256)),
    ).to(device)
    mask_model.load_state_dict(mask_ckpt["model_state"])
    mask_model.eval()
    factors = [str(x) for x in ckpt["factors"]]
    editor = JointFutureEditor(
        future_dim=int(ckpt["future_dim"]),
        z_dim=int(ckpt["z_dim"]),
        task_dim=int(ckpt["task_dim"]),
        num_factors=len(factors),
        hidden_dim=int(ckpt["hidden_dim"]),
        token_dim=int(ckpt.get("editor_token_dim", 256)),
    ).to(device)
    editor.load_state_dict(ckpt["model_state"])
    editor.eval()
    bundle = {
        "path": path,
        "encoder": encoder,
        "encoder_ckpt": enc_ckpt,
        "mask_head": mask_model,
        "mask_ckpt": mask_ckpt,
        "editor": editor,
        "factors": factors,
        "factor_to_idx": {factor: idx for idx, factor in enumerate(factors)},
        "gate_scale": float(getattr(cfg, "joint_future_editor_gate_scale", ckpt.get("gate_scale", 0.05))),
    }
    _JOINT_FUTURE_EDITOR_CACHE[path] = bundle
    print(f"[INFO] Loaded joint future editor: {path}", flush=True)
    return bundle


def apply_joint_future_editor(cfg, base_future, task, subtask_index, factor):
    bundle = load_joint_future_editor(cfg, base_future.device)
    if bundle is None:
        return base_future, {"joint_future_editor_used": False, "joint_future_editor_reason": "missing"}
    factor = str(factor)
    if factor not in bundle["factor_to_idx"] or factor in {"", "none", "unknown"}:
        return base_future, {"joint_future_editor_used": False, "joint_future_editor_reason": f"unsupported_factor:{factor}"}
    task_vec = torch.from_numpy(task_hash(str(task), int(bundle["encoder_ckpt"].get("task_dim", 128)))).to(base_future.device, dtype=torch.float32).unsqueeze(0)
    sub = torch.tensor([float(subtask_index) / 5.0], device=base_future.device, dtype=torch.float32)
    factor_t = torch.tensor([bundle["factor_to_idx"][factor]], device=base_future.device, dtype=torch.long)
    start, end = program_time_window_for_factor(factor)
    window = torch.tensor([[start, end]], device=base_future.device, dtype=torch.float32)
    with torch.no_grad():
        base_b = base_future.unsqueeze(0).float()
        z = bundle["encoder"](base_b, task_vec, sub)["z"]
        mask = bundle["mask_head"](base_b, factor_t, window)
        if bool(getattr(cfg, "joint_future_editor_hard_mask", True)):
            density = float(getattr(cfg, "joint_future_editor_mask_density", 0.02))
            flat = mask.reshape(-1)
            k = max(1, int(round(flat.numel() * float(np.clip(density, 1e-6, 1.0)))))
            threshold = torch.topk(flat, k).values[-1]
            mask = (mask >= threshold).to(mask.dtype)
        delta, gate = bundle["editor"](base_b, z, task_vec, factor_t, window)
        if bool(getattr(cfg, "joint_future_editor_fixed_gate", True)):
            gate = torch.full_like(gate, float(getattr(cfg, "joint_future_editor_fixed_gate_value", 1.0)))
        edit = bundle["gate_scale"] * gate * mask * delta
        final = (base_b + edit).squeeze(0).to(dtype=base_future.dtype)
    return final, {
        "joint_future_editor_used": True,
        "joint_future_editor_ckpt": bundle["path"],
        "joint_future_editor_factor": factor,
        "joint_future_editor_gate_scale": float(bundle["gate_scale"]),
        "joint_future_editor_gate_mean": float(gate.mean().item()),
        "joint_future_editor_mask_mean": float(mask.mean().item()),
        "joint_future_editor_edit_norm": float(torch.norm((final - base_future).reshape(-1), p=2).item()),
        "joint_future_editor_delta_norm": float(torch.norm(delta.reshape(-1), p=2).item()),
    }


def load_joint_editor_trigger(cfg, device):
    path = str(getattr(cfg, "joint_editor_trigger_ckpt", "") or "")
    if not path:
        return None
    cached = _JOINT_EDITOR_TRIGGER_CACHE.get(path)
    if cached is not None:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    model = JointEditorTriggerMLP(
        int(ckpt["input_dim"]),
        int(ckpt.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bundle = {
        "path": path,
        "model": model,
        "feature_mean": np.asarray(ckpt["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(ckpt["feature_std"], dtype=np.float32),
        "task_to_idx": dict(ckpt.get("task_to_idx", {})),
        "future_key": str(ckpt.get("future_key", "base_future")),
        "online_only_features": bool(ckpt.get("online_only_features", False)),
    }
    _JOINT_EDITOR_TRIGGER_CACHE[path] = bundle
    print(f"[INFO] Loaded joint editor trigger: {path}", flush=True)
    return bundle


def predict_joint_editor_trigger(cfg, row, base_future, task):
    bundle = load_joint_editor_trigger(cfg, base_future.device)
    if bundle is None:
        return None, {
            "joint_editor_trigger_model_used": False,
            "joint_editor_trigger_model_reason": "missing_ckpt",
        }
    temp_dir = Path(str(getattr(cfg, "future_trace_dir", "") or "/tmp")) / "_trigger_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"trigger_{os.getpid()}_{time.time_ns()}.npz"
    np.savez_compressed(temp_path, base_future=base_future.detach().cpu().numpy().astype(np.float32))
    try:
        stats = joint_editor_trigger_future_stats(temp_path, bundle["future_key"])
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
    probe_row = dict(row)
    probe_row["task"] = task
    probe_row.setdefault("failure_factor", row.get("failure_factor", "unknown"))
    feature = build_joint_editor_trigger_feature(
        probe_row,
        stats,
        bundle["task_to_idx"],
        online_only=bool(bundle.get("online_only_features", False)),
    )
    feature = (feature - bundle["feature_mean"]) / np.maximum(bundle["feature_std"], 1e-6)
    x = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0).to(base_future.device)
    with torch.no_grad():
        probability = float(torch.sigmoid(bundle["model"](x)).item())
    threshold = float(getattr(cfg, "joint_editor_trigger_threshold", 0.5))
    return probability, {
        "joint_editor_trigger_model_used": True,
        "joint_editor_trigger_model_ckpt": str(bundle["path"]),
        "joint_editor_trigger_probability": probability,
        "joint_editor_trigger_threshold": threshold,
    }


def apply_factor_mask_to_future(cfg, base_future, corrected_future, factor):
    mode = str(getattr(cfg, "factor_mask_mode", "global") or "global")
    info = {
        "factor_mask_mode": mode,
        "selected_failure_factor": str(factor),
        "applied_factor_mask": "program" if mode == "program" else "global",
        "factor_mask_applied": False,
    }
    if mode == "program":
        mask, mask_name = compute_program_conditioned_mask(cfg, base_future, factor)
        info.update(
            {
                "applied_factor_mask": str(mask_name),
                "program_conditioned_mask_ckpt": str(getattr(cfg, "program_conditioned_mask_ckpt", "") or ""),
            }
        )
        if mask is None:
            return corrected_future, info
        if tuple(mask.shape[-2:]) != tuple(corrected_future.shape[-2:]):
            info["factor_mask_skip_reason"] = f"shape_mismatch:{tuple(mask.shape)} vs {tuple(corrected_future.shape)}"
            return corrected_future, info
        masked_future = base_future + (corrected_future - base_future) * mask
        info.update(
            {
                "factor_mask_applied": True,
                "factor_mask_density": float((mask > 0).float().mean().item()),
                "factor_mask_mean": float(mask.float().mean().item()),
                "factor_masked_shift_norm": float(torch.norm((masked_future - base_future).reshape(-1), p=2).item()),
            }
        )
        return masked_future, info
    mask_np, original_factor, mask_name = choose_factor_mask(cfg, factor)
    info.update(
        {
            "selected_failure_factor": str(original_factor),
            "applied_factor_mask": str(mask_name),
        }
    )
    if mask_np is None:
        return corrected_future, info
    mask = torch.from_numpy(mask_np).to(device=corrected_future.device, dtype=corrected_future.dtype)
    while mask.ndim < corrected_future.ndim:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape[-2:]) != tuple(corrected_future.shape[-2:]):
        info["factor_mask_skip_reason"] = f"shape_mismatch:{tuple(mask.shape)} vs {tuple(corrected_future.shape)}"
        return corrected_future, info
    masked_future = base_future + (corrected_future - base_future) * mask
    info.update(
        {
            "factor_mask_applied": True,
            "factor_mask_density": float((mask_np > 0).mean()),
            "factor_mask_mean": float(mask_np.mean()),
            "factor_masked_shift_norm": float(torch.norm((masked_future - base_future).reshape(-1), p=2).item()),
        }
    )
    return masked_future, info


def apply_immune_factor_mask(cfg, original_future, edited_future, task):
    """Restrict an immune edit to a task-inferred factor mask."""
    factor_npz = str(getattr(cfg, "immune_factor_mask_npz", "") or "")
    if not factor_npz:
        return edited_future, {"immune_factor_mask_applied": False, "immune_factor_mask_reason": "disabled"}
    old_npz = getattr(cfg, "factor_mask_npz", "")
    old_mode = getattr(cfg, "factor_mask_mode", "global")
    try:
        cfg.factor_mask_npz = factor_npz
        cfg.factor_mask_mode = str(getattr(cfg, "immune_factor_mask_mode", "correct") or "correct")
        factor = infer_diagnosis_factor_from_task(task)
        masked, info = apply_factor_mask_to_future(cfg, original_future, edited_future, factor)
    finally:
        cfg.factor_mask_npz = old_npz
        cfg.factor_mask_mode = old_mode
    return masked, {
        "immune_factor_mask_applied": bool(info.get("factor_mask_applied", False)),
        "immune_factor_mask_npz": factor_npz,
        "immune_factor_mask_mode": str(getattr(cfg, "immune_factor_mask_mode", "correct") or "correct"),
        "immune_factor_mask_factor": str(info.get("selected_failure_factor", infer_diagnosis_factor_from_task(task))),
        "immune_factor_mask_name": str(info.get("applied_factor_mask", "")),
        "immune_factor_mask_density": info.get("factor_mask_density"),
        "immune_factor_mask_shift_norm": info.get("factor_masked_shift_norm"),
        "immune_factor_mask_skip_reason": info.get("factor_mask_skip_reason"),
    }


def should_apply_risk_gated_intervention(cfg, decision, factor, gate_trace):
    if not bool(getattr(cfg, "risk_gated_intervention", False)):
        return True, {
            "risk_gated_intervention": False,
            "risk_gate_allowed": True,
            "risk_gate_reason": "disabled",
        }
    factor = str(factor or "")
    payload = parse_reflection_json(getattr(decision, "raw_text", ""))
    gamma = payload.get("gamma", None)
    try:
        gamma_value = float(gamma)
    except Exception:
        gamma_value = float(gate_trace.get("gate", 0.0))
    trust = str(getattr(decision, "trust", "") or payload.get("trust", "") or "").lower()
    recoverability = str(getattr(decision, "recoverability", "") or payload.get("recoverability", "") or "").lower()
    memory_support = float(gate_trace.get("memory_support", 0.0))
    gate = float(gate_trace.get("gate", 0.0))
    min_gamma = float(getattr(cfg, "risk_gate_min_gamma", 0.08))
    min_memory_support = float(getattr(cfg, "risk_gate_min_memory_support", 0.75))
    min_gate = float(getattr(cfg, "risk_gate_min_gate", 0.08))

    reasons = []
    if factor in {"", "none"}:
        reasons.append("factor_none")
    if gamma_value < min_gamma:
        reasons.append(f"gamma<{min_gamma:g}")
    if memory_support < min_memory_support:
        reasons.append(f"memory_support<{min_memory_support:g}")
    if gate < min_gate:
        reasons.append(f"gate<{min_gate:g}")
    if trust == "high" and recoverability in {"not_needed", "none"}:
        reasons.append("success_preserve")
    allowed = len(reasons) == 0
    return allowed, {
        "risk_gated_intervention": True,
        "risk_gate_allowed": bool(allowed),
        "risk_gate_reason": "allow" if allowed else ";".join(reasons),
        "risk_gate_factor": factor,
        "risk_gate_gamma": float(gamma_value),
        "risk_gate_memory_support": float(memory_support),
        "risk_gate_raw_gate": float(gate),
        "risk_gate_trust": trust,
        "risk_gate_recoverability": recoverability,
        "risk_gate_min_gamma": min_gamma,
        "risk_gate_min_memory_support": min_memory_support,
        "risk_gate_min_gate": min_gate,
    }


def cap_counterfactual_gate(cfg, base_future, corrected_future, gate_trace):
    if not bool(getattr(cfg, "risk_gated_intervention", False)):
        return corrected_future, gate_trace, {"risk_gate_gate_capped": False}
    max_gate = float(getattr(cfg, "risk_gate_max_gate", 0.12))
    gate = float(gate_trace.get("gate", 0.0))
    if gate <= max_gate or gate <= 1e-8:
        return corrected_future, gate_trace, {"risk_gate_gate_capped": False, "risk_gate_effective_gate": gate}
    scale = max_gate / gate
    capped_future = base_future + (corrected_future - base_future) * scale
    new_trace = dict(gate_trace)
    new_trace["gate"] = max_gate
    new_trace["corrected_shift_norm"] = float(torch.norm((capped_future - base_future).reshape(-1), p=2).item())
    return capped_future, new_trace, {
        "risk_gate_gate_capped": True,
        "risk_gate_original_gate": gate,
        "risk_gate_effective_gate": max_gate,
        "risk_gate_cap_scale": float(scale),
    }


def topk_future_mask(mask_np, density, reference_future):
    if mask_np is None:
        return None
    mask = np.asarray(mask_np, dtype=np.float32)
    if mask.shape != tuple(reference_future.shape[-2:]):
        return None
    density = float(np.clip(float(density), 1e-6, 1.0))
    k = max(1, int(round(mask.size * density)))
    flat = mask.reshape(-1)
    if float(flat.max()) <= 0:
        return None
    threshold = np.partition(flat, -k)[-k]
    hard = (mask >= threshold).astype(np.float32)
    return torch.from_numpy(hard).to(device=reference_future.device, dtype=reference_future.dtype)


def select_semantic_future_beam(cfg, base_future, proposal_future, gated_future, memory_target_future, factor):
    if not bool(getattr(cfg, "semantic_future_beam", False)) or memory_target_future is None:
        return gated_future, {
            "semantic_future_beam": bool(getattr(cfg, "semantic_future_beam", False)),
            "semantic_beam_selected": "disabled_or_no_memory_target",
        }
    density = float(getattr(cfg, "semantic_beam_patch_density", 0.01))
    margin = float(getattr(cfg, "semantic_beam_min_margin", 1e-4))
    beams = {
        "base": base_future,
        "proposal": proposal_future,
        "gated": gated_future,
    }

    bundle = load_factor_masks(cfg)
    masks = {} if bundle is None else bundle.get("masks", {})
    for name, mask_name in (("memory_patch_global", "global"), ("memory_patch_factor", factor if factor in masks else "goal_completion")):
        mask = topk_future_mask(masks.get(mask_name), density, base_future)
        if mask is not None:
            while mask.ndim < base_future.ndim:
                mask = mask.unsqueeze(0)
            beams[name] = base_future + mask * (memory_target_future - base_future)

    scores = {name: tensor_cosine(future, memory_target_future) for name, future in beams.items()}
    base_score = scores["base"]
    selected = max(scores, key=scores.get)
    selected_gain = float(scores[selected] - base_score)
    if selected == "base" or selected_gain < margin:
        selected = "base"
        selected_future = base_future
    else:
        selected_future = beams[selected]
    return selected_future, {
        "semantic_future_beam": True,
        "semantic_beam_selected": selected,
        "semantic_beam_base_score": float(base_score),
        "semantic_beam_selected_score": float(scores[selected]),
        "semantic_beam_selected_gain": float(scores[selected] - base_score),
        "semantic_beam_scores": {key: float(value) for key, value in scores.items()},
        "semantic_beam_patch_density": density,
        "semantic_beam_min_margin": margin,
    }


def apply_safe_intervention_selector(cfg, adapter_info, base_future, proposal_future, corrected_future, target_future):
    if not bool(getattr(cfg, "safe_intervention_selector", False)):
        return corrected_future, {
            "safe_intervention_selector": False,
            "safe_selector_allowed": True,
            "safe_selector_reason": "disabled",
        }
    max_delta = float(getattr(cfg, "safe_selector_max_adapter_delta_norm", 300.0))
    max_gate = float(getattr(cfg, "safe_selector_max_adapter_gate_mean", 0.08))
    min_memory = float(getattr(cfg, "safe_selector_min_memory_support", 0.75))
    min_gain = float(getattr(cfg, "safe_selector_min_proposal_cos_gain", 0.0))

    adapter_delta = safe_float(adapter_info.get("adapter_delta_norm"))
    adapter_gate = safe_float(adapter_info.get("adapter_gate_mean"))
    memory_support = safe_float(adapter_info.get("counterfactual_memory_support"))
    if target_future is not None:
        proposal_gain = tensor_cosine(proposal_future, target_future) - tensor_cosine(base_future, target_future)
    else:
        proposal_gain = safe_float(adapter_info.get("proposal_cos_gain"), 0.0)

    reasons = []
    if adapter_delta >= max_delta:
        reasons.append(f"adapter_delta_norm>={max_delta:g}")
    if adapter_gate >= max_gate:
        reasons.append(f"adapter_gate_mean>={max_gate:g}")
    if memory_support <= min_memory:
        reasons.append(f"memory_support<={min_memory:g}")
    if proposal_gain <= min_gain:
        reasons.append(f"proposal_cos_gain<={min_gain:g}")

    allowed = not reasons
    return (corrected_future if allowed else base_future), {
        "safe_intervention_selector": True,
        "safe_selector_allowed": bool(allowed),
        "safe_selector_reason": "allow" if allowed else ";".join(reasons),
        "safe_selector_adapter_delta_norm": float(adapter_delta),
        "safe_selector_adapter_gate_mean": float(adapter_gate),
        "safe_selector_memory_support": float(memory_support),
        "safe_selector_proposal_cos_gain": float(proposal_gain),
        "safe_selector_max_adapter_delta_norm": max_delta,
        "safe_selector_max_adapter_gate_mean": max_gate,
        "safe_selector_min_memory_support": min_memory,
        "safe_selector_min_proposal_cos_gain": min_gain,
    }


def update_runtime_state_memory(recent_memory, key_state_memory, row, recent_k, key_max):
    recent_memory.append(row)
    del recent_memory[:-recent_k]
    if row.get("success"):
        key_state_memory.append(dict(row, memory_type="key_state"))
        if key_max > 0:
            del key_state_memory[:-key_max]


def rollout_with_lang_text(
    env,
    model,
    task_oracle,
    cfg,
    subtask,
    lang_embeddings,
    val_annotations,
    lang_text,
    override_future_feature=None,
    override_action_intent=None,
    collect_strong_gt=False,
    subtask_index=0,
    retry_mode: bool = False,
):
    old_mode = getattr(model, "future_feature_mode", None)
    old_override = getattr(model, "override_future_feature", None)
    old_action_intent = getattr(model, "override_action_intent", None)
    old_dynamic_coupling_info = getattr(model, "_last_dynamic_coupling_info", None)
    try:
        model._last_dynamic_coupling_info = {}
        model._last_dynamic_coupling_info_persist = {}
        model._last_dynamic_coupling_future_persist = None
        model._last_dynamic_coupling_action_intent_persist = None
        model._last_joint_hypothesis_init_info = {}
        if override_future_feature is not None:
            model.future_feature_mode = "override"
            model.override_future_feature = override_future_feature
        obs = env.get_obs()
        coupling_first_pass = bool(getattr(cfg, "dynamic_coupling_first_pass", False))
        coupling_control_enabled = bool(retry_mode or coupling_first_pass)
        clean_controller = bool(getattr(cfg, "joint_belief_clean_controller", False))
        strong_gt_start_raw = snapshot_env_raw(env) if collect_strong_gt else None
        base_annotation = val_annotations[subtask][0]
        goal = lang_embeddings.get_lang_goal(base_annotation)
        goal["lang_text"] = lang_text

        model.reset()
        retry_score_base_future = None
        if coupling_control_enabled and override_action_intent is not None:
            model.override_action_intent = override_action_intent
        elif (
            coupling_control_enabled
            and override_future_feature is not None
            and bool(joint_pair_memory_free_mode(cfg))
            and bool(getattr(cfg, "dynamic_coupling_online", False))
        ):
            init_action_intent, init_info = generate_initial_joint_action_intent(
                cfg,
                env,
                model,
                obs,
                lang_text,
                subtask,
                int(subtask_index or 0),
                override_future_feature,
            )
            model._last_joint_hypothesis_init_info = dict(init_info)
            if init_action_intent is not None:
                model.override_action_intent = init_action_intent
                model._last_dynamic_coupling_action_intent_persist = init_action_intent.detach().float().cpu()
        # Persist joint-init metadata even if dynamic coupling never gets a chance
        # to run (for example, very short successful subtasks).
        model._last_dynamic_coupling_info_persist = dict(
            getattr(model, "_last_joint_hypothesis_init_info", {}) or {}
        )
        first_pass_base_future = None
        if (
            coupling_first_pass
            and override_future_feature is None
            and bool(getattr(cfg, "dynamic_coupling_online", False))
        ):
            try:
                from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

                first_pass_base_future = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
                finite_info = {}
                first_pass_base_future = _finite_tensor_or_none(
                    first_pass_base_future,
                    name="dynamic_coupling_first_pass_future",
                    info=finite_info,
                )
                model._last_dynamic_coupling_info_persist.update(finite_info)
                if first_pass_base_future is not None and not clean_controller:
                    model.future_feature_mode = "override"
                    model.override_future_feature = first_pass_base_future
                if first_pass_base_future is not None and bool(getattr(cfg, "dynamic_coupling_generate_action_intent", False)):
                    init_action_intent, init_info = generate_initial_joint_action_intent(
                        cfg,
                        env,
                        model,
                        obs,
                        lang_text,
                        subtask,
                        int(subtask_index or 0),
                        first_pass_base_future,
                    )
                    model._last_joint_hypothesis_init_info = dict(init_info)
                    if init_action_intent is not None:
                        # z_a is normally used as the persistent belief/action
                        # latent for T/D. Only write it into the DeFi policy
                        # decoder when explicitly requested.
                        if bool(getattr(cfg, "dynamic_coupling_apply_action_intent_to_policy", False)):
                            model.override_action_intent = init_action_intent
                        model._last_dynamic_coupling_action_intent_persist = init_action_intent.detach().float().cpu()
                model._last_dynamic_coupling_info_persist.update(
                    {
                        "dynamic_coupling_first_pass_target_initialized": first_pass_base_future is not None,
                        "dynamic_coupling_first_pass_target_source": "gfdm_future",
                    }
                )
            except Exception as exc:
                model._last_dynamic_coupling_info_persist.update(
                    {
                        "dynamic_coupling_first_pass_target_initialized": False,
                        "dynamic_coupling_first_pass_target_error": str(exc),
                    }
                )
        if retry_mode and override_future_feature is not None:
            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

            retry_score_base_future = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
            current_state = _joint_pair_state_from_raw(snapshot_env_raw(env))
            initial_action_for_score = getattr(model, "override_action_intent", None)
            initial_score_info = score_joint_hypothesis_candidate(
                cfg,
                base_future=retry_score_base_future,
                target_future=override_future_feature,
                action_chunk=initial_action_for_score,
                task=subtask,
                subtask_index=int(subtask_index or 0),
                current_state=current_state,
                prefix="retry_initial_score",
            )
            model._last_dynamic_coupling_info_persist.update(initial_score_info)
        start_info = env.get_info()
        last_info = start_info
        action_trace = []
        recent_actions = []
        coupling_last_info = {}
        coupling_agg_info = {
            "dynamic_coupling_used_count": 0,
            "dynamic_coupling_applied_count": 0,
            "dynamic_coupling_action_generated_count": 0,
            "dynamic_coupling_delta_action_patch_count": 0,
            "dynamic_coupling_any_used": False,
            "dynamic_coupling_any_applied": False,
            "dynamic_coupling_any_action_generated": False,
            "dynamic_coupling_any_delta_action_patch": False,
            "dynamic_coupling_first_applied_step": None,
            "dynamic_coupling_last_applied_step": None,
            "dynamic_coupling_max_final_shift_norm": 0.0,
            "chunk_effect_used_count": 0,
            "chunk_effect_decision_count": 0,
            "chunk_effect_any_used": False,
            "chunk_effect_last_runtime_type": None,
            "chunk_effect_last_decision": None,
            "chunk_effect_last_residual_norm": None,
            "chunk_effect_min_residual_norm": None,
            "chunk_effect_max_residual_norm": None,
            "chunk_effect_last_transition_cos": None,
            "chunk_effect_min_transition_cos": None,
            "chunk_effect_max_transition_cos": None,
            "chunk_effect_last_success_prob": None,
            "chunk_effect_min_success_prob": None,
            "chunk_effect_max_success_prob": None,
        }
        if override_future_feature is not None:
            coupling_target_future = override_future_feature.detach().clone()
        elif first_pass_base_future is not None:
            coupling_target_future = first_pass_base_future.detach().clone()
        else:
            coupling_target_future = None
        persistent_hypothesis = bool(getattr(cfg, "dynamic_coupling_persistent_hypothesis", False))
        joint_execution_update = bool(getattr(cfg, "joint_execution_update", False))
        hypothesis_future_t = coupling_target_future.detach().clone() if coupling_target_future is not None else None
        hypothesis_action_t = None
        initial_action_intent = getattr(model, "override_action_intent", None)
        if not torch.is_tensor(initial_action_intent):
            initial_action_intent = getattr(model, "_last_dynamic_coupling_action_intent_persist", None)
        if torch.is_tensor(initial_action_intent):
            hypothesis_action_t = initial_action_intent.detach().clone().to(model.device)
        joint_belief_patch_count = 0
        joint_belief_last_patch_step = -10**9
        for step in range(cfg.ep_len):
            collect_jbt = bool(getattr(cfg, "collect_joint_belief_transition", False))
            jbt_saved = False
            active_target_future = hypothesis_future_t if persistent_hypothesis and hypothesis_future_t is not None else coupling_target_future
            if (
                active_target_future is not None
                and bool(getattr(cfg, "dynamic_coupling_online", False))
                and not joint_execution_update
            ):
                every_k = max(1, int(getattr(cfg, "dynamic_coupling_gate_every_k", 1)))
                start_step = max(0, int(getattr(cfg, "dynamic_coupling_gate_start_step", 0)))
                # Gate relative to start_step so start_step=1,every_k=4 means
                # steps 1,5,9,... instead of 4,8,12,...
                if step < start_step or ((step - start_step) % every_k) != 0:
                    coupling_last_info = {
                        "dynamic_coupling_used": False,
                        "dynamic_coupling_applied": False,
                        "dynamic_coupling_reason": f"step_gate_every_{every_k}",
                        "dynamic_coupling_gate_start_step": int(start_step),
                        "dynamic_coupling_gate_current_step": int(step),
                    }
                    model._last_dynamic_coupling_info = dict(coupling_last_info)
                    persist_info = dict(coupling_last_info)
                    persist_info.update(dict(getattr(model, "_last_joint_hypothesis_init_info", {}) or {}))
                    persist_info.update(coupling_agg_info)
                    model._last_dynamic_coupling_info_persist = persist_info
                else:
                    corrected_future_t, generated_action_intent_t, coupling_info = apply_coupled_correction(
                        cfg,
                        mode="fast",
                        task=subtask,
                        subtask_index=int(subtask_index or 0),
                        env=env,
                        model=model,
                        obs=obs,
                        lang_text=lang_text,
                        slow_target_future=active_target_future,
                        recent_actions=recent_actions,
                    )
                    if persistent_hypothesis and not joint_execution_update:
                        future_update_mix = float(getattr(cfg, "dynamic_coupling_hypothesis_future_update_mix", 1.0))
                        action_update_mix = float(getattr(cfg, "dynamic_coupling_hypothesis_action_update_mix", 1.0))
                        future_update_mix = float(np.clip(future_update_mix, 0.0, 1.0))
                        action_update_mix = float(np.clip(action_update_mix, 0.0, 1.0))
                        prev_future_t = hypothesis_future_t.detach().clone() if hypothesis_future_t is not None else active_target_future.detach().clone()
                        if corrected_future_t is not None:
                            hypothesis_future_t = prev_future_t + future_update_mix * (corrected_future_t.detach() - prev_future_t)
                            corrected_future_t = hypothesis_future_t
                        if generated_action_intent_t is not None:
                            candidate_action_t = generated_action_intent_t.detach().to(model.device)
                            if hypothesis_action_t is None:
                                prev_action_t = candidate_action_t.clone()
                                hypothesis_action_t = candidate_action_t.clone()
                            else:
                                prev_action_t = hypothesis_action_t.detach().clone()
                                hypothesis_action_t = hypothesis_action_t + action_update_mix * (candidate_action_t - hypothesis_action_t)
                            generated_action_intent_t = hypothesis_action_t
                            action_delta_norm = float(torch.norm((hypothesis_action_t - prev_action_t).reshape(-1), p=2).item())
                        else:
                            action_delta_norm = 0.0
                        coupling_info = dict(coupling_info)
                        coupling_info.update(
                            {
                                "persistent_hypothesis_used": True,
                                "persistent_hypothesis_future_update_mix": future_update_mix,
                                "persistent_hypothesis_action_update_mix": action_update_mix,
                                "persistent_hypothesis_future_delta_norm": float(
                                    torch.norm((hypothesis_future_t - prev_future_t).reshape(-1), p=2).item()
                                )
                                if hypothesis_future_t is not None
                                else 0.0,
                                "persistent_hypothesis_action_delta_norm": action_delta_norm,
                            }
                        )
                    else:
                        coupling_info = dict(coupling_info)
                        coupling_info["persistent_hypothesis_used"] = bool(persistent_hypothesis and joint_execution_update)
                        if joint_execution_update:
                            coupling_info["joint_execution_update_deferred"] = True
                    corrected_future_t = _finite_tensor_or_none(
                        corrected_future_t,
                        name="dynamic_coupling_corrected_future",
                        info=coupling_last_info,
                    )
                    generated_action_intent_t = _finite_tensor_or_none(
                        generated_action_intent_t,
                        name="dynamic_coupling_generated_action_intent",
                        info=coupling_last_info,
                    )
                    if corrected_future_t is not None and not bool(
                        getattr(cfg, "joint_belief_clean_controller", False)
                    ):
                        model.future_feature_mode = "override"
                        model.override_future_feature = corrected_future_t
                    else:
                        if bool(getattr(cfg, "joint_belief_clean_controller", False)):
                            model.future_feature_mode = None
                        model.override_future_feature = None
                    coupling_last_info = dict(coupling_info)
                    if coupling_control_enabled and generated_action_intent_t is not None:
                        if not joint_execution_update:
                            model._last_dynamic_coupling_action_intent_persist = generated_action_intent_t.detach().float().cpu()
                            if bool(getattr(cfg, "dynamic_coupling_apply_action_intent_to_policy", False)):
                                model.override_action_intent = generated_action_intent_t
                        remaining_action_chunk = model.get_remaining_action_seq(pad_to_full=True) if hasattr(model, "get_remaining_action_seq") else None
                        if remaining_action_chunk is not None:
                            current_state_for_delta = _joint_pair_state_from_raw(snapshot_env_raw(env))
                            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

                            current_future_for_delta = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
                            base_summary_exec = _joint_pair_future_summary_exec(
                                current_future_for_delta.detach().float().cpu().numpy().astype(np.float32)
                            )
                            target_summary_exec = _joint_pair_future_summary_exec(
                                corrected_future_t.detach().float().cpu().numpy().astype(np.float32)
                            )
                            if joint_update_payload is not None:
                                # Unified U(h_t, s_t, a_t, s_{t+1}) replaces the separate phi scorer.
                                chunk_effect_payload = None
                                chunk_effect_info = {
                                    "chunk_effect_used": False,
                                    "chunk_effect_reason": "replaced_by_joint_execution_update",
                                }
                            else:
                                chunk_effect_payload, chunk_effect_info = predict_remaining_chunk_effect(
                                    cfg,
                                    model,
                                    task=subtask,
                                    subtask_index=int(subtask_index or 0),
                                    current_state=current_state_for_delta,
                                    base_summary_exec=base_summary_exec,
                                    target_summary_exec=target_summary_exec,
                                    remaining_action_chunk=remaining_action_chunk,
                                )
                            coupling_last_info.update(chunk_effect_info)
                            if chunk_effect_info.get("chunk_effect_used", False):
                                coupling_agg_info["chunk_effect_used_count"] += 1
                                coupling_agg_info["chunk_effect_any_used"] = True
                                coupling_agg_info["chunk_effect_last_runtime_type"] = chunk_effect_info.get(
                                    "chunk_effect_runtime_type"
                                )
                                residual_norm = chunk_effect_info.get("chunk_effect_residual_norm")
                                transition_cos = chunk_effect_info.get("chunk_effect_transition_cos")
                                success_prob = chunk_effect_info.get("chunk_effect_success_prob")
                                if isinstance(residual_norm, (int, float)) and np.isfinite(float(residual_norm)):
                                    residual_norm = float(residual_norm)
                                    coupling_agg_info["chunk_effect_last_residual_norm"] = residual_norm
                                    coupling_agg_info["chunk_effect_min_residual_norm"] = (
                                        residual_norm
                                        if coupling_agg_info["chunk_effect_min_residual_norm"] is None
                                        else min(float(coupling_agg_info["chunk_effect_min_residual_norm"]), residual_norm)
                                    )
                                    coupling_agg_info["chunk_effect_max_residual_norm"] = (
                                        residual_norm
                                        if coupling_agg_info["chunk_effect_max_residual_norm"] is None
                                        else max(float(coupling_agg_info["chunk_effect_max_residual_norm"]), residual_norm)
                                    )
                                if isinstance(transition_cos, (int, float)) and np.isfinite(float(transition_cos)):
                                    transition_cos = float(transition_cos)
                                    coupling_agg_info["chunk_effect_last_transition_cos"] = transition_cos
                                    coupling_agg_info["chunk_effect_min_transition_cos"] = (
                                        transition_cos
                                        if coupling_agg_info["chunk_effect_min_transition_cos"] is None
                                        else min(float(coupling_agg_info["chunk_effect_min_transition_cos"]), transition_cos)
                                    )
                                    coupling_agg_info["chunk_effect_max_transition_cos"] = (
                                        transition_cos
                                        if coupling_agg_info["chunk_effect_max_transition_cos"] is None
                                        else max(float(coupling_agg_info["chunk_effect_max_transition_cos"]), transition_cos)
                                    )
                                if isinstance(success_prob, (int, float)) and np.isfinite(float(success_prob)):
                                    success_prob = float(success_prob)
                                    coupling_agg_info["chunk_effect_last_success_prob"] = success_prob
                                    coupling_agg_info["chunk_effect_min_success_prob"] = (
                                        success_prob
                                        if coupling_agg_info["chunk_effect_min_success_prob"] is None
                                        else min(float(coupling_agg_info["chunk_effect_min_success_prob"]), success_prob)
                                    )
                                    coupling_agg_info["chunk_effect_max_success_prob"] = (
                                        success_prob
                                        if coupling_agg_info["chunk_effect_max_success_prob"] is None
                                        else max(float(coupling_agg_info["chunk_effect_max_success_prob"]), success_prob)
                                    )
                            delta_action_t, delta_action_info = predict_delta_action_repair(
                                cfg,
                                model,
                                task=subtask,
                                subtask_index=int(subtask_index or 0),
                                current_state=current_state_for_delta,
                                current_action_chunk=remaining_action_chunk,
                                base_summary_exec=base_summary_exec,
                                target_summary_exec=target_summary_exec,
                            )
                            coupling_last_info.update(delta_action_info)
                            if joint_execution_update:
                                chunk_decision = decide_joint_execution_suffix(cfg, chunk_effect_payload, delta_action_t)
                            else:
                                chunk_decision = decide_chunk_intervention(cfg, chunk_effect_payload, delta_action_t)
                            if persistent_hypothesis and joint_execution_update:
                                joint_gate = float(chunk_decision.get("joint_execution_gate", 0.0) or 0.0)
                                prev_future_t = hypothesis_future_t.detach().clone() if hypothesis_future_t is not None else active_target_future.detach().clone()
                                if corrected_future_t is not None:
                                    hypothesis_future_t = prev_future_t + joint_gate * (corrected_future_t.detach() - prev_future_t)
                                    corrected_future_t = hypothesis_future_t
                                    if not bool(getattr(cfg, "joint_belief_clean_controller", False)):
                                        model.override_future_feature = corrected_future_t
                                if generated_action_intent_t is not None:
                                    candidate_action_t = generated_action_intent_t.detach().to(model.device)
                                    if hypothesis_action_t is None:
                                        prev_action_t = candidate_action_t.clone()
                                        hypothesis_action_t = candidate_action_t.clone()
                                    else:
                                        prev_action_t = hypothesis_action_t.detach().clone()
                                        hypothesis_action_t = hypothesis_action_t + joint_gate * (candidate_action_t - hypothesis_action_t)
                                    generated_action_intent_t = hypothesis_action_t
                                    model._last_dynamic_coupling_action_intent_persist = hypothesis_action_t.detach().float().cpu()
                                    if (
                                        bool(getattr(cfg, "dynamic_coupling_apply_action_intent_to_policy", False))
                                        and not bool(getattr(cfg, "joint_belief_clean_controller", False))
                                    ):
                                        model.override_action_intent = generated_action_intent_t
                                    action_delta_norm = float(torch.norm((hypothesis_action_t - prev_action_t).reshape(-1), p=2).item())
                                else:
                                    action_delta_norm = 0.0
                                coupling_last_info.update(
                                    {
                                        "persistent_hypothesis_used": True,
                                        "joint_execution_update_applied": True,
                                        "persistent_hypothesis_update_gate": joint_gate,
                                        "persistent_hypothesis_future_delta_norm": float(
                                            torch.norm((hypothesis_future_t - prev_future_t).reshape(-1), p=2).item()
                                        )
                                        if hypothesis_future_t is not None
                                        else 0.0,
                                        "persistent_hypothesis_action_delta_norm": action_delta_norm,
                                    }
                                )
                            coupling_agg_info["chunk_effect_decision_count"] += 1
                            coupling_agg_info["chunk_effect_last_decision"] = str(chunk_decision["decision"])
                            coupling_last_info.update(
                                {
                                    "dynamic_coupling_chunk_decision": str(chunk_decision["decision"]),
                                    "dynamic_coupling_chunk_decision_reason": str(chunk_decision["decision_reason"]),
                                    "dynamic_coupling_chunk_residual_norm": float(chunk_decision["residual_norm"]),
                                    "dynamic_coupling_chunk_compat": float(chunk_decision["compat"]),
                                    "dynamic_coupling_chunk_success_prob": float(chunk_decision["success_prob"]),
                                    "dynamic_coupling_chunk_keep_threshold": float(chunk_decision["keep_threshold"]),
                                    "dynamic_coupling_chunk_replan_threshold": float(chunk_decision["replan_threshold"]),
                                    "dynamic_coupling_chunk_success_threshold": float(chunk_decision["success_threshold"]),
                                    "dynamic_coupling_chunk_low_compat_threshold": float(chunk_decision["low_compat_threshold"]),
                                }
                            )
                            if (
                                chunk_decision["decision"] == "repair_suffix"
                                and delta_action_t is not None
                                and hasattr(model, "patch_remaining_action_seq")
                            ):
                                patch_mix = float(getattr(cfg, "dynamic_coupling_delta_action_repair_mix", 1.0))
                                patch_info = model.patch_remaining_action_seq(
                                    delta_action_t,
                                    mode="delta",
                                    mix=patch_mix,
                                )
                                coupling_last_info.update(
                                    {
                                        "dynamic_coupling_delta_action_patch_requested": True,
                                        "dynamic_coupling_delta_action_patch_mix": patch_mix,
                                        **{f"dynamic_coupling_delta_action_patch_{k}": v for k, v in patch_info.items()},
                                    }
                                )
                                if bool(patch_info.get("patched", False)):
                                    joint_belief_patch_count += 1
                                    joint_belief_last_patch_step = int(step)
                                    coupling_last_info.update(
                                        {
                                            "joint_belief_patch_count": int(joint_belief_patch_count),
                                            "joint_belief_last_patch_step": int(joint_belief_last_patch_step),
                                        }
                                    )
                            elif chunk_decision["decision"] == "replan":
                                model.pred_action_seq = None
                                model.rollout_step_counter = 0
                                if persistent_hypothesis:
                                    hypothesis_future_t = corrected_future_t.detach().clone() if corrected_future_t is not None else coupling_target_future
                                    hypothesis_action_t = generated_action_intent_t.detach().clone() if generated_action_intent_t is not None else None
                                coupling_last_info.update(
                                    {
                                        "dynamic_coupling_full_replan_requested": True,
                                        "dynamic_coupling_replan_from_step": int(step),
                                        "persistent_hypothesis_reset_by_replan": bool(persistent_hypothesis),
                                    }
                                )
                            else:
                                coupling_last_info.update(
                                    {
                                        "dynamic_coupling_delta_action_patch_requested": False,
                                        "dynamic_coupling_full_replan_requested": False,
                                    }
                                )
                        use_delta_action_repair = bool(getattr(cfg, "dynamic_coupling_delta_action_repair_ckpt", "") or "")
                        if (
                            not bool(getattr(cfg, "dynamic_coupling_chunk_effect_ckpt", "") or "")
                            and not use_delta_action_repair
                            and bool(getattr(cfg, "dynamic_coupling_patch_remaining_actions", False))
                            and hasattr(model, "patch_remaining_action_seq")
                        ):
                            patch_mode = str(getattr(cfg, "dynamic_coupling_patch_mode", "replace") or "replace")
                            patch_mix = float(getattr(cfg, "dynamic_coupling_patch_mix", 1.0))
                            patch_info = model.patch_remaining_action_seq(
                                generated_action_intent_t,
                                mode=patch_mode,
                                mix=patch_mix,
                            )
                            coupling_last_info.update(
                                {
                                    "dynamic_coupling_action_patch_requested": True,
                                    "dynamic_coupling_action_patch_mode": patch_mode,
                                    "dynamic_coupling_action_patch_mix": patch_mix,
                                    **{f"dynamic_coupling_action_patch_{k}": v for k, v in patch_info.items()},
                                }
                            )
                    elif coupling_control_enabled and override_action_intent is not None:
                        model.override_action_intent = override_action_intent
                        if torch.is_tensor(override_action_intent):
                            model._last_dynamic_coupling_action_intent_persist = override_action_intent.detach().float().cpu()
                    model._last_dynamic_coupling_info = dict(coupling_last_info)
                    if coupling_last_info.get("dynamic_coupling_used", False):
                        coupling_agg_info["dynamic_coupling_used_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_used"] = True
                    if coupling_last_info.get("dynamic_coupling_applied", False):
                        coupling_agg_info["dynamic_coupling_applied_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_applied"] = True
                        if coupling_agg_info["dynamic_coupling_first_applied_step"] is None:
                            coupling_agg_info["dynamic_coupling_first_applied_step"] = int(step)
                        coupling_agg_info["dynamic_coupling_last_applied_step"] = int(step)
                        coupling_agg_info["dynamic_coupling_max_final_shift_norm"] = max(
                            float(coupling_agg_info["dynamic_coupling_max_final_shift_norm"]),
                            float(coupling_last_info.get("dynamic_coupling_final_shift_norm", 0.0) or 0.0),
                        )
                        model._last_dynamic_coupling_future_persist = corrected_future_t.detach().float().cpu()
                    if coupling_last_info.get("dynamic_coupling_action_generated", False):
                        coupling_agg_info["dynamic_coupling_action_generated_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_action_generated"] = True
                    persist_info = dict(coupling_last_info)
                    persist_info.update(dict(getattr(model, "_last_joint_hypothesis_init_info", {}) or {}))
                    if retry_mode and corrected_future_t is not None:
                        final_action_for_score = getattr(model, "override_action_intent", None)
                        final_score_info = score_joint_hypothesis_candidate(
                            cfg,
                            base_future=retry_score_base_future,
                            target_future=corrected_future_t,
                            action_chunk=final_action_for_score,
                            task=subtask,
                            subtask_index=int(subtask_index or 0),
                            current_state=current_state,
                            prefix="retry_final_score",
                        )
                        persist_info.update(final_score_info)
                        if persist_info.get("retry_initial_score_used") and final_score_info.get("retry_final_score_used"):
                            persist_info["retry_score_improved"] = (
                                float(final_score_info["retry_final_score_energy"])
                                < float(persist_info.get("retry_initial_score_energy", 1e18))
                            )
                            persist_info["retry_score_delta_energy"] = (
                                float(persist_info.get("retry_initial_score_energy", 0.0))
                                - float(final_score_info["retry_final_score_energy"])
                            )
                    persist_info.update(coupling_agg_info)
                    model._last_dynamic_coupling_info_persist = persist_info
            jbt_future_before = hypothesis_future_t.detach().clone() if torch.is_tensor(hypothesis_future_t) else None
            if torch.is_tensor(hypothesis_action_t):
                jbt_action_before = hypothesis_action_t.detach().clone()
            elif torch.is_tensor(getattr(model, "override_action_intent", None)):
                jbt_action_before = getattr(model, "override_action_intent").detach().clone()
            else:
                jbt_action_before = None
            jbt_shadow_info = {}
            if collect_jbt and (jbt_future_before is None or jbt_action_before is None):
                shadow_future, shadow_action, jbt_shadow_info = compute_shadow_joint_hypothesis(
                    cfg,
                    env,
                    model,
                    obs,
                    lang_text,
                    subtask,
                    int(subtask_index or 0),
                )
                if jbt_future_before is None:
                    jbt_future_before = shadow_future
                if jbt_action_before is None:
                    jbt_action_before = shadow_action
            action = model.step(obs, goal)
            try:
                action_dbg = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
            except Exception:
                action_dbg = np.asarray(action)
            action_dbg = np.asarray(action_dbg, dtype=np.float32)
            if bool(getattr(cfg, "debug_action_before_env_step", False)) and step < int(
                getattr(cfg, "debug_action_steps", 5)
            ):
                print(f"\n[DEBUG ACTION] step={step} value={action_dbg}", flush=True)
                print(f"[DEBUG ACTION] step={step} shape={action_dbg.shape}", flush=True)
                print(f"[DEBUG ACTION] step={step} finite={np.isfinite(action_dbg).all()}", flush=True)
                print(
                    f"[DEBUG ACTION] step={step} minmax={np.nanmin(action_dbg)} {np.nanmax(action_dbg)}",
                    flush=True,
                )
            if not np.isfinite(action_dbg).all():
                raise FloatingPointError(
                    f"non-finite action before env.step: subtask={subtask} step={step} value={action_dbg}"
                )
            if bool(getattr(cfg, "collect_action_trace", False)):
                action_trace.append(_flatten_action_for_probe(action))
            recent_actions.append(_flatten_action_for_probe(action))
            keep_actions = int(getattr(cfg, "dynamic_coupling_keep_actions", 64))
            if keep_actions > 0 and len(recent_actions) > keep_actions:
                recent_actions = recent_actions[-keep_actions:]
            state_before_step = _joint_pair_state_from_raw(snapshot_env_raw(env))
            jbt_action_remain_before = (
                model.get_remaining_action_seq(pad_to_full=True)
                if bool(getattr(cfg, "collect_joint_belief_transition", False)) and hasattr(model, "get_remaining_action_seq")
                else None
            )
            obs, _, _, current_info = env.step(action)
            state_after_step = _joint_pair_state_from_raw(snapshot_env_raw(env))
            last_info = current_info
            jbt_future_after_shadow = None
            jbt_action_after_shadow = None
            jbt_shadow_after_info = {}
            if collect_jbt and not (
                joint_execution_update
                and hypothesis_future_t is not None
                and bool(getattr(cfg, "dynamic_coupling_online", False))
            ):
                jbt_future_after_shadow, jbt_action_after_shadow, jbt_shadow_after_info = compute_shadow_joint_hypothesis(
                    cfg,
                    env,
                    model,
                    obs,
                    lang_text,
                    subtask,
                    int(subtask_index or 0),
                )
            if (
                joint_execution_update
                and hypothesis_future_t is not None
                and bool(getattr(cfg, "dynamic_coupling_online", False))
            ):
                every_k = max(1, int(getattr(cfg, "dynamic_coupling_gate_every_k", 1)))
                start_step = max(0, int(getattr(cfg, "dynamic_coupling_gate_start_step", 0)))
                if step >= start_step and ((step - start_step) % every_k) == 0:
                    corrected_future_t, generated_action_intent_t, coupling_info = apply_coupled_correction(
                        cfg,
                        mode="fast",
                        task=subtask,
                        subtask_index=int(subtask_index or 0),
                        env=env,
                        model=model,
                        obs=obs,
                        lang_text=lang_text,
                        slow_target_future=hypothesis_future_t,
                        recent_actions=recent_actions,
                    )
                    coupling_last_info = dict(coupling_info)
                    chunk_decision = None
                    if coupling_control_enabled:
                        remaining_action_chunk = (
                            model.get_remaining_action_seq(pad_to_full=True)
                            if hasattr(model, "get_remaining_action_seq")
                            else None
                        )
                        clean_controller = bool(getattr(cfg, "joint_belief_clean_controller", False))
                        target_future_for_delta = (
                            hypothesis_future_t
                            if clean_controller
                            else (corrected_future_t if corrected_future_t is not None else hypothesis_future_t)
                        )
                        if remaining_action_chunk is not None and target_future_for_delta is not None:
                            current_state_for_delta = _joint_pair_state_from_raw(snapshot_env_raw(env))
                            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

                            current_future_for_delta = _call_preserving_rng(defi_future_feature, model, obs, lang_text).detach().to(model.device)
                            base_summary_exec = _joint_pair_future_summary_exec(
                                current_future_for_delta.detach().float().cpu().numpy().astype(np.float32)
                            )
                            target_summary_exec = _joint_pair_future_summary_exec(
                                target_future_for_delta.detach().float().cpu().numpy().astype(np.float32)
                            )
                            joint_update_payload, joint_update_info = predict_joint_execution_update(
                                cfg,
                                model,
                                task=subtask,
                                subtask_index=int(subtask_index or 0),
                                state_start=state_before_step,
                                state_end=current_state_for_delta,
                                base_summary_exec=base_summary_exec,
                                target_summary_exec=target_summary_exec,
                                remaining_action_chunk=remaining_action_chunk,
                                executed_action=action_dbg,
                            )
                            coupling_last_info.update(joint_update_info)
                            if joint_update_payload is not None:
                                chunk_effect_payload = None
                                chunk_effect_info = {
                                    "chunk_effect_used": False,
                                    "chunk_effect_reason": "replaced_by_joint_belief_transition",
                                }
                            else:
                                chunk_effect_payload, chunk_effect_info = predict_remaining_chunk_effect(
                                    cfg,
                                    model,
                                    task=subtask,
                                    subtask_index=int(subtask_index or 0),
                                    current_state=current_state_for_delta,
                                    base_summary_exec=base_summary_exec,
                                    target_summary_exec=target_summary_exec,
                                    remaining_action_chunk=remaining_action_chunk,
                                )
                            coupling_last_info.update(chunk_effect_info)
                            if chunk_effect_info.get("chunk_effect_used", False):
                                coupling_agg_info["chunk_effect_used_count"] += 1
                                coupling_agg_info["chunk_effect_any_used"] = True
                                coupling_agg_info["chunk_effect_last_runtime_type"] = chunk_effect_info.get(
                                    "chunk_effect_runtime_type"
                                )
                                residual_norm = chunk_effect_info.get("chunk_effect_residual_norm")
                                transition_cos = chunk_effect_info.get("chunk_effect_transition_cos")
                                success_prob = chunk_effect_info.get("chunk_effect_success_prob")
                                if isinstance(residual_norm, (int, float)) and np.isfinite(float(residual_norm)):
                                    residual_norm = float(residual_norm)
                                    coupling_agg_info["chunk_effect_last_residual_norm"] = residual_norm
                                    coupling_agg_info["chunk_effect_min_residual_norm"] = (
                                        residual_norm
                                        if coupling_agg_info["chunk_effect_min_residual_norm"] is None
                                        else min(float(coupling_agg_info["chunk_effect_min_residual_norm"]), residual_norm)
                                    )
                                    coupling_agg_info["chunk_effect_max_residual_norm"] = (
                                        residual_norm
                                        if coupling_agg_info["chunk_effect_max_residual_norm"] is None
                                        else max(float(coupling_agg_info["chunk_effect_max_residual_norm"]), residual_norm)
                                    )
                                if isinstance(transition_cos, (int, float)) and np.isfinite(float(transition_cos)):
                                    transition_cos = float(transition_cos)
                                    coupling_agg_info["chunk_effect_last_transition_cos"] = transition_cos
                                    coupling_agg_info["chunk_effect_min_transition_cos"] = (
                                        transition_cos
                                        if coupling_agg_info["chunk_effect_min_transition_cos"] is None
                                        else min(float(coupling_agg_info["chunk_effect_min_transition_cos"]), transition_cos)
                                    )
                                    coupling_agg_info["chunk_effect_max_transition_cos"] = (
                                        transition_cos
                                        if coupling_agg_info["chunk_effect_max_transition_cos"] is None
                                        else max(float(coupling_agg_info["chunk_effect_max_transition_cos"]), transition_cos)
                                    )
                                if isinstance(success_prob, (int, float)) and np.isfinite(float(success_prob)):
                                    success_prob = float(success_prob)
                                    coupling_agg_info["chunk_effect_last_success_prob"] = success_prob
                                    coupling_agg_info["chunk_effect_min_success_prob"] = (
                                        success_prob
                                        if coupling_agg_info["chunk_effect_min_success_prob"] is None
                                        else min(float(coupling_agg_info["chunk_effect_min_success_prob"]), success_prob)
                                    )
                                    coupling_agg_info["chunk_effect_max_success_prob"] = (
                                        success_prob
                                        if coupling_agg_info["chunk_effect_max_success_prob"] is None
                                        else max(float(coupling_agg_info["chunk_effect_max_success_prob"]), success_prob)
                                    )
                            suffix_mask_ablation_mode = str(getattr(cfg, "suffix_mask_ablation_mode", "off") or "off")
                            suffix_ablation_teacher_info = None
                            suffix_ablation_needs_teacher = suffix_mask_ablation_mode in {
                                "gt_selective",
                                "random_mask",
                                "complement_mask",
                                "whole_suffix",
                            }
                            suffix_ablation_scheduled = _suffix_mask_schedule_allows(
                                cfg,
                                sequence_index=int(getattr(cfg, "_runtime_sequence_index", -1)),
                                subtask_index=int(subtask_index or 0),
                                step=int(step),
                            )
                            if suffix_ablation_needs_teacher and not suffix_ablation_scheduled:
                                suffix_ablation_needs_teacher = False
                            if joint_update_payload is not None:
                                delta_action_t = joint_update_payload["action_delta"]
                                delta_action_t = _finite_tensor_or_none(
                                    delta_action_t,
                                    name="joint_belief_transition_delta_action",
                                    info=coupling_last_info,
                                )
                                if bool(getattr(cfg, "directional_sensitivity_tcn_transport", False)):
                                    tcn_delta_t, tcn_delta_info = predict_directional_sensitivity_tcn(
                                        cfg,
                                        model,
                                        task=subtask,
                                        subtask_index=int(subtask_index or 0),
                                        current_state=current_state_for_delta,
                                        belief_before=base_summary_exec,
                                        belief_after=joint_update_payload.get("h_posterior"),
                                        innovation=joint_update_payload.get("innovation"),
                                        remaining_action_chunk=remaining_action_chunk,
                                    )
                                    coupling_last_info.update(tcn_delta_info)
                                    if tcn_delta_t is not None and str(
                                        tcn_delta_info.get("directional_sensitivity_tcn_decision_hint", "")
                                    ) == "repair_suffix":
                                        delta_action_t = tcn_delta_t
                                    elif str(joint_update_info.get("joint_execution_update_runtime_type", "")) in {
                                        "rssm_plan_compatibility",
                                        "full_rssm_plan_compatibility",
                                    }:
                                        delta_action_t = None
                                if bool(getattr(cfg, "direct_action_correction_tcn_transport", False)):
                                    direct_delta_t, direct_delta_info = predict_direct_action_correction_tcn(
                                        cfg,
                                        model,
                                        task=subtask,
                                        subtask_index=int(subtask_index or 0),
                                        state_before=state_before_step,
                                        state_after=state_after_step,
                                        executed_action=action_dbg,
                                        remaining_action_chunk=remaining_action_chunk,
                                    )
                                    coupling_last_info.update(direct_delta_info)
                                    if str(direct_delta_info.get("directional_sensitivity_tcn_decision_hint", "")) == "repair_suffix":
                                        delta_action_t = direct_delta_t
                                    else:
                                        delta_action_t = None
                                if (
                                    suffix_ablation_needs_teacher
                                    and (
                                        hasattr(model, "project_sensitivity_delta_action")
                                        or hasattr(model, "project_paired_counterfactual_delta_action")
                                    )
                                    and jbt_future_before is not None
                                    and jbt_action_before is not None
                                    and hypothesis_future_t is not None
                                    and hypothesis_action_t is not None
                                ):
                                    teacher_mode = str(getattr(cfg, "sensitivity_teacher_mode", "jvp") or "jvp")
                                    teacher_fn = (
                                        model.project_paired_counterfactual_delta_action
                                        if teacher_mode == "paired_counterfactual"
                                        and hasattr(model, "project_paired_counterfactual_delta_action")
                                        else model.project_sensitivity_delta_action
                                    )
                                    teacher_kwargs = {
                                        "action_intent_expected": jbt_action_before,
                                        "action_intent_posterior": hypothesis_action_t,
                                        "latent_goal_expected": jbt_future_before,
                                        "latent_goal_posterior": hypothesis_future_t,
                                        "joint_delta": (
                                            joint_update_payload.get("innovation")
                                            if isinstance(joint_update_payload, dict)
                                            else None
                                        ),
                                        "edit_step_size": float(getattr(cfg, "sensitivity_teacher_step_size", 1.0)),
                                        "max_abs_delta": float(getattr(cfg, "sensitivity_teacher_max_abs_delta", 0.0)),
                                    }
                                    if teacher_mode == "paired_counterfactual":
                                        teacher_kwargs["joint_delta_to_action_intent_scale"] = float(
                                            getattr(cfg, "sensitivity_teacher_joint_delta_to_action_scale", 1.0)
                                        )
                                        teacher_kwargs["condition_ablation"] = str(
                                            getattr(cfg, "sensitivity_teacher_condition_ablation", "joint") or "joint"
                                        )
                                    suffix_ablation_teacher_info = teacher_fn(**teacher_kwargs)
                                ablation_delta_t, ablation_info = _suffix_control_ablation_delta_from_teacher(
                                    cfg,
                                    suffix_ablation_teacher_info,
                                    device=model.device,
                                    sequence_index=int(getattr(cfg, "_runtime_sequence_index", -1)),
                                    subtask_index=int(subtask_index or 0),
                                    step=int(step),
                                )
                                if suffix_mask_ablation_mode != "off" and suffix_ablation_scheduled:
                                    delta_action_t = ablation_delta_t
                                    coupling_last_info.update(ablation_info)
                                elif suffix_mask_ablation_mode != "off" and not suffix_ablation_scheduled:
                                    delta_action_t = None
                                    coupling_last_info.update(
                                        {
                                            "suffix_mask_ablation_mode": suffix_mask_ablation_mode,
                                            "suffix_mask_ablation_used": True,
                                            "suffix_mask_ablation_scheduled": False,
                                            "directional_sensitivity_tcn_used": True,
                                            "directional_sensitivity_tcn_applied": False,
                                            "directional_sensitivity_tcn_decision_hint": "keep",
                                            "directional_sensitivity_tcn_reason": "ablation_step_not_in_schedule",
                                        }
                                    )
                                delta_action_info = {
                                    "delta_action_repair_used": True,
                                    "delta_action_repair_applied": delta_action_t is not None,
                                    "delta_action_repair_source": (
                                        f"suffix_mask_ablation_{suffix_mask_ablation_mode}"
                                        if suffix_mask_ablation_mode != "off"
                                        else "directional_sensitivity_tcn"
                                    )
                                    if bool(coupling_last_info.get("directional_sensitivity_tcn_applied", False))
                                    else "joint_execution_update_mlp",
                                    "delta_action_repair_delta_norm": float(
                                        torch.norm(delta_action_t.reshape(-1), p=2).item()
                                    )
                                    if delta_action_t is not None
                                    else 0.0,
                                }
                            else:
                                delta_action_t, delta_action_info = predict_delta_action_repair(
                                    cfg,
                                    model,
                                    task=subtask,
                                    subtask_index=int(subtask_index or 0),
                                    current_state=current_state_for_delta,
                                    current_action_chunk=remaining_action_chunk,
                                    base_summary_exec=base_summary_exec,
                                    target_summary_exec=target_summary_exec,
                                )
                            coupling_last_info.update(delta_action_info)
                            if joint_update_payload is not None:
                                error = float(joint_update_payload["error"])
                                direct_suffix = bool(getattr(cfg, "joint_belief_transition_direct_suffix", False))
                                heuristic_gate = float(
                                    np.clip(
                                        max(0.0, error - float(getattr(cfg, "joint_execution_update_threshold", 1.0)))
                                        * float(getattr(cfg, "joint_execution_update_gain", 1.0)),
                                        float(getattr(cfg, "joint_execution_update_min_gate", 0.0)),
                                        float(getattr(cfg, "joint_execution_update_max_gate", 0.3)),
                                    )
                                )
                                learned_gate_raw = float(joint_update_payload.get("learned_gate", float("nan")))
                                gate_mode = str(getattr(cfg, "joint_execution_gate_mode", "heuristic"))
                                if gate_mode in {"learned", "learned_or_heuristic"} and np.isfinite(learned_gate_raw):
                                    joint_gate = float(
                                        np.clip(
                                            learned_gate_raw * float(getattr(cfg, "joint_execution_update_max_gate", 0.3)),
                                            float(getattr(cfg, "joint_execution_update_min_gate", 0.0)),
                                            float(getattr(cfg, "joint_execution_update_max_gate", 0.3)),
                                        )
                                    )
                                    gate_source = "learned"
                                else:
                                    joint_gate = heuristic_gate
                                    gate_source = "heuristic"
                                success_prob = float(joint_update_payload["success_prob"])
                                sparse_suffix = bool(getattr(cfg, "joint_belief_transition_sparse_suffix", False))
                                min_learned_gate = float(getattr(cfg, "joint_belief_transition_min_learned_gate", 0.08))
                                max_success_prob = float(getattr(cfg, "joint_belief_transition_max_success_prob", 0.25))
                                predicted_mode = str(joint_update_payload.get("mode", "") or "")
                                suffix_decision = "keep"
                                suffix_reason = "joint_belief_transition_default_keep"
                                if clean_controller and (
                                    bool(getattr(cfg, "directional_sensitivity_tcn_transport", False))
                                    or bool(getattr(cfg, "direct_action_correction_tcn_transport", False))
                                    or suffix_mask_ablation_mode != "off"
                                ):
                                    tcn_hint = str(
                                        coupling_last_info.get("directional_sensitivity_tcn_decision_hint", "keep")
                                        or "keep"
                                    )
                                    if tcn_hint == "repair_suffix" and delta_action_t is not None:
                                        sparse_triggered = True
                                        suffix_decision = "repair_suffix"
                                        suffix_reason = "directional_sensitivity_affected_mask_patch"
                                    elif (
                                        tcn_hint == "replan"
                                        and not bool(getattr(cfg, "dynamic_coupling_disable_replan", False))
                                    ):
                                        sparse_triggered = False
                                        suffix_decision = "replan"
                                        suffix_reason = "directional_sensitivity_affected_mask_replan"
                                    else:
                                        sparse_triggered = False
                                        suffix_decision = "keep"
                                        suffix_reason = "directional_sensitivity_affected_mask_keep"
                                    if suffix_decision == "repair_suffix":
                                        patch_min_interval = max(
                                            1,
                                            int(getattr(cfg, "joint_belief_patch_min_interval_steps", 1)),
                                        )
                                        patch_max_per_subtask = int(
                                            getattr(cfg, "joint_belief_patch_max_per_subtask", 0)
                                        )
                                        force_suffix_ablation = bool(
                                            getattr(cfg, "suffix_mask_ablation_force_writeback", False)
                                            and suffix_mask_ablation_mode != "off"
                                        )
                                        patch_budget_ok = (
                                            force_suffix_ablation
                                            or
                                            patch_max_per_subtask <= 0
                                            or joint_belief_patch_count < patch_max_per_subtask
                                        )
                                        patch_interval_ok = (
                                            force_suffix_ablation
                                            or int(step) - int(joint_belief_last_patch_step) >= patch_min_interval
                                        )
                                        if not patch_budget_ok:
                                            sparse_triggered = False
                                            suffix_decision = "keep"
                                            suffix_reason = "directional_sensitivity_patch_budget_exhausted"
                                        elif not patch_interval_ok:
                                            sparse_triggered = False
                                            suffix_decision = "keep"
                                            suffix_reason = "directional_sensitivity_patch_interval_blocked"
                                    trigger_payload = {"used": False}
                                    trigger_threshold = float("nan")
                                    suffix_trigger_source = "directional_sensitivity_temporal_attribution"
                                elif clean_controller and predicted_mode:
                                    confidence = float(joint_update_payload.get("confidence", success_prob))
                                    min_confidence = float(getattr(cfg, "joint_belief_clean_min_confidence", 0.0))
                                    mode_prob = joint_update_payload.get("mode_prob", None)
                                    if torch.is_tensor(mode_prob):
                                        keep_prob = float(mode_prob[:, 0].reshape(-1)[0].item())
                                        patch_prob = float(mode_prob[:, 1].reshape(-1)[0].item())
                                        replan_prob = float(mode_prob[:, 2].reshape(-1)[0].item())
                                    else:
                                        keep_prob = patch_prob = replan_prob = float("nan")
                                    patch_min_prob = float(getattr(cfg, "joint_belief_clean_patch_min_prob", 0.6))
                                    patch_margin = float(getattr(cfg, "joint_belief_clean_patch_margin", 0.1))
                                    replan_min_prob = float(getattr(cfg, "joint_belief_clean_replan_min_prob", 0.7))
                                    if confidence < min_confidence and not bool(getattr(cfg, "dynamic_coupling_disable_replan", False)):
                                        sparse_triggered = False
                                        suffix_decision = "replan"
                                        suffix_reason = "clean_controller_low_confidence_replan"
                                    elif (
                                        predicted_mode == "patch"
                                        and
                                        delta_action_t is not None
                                        and np.isfinite(patch_prob)
                                        and np.isfinite(keep_prob)
                                        and patch_prob >= patch_min_prob
                                        and patch_prob >= keep_prob + patch_margin
                                    ):
                                        sparse_triggered = True
                                        suffix_decision = "repair_suffix"
                                        suffix_reason = "clean_controller_D_mode_patch_confident"
                                    elif (
                                        predicted_mode == "replan"
                                        and np.isfinite(replan_prob)
                                        and replan_prob >= replan_min_prob
                                        and not bool(getattr(cfg, "dynamic_coupling_disable_replan", False))
                                    ):
                                        sparse_triggered = False
                                        suffix_decision = "replan"
                                        suffix_reason = "clean_controller_D_mode_replan_confident"
                                    else:
                                        sparse_triggered = False
                                        suffix_decision = "keep"
                                        suffix_reason = f"clean_controller_D_mode_{predicted_mode}"
                                    if suffix_decision == "repair_suffix":
                                        patch_min_interval = max(
                                            1,
                                            int(getattr(cfg, "joint_belief_patch_min_interval_steps", 1)),
                                        )
                                        patch_max_per_subtask = int(
                                            getattr(cfg, "joint_belief_patch_max_per_subtask", 0)
                                        )
                                        patch_budget_ok = (
                                            patch_max_per_subtask <= 0
                                            or joint_belief_patch_count < patch_max_per_subtask
                                        )
                                        patch_interval_ok = int(step) - int(joint_belief_last_patch_step) >= patch_min_interval
                                        if not patch_budget_ok:
                                            sparse_triggered = False
                                            suffix_decision = "keep"
                                            suffix_reason = "clean_controller_patch_budget_exhausted"
                                        elif not patch_interval_ok:
                                            sparse_triggered = False
                                            suffix_decision = "keep"
                                            suffix_reason = "clean_controller_patch_interval_blocked"
                                    trigger_payload = {"used": False}
                                    trigger_threshold = float("nan")
                                    suffix_trigger_source = "factored_D_mode"
                                elif predicted_mode:
                                    sparse_triggered = predicted_mode == "patch"
                                    trigger_payload = {"used": False}
                                    trigger_threshold = float("nan")
                                    suffix_trigger_source = "factored_D_mode"
                                else:
                                    sparse_triggered = (
                                        (np.isfinite(learned_gate_raw) and learned_gate_raw >= min_learned_gate)
                                        or success_prob <= max_success_prob
                                    )
                                    trigger_payload = predict_suffix_trigger_runtime(
                                        load_suffix_trigger_runtime(cfg, model.device),
                                        task=subtask,
                                        subtask_index=int(subtask_index or 0),
                                        chunk_decision={
                                            "residual_norm": error,
                                            "compat": float("nan"),
                                            "success_prob": success_prob,
                                            "joint_execution_learned_gate_raw": learned_gate_raw,
                                            "joint_execution_gate": joint_gate,
                                            "joint_execution_error": error,
                                        },
                                        coupling_info=coupling_last_info,
                                    )
                                    trigger_threshold = float(getattr(cfg, "suffix_trigger_mlp_threshold", 0.5))
                                    if trigger_payload.get("used", False) and "prob" in trigger_payload:
                                        sparse_triggered = bool(float(trigger_payload["prob"]) >= trigger_threshold)
                                        suffix_trigger_source = "suffix_trigger_mlp"
                                    else:
                                        suffix_trigger_source = "joint_belief_thresholds"
                                if not clean_controller:
                                    suffix_decision = "repair_suffix" if delta_action_t is not None else "keep"
                                    suffix_reason = "joint_belief_transition_direct_suffix"
                                if predicted_mode and not clean_controller:
                                    if predicted_mode == "patch" and delta_action_t is not None:
                                        suffix_decision = "repair_suffix"
                                    elif predicted_mode == "replan" and not bool(getattr(cfg, "dynamic_coupling_disable_replan", False)):
                                        suffix_decision = "replan"
                                    else:
                                        suffix_decision = "keep"
                                    suffix_reason = f"factored_D_mode_{predicted_mode}"
                                if not clean_controller and not predicted_mode and direct_suffix and sparse_suffix:
                                    suffix_decision = "repair_suffix" if delta_action_t is not None and sparse_triggered else "keep"
                                    suffix_reason = (
                                        f"{suffix_trigger_source}_trigger"
                                        if sparse_triggered
                                        else f"{suffix_trigger_source}_keep"
                                    )
                                if not clean_controller and not direct_suffix:
                                    suffix_decision = (
                                        "repair_suffix"
                                        if delta_action_t is not None
                                        and error >= float(getattr(cfg, "joint_execution_suffix_error_threshold", 1.0))
                                        else "keep"
                                    )
                                    suffix_reason = "joint_execution_update_mlp_error_gate"
                                chunk_decision = {
                                    "decision": suffix_decision,
                                    "decision_reason": suffix_reason,
                                    "residual_norm": float(error),
                                    "compat": float("nan"),
                                    "success_prob": success_prob,
                                    "keep_threshold": float(getattr(cfg, "dynamic_coupling_chunk_keep_threshold", 0.35)),
                                    "replan_threshold": float(getattr(cfg, "dynamic_coupling_chunk_replan_threshold", 0.75)),
                                    "success_threshold": float(getattr(cfg, "dynamic_coupling_chunk_success_threshold", 0.5)),
                                    "low_compat_threshold": float(getattr(cfg, "dynamic_coupling_chunk_low_compat_threshold", 0.0)),
                                    "disable_replan": True,
                                    "joint_execution_update_used": True,
                                    "joint_execution_error": error,
                                    "joint_execution_gate": joint_gate,
                                    "joint_execution_gate_mode": gate_mode,
                                    "joint_execution_gate_source": gate_source,
                                    "joint_execution_heuristic_gate": heuristic_gate,
                                    "joint_execution_learned_gate_raw": learned_gate_raw,
                                    "joint_belief_transition_used": True,
                                    "joint_belief_clean_controller": clean_controller,
                                    "joint_belief_transition_direct_suffix": direct_suffix,
                                    "joint_belief_transition_sparse_suffix": sparse_suffix,
                                    "joint_belief_transition_sparse_triggered": sparse_triggered,
                                    "joint_belief_transition_min_learned_gate": min_learned_gate,
                                    "joint_belief_transition_max_success_prob": max_success_prob,
                                    "joint_belief_transition_predicted_mode": predicted_mode,
                                    "joint_belief_patch_min_interval_steps": int(
                                        getattr(cfg, "joint_belief_patch_min_interval_steps", 1)
                                    ),
                                    "joint_belief_patch_max_per_subtask": int(
                                        getattr(cfg, "joint_belief_patch_max_per_subtask", 0)
                                    ),
                                    "joint_belief_patch_count_before_decision": int(joint_belief_patch_count),
                                    "joint_belief_transition_mode_keep_prob": float(joint_update_payload.get("mode_prob", torch.zeros((1, 3), device=model.device))[:, 0].reshape(-1)[0].item()) if torch.is_tensor(joint_update_payload.get("mode_prob", None)) else float("nan"),
                                    "joint_belief_transition_mode_patch_prob": float(joint_update_payload.get("mode_prob", torch.zeros((1, 3), device=model.device))[:, 1].reshape(-1)[0].item()) if torch.is_tensor(joint_update_payload.get("mode_prob", None)) else float("nan"),
                                    "joint_belief_transition_mode_replan_prob": float(joint_update_payload.get("mode_prob", torch.zeros((1, 3), device=model.device))[:, 2].reshape(-1)[0].item()) if torch.is_tensor(joint_update_payload.get("mode_prob", None)) else float("nan"),
                                    "suffix_trigger_mlp_used": bool(trigger_payload.get("used", False)),
                                    "suffix_trigger_mlp_prob": float(trigger_payload.get("prob", float("nan"))),
                                    "suffix_trigger_mlp_threshold": trigger_threshold,
                                    "suffix_trigger_source": suffix_trigger_source,
                                }
                            else:
                                chunk_decision = decide_joint_execution_suffix(cfg, chunk_effect_payload, delta_action_t)
                            coupling_agg_info["chunk_effect_decision_count"] += 1
                            coupling_agg_info["chunk_effect_last_decision"] = str(chunk_decision["decision"])
                            coupling_last_info.update(
                                {
                                    "dynamic_coupling_chunk_decision": str(chunk_decision["decision"]),
                                    "dynamic_coupling_chunk_decision_reason": str(chunk_decision["decision_reason"]),
                                    "dynamic_coupling_chunk_residual_norm": float(chunk_decision["residual_norm"]),
                                    "dynamic_coupling_chunk_compat": float(chunk_decision["compat"]),
                                    "dynamic_coupling_chunk_success_prob": float(chunk_decision["success_prob"]),
                                    "joint_execution_update_used": True,
                                    "joint_execution_update_after_step": True,
                                    "joint_belief_transition_used": bool(joint_update_payload is not None),
                                    "joint_belief_transition_direct_suffix": bool(getattr(cfg, "joint_belief_transition_direct_suffix", False)),
                                    "joint_belief_transition_sparse_suffix": bool(getattr(cfg, "joint_belief_transition_sparse_suffix", False)),
                                    "joint_belief_transition_sparse_triggered": bool(chunk_decision.get("joint_belief_transition_sparse_triggered", False)),
                                    "joint_execution_error": float(chunk_decision.get("joint_execution_error", 0.0)),
                                    "joint_execution_gate": float(chunk_decision.get("joint_execution_gate", 0.0)),
                                }
                            )
                            joint_gate = float(chunk_decision.get("joint_execution_gate", 0.0) or 0.0)
                            if bool(getattr(cfg, "joint_belief_clean_controller", False)):
                                # Clean controller never injects h directly into the decoder
                                # during keep/patch, but h itself can still be updated as the
                                # persistent execution belief used by TCN/D decisions.
                                joint_gate *= float(
                                    np.clip(getattr(cfg, "joint_belief_clean_h_update_scale", 0.0), 0.0, 1.0)
                                )
                            prev_future_t = hypothesis_future_t.detach().clone()
                            if bool(getattr(cfg, "joint_belief_summary_future_adapter_update", False)):
                                adapter_future_t, adapter_runtime_info = apply_summary_adapter_to_runtime_future(
                                    cfg,
                                    base_future=prev_future_t,
                                    joint_update_payload=joint_update_payload,
                                    gate=joint_gate,
                                )
                                coupling_last_info.update(adapter_runtime_info)
                                if adapter_future_t is not None:
                                    corrected_future_t = adapter_future_t
                            if corrected_future_t is not None:
                                hypothesis_future_t = prev_future_t + joint_gate * (corrected_future_t.detach() - prev_future_t)
                                if not bool(getattr(cfg, "joint_belief_clean_controller", False)):
                                    model.future_feature_mode = "override"
                                    model.override_future_feature = hypothesis_future_t
                            if generated_action_intent_t is not None:
                                candidate_action_t = generated_action_intent_t.detach().to(model.device)
                                if hypothesis_action_t is None:
                                    prev_action_t = candidate_action_t.clone()
                                    hypothesis_action_t = candidate_action_t.clone()
                                else:
                                    prev_action_t = hypothesis_action_t.detach().clone()
                                    hypothesis_action_t = hypothesis_action_t + joint_gate * (candidate_action_t - hypothesis_action_t)
                                model._last_dynamic_coupling_action_intent_persist = hypothesis_action_t.detach().float().cpu()
                                if (
                                    bool(getattr(cfg, "dynamic_coupling_apply_action_intent_to_policy", False))
                                    and not bool(getattr(cfg, "joint_belief_clean_controller", False))
                                ):
                                    model.override_action_intent = hypothesis_action_t
                                action_delta_norm = float(torch.norm((hypothesis_action_t - prev_action_t).reshape(-1), p=2).item())
                            else:
                                action_delta_norm = 0.0
                            coupling_last_info.update(
                                {
                                    "persistent_hypothesis_used": True,
                                    "joint_execution_update_applied": True,
                                    "persistent_hypothesis_update_gate": joint_gate,
                                    "persistent_hypothesis_future_delta_norm": float(
                                        torch.norm((hypothesis_future_t - prev_future_t).reshape(-1), p=2).item()
                                    ),
                                    "persistent_hypothesis_action_delta_norm": action_delta_norm,
                                }
                            )
                            if (
                                (bool(getattr(cfg, "collect_sensitivity_teacher", False)) or suffix_ablation_needs_teacher)
                                and suffix_ablation_scheduled
                                and (
                                    hasattr(model, "project_sensitivity_delta_action")
                                    or hasattr(model, "project_paired_counterfactual_delta_action")
                                )
                                and jbt_future_before is not None
                                and jbt_action_before is not None
                                and hypothesis_future_t is not None
                                and hypothesis_action_t is not None
                            ):
                                # Teacher rows distill frozen GIDM finite-difference JVP:
                                # same suffix, only h_exp -> h_post changes.
                                teacher_info = suffix_ablation_teacher_info
                                if teacher_info is None:
                                    teacher_mode = str(getattr(cfg, "sensitivity_teacher_mode", "jvp") or "jvp")
                                    teacher_fn = (
                                        model.project_paired_counterfactual_delta_action
                                        if teacher_mode == "paired_counterfactual"
                                        and hasattr(model, "project_paired_counterfactual_delta_action")
                                        else model.project_sensitivity_delta_action
                                    )
                                    teacher_kwargs = {
                                        "action_intent_expected": jbt_action_before,
                                        "action_intent_posterior": hypothesis_action_t,
                                        "latent_goal_expected": jbt_future_before,
                                        "latent_goal_posterior": hypothesis_future_t,
                                        "joint_delta": (
                                            joint_update_payload.get("innovation")
                                            if isinstance(joint_update_payload, dict)
                                            else None
                                        ),
                                        "edit_step_size": float(getattr(cfg, "sensitivity_teacher_step_size", 1.0)),
                                        "max_abs_delta": float(getattr(cfg, "sensitivity_teacher_max_abs_delta", 0.0)),
                                    }
                                    if teacher_mode == "paired_counterfactual":
                                        teacher_kwargs["joint_delta_to_action_intent_scale"] = float(
                                            getattr(cfg, "sensitivity_teacher_joint_delta_to_action_scale", 1.0)
                                        )
                                        teacher_kwargs["condition_ablation"] = str(
                                            getattr(cfg, "sensitivity_teacher_condition_ablation", "joint") or "joint"
                                        )
                                    teacher_info = teacher_fn(**teacher_kwargs)
                                if isinstance(teacher_info, dict):
                                    coupling_last_info.update(
                                        {
                                            "sensitivity_teacher_available": bool(
                                                teacher_info.get("available", False)
                                            ),
                                            "sensitivity_teacher_reason": str(
                                                teacher_info.get("reason", "")
                                            ),
                                            "sensitivity_teacher_rank": int(
                                                teacher_info.get("rank", 0) or 0
                                            ),
                                            "sensitivity_teacher_delta_h_norm": float(
                                                teacher_info.get("delta_h_norm", 0.0) or 0.0
                                            ),
                                            "sensitivity_teacher_delta_action_norm": float(
                                                teacher_info.get("delta_action_norm", 0.0) or 0.0
                                            ),
                                            "sensitivity_teacher_suffix_delta_norm": float(
                                                teacher_info.get("suffix_delta_norm", 0.0) or 0.0
                                            ),
                                            "sensitivity_teacher_step_score_max": float(
                                                teacher_info.get("step_score_max", 0.0) or 0.0
                                            ),
                                            "sensitivity_teacher_mode": str(
                                                teacher_info.get("teacher_mode", "low_rank_jvp")
                                            ),
                                        }
                                    )
                                    maybe_save_sensitivity_teacher_row(
                                        cfg,
                                        sequence_index=int(getattr(cfg, "_runtime_sequence_index", -1)),
                                        subtask_index=int(subtask_index or 0),
                                        task=subtask,
                                        lang_text=lang_text,
                                        step=step,
                                        state_before=state_before_step,
                                        state_after=state_after_step,
                                        action=action_dbg,
                                        h_future_before=jbt_future_before,
                                        h_action_before=jbt_action_before,
                                        h_future_after=hypothesis_future_t,
                                        h_action_after=hypothesis_action_t,
                                        action_remain_before=jbt_action_remain_before,
                                        teacher=teacher_info,
                                        info=coupling_last_info,
                                    )
                            if (
                                chunk_decision["decision"] == "repair_suffix"
                                and delta_action_t is not None
                                and hasattr(model, "patch_remaining_action_seq")
                            ):
                                patch_mix = float(getattr(cfg, "dynamic_coupling_delta_action_repair_mix", 1.0))
                                if bool(getattr(cfg, "joint_belief_transition_direct_suffix", False)) and joint_update_payload is not None:
                                    # In direct transition mode, U's gate is the continuous suffix update magnitude.
                                    patch_mix *= joint_gate
                                transport_info = {"applied": False, "reason": "disabled"}
                                if bool(getattr(cfg, "linear_suffix_sensitivity_transport", False)) and hasattr(
                                    model,
                                    "sensitivity_transport_remaining_action_seq",
                                ):
                                    transport_info = model.sensitivity_transport_remaining_action_seq(
                                        action_intent_expected=jbt_action_before,
                                        action_intent_posterior=hypothesis_action_t,
                                        latent_goal_expected=jbt_future_before,
                                        latent_goal_posterior=hypothesis_future_t,
                                        edit_step_size=float(
                                            getattr(cfg, "linear_suffix_sensitivity_step_size", 1.0)
                                        ),
                                        max_abs_delta=float(
                                            getattr(cfg, "linear_suffix_sensitivity_max_abs_delta", 0.03)
                                        ),
                                        min_step_score=float(
                                            getattr(cfg, "linear_suffix_sensitivity_min_step_score", 0.0)
                                        ),
                                        selective_step_threshold=float(
                                            getattr(cfg, "linear_suffix_sensitivity_selective_step_threshold", 0.0)
                                        ),
                                        max_patch_fraction=float(
                                            getattr(cfg, "linear_suffix_sensitivity_max_patch_fraction", 1.0)
                                        ),
                                        replan_fraction=float(
                                            getattr(cfg, "linear_suffix_sensitivity_replan_fraction", 1.01)
                                        ),
                                        replan_delta_h_norm=float(
                                            getattr(cfg, "linear_suffix_sensitivity_replan_delta_h_norm", 0.0)
                                        ),
                                        smooth_weight=float(
                                            getattr(cfg, "linear_suffix_sensitivity_smooth_weight", 0.0)
                                        ),
                                        continuity_weight=float(
                                            getattr(cfg, "linear_suffix_sensitivity_continuity_weight", 0.0)
                                        ),
                                        mpc_iters=int(getattr(cfg, "linear_suffix_sensitivity_mpc_iters", 1)),
                                    )
                                    if not isinstance(transport_info, dict):
                                        transport_info = {
                                            "applied": False,
                                            "reason": "linear_suffix_transport_returned_none",
                                        }
                                if bool(getattr(cfg, "counterfactual_suffix_transport", False)) and hasattr(
                                    model,
                                    "counterfactual_transport_remaining_action_seq",
                                ) and not bool(transport_info.get("applied", False)):
                                    transport_info = model.counterfactual_transport_remaining_action_seq(
                                        action_intent_expected=jbt_action_before,
                                        action_intent_posterior=hypothesis_action_t,
                                        sigma=float(getattr(cfg, "counterfactual_suffix_transport_sigma", 0.5)),
                                        edit_steps=int(getattr(cfg, "counterfactual_suffix_transport_edit_steps", 1)),
                                        edit_step_size=float(getattr(cfg, "counterfactual_suffix_transport_step_size", 0.05)),
                                        max_abs_delta=float(getattr(cfg, "counterfactual_suffix_transport_max_abs_delta", 0.03)),
                                    )
                                    if not isinstance(transport_info, dict):
                                        transport_info = {
                                            "applied": False,
                                            "reason": "counterfactual_suffix_transport_returned_none",
                                        }
                                if (
                                    bool(coupling_last_info.get("directional_sensitivity_tcn_applied", False))
                                    and not bool(transport_info.get("applied", False))
                                ):
                                    patch_info = model.patch_remaining_action_seq(delta_action_t, mode="delta", mix=patch_mix)
                                    transport_info = {
                                        "applied": bool(patch_info.get("patched", False)),
                                        "transport_source": "directional_sensitivity_tcn",
                                        "reason": "directional_sensitivity_tcn_delta_patch",
                                    }
                                else:
                                    patch_info = None
                                if (
                                    str(transport_info.get("reason", ""))
                                    in {
                                        "sensitivity_affects_too_much_replan_preferred",
                                        "joint_hypothesis_shift_too_large_replan_preferred",
                                    }
                                    and hasattr(model, "request_full_replan_next_step")
                                    and not bool(getattr(cfg, "dynamic_coupling_disable_replan", False))
                                ):
                                    replan_info = model.request_full_replan_next_step()
                                    patch_info = {
                                        "patched": False,
                                        "mode": "selective_sensitivity_full_replan",
                                        "reason": "sensitivity_affects_too_much_replan_preferred",
                                        **replan_info,
                                    }
                                elif bool(transport_info.get("applied", False)):
                                    patch_info = {
                                        "patched": True,
                                        "mode": str(transport_info.get("transport_source", "counterfactual_velocity_transport")),
                                        "transport_reason": transport_info.get("reason", ""),
                                        **(patch_info or {}),
                                    }
                                elif (
                                    joint_update_payload is not None
                                    and str(coupling_last_info.get("joint_execution_update_runtime_type", ""))
                                    in {"rssm_plan_compatibility", "full_rssm_plan_compatibility"}
                                ):
                                    patch_info = model.patch_remaining_action_seq(
                                        delta_action_t,
                                        mode="delta",
                                        mix=patch_mix,
                                    )
                                    patch_info.update(
                                        {
                                            "reason": "rssm_plan_head_delta_patch",
                                            "transport_fallback_reason": transport_info.get("reason", ""),
                                        }
                                    )
                                else:
                                    patch_info = model.patch_remaining_action_seq(delta_action_t, mode="delta", mix=patch_mix)
                                coupling_last_info.update(
                                    {
                                        "dynamic_coupling_delta_action_patch_requested": True,
                                        "dynamic_coupling_delta_action_patch_written": bool(
                                            patch_info.get("patched", False)
                                        ),
                                        "dynamic_coupling_delta_action_patch_mix": patch_mix,
                                        "counterfactual_suffix_transport_enabled": bool(
                                            getattr(cfg, "counterfactual_suffix_transport", False)
                                        ),
                                        "linear_suffix_sensitivity_transport_enabled": bool(
                                            getattr(cfg, "linear_suffix_sensitivity_transport", False)
                                        ),
                                        **{
                                            f"counterfactual_suffix_transport_{k}": v
                                            for k, v in transport_info.items()
                                        },
                                        **{f"dynamic_coupling_delta_action_patch_{k}": v for k, v in patch_info.items()},
                                    }
                                )
                                if bool(patch_info.get("patched", False)):
                                    joint_belief_patch_count += 1
                                    joint_belief_last_patch_step = int(step)
                                    coupling_last_info.update(
                                        {
                                            "joint_belief_patch_count": int(joint_belief_patch_count),
                                            "joint_belief_last_patch_step": int(joint_belief_last_patch_step),
                                        }
                                    )
                            elif chunk_decision["decision"] == "replan":
                                model.pred_action_seq = None
                                model.rollout_step_counter = 0
                                coupling_last_info.update(
                                    {
                                        "dynamic_coupling_delta_action_patch_requested": False,
                                        "dynamic_coupling_delta_action_patch_written": False,
                                        "dynamic_coupling_full_replan_requested": True,
                                        "dynamic_coupling_replan_from_step": int(step),
                                    }
                                )
                            else:
                                coupling_last_info.update(
                                    {
                                        "dynamic_coupling_delta_action_patch_requested": False,
                                        "dynamic_coupling_delta_action_patch_written": False,
                                        "dynamic_coupling_full_replan_requested": False,
                                    }
                                )
                    maybe_save_joint_belief_transition_row(
                        cfg,
                        sequence_index=int(getattr(cfg, "_runtime_sequence_index", -1)),
                        subtask_index=int(subtask_index or 0),
                        task=subtask,
                        lang_text=lang_text,
                        step=step,
                        state_before=state_before_step,
                        state_after=state_after_step,
                        action=action_dbg,
                        h_future_before=jbt_future_before,
                        h_action_before=jbt_action_before,
                        h_future_after=hypothesis_future_t,
                        h_action_after=hypothesis_action_t,
                        action_remain_before=jbt_action_remain_before,
                        action_remain_after=(
                            model.get_remaining_action_seq(pad_to_full=True)
                            if hasattr(model, "get_remaining_action_seq")
                            else None
                        ),
                        info=coupling_last_info,
                    )
                    jbt_saved = True
                    model._last_dynamic_coupling_info = dict(coupling_last_info)
                    if coupling_last_info.get("dynamic_coupling_used", False):
                        coupling_agg_info["dynamic_coupling_used_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_used"] = True
                    if coupling_last_info.get("dynamic_coupling_applied", False):
                        coupling_agg_info["dynamic_coupling_applied_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_applied"] = True
                        if coupling_agg_info["dynamic_coupling_first_applied_step"] is None:
                            coupling_agg_info["dynamic_coupling_first_applied_step"] = int(step)
                        coupling_agg_info["dynamic_coupling_last_applied_step"] = int(step)
                        coupling_agg_info["dynamic_coupling_max_final_shift_norm"] = max(
                            float(coupling_agg_info["dynamic_coupling_max_final_shift_norm"]),
                            float(coupling_last_info.get("dynamic_coupling_final_shift_norm", 0.0) or 0.0),
                        )
                        model._last_dynamic_coupling_future_persist = hypothesis_future_t.detach().float().cpu()
                    if coupling_last_info.get("dynamic_coupling_action_generated", False):
                        coupling_agg_info["dynamic_coupling_action_generated_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_action_generated"] = True
                    if coupling_last_info.get("dynamic_coupling_delta_action_patch_requested", False):
                        coupling_agg_info["dynamic_coupling_delta_action_patch_count"] += 1
                        coupling_agg_info["dynamic_coupling_any_delta_action_patch"] = True
                    persist_info = dict(coupling_last_info)
                    persist_info.update(dict(getattr(model, "_last_joint_hypothesis_init_info", {}) or {}))
                    persist_info.update(coupling_agg_info)
                    model._last_dynamic_coupling_info_persist = persist_info
            if collect_jbt and not jbt_saved:
                shadow_info = dict(coupling_last_info or {})
                shadow_info.update(jbt_shadow_info)
                shadow_info.update({f"after_{k}": v for k, v in jbt_shadow_after_info.items()})
                shadow_info.update(
                    {
                        "joint_belief_transition_shadow_only": True,
                        "joint_execution_update_used": False,
                    }
                )
                maybe_save_joint_belief_transition_row(
                    cfg,
                    sequence_index=int(getattr(cfg, "_runtime_sequence_index", -1)),
                    subtask_index=int(subtask_index or 0),
                    task=subtask,
                    lang_text=lang_text,
                    step=step,
                    state_before=state_before_step,
                    state_after=state_after_step,
                    action=action_dbg,
                    h_future_before=jbt_future_before,
                    h_action_before=jbt_action_before,
                    h_future_after=jbt_future_after_shadow,
                    h_action_after=jbt_action_after_shadow,
                    action_remain_before=jbt_action_remain_before,
                    action_remain_after=(
                        model.get_remaining_action_seq(pad_to_full=True)
                        if hasattr(model, "get_remaining_action_seq")
                        else None
                    ),
                    info=shadow_info,
                )
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if collect_strong_gt:
                    payload = build_strong_gt_payload(
                        env,
                        task_oracle,
                        subtask,
                        strong_gt_start_raw,
                        start_info,
                        current_info,
                        True,
                        int(step + 1),
                    )
                    if bool(getattr(cfg, "collect_action_trace", False)):
                        payload["action_trace_path"] = maybe_save_action_trace(cfg, action_trace)
                        payload["action_trace_summary"] = action_trace_summary(action_trace)
                    if coupling_last_info:
                        payload["dynamic_coupling"] = coupling_last_info
                    return (
                        True,
                        int(step + 1),
                        payload,
                    )
                return True, int(step + 1)
        if collect_strong_gt:
            payload = build_strong_gt_payload(
                env,
                task_oracle,
                subtask,
                strong_gt_start_raw,
                start_info,
                last_info,
                False,
                int(cfg.ep_len),
            )
            if bool(getattr(cfg, "collect_action_trace", False)):
                payload["action_trace_path"] = maybe_save_action_trace(cfg, action_trace)
                payload["action_trace_summary"] = action_trace_summary(action_trace)
            if coupling_last_info:
                payload["dynamic_coupling"] = coupling_last_info
            return (
                False,
                int(cfg.ep_len),
                payload,
            )
        return False, int(cfg.ep_len)
    finally:
        if old_mode is not None:
            model.future_feature_mode = old_mode
        if old_override is not None:
            model.override_future_feature = old_override
        elif hasattr(model, "override_future_feature"):
            model.override_future_feature = None
        if old_action_intent is not None:
            model.override_action_intent = old_action_intent
        elif hasattr(model, "override_action_intent"):
            model.override_action_intent = None
        if hasattr(model, "_last_dynamic_coupling_action_intent_persist"):
            model._last_dynamic_coupling_action_intent_persist = None
        if old_dynamic_coupling_info is not None:
            model._last_dynamic_coupling_info = old_dynamic_coupling_info
        elif hasattr(model, "_last_dynamic_coupling_info"):
            model._last_dynamic_coupling_info = {}


def evaluate_sequence_memory(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    lang_embeddings,
    val_annotations,
    cfg,
    record,
    rollout_video,
    seq_idx,
    qwen_getter,
    adapter_getter,
    key_memory_rows,
    runtime_key_state_memory,
    logs,
    strong_gt_logs=None,
    online_acceptance_logs=None,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(seq_idx), caption=caption)

    success_counter = 0
    recent_memory = []
    preemptive_counterfactual = bool(getattr(cfg, "counterfactual_refinement", False))
    memory_factor_by_id = build_memory_factor_by_id(key_memory_rows)
    qwen_diagnosis_pending_rows = []
    strong_gt_pending_rows = []

    def queue_strong_gt(row, payload):
        if strong_gt_logs is None or not bool(getattr(cfg, "collect_strong_gt", False)):
            return
        strong_gt_row = {
            "sequence_index": int(row.get("sequence_index", seq_idx)),
            "subtask_index": int(row.get("subtask_index", -1)),
            "task": str(row.get("task", "")),
            "success": bool(row.get("success", False)),
            "steps": int(row.get("steps", -1)),
            "lang_text": row.get("lang_text", ""),
            "collection_trace_path": row.get("collection_trace_path"),
            "collection_trace_saved": bool(row.get("collection_trace_saved", False)),
            "base_future_norm": row.get("base_future_norm"),
            "final_shift_norm": row.get("final_shift_norm", row.get("base_shift_norm")),
            "source": row.get("source", "rollout"),
            **(payload or {}),
        }
        strong_gt_pending_rows.append(strong_gt_row)

    def queue_online_acceptance(row):
        if online_acceptance_logs is None or not bool(getattr(cfg, "collect_online_acceptance_rows", False)):
            return
        acceptance_row = maybe_make_online_acceptance_row(row)
        if acceptance_row is not None:
            online_acceptance_logs.append(acceptance_row)

    def flush_strong_gt_on_failure():
        if strong_gt_logs is None or not bool(getattr(cfg, "collect_strong_gt", False)):
            return
        for row in strong_gt_pending_rows:
            strong_gt_logs.append(row)
            if bool(getattr(cfg, "print_strong_gt_rows", False)):
                print(
                    "[strong-gt-row] " + json.dumps(
                        {
                            "seq": row["sequence_index"],
                            "sub": row["subtask_index"],
                            "task": row["task"],
                            "success": row["success"],
                            "steps": row["steps"],
                            "trace": row.get("collection_trace_path"),
                            "robot_delta": (row.get("strong_gt_robot_obs_delta") or {}).get("delta_norm"),
                            "scene_delta": (row.get("strong_gt_scene_obs_delta") or {}).get("delta_norm"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    def flush_strong_gt_on_full_success():
        if strong_gt_logs is None or not bool(getattr(cfg, "collect_strong_gt", False)):
            return
        if strong_gt_pending_rows:
            row = strong_gt_pending_rows[-1]
            strong_gt_logs.append(row)
            if bool(getattr(cfg, "print_strong_gt_rows", False)):
                print(
                    "[strong-gt-row] " + json.dumps(
                        {
                            "seq": row["sequence_index"],
                            "sub": row["subtask_index"],
                            "task": row["task"],
                            "success": row["success"],
                            "steps": row["steps"],
                            "trace": row.get("collection_trace_path"),
                            "full_success_last_only": True,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    def flush_qwen_diagnosis_pending(rows_to_keep):
        for pending in rows_to_keep:
            row = dict(pending["row"])
            trace_path = maybe_save_future_trace(cfg, pending["trace_payload"])
            row["collection_trace_path"] = trace_path
            row["collection_trace_saved"] = bool(trace_path)
            for strong_row in strong_gt_pending_rows:
                if (
                    int(strong_row.get("sequence_index", -1)) == int(row["sequence_index"])
                    and int(strong_row.get("subtask_index", -1)) == int(row["subtask_index"])
                ):
                    strong_row["collection_trace_path"] = trace_path
                    strong_row["collection_trace_saved"] = bool(trace_path)
            logs.append(row)
            queue_online_acceptance(row)
            if bool(getattr(cfg, "print_collection_rows", False)):
                print(
                    "[diagnosis-row] "
                    + json.dumps(
                        {
                            "seq": int(row["sequence_index"]),
                            "sub": int(row["subtask_index"]),
                            "task": row["task"],
                            "success": bool(row["success"]),
                            "steps": int(row["steps"]),
                            "component": row.get("qwen_diagnosis_wrong_component"),
                            "mechanism": row.get("qwen_diagnosis_failure_mechanism"),
                            "object": row.get("qwen_diagnosis_object"),
                            "time": [
                                row.get("qwen_diagnosis_time_start"),
                                row.get("qwen_diagnosis_time_end"),
                            ],
                            "trace": trace_path,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    for sub_idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()

        step_start_raw = snapshot_env_raw(env)
        base_lang_text = val_annotations[subtask][0]
        next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
        runtime_key_rows = retrieve_key_memory(runtime_key_state_memory, subtask, cfg.memory_key_topk)
        historical_key_rows = retrieve_key_memory(key_memory_rows, subtask, max(0, cfg.memory_key_topk - len(runtime_key_rows)))
        key_rows = (runtime_key_rows + historical_key_rows)[: cfg.memory_key_topk]
        recent_summary = summarize_recent_memory(recent_memory, cfg.memory_recent_k)
        key_summary = summarize_key_memory(key_rows)
        important_summary = summarize_important_key_memory(key_rows)
        rule_text = RULE_MEMORY.get(subtask, RULE_MEMORY["default"])
        if preemptive_counterfactual:
            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature, parse_memory_reflection_output

            decision_obs = env.get_obs()
            base_future = _call_preserving_rng(defi_future_feature, model, decision_obs, base_lang_text).detach().to(model.device)
            original_base_future = base_future
            base_future, immune_info = apply_immune_repulsion(cfg, base_future, subtask, sub_idx)
            base_future, energy_info = apply_future_energy_descent(cfg, base_future, subtask, sub_idx)
            if bool(getattr(cfg, "no_qwen_reflection", False)) and not bool(getattr(cfg, "joint_pair_retry_only", False)):
                reflection_row = {
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "lang_text": base_lang_text,
                    "failure_factor": infer_diagnosis_factor_from_task(subtask),
                    "_base_future_for_trigger": base_future,
                }
                trigger_info = joint_editor_trigger_decision(
                    cfg,
                    subtask,
                    key_rows,
                    reflection_row,
                    memory_factor_by_id,
                    base_future.device,
                )
                reflection_row.pop("_base_future_for_trigger", None)
                joint_pair_target_future, joint_pair_action_intent, joint_pair_info = (None, None, {})
                direct_joint_generate_only = joint_pair_memory_free_mode(cfg)
                run_joint_pair = bool(trigger_info.get("joint_editor_triggered", False)) or bool(
                    getattr(cfg, "joint_pair_without_trigger", False)
                )
                if run_joint_pair:
                    if bool(trigger_info.get("joint_editor_force_non_none", False)) and reflection_row["failure_factor"] in {"", "none", "unknown"}:
                        reflection_row["failure_factor"] = str(trigger_info.get("joint_editor_fallback_factor", "goal_completion"))
                    if not direct_joint_generate_only:
                        init_joint_action_intent = None
                        init_joint_info = {}
                        try:
                            init_joint_action_intent, init_joint_info = generate_initial_joint_action_intent(
                                cfg,
                                env,
                                model,
                                obs,
                                base_lang_text,
                                subtask,
                                sub_idx,
                                base_future,
                            )
                        except Exception as exc:
                            init_joint_info = {
                                "joint_hypothesis_init_used": False,
                                "joint_hypothesis_init_reason": f"precompute_error:{exc}",
                            }
                        joint_pair_target_future, joint_pair_action_intent, joint_pair_info = select_joint_pair_target_future(
                            cfg,
                            base_future,
                            subtask,
                            sub_idx,
                            current_state=_joint_pair_state_from_raw(step_start_raw),
                            next_task=next_task,
                            initial_action_intent=init_joint_action_intent,
                        )
                        if joint_pair_info:
                            joint_pair_info = {**init_joint_info, **joint_pair_info}
                    else:
                        joint_pair_info = {
                            "joint_pair_direct_generate_only": True,
                            "joint_pair_no_memory": True,
                            "joint_pair_selector_used": False,
                            "joint_pair_reason": "memory_free_generate_only",
                        }
                corrected_future = joint_pair_target_future if joint_pair_target_future is not None else base_future
                rollout_result = rollout_with_lang_text(
                    env,
                    model,
                    task_checker,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    base_lang_text,
                corrected_future,
                joint_pair_action_intent,
                collect_strong_gt=bool(getattr(cfg, "collect_strong_gt", False)),
                subtask_index=sub_idx,
                retry_mode=False,
            )
                if bool(getattr(cfg, "collect_strong_gt", False)):
                    success, steps, strong_gt_payload = rollout_result
                else:
                    success, steps = rollout_result
                    strong_gt_payload = None
                persisted_coupling_future = getattr(model, "_last_dynamic_coupling_future_persist", None)
                effective_final_future = (
                    persisted_coupling_future
                    if persisted_coupling_future is not None
                    else corrected_future
                )
                trace_payload = {
                    "original_base_future": original_base_future,
                    "base_future": base_future,
                    "final_future": effective_final_future,
                }
                if joint_pair_target_future is not None:
                    trace_payload["target_proxy_future"] = joint_pair_target_future
                if persisted_coupling_future is not None:
                    trace_payload["dynamic_coupling_final_future"] = persisted_coupling_future
                coupling_trace_info = dict(getattr(model, "_last_dynamic_coupling_info_persist", {}) or {})
                for key in (
                    "dynamic_coupling_used",
                    "dynamic_coupling_applied",
                    "dynamic_coupling_reason",
                    "dynamic_coupling_target_key",
                    "dynamic_coupling_committed_norm",
                    "dynamic_coupling_committed_scalar",
                    "dynamic_coupling_dneed_norm",
                    "dynamic_coupling_dt_norm",
                    "dynamic_coupling_need_scale",
                    "dynamic_coupling_final_shift_norm",
                    "dynamic_coupling_shift_norm_vs_slow",
                    "dynamic_coupling_shift_ratio_vs_slow",
                    "dynamic_coupling_raw_shift_norm_vs_slow",
                    "dynamic_coupling_shrunk",
                    "dynamic_coupling_shrink_scale",
                    "dynamic_coupling_vector_mode",
                    "dynamic_coupling_future_mix",
                ):
                    if key in coupling_trace_info:
                        trace_payload[key] = coupling_trace_info[key]
                trace_path = maybe_save_future_trace(cfg, trace_payload)
                coupling_row_info = dict(getattr(model, "_last_dynamic_coupling_info_persist", {}) or {})
                row = {
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": bool(success),
                    "steps": int(steps),
                    "used_reflection_retry": False,
                    "lang_text": base_lang_text,
                    "preemptive_counterfactual": True,
                    "no_qwen_reflection": True,
                    "joint_pair_rollout": True,
                    "collection_trace_path": trace_path,
                    "collection_trace_saved": bool(trace_path),
                    "base_future_norm": float(torch.norm(base_future.reshape(-1), p=2).item()),
                    "final_shift_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                    "failure_factor": reflection_row.get("failure_factor"),
                    "joint_pair_direct_generate_only": bool(direct_joint_generate_only),
                    "joint_pair_no_memory": bool(direct_joint_generate_only),
                    "joint_pair_retry_only": bool(getattr(cfg, "joint_pair_retry_only", False)),
                    **immune_info,
                    **energy_info,
                    **trigger_info,
                    **joint_pair_info,
                    **coupling_row_info,
                }
                logs.append(row)
                queue_online_acceptance(row)
                if bool(getattr(cfg, "collect_strong_gt", False)):
                    queue_strong_gt(row, strong_gt_payload)
                if record:
                    rollout_video.draw_outcome(success)
                if success:
                    success_counter += 1
                    continue
                return success_counter
            if bool(getattr(cfg, "channel_causal_patch", False)):
                corrected_future, channel_patch_info = apply_channel_causal_patch(cfg, base_future, subtask, sub_idx)
                rollout_result = rollout_with_lang_text(
                    env,
                    model,
                    task_checker,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    base_lang_text,
                    corrected_future,
                    collect_strong_gt=bool(getattr(cfg, "collect_strong_gt", False)),
                    subtask_index=sub_idx,
                )
                if bool(getattr(cfg, "collect_strong_gt", False)):
                    success, steps, strong_gt_payload = rollout_result
                else:
                    success, steps = rollout_result
                    strong_gt_payload = None
                trace_path = maybe_save_future_trace(
                    cfg,
                    {
                        "original_base_future": original_base_future,
                        "base_future": base_future,
                        "final_future": corrected_future,
                    },
                )
                row = {
                    "sequence_index": int(seq_idx),
                    "subtask_index": int(sub_idx),
                    "task": subtask,
                    "success": bool(success),
                    "steps": int(steps),
                    "used_reflection_retry": False,
                    "language_augmented": False,
                    "lang_text": base_lang_text,
                    "preemptive_counterfactual": True,
                    "channel_causal_patch_eval": True,
                    "collection_trace_path": trace_path,
                    "collection_trace_saved": bool(trace_path),
                    "base_future_norm": float(torch.norm(base_future.reshape(-1), p=2).item()),
                    "original_base_future_norm": float(torch.norm(original_base_future.reshape(-1), p=2).item()),
                    "final_shift_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                    **immune_info,
                    **energy_info,
                    **channel_patch_info,
                }
                logs.append(row)
                queue_online_acceptance(row)
                queue_strong_gt(row, strong_gt_payload)
                if bool(getattr(cfg, "print_collection_rows", False)):
                    print(
                        "[channel-patch-row] "
                        + json.dumps(
                            {
                                "seq": int(seq_idx),
                                "sub": int(sub_idx),
                                "task": subtask,
                                "success": bool(success),
                                "steps": int(steps),
                                "mode": channel_patch_info.get("channel_causal_patch_mode"),
                                "used": channel_patch_info.get("channel_causal_patch_used"),
                                "group": [
                                    channel_patch_info.get("channel_causal_patch_group_start"),
                                    channel_patch_info.get("channel_causal_patch_group_end"),
                                ],
                                "proxy_margin_gain": channel_patch_info.get("channel_causal_patch_margin_gain_proxy"),
                                "shift": channel_patch_info.get("channel_causal_patch_shift_norm"),
                                "trace": trace_path,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                state_row = make_state_memory_row(
                    f"R_seq{seq_idx}_chpatch{sub_idx}",
                    seq_idx,
                    sub_idx,
                    subtask,
                    bool(success),
                    int(steps),
                    "recent_state",
                )
                update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
                if record:
                    rollout_video.draw_outcome(success)
                if success:
                    success_counter += 1
                    continue
                flush_strong_gt_on_failure()
                return success_counter
            if bool(getattr(cfg, "qwen_diagnosis_only", False)):
                qwen = qwen_getter()
                diagnosis_prompt = build_qwen_failure_diagnosis_prompt(
                    subtask,
                    base_lang_text,
                    seq_idx,
                    sub_idx,
                    base_future,
                    key_rows,
                    recent_memory,
                    energy_info,
                )
                if qwen is None:
                    diagnosis_raw = ""
                    diagnosis_info = {
                        "qwen_diagnosis_wrong_component": infer_diagnosis_factor_from_task(subtask),
                        "qwen_diagnosis_edit_intent": "stabilize contact",
                        "qwen_diagnosis_failure_mechanism": rule_text,
                    }
                else:
                    diagnosis_raw = qwen.generate_reflection(
                        diagnosis_prompt,
                        max_new_tokens=int(getattr(cfg, "qwen_diagnosis_max_new_tokens", 128)),
                    )
                    diagnosis_info = parse_qwen_failure_diagnosis(diagnosis_raw)
                rollout_result = rollout_with_lang_text(
                    env,
                    model,
                    task_checker,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    base_lang_text,
                    base_future,
                    collect_strong_gt=bool(getattr(cfg, "collect_strong_gt", False)),
                    subtask_index=sub_idx,
                )
                if bool(getattr(cfg, "collect_strong_gt", False)):
                    success, steps, strong_gt_payload = rollout_result
                else:
                    success, steps = rollout_result
                    strong_gt_payload = None
                target_proxy_future = None
                if bool(success) and bool(getattr(cfg, "qwen_diagnosis_save_target_proxy", False)):
                    target_proxy_future = _call_preserving_rng(defi_future_feature, model, env.get_obs(), base_lang_text).detach().to(model.device)
                reflection_row = {
                    "sequence_index": int(seq_idx),
                    "subtask_index": int(sub_idx),
                    "task": subtask,
                    "success": bool(success),
                    "steps": int(steps),
                    "used_reflection_retry": False,
                    "language_augmented": False,
                    "lang_text": base_lang_text,
                    "preemptive_counterfactual": False,
                    "qwen_diagnosis_only": True,
                    "base_future_norm": float(torch.norm(base_future.reshape(-1), p=2).item()),
                    "original_base_future_norm": float(torch.norm(original_base_future.reshape(-1), p=2).item()),
                    "base_shift_norm": float(torch.norm((base_future - original_base_future).reshape(-1), p=2).item()),
                    "collection_trace_path": None,
                    "collection_trace_saved": False,
                    **diagnosis_info,
                    **immune_info,
                    **energy_info,
                }
                qwen_diagnosis_pending_rows.append(
                    {
                        "row": reflection_row,
                        "trace_payload": {
                            "original_base_future": original_base_future,
                            "base_future": base_future,
                            "final_future": base_future,
                            "target_proxy_future": target_proxy_future,
                        },
                    }
                )
                queue_strong_gt(reflection_row, strong_gt_payload)
                state_row = make_state_memory_row(
                    f"R_seq{seq_idx}_diag{sub_idx}",
                    seq_idx,
                    sub_idx,
                    subtask,
                    bool(success),
                    int(steps),
                    "recent_state",
                )
                update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
                if record:
                    rollout_video.draw_outcome(success)
                if success:
                    success_counter += 1
                    continue
                flush_qwen_diagnosis_pending(qwen_diagnosis_pending_rows)
                flush_strong_gt_on_failure()
                return success_counter
            if bool(getattr(cfg, "immune_only", False)):
                base_future, immune_mask_info = apply_immune_factor_mask(cfg, original_base_future, base_future, subtask)
                action_sensitivity_info = {}
                if bool(getattr(cfg, "action_sensitivity_probe", False)):
                    probe_goal = lang_embeddings.get_lang_goal(base_lang_text)
                    probe_goal["lang_text"] = base_lang_text
                    eps_values = [
                        float(x)
                        for x in str(getattr(cfg, "action_sensitivity_eps", "0,0.25,0.5,1.0,2.0")).split(",")
                        if str(x).strip()
                    ]
                    if not eps_values or abs(eps_values[0]) > 1e-12:
                        eps_values = [0.0] + eps_values
                    action_sensitivity_info = probe_action_sensitivity_to_future(
                        model,
                        decision_obs,
                        probe_goal,
                        original_base_future,
                        base_future - original_base_future,
                        eps_values,
                    )
                rollout_result = rollout_with_lang_text(
                    env,
                    model,
                    task_checker,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    base_lang_text,
                    base_future,
                    collect_strong_gt=bool(getattr(cfg, "collect_strong_gt", False)),
                    subtask_index=sub_idx,
                )
                if bool(getattr(cfg, "collect_strong_gt", False)):
                    success, steps, strong_gt_payload = rollout_result
                else:
                    success, steps = rollout_result
                    strong_gt_payload = None
                trace_path = maybe_save_future_trace(
                    cfg,
                    {
                        "original_base_future": original_base_future,
                        "base_future": original_base_future,
                        "final_future": base_future,
                        "edit_direction": base_future - original_base_future,
                        **(
                            observation_image_payload(decision_obs, prefix="decision_obs")
                            if bool(getattr(cfg, "save_trace_observation", False))
                            else {}
                        ),
                        **(
                            raw_rgb_payload(step_start_raw, prefix="raw_obs")
                            if bool(getattr(cfg, "save_trace_observation", False))
                            else {}
                        ),
                    },
                )
                reflection_row = {
                    "sequence_index": int(seq_idx),
                    "subtask_index": int(sub_idx),
                    "task": subtask,
                    "success": bool(success),
                    "steps": int(steps),
                    "used_reflection_retry": False,
                    "language_augmented": False,
                    "lang_text": base_lang_text,
                    "preemptive_counterfactual": False,
                    "immune_only": True,
                    "base_future_norm": float(torch.norm(base_future.reshape(-1), p=2).item()),
                    "original_base_future_norm": float(torch.norm(original_base_future.reshape(-1), p=2).item()),
                    "immune_base_shift_norm": float(torch.norm((base_future - original_base_future).reshape(-1), p=2).item()),
                    "collection_trace_path": trace_path,
                    "collection_trace_saved": bool(trace_path),
                    **action_sensitivity_info,
                    **immune_info,
                    **immune_mask_info,
                    **energy_info,
                }
                logs.append(reflection_row)
                queue_online_acceptance(reflection_row)
                queue_strong_gt(reflection_row, strong_gt_payload)
                if bool(getattr(cfg, "print_collection_rows", False)):
                    print(
                        "[collect-row] "
                        + json.dumps(
                            {
                                "seq": int(seq_idx),
                                "sub": int(sub_idx),
                                "task": subtask,
                                "success": bool(success),
                                "steps": int(steps),
                                "trace": trace_path,
                                "energy": reflection_row.get("future_energy_before"),
                                "energy_after": reflection_row.get("future_energy_after"),
                                "energy_reason": reflection_row.get("future_energy_reason"),
                                "energy_shift": reflection_row.get("future_energy_shift_norm"),
                                "energy_factor": reflection_row.get("future_energy_selected_factor"),
                                "immune_gate": reflection_row.get("immune_gate"),
                                "immune_shift": reflection_row.get("immune_base_shift_norm"),
                                "action_delta_eps_1": reflection_row.get("action_delta_eps_1p0"),
                                "action_delta_rel_eps_1": reflection_row.get("action_delta_rel_eps_1p0"),
                                "action_sensitivity_max_delta": reflection_row.get("action_sensitivity_max_delta"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                state_row = make_state_memory_row(
                    f"R_seq{seq_idx}_immune{sub_idx}",
                    seq_idx,
                    sub_idx,
                    subtask,
                    bool(success),
                    int(steps),
                    "recent_state",
                )
                update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
                if record:
                    rollout_video.draw_outcome(success)
                if success:
                    success_counter += 1
                    continue
                return success_counter
            qwen = qwen_getter()
            if getattr(cfg, "joint_future_editor_ckpt", ""):
                from policy_evaluation.oracle_hypothesis_rollout import MemoryReflectionDecision

                diagnosis_prompt = build_qwen_failure_diagnosis_prompt(
                    subtask,
                    base_lang_text,
                    seq_idx,
                    sub_idx,
                    base_future,
                    key_rows,
                    recent_memory,
                    energy_info=energy_info,
                )
                if qwen is None:
                    diagnosis_raw = ""
                    diagnosis_info = {
                        "qwen_diagnosis_wrong_component": infer_diagnosis_factor_from_task(subtask),
                        "qwen_diagnosis_edit_intent": "stabilize contact",
                        "qwen_diagnosis_failure_mechanism": rule_text,
                    }
                else:
                    diagnosis_raw = qwen.generate_reflection(
                        diagnosis_prompt,
                        max_new_tokens=int(getattr(cfg, "qwen_diagnosis_max_new_tokens", 128)),
                    )
                    diagnosis_info = parse_qwen_failure_diagnosis(diagnosis_raw)
                diagnosis_factor = str(diagnosis_info.get("qwen_diagnosis_wrong_component", "unknown") or "unknown")
                mismatch_type = infer_mismatch_type(subtask, False, int(cfg.ep_len), cfg.ep_len)
                decision = MemoryReflectionDecision(
                    trust_score=0.5 if diagnosis_factor not in {"", "none", "unknown"} else 0.85,
                    mismatch_type=mismatch_type,
                    correction_direction=str(diagnosis_info.get("qwen_diagnosis_edit_intent", "") or ""),
                    recoverability="repairable" if diagnosis_factor not in {"", "none", "unknown"} else "not_needed",
                    explanation=str(diagnosis_info.get("qwen_diagnosis_failure_mechanism", "") or ""),
                    raw_text=str(diagnosis_raw or ""),
                    hypothetical_failure=str(diagnosis_info.get("qwen_diagnosis_failure_mechanism", "") or ""),
                    counterfactual_future=str(diagnosis_info.get("qwen_diagnosis_edit_intent", "") or ""),
                    failure_factor=diagnosis_factor,
                    intervention=str(diagnosis_info.get("qwen_diagnosis_edit_intent", "") or ""),
                )
                reflection_row = make_reflection_row(
                    seq_idx,
                    sub_idx,
                    subtask,
                    False,
                    int(cfg.ep_len),
                    base_lang_text,
                    decision,
                    mismatch_type,
                    recent_summary,
                    key_summary,
                    important_summary,
                    rule_text,
                    baseline_success=None,
                    baseline_steps=None,
                )
                reflection_row.update(diagnosis_info)
                reflection_row["qwen_diagnosis_prompt"] = diagnosis_prompt[:4000]
                reflection_row["qwen_diagnosis_schema"] = "pre_rollout_wrong_component"
                reflection_row["failure_factor"] = diagnosis_factor
                reflection_row["_base_future_for_trigger"] = base_future
                trigger_info = joint_editor_trigger_decision(
                    cfg,
                    subtask,
                    key_rows,
                    reflection_row,
                    memory_factor_by_id,
                    base_future.device,
                )
                reflection_row.pop("_base_future_for_trigger", None)
                if bool(getattr(cfg, "joint_editor_risk_gate", False)):
                    if not bool(trigger_info.get("joint_editor_triggered", False)):
                        reflection_row["failure_factor"] = "none"
                    elif bool(trigger_info.get("joint_editor_force_non_none", False)) and reflection_row["failure_factor"] in {"", "none", "unknown"}:
                        reflection_row["failure_factor"] = str(trigger_info.get("joint_editor_fallback_factor", "goal_completion"))
                reflection_row.update(trigger_info)
            else:
                prompt = build_counterfactual_reflection_prompt(
                    subtask,
                    next_task,
                    base_lang_text,
                    recent_summary,
                    key_summary,
                    important_summary,
                    rule_text,
                )
                if qwen is None:
                    decision = make_rule_based_reflection_decision(
                        subtask,
                        False,
                        int(cfg.ep_len),
                        int(cfg.ep_len),
                        rule_text,
                    )
                else:
                    decision = parse_memory_reflection_output(qwen.generate_reflection(prompt, max_new_tokens=192))
                reflection_row = make_reflection_row(
                    seq_idx,
                    sub_idx,
                    subtask,
                    False,
                    int(cfg.ep_len),
                    base_lang_text,
                    decision,
                    infer_mismatch_type(subtask, False, int(cfg.ep_len), cfg.ep_len),
                    recent_summary,
                    key_summary,
                    important_summary,
                    rule_text,
                    baseline_success=None,
                    baseline_steps=None,
                )
                reflection_row["failure_factor"] = failure_factor_from_decision(decision)
            reflection_row["preemptive_counterfactual"] = True
            factor_intervention_allowed = str(reflection_row["failure_factor"]) not in {"", "none", "unknown"}
            proposal_lang_text = build_counterfactual_instruction(base_lang_text, reflection_row, important_summary)
            proposal_future = _call_preserving_rng(defi_future_feature, model, decision_obs, proposal_lang_text).detach().to(model.device)
            adapter_info = {
                "future_adapter_enabled": bool(cfg.future_adapter_ckpt),
                "future_adapter_used": False,
                "preemptive_counterfactual": True,
                "base_future_norm": float(torch.norm(base_future.reshape(-1), p=2).item()),
                "original_base_future_norm": float(torch.norm(original_base_future.reshape(-1), p=2).item()),
                "immune_base_shift_norm": float(torch.norm((base_future - original_base_future).reshape(-1), p=2).item()),
                "proposal_future_norm": float(torch.norm(proposal_future.reshape(-1), p=2).item()),
                **immune_info,
                **energy_info,
            }
            if getattr(cfg, "joint_future_editor_ckpt", ""):
                corrected_future, joint_info = apply_joint_future_editor(
                    cfg,
                    base_future,
                    subtask,
                    sub_idx,
                    reflection_row["failure_factor"],
                )
                adapter_info.update(joint_info)
                trace_path = maybe_save_future_trace(
                    cfg,
                    {
                        "base_future": base_future,
                        "proposal_future": proposal_future,
                        "final_future": corrected_future,
                    },
                )
                adapter_info.update(
                    {
                        "target_proxy_available": False,
                        "collection_trace_path": trace_path,
                        "collection_trace_saved": bool(trace_path),
                        "final_shift_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                    }
                )
                success, steps = rollout_with_lang_text(
                    env,
                    model,
                    task_checker,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    base_lang_text,
                    corrected_future,
                    subtask_index=sub_idx,
                )
                reflection_row["success"] = bool(success)
                reflection_row["steps"] = int(steps)
                reflection_row["used_reflection_retry"] = False
                reflection_row["language_augmented"] = True
                reflection_row["lang_text"] = base_lang_text
                reflection_row["counterfactual_lang_text"] = proposal_lang_text
                reflection_row["preemptive_counterfactual"] = True
                reflection_row.update(adapter_info)
                logs.append(reflection_row)
                queue_online_acceptance(reflection_row)
                if bool(getattr(cfg, "print_collection_rows", False)):
                    print(
                        "[joint-editor-row] "
                        + json.dumps(
                            {
                                "seq": int(seq_idx),
                                "sub": int(sub_idx),
                                "task": subtask,
                                "success": bool(success),
                                "steps": int(steps),
                                "factor": reflection_row["failure_factor"],
                                "component": reflection_row.get("qwen_diagnosis_wrong_component"),
                                "mechanism": reflection_row.get("qwen_diagnosis_failure_mechanism"),
                                "time_window": [
                                    reflection_row.get("qwen_diagnosis_time_start"),
                                    reflection_row.get("qwen_diagnosis_time_end"),
                                ],
                                "parse_ok": reflection_row.get("qwen_diagnosis_parse_ok"),
                                "triggered": reflection_row.get("joint_editor_triggered"),
                                "trigger_reason": reflection_row.get("joint_editor_trigger_reason"),
                                "risk_probability": reflection_row.get("joint_editor_risk_probability"),
                                "trigger_probability": reflection_row.get("joint_editor_trigger_probability"),
                                "trigger_threshold": reflection_row.get("joint_editor_trigger_threshold"),
                                "fallback_factor": reflection_row.get("joint_editor_fallback_factor"),
                                "memory_vote_factor": reflection_row.get("joint_editor_memory_vote_factor"),
                                "memory_vote_count": reflection_row.get("joint_editor_memory_vote_count"),
                                "used": bool(joint_info.get("joint_future_editor_used", False)),
                                "edit_norm": joint_info.get("joint_future_editor_edit_norm"),
                                "mask_mean": joint_info.get("joint_future_editor_mask_mean"),
                                "trace": trace_path,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                state_row = make_state_memory_row(
                    f"R_seq{seq_idx}_joint{sub_idx}",
                    seq_idx,
                    sub_idx,
                    subtask,
                    bool(success),
                    int(steps),
                    "recent_state",
                )
                update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
                if record:
                    rollout_video.draw_outcome(success)
                if success:
                    success_counter += 1
                    continue
                return success_counter
            corrected_future = base_future
            gated_future = base_future
            target_future = None
            gate_trace = {
                "gate": 0.0,
                "alignment": 0.0,
                "memory_support": 0.0,
                "residual_norm": float(torch.norm((proposal_future - base_future).reshape(-1), p=2).item()),
                "corrected_shift_norm": 0.0,
            }
            joint_pair_info = {}
            adapter_bundle = adapter_getter()
            if adapter_bundle is not None and bool(adapter_bundle.get("token_level", False)):
                memory_features, memory_sims, memory_ids, memory_mask = retrieve_adapter_memory_topk(
                    subtask,
                    proposal_future.detach().cpu().numpy().astype(np.float32),
                    adapter_bundle["memory_arrays"],
                    int(adapter_bundle["topk"]),
                )
                if memory_ids:
                    corrected_future, gate_trace = blend_counterfactual_future(
                        base_future,
                        proposal_future,
                        decision,
                        memory_sims,
                        memory_ids,
                    )
                    risk_probe_row = {
                        **reflection_row,
                        "adapter_memory_ids": list(memory_ids),
                        "adapter_memory_scores": [float(x) for x in memory_sims[: len(memory_ids)].tolist()],
                        "counterfactual_gate": float(gate_trace["gate"]),
                        "counterfactual_alignment": float(gate_trace["alignment"]),
                        "counterfactual_memory_support": float(gate_trace["memory_support"]),
                        "counterfactual_residual_norm": float(gate_trace["residual_norm"]),
                        "counterfactual_shift_norm": float(gate_trace["corrected_shift_norm"]),
                        "base_future_norm": adapter_info["base_future_norm"],
                        "proposal_future_norm": adapter_info["proposal_future_norm"],
                    }
                    risk_probability = predict_risk_probability(cfg, risk_probe_row, memory_factor_by_id, base_future.device)
                    if risk_probability is None:
                        allowed_edit, risk_gate_info = should_apply_risk_gated_intervention(
                            cfg,
                            decision,
                            reflection_row["failure_factor"],
                            gate_trace,
                        )
                    else:
                        threshold = float(getattr(cfg, "risk_detector_threshold", 0.5))
                        allowed_edit = bool(risk_probability >= threshold)
                        risk_gate_info = {
                            "risk_gated_intervention": True,
                            "risk_gate_allowed": allowed_edit,
                            "risk_gate_reason": "learned_detector_allow" if allowed_edit else "learned_detector_block",
                            "risk_detector_probability": float(risk_probability),
                            "risk_detector_threshold": threshold,
                            "risk_detector_ckpt": str(getattr(cfg, "risk_detector_ckpt", "") or ""),
                        }
                    if not factor_intervention_allowed:
                        corrected_future = base_future
                        gate_trace = dict(gate_trace)
                        gate_trace["gate"] = 0.0
                        gate_trace["corrected_shift_norm"] = 0.0
                        factor_mask_info = {
                            "factor_mask_mode": str(getattr(cfg, "factor_mask_mode", "global") or "global"),
                            "selected_failure_factor": str(reflection_row["failure_factor"]),
                            "applied_factor_mask": "factor_none_preserve",
                            "factor_mask_applied": False,
                        }
                        risk_gate_info = {
                            "risk_gated_intervention": True,
                            "risk_gate_allowed": False,
                            "risk_gate_reason": "factor_none_preserve",
                        }
                        cap_info = {"risk_gate_gate_capped": False, "risk_gate_effective_gate": 0.0}
                        allowed_edit = False
                    elif allowed_edit:
                        corrected_future, gate_trace, cap_info = cap_counterfactual_gate(
                            cfg,
                            base_future,
                            corrected_future,
                            gate_trace,
                        )
                        corrected_future, factor_mask_info = apply_factor_mask_to_future(
                            cfg,
                            base_future,
                            corrected_future,
                            reflection_row["failure_factor"],
                        )
                    else:
                        corrected_future = base_future
                        gate_trace = dict(gate_trace)
                        gate_trace["gate"] = 0.0
                        gate_trace["corrected_shift_norm"] = 0.0
                        factor_mask_info = {
                            "factor_mask_mode": str(getattr(cfg, "factor_mask_mode", "global") or "global"),
                            "selected_failure_factor": str(reflection_row["failure_factor"]),
                            "applied_factor_mask": "risk_gate_blocked",
                            "factor_mask_applied": False,
                        }
                        cap_info = {"risk_gate_gate_capped": False, "risk_gate_effective_gate": 0.0}
                    gated_future = corrected_future
                    target_future = get_memory_target_future(adapter_bundle, memory_ids, memory_sims, base_future.device)
                    target_action_intent = None
                    init_joint_action_intent = None
                    init_joint_info = {}
                    try:
                        init_joint_action_intent, init_joint_info = generate_initial_joint_action_intent(
                            cfg,
                            env,
                            model,
                            decision_obs,
                            base_lang_text,
                            subtask,
                            reflection_row.get("subtask_index", 0),
                            base_future,
                        )
                    except Exception as exc:
                        init_joint_info = {
                            "joint_hypothesis_init_used": False,
                            "joint_hypothesis_init_reason": f"precompute_error:{exc}",
                        }
                    joint_pair_target_future, joint_pair_action_intent, joint_pair_info = select_joint_pair_target_future(
                        cfg,
                        base_future,
                        subtask,
                        reflection_row.get("subtask_index", 0),
                        current_state=_joint_pair_state_from_raw(step_start_raw),
                        next_task=next_task,
                        initial_action_intent=init_joint_action_intent,
                    )
                    if joint_pair_info:
                        joint_pair_info = {**init_joint_info, **joint_pair_info}
                    if joint_pair_target_future is not None:
                        target_future = joint_pair_target_future
                    target_action_intent = joint_pair_action_intent if joint_pair_target_future is not None else None
                    if allowed_edit:
                        corrected_future, semantic_beam_info = select_semantic_future_beam(
                            cfg,
                            base_future,
                            proposal_future,
                            corrected_future,
                            target_future,
                            reflection_row["failure_factor"],
                        )
                        gated_future = corrected_future
                        device = next(model.parameters()).device
                        if adapter_bundle.get("factor_adapter") is not None:
                            factor = str(reflection_row["failure_factor"])
                            factor_to_idx = adapter_bundle.get("factor_to_idx", {})
                            allowed_factor_adapter_factors = {
                                item.strip()
                                for item in str(getattr(cfg, "factor_conditioned_allowed_factors", "goal_completion,drawer_slider_progress")).split(",")
                                if item.strip()
                            }
                            if factor not in factor_to_idx or factor not in allowed_factor_adapter_factors:
                                corrected_future = base_future
                                factor_mask_info.update(
                                    {
                                        "factor_conditioned_adapter_used": False,
                                        "factor_adapter_skip_reason": f"unsupported_factor:{factor}",
                                    }
                                )
                                adapter_specific_info = {
                                    "future_adapter_used": False,
                                    "adapter_type": "factor_conditioned",
                                    "future_adapter_skip_reason": f"unsupported_factor:{factor}",
                                    "adapter_delta_norm": 0.0,
                                }
                            else:
                                factor_idx = int(factor_to_idx[factor])
                                with torch.no_grad():
                                    corrected, adapter_aux = adapter_bundle["factor_adapter"](
                                        base_future.unsqueeze(0).to(device),
                                        proposal_future.unsqueeze(0).to(device),
                                        torch.tensor([factor_idx], device=device),
                                    )
                                corrected_future = corrected.squeeze(0).detach().to(model.device)
                                corrected_future, factor_adapter_mask_info = apply_factor_mask_to_future(
                                    cfg,
                                    base_future,
                                    corrected_future,
                                    reflection_row["failure_factor"],
                                )
                                factor_mask_info.update(
                                    {
                                        "factor_conditioned_adapter_used": True,
                                        "factor_adapter_factor": factor,
                                        "factor_adapter_delta_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                                        **{
                                            f"factor_adapter_{key}": value
                                            for key, value in factor_adapter_mask_info.items()
                                            if key not in factor_mask_info
                                        },
                                    }
                                )
                                adapter_specific_info = {
                                    "future_adapter_used": True,
                                    "adapter_type": "factor_conditioned",
                                    "adapter_gate_mean": float(adapter_aux["gate"].detach().mean().item()),
                                    "adapter_delta_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                                }
                        else:
                            qwen_weight_map = {memory_id: 1.0 / float(len(memory_ids)) for memory_id in memory_ids}
                            similarity_map = {memory_id: float(memory_sims[idx]) for idx, memory_id in enumerate(memory_ids)}
                            qwen_weights, sims, mask = pad_weight_inputs(
                                qwen_weight_map,
                                similarity_map,
                                memory_ids,
                                int(adapter_bundle["topk"]),
                            )
                            future_pooled = mean_pool_feature(proposal_future.detach().cpu().numpy().astype(np.float32))
                            with torch.no_grad():
                                _, memory_state, cal_aux = adapter_bundle["calibrator"](
                                    torch.from_numpy(future_pooled).unsqueeze(0).to(device),
                                    torch.from_numpy(memory_features).unsqueeze(0).to(device),
                                    qwen_weights.unsqueeze(0).to(device),
                                    sims.unsqueeze(0).to(device),
                                    mask.unsqueeze(0).to(device),
                                )
                                corrected, _, adapter_aux = adapter_bundle["adapter"](
                                    corrected_future.unsqueeze(0).to(device),
                                    memory_state,
                                )
                            corrected_future = corrected.squeeze(0).detach().to(model.device)
                            adapter_specific_info = {
                                "future_adapter_used": True,
                                "adapter_type": "memory_state_guided",
                                "adapter_weight_mean": float(cal_aux["aggregated_memory"].detach().mean().item()),
                                "adapter_gate_mean": float(adapter_aux["gate"].detach().mean().item()),
                                "adapter_delta_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                            }
                        verifier_row = {
                            **risk_probe_row,
                            **factor_mask_info,
                            **adapter_specific_info,
                        }
                        verifier_row.update(
                            {
                                "counterfactual_gate": float(gate_trace["gate"]),
                                "counterfactual_alignment": float(gate_trace["alignment"]),
                                "counterfactual_memory_support": float(gate_trace["memory_support"]),
                                "counterfactual_residual_norm": float(gate_trace["residual_norm"]),
                                "counterfactual_shift_norm": float(gate_trace["corrected_shift_norm"]),
                            }
                        )
                        corrected_future, safe_selector_info = apply_safe_intervention_selector(
                            cfg,
                            verifier_row,
                            base_future,
                            proposal_future,
                            corrected_future,
                            target_future,
                        )
                        adapter_specific_info.update(safe_selector_info)
                        if not safe_selector_info.get("safe_selector_allowed", True):
                            adapter_specific_info.update(
                                {
                                    "future_adapter_used": False,
                                    "future_adapter_skip_reason": "safe_selector_block",
                                    "adapter_delta_norm": 0.0,
                                }
                            )
                            verifier_row.update(adapter_specific_info)
                        verifier_probability, verifier_debug = predict_counterfactual_verifier_probability(
                            cfg,
                            verifier_row,
                            base_future,
                            proposal_future,
                            gated_future,
                            corrected_future,
                            target_future,
                            base_future.device,
                        )
                        if verifier_probability is not None:
                            verifier_threshold = float(getattr(cfg, "counterfactual_verifier_threshold", 0.8))
                            verifier_allowed = bool(verifier_probability >= verifier_threshold)
                            adapter_specific_info.update(
                                {
                                    "counterfactual_verifier_probability": float(verifier_probability),
                                    "counterfactual_verifier_threshold": verifier_threshold,
                                    "counterfactual_verifier_allowed": verifier_allowed,
                                    "counterfactual_verifier_ckpt": str(getattr(cfg, "counterfactual_verifier_ckpt", "") or ""),
                                    "counterfactual_verifier_final_cos_gain": float(verifier_debug.get("final_cos_gain", 0.0)),
                                    "counterfactual_verifier_logit": float(verifier_debug.get("counterfactual_verifier_logit", 0.0)),
                                    "counterfactual_verifier_temperature": float(verifier_debug.get("counterfactual_verifier_temperature", 1.0)),
                                }
                            )
                            if not verifier_allowed:
                                corrected_future = base_future
                                adapter_specific_info.update(
                                    {
                                        "future_adapter_used": False,
                                        "future_adapter_skip_reason": "counterfactual_verifier_block",
                                        "adapter_delta_norm": 0.0,
                                    }
                                )
                    else:
                        semantic_beam_info = {
                            "semantic_future_beam": bool(getattr(cfg, "semantic_future_beam", False)),
                            "semantic_beam_selected": "risk_gate_blocked",
                        }
                        adapter_specific_info = {
                            "future_adapter_used": False,
                            "future_adapter_skip_reason": "risk_gate_blocked",
                            "adapter_delta_norm": 0.0,
                        }
                    adapter_info.update(
                        {
                            "adapter_memory_ids": list(memory_ids),
                            "adapter_memory_scores": [float(x) for x in memory_sims[: len(memory_ids)].tolist()],
                            **(joint_pair_info if 'joint_pair_info' in locals() else {}),
                            **factor_mask_info,
                            **risk_gate_info,
                            **cap_info,
                            **semantic_beam_info,
                            **adapter_specific_info,
                        }
                    )
                else:
                    adapter_info["future_adapter_skip_reason"] = "no_memory_candidates"
            elif adapter_bundle is not None:
                adapter_info["future_adapter_skip_reason"] = "adapter_checkpoint_not_token_level"
            else:
                corrected_future, gate_trace = blend_counterfactual_future(
                    base_future,
                    proposal_future,
                    decision,
                    np.zeros((0,), dtype=np.float32),
                    [],
                )
                risk_probe_row = {
                    **reflection_row,
                    "adapter_memory_ids": [],
                    "adapter_memory_scores": [],
                    "counterfactual_gate": float(gate_trace["gate"]),
                    "counterfactual_alignment": float(gate_trace["alignment"]),
                    "counterfactual_memory_support": float(gate_trace["memory_support"]),
                    "counterfactual_residual_norm": float(gate_trace["residual_norm"]),
                    "counterfactual_shift_norm": float(gate_trace["corrected_shift_norm"]),
                    "base_future_norm": adapter_info["base_future_norm"],
                    "proposal_future_norm": adapter_info["proposal_future_norm"],
                }
                risk_probability = predict_risk_probability(cfg, risk_probe_row, memory_factor_by_id, base_future.device)
                if risk_probability is None:
                    allowed_edit, risk_gate_info = should_apply_risk_gated_intervention(
                        cfg,
                        decision,
                        reflection_row["failure_factor"],
                        gate_trace,
                    )
                else:
                    threshold = float(getattr(cfg, "risk_detector_threshold", 0.5))
                    allowed_edit = bool(risk_probability >= threshold)
                    risk_gate_info = {
                        "risk_gated_intervention": True,
                        "risk_gate_allowed": allowed_edit,
                        "risk_gate_reason": "learned_detector_allow" if allowed_edit else "learned_detector_block",
                        "risk_detector_probability": float(risk_probability),
                        "risk_detector_threshold": threshold,
                        "risk_detector_ckpt": str(getattr(cfg, "risk_detector_ckpt", "") or ""),
                    }
                if allowed_edit:
                    corrected_future, gate_trace, cap_info = cap_counterfactual_gate(
                        cfg,
                        base_future,
                        corrected_future,
                        gate_trace,
                    )
                    corrected_future, factor_mask_info = apply_factor_mask_to_future(
                        cfg,
                        base_future,
                        corrected_future,
                        reflection_row["failure_factor"],
                    )
                else:
                    corrected_future = base_future
                    gate_trace = dict(gate_trace)
                    gate_trace["gate"] = 0.0
                    gate_trace["corrected_shift_norm"] = 0.0
                    factor_mask_info = {
                        "factor_mask_mode": str(getattr(cfg, "factor_mask_mode", "global") or "global"),
                        "selected_failure_factor": str(reflection_row["failure_factor"]),
                        "applied_factor_mask": "risk_gate_blocked",
                        "factor_mask_applied": False,
                    }
                    cap_info = {"risk_gate_gate_capped": False, "risk_gate_effective_gate": 0.0}
                adapter_info.update({**factor_mask_info, **risk_gate_info, **cap_info})
                gated_future = corrected_future
            adapter_info.update(
                {
                    "counterfactual_gate": float(gate_trace["gate"]),
                    "counterfactual_alignment": float(gate_trace["alignment"]),
                    "counterfactual_memory_support": float(gate_trace["memory_support"]),
                    "counterfactual_residual_norm": float(gate_trace["residual_norm"]),
                    "counterfactual_shift_norm": float(gate_trace["corrected_shift_norm"]),
                }
            )
            if target_future is not None:
                trace_path = maybe_save_future_trace(
                    cfg,
                    {
                        "base_future": base_future,
                        "proposal_future": proposal_future,
                        "gated_future": gated_future,
                        "final_future": corrected_future,
                        "target_proxy_future": target_future,
                    },
                )
                adapter_info.update(
                    {
                        "target_proxy_available": True,
                        "target_proxy_trace_path": trace_path,
                        "base_to_target_cos": tensor_cosine(base_future, target_future),
                        "proposal_to_target_cos": tensor_cosine(proposal_future, target_future),
                        "gated_to_target_cos": tensor_cosine(gated_future, target_future),
                        "final_to_target_cos": tensor_cosine(corrected_future, target_future),
                        "proposal_direction_to_target_cos": tensor_cosine(proposal_future - base_future, target_future - base_future),
                        "gated_direction_to_target_cos": tensor_cosine(gated_future - base_future, target_future - base_future),
                        "final_direction_to_target_cos": tensor_cosine(corrected_future - base_future, target_future - base_future),
                    }
                )
            else:
                adapter_info["target_proxy_available"] = False

            success, steps = rollout_with_lang_text(
                env,
                model,
                task_checker,
                cfg,
                subtask,
                lang_embeddings,
                val_annotations,
                base_lang_text,
                corrected_future,
                target_action_intent if 'target_action_intent' in locals() else None,
                subtask_index=sub_idx,
            )
            reflection_row["success"] = bool(success)
            reflection_row["steps"] = int(steps)
            reflection_row["used_reflection_retry"] = False
            reflection_row["language_augmented"] = True
            reflection_row["lang_text"] = base_lang_text
            reflection_row["counterfactual_lang_text"] = proposal_lang_text
            reflection_row["preemptive_counterfactual"] = True
            reflection_row["counterfactual_gate"] = float(gate_trace["gate"])
            reflection_row["counterfactual_alignment"] = float(gate_trace["alignment"])
            reflection_row["counterfactual_memory_support"] = float(gate_trace["memory_support"])
            reflection_row.update(adapter_info)
            logs.append(reflection_row)
            queue_online_acceptance(reflection_row)

            state_row = make_state_memory_row(
                f"R_seq{seq_idx}_cf{sub_idx}",
                seq_idx,
                sub_idx,
                subtask,
                bool(success),
                int(steps),
                "recent_state",
            )
            update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
            if record:
                rollout_video.draw_outcome(success)
            if success:
                success_counter += 1
                continue
            return success_counter

        first_pass_coupling = bool(getattr(cfg, "dynamic_coupling_first_pass", False))
        # Shadow transition collection must use this rollout path because the
        # official baseline rollout has no hooks for h_t / action-buffer logging.
        shadow_transition_collection = bool(getattr(cfg, "collect_joint_belief_transition", False))
        if first_pass_coupling or shadow_transition_collection:
            success, first_pass_steps = rollout_with_lang_text(
                env,
                model,
                task_checker,
                cfg,
                subtask,
                lang_embeddings,
                val_annotations,
                base_lang_text,
                None,
                None,
                subtask_index=sub_idx,
                retry_mode=False,
            )
            coupling_row_info = dict(getattr(model, "_last_dynamic_coupling_info_persist", {}) or {})
            logs.append(
                {
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": bool(success),
                    "steps": int(first_pass_steps),
                    "used_reflection_retry": False,
                    "lang_text": base_lang_text,
                    "dynamic_coupling_first_pass": bool(first_pass_coupling),
                    "joint_belief_transition_shadow_collection": bool(shadow_transition_collection),
                    **coupling_row_info,
                }
            )
        else:
            success = original_rollout(
                env,
                model,
                task_checker,
                cfg,
                subtask,
                lang_embeddings,
                val_annotations,
                record,
                rollout_video,
            )
            first_pass_steps = -1
        if success:
            row = make_state_memory_row(
                f"R_seq{seq_idx}_sub{sub_idx}",
                seq_idx,
                sub_idx,
                subtask,
                True,
                int(first_pass_steps),
                "recent_state",
            )
            update_runtime_state_memory(recent_memory, runtime_key_state_memory, row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
            if not first_pass_coupling and not shadow_transition_collection:
                logs.append(
                    {
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": True,
                        "used_reflection_retry": False,
                        "lang_text": base_lang_text,
                    }
                )
            success_counter += 1
            if record:
                rollout_video.draw_outcome(True)
            continue

        if bool(getattr(cfg, "disable_retry_after_first_pass", False)):
            if record:
                rollout_video.draw_outcome(False)
            return success_counter

        retry_rollback_mode = str(getattr(cfg, "retry_rollback_mode", "subtask_start") or "subtask_start")
        if retry_rollback_mode == "subtask_start":
            reset_env_raw(env, step_start_raw)
        elif retry_rollback_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported retry_rollback_mode: {retry_rollback_mode}")
        if bool(getattr(cfg, "no_qwen_reflection", False)):
            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

            decision_obs = env.get_obs()
            base_future = _call_preserving_rng(defi_future_feature, model, decision_obs, base_lang_text).detach().to(model.device)
            reflection_row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "lang_text": base_lang_text,
                "failure_factor": infer_diagnosis_factor_from_task(subtask),
                "baseline_success": False,
                "baseline_steps": int(cfg.ep_len),
                "_base_future_for_trigger": base_future,
            }
            trigger_info = joint_editor_trigger_decision(
                cfg,
                subtask,
                key_rows,
                reflection_row,
                memory_factor_by_id,
                base_future.device,
            )
            reflection_row.pop("_base_future_for_trigger", None)
            joint_pair_target_future, joint_pair_action_intent, joint_pair_info = (None, None, {})
            direct_joint_generate_only = joint_pair_memory_free_mode(cfg)
            run_joint_pair = bool(trigger_info.get("joint_editor_triggered", False)) or bool(
                getattr(cfg, "joint_pair_without_trigger", False)
            )
            if run_joint_pair:
                if bool(trigger_info.get("joint_editor_force_non_none", False)) and reflection_row["failure_factor"] in {"", "none", "unknown"}:
                    reflection_row["failure_factor"] = str(trigger_info.get("joint_editor_fallback_factor", "goal_completion"))
                if not direct_joint_generate_only:
                    joint_pair_target_future, joint_pair_action_intent, joint_pair_info = select_joint_pair_target_future(
                        cfg,
                        base_future,
                        subtask,
                        sub_idx,
                        current_state=_joint_pair_state_from_raw(step_start_raw),
                        next_task=next_task,
                    )
                else:
                    current_state = _joint_pair_state_from_raw(step_start_raw)
                    init_action_intent, init_info = generate_initial_joint_action_intent(
                        cfg,
                        env,
                        model,
                        decision_obs,
                        base_lang_text,
                        subtask,
                        int(sub_idx or 0),
                        base_future,
                    )
                    joint_pair_target_future = base_future
                    joint_pair_action_intent = init_action_intent
                    joint_pair_info = {
                        "joint_pair_direct_generate_only": True,
                        "joint_pair_no_memory": True,
                        "joint_pair_selector_used": False,
                        "joint_pair_reason": "memory_free_gfdm_init",
                    }
                    joint_pair_info.update(init_info)
                    repaired_future, repaired_action, repair_info = maybe_apply_initial_slow_correction(
                        cfg,
                        task=subtask,
                        subtask_index=int(sub_idx or 0),
                        current_state=current_state,
                        base_future=base_future,
                        initial_future=base_future,
                        initial_action=joint_pair_action_intent,
                        target_transition=None,
                    )
                    joint_pair_info.update(repair_info)
                    if repaired_future is not None:
                        joint_pair_target_future = repaired_future
                    if repaired_action is not None:
                        joint_pair_action_intent = repaired_action
                    if bool(repair_info.get("joint_repair_advantage_used", False)):
                        if bool(repair_info.get("joint_repair_advantage_passed", False)):
                            joint_pair_info["joint_pair_reason"] = "memory_free_gfdm_slow_repair_accept"
                        else:
                            joint_pair_info["joint_pair_reason"] = str(
                                repair_info.get("joint_repair_advantage_reason", "memory_free_gfdm_slow_repair_reject")
                            )
            corrected_future = joint_pair_target_future if joint_pair_target_future is not None else base_future
            retry_max_attempts = max(1, int(getattr(cfg, "retry_max_attempts", 1)))
            retry_success = False
            retry_steps = -1
            retry_patch_count = 0
            retry_attempt_count = 0
            retry_attempt_patch_counts = []
            retry_attempt_successes = []
            retry_attempt_steps = []
            retry_coupling_info = {}
            for retry_attempt_idx in range(retry_max_attempts):
                if retry_attempt_idx > 0:
                    if retry_rollback_mode == "subtask_start":
                        reset_env_raw(env, step_start_raw)
                    elif retry_rollback_mode == "none":
                        pass
                    else:
                        raise ValueError(f"Unsupported retry_rollback_mode: {retry_rollback_mode}")
                retry_attempt_count += 1
                attempt_success, attempt_steps = rollout_with_lang_text(
                    env,
                    model,
                    task_checker,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    base_lang_text,
                    corrected_future,
                    joint_pair_action_intent,
                    subtask_index=sub_idx,
                    retry_mode=True,
                )
                attempt_coupling_info = dict(getattr(model, "_last_dynamic_coupling_info_persist", {}) or {})
                attempt_patch_count = int(
                    attempt_coupling_info.get("dynamic_coupling_delta_action_patch_count", 0) or 0
                )
                retry_attempt_patch_counts.append(int(attempt_patch_count))
                retry_attempt_successes.append(bool(attempt_success))
                retry_attempt_steps.append(int(attempt_steps))
                retry_patch_count += int(attempt_patch_count)
                retry_coupling_info = attempt_coupling_info
                retry_success = bool(attempt_success)
                retry_steps = int(attempt_steps)
                if retry_success:
                    break
            retry_fallback_used = False
            retry_fallback_success = False
            retry_fallback_steps = -1
            retry_fallback_reason = ""
            if (
                not bool(retry_success)
                and bool(getattr(cfg, "retry_baseline_fallback_on_failure", False))
            ):
                retry_fallback_used = True
                retry_fallback_reason = "patch_retry_failed_then_baseline_retry"
                if retry_rollback_mode == "subtask_start":
                    reset_env_raw(env, step_start_raw)
                elif retry_rollback_mode == "none":
                    pass
                else:
                    raise ValueError(f"Unsupported retry_rollback_mode: {retry_rollback_mode}")
                retry_fallback_success = bool(
                    original_rollout(
                        env,
                        model,
                        task_checker,
                        cfg,
                        subtask,
                        lang_embeddings,
                        val_annotations,
                        record,
                        rollout_video,
                    )
                )
                retry_fallback_steps = -1
                if retry_fallback_success:
                    retry_success = True
            retry_requires_suffix = bool(getattr(cfg, "retry_requires_suffix_patch", False))
            retry_effective_success = bool(retry_success)
            # A retry that succeeds while the controller decides to keep the
            # suffix is still a valid recovery. Keep retry_requires_suffix_patch
            # as a diagnostic flag only; do not discard raw retry success.
            retry_skipped_no_suffix_patch = bool(retry_requires_suffix and retry_patch_count <= 0)
            persisted_coupling_future = getattr(model, "_last_dynamic_coupling_future_persist", None)
            effective_final_future = (
                persisted_coupling_future
                if persisted_coupling_future is not None
                else corrected_future
            )
            trace_payload = {"base_future": base_future, "final_future": effective_final_future}
            if joint_pair_target_future is not None:
                trace_payload["target_proxy_future"] = joint_pair_target_future
            if persisted_coupling_future is not None:
                trace_payload["dynamic_coupling_final_future"] = persisted_coupling_future
            coupling_trace_info = dict(getattr(model, "_last_dynamic_coupling_info_persist", {}) or {})
            for key in (
                "dynamic_coupling_used",
                "dynamic_coupling_applied",
                "dynamic_coupling_reason",
                "dynamic_coupling_target_key",
                "dynamic_coupling_committed_norm",
                "dynamic_coupling_committed_scalar",
                "dynamic_coupling_dneed_norm",
                "dynamic_coupling_dt_norm",
                "dynamic_coupling_need_scale",
                "dynamic_coupling_final_shift_norm",
                "dynamic_coupling_shift_norm_vs_slow",
                "dynamic_coupling_shift_ratio_vs_slow",
                "dynamic_coupling_raw_shift_norm_vs_slow",
                "dynamic_coupling_shrunk",
                "dynamic_coupling_shrink_scale",
                "dynamic_coupling_vector_mode",
                "dynamic_coupling_future_mix",
            ):
                if key in coupling_trace_info:
                    trace_payload[key] = coupling_trace_info[key]
            trace_path = maybe_save_future_trace(cfg, trace_payload)
            reflection_row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "success": bool(retry_effective_success),
                "steps": int(retry_steps),
                "used_reflection_retry": True,
                "retry_raw_success": bool(retry_success),
                "retry_effective_success": bool(retry_effective_success),
                "retry_rollback_mode": str(retry_rollback_mode),
                "retry_requires_suffix_patch": bool(retry_requires_suffix),
                "retry_suffix_patch_count": int(retry_patch_count),
                "retry_max_attempts": int(retry_max_attempts),
                "retry_attempt_count": int(retry_attempt_count),
                "retry_attempt_patch_counts": retry_attempt_patch_counts,
                "retry_attempt_successes": retry_attempt_successes,
                "retry_attempt_steps": retry_attempt_steps,
                "retry_baseline_fallback_used": bool(retry_fallback_used),
                "retry_baseline_fallback_success": bool(retry_fallback_success),
                "retry_baseline_fallback_steps": int(retry_fallback_steps),
                "retry_baseline_fallback_reason": str(retry_fallback_reason),
                "retry_skipped_no_suffix_patch": bool(retry_skipped_no_suffix_patch),
                "retry_no_suffix_patch_preserved": bool(retry_skipped_no_suffix_patch and retry_success),
                "lang_text": base_lang_text,
                "language_augmented": False,
                "no_qwen_reflection": True,
                "joint_pair_rollout": True,
                "joint_pair_direct_generate_only": bool(direct_joint_generate_only),
                "joint_pair_no_memory": bool(direct_joint_generate_only),
                "joint_pair_retry_only": bool(getattr(cfg, "joint_pair_retry_only", False)),
                "failure_factor": reflection_row.get("failure_factor"),
                "baseline_success": False,
                "baseline_steps": int(cfg.ep_len),
                "collection_trace_path": trace_path,
                "collection_trace_saved": bool(trace_path),
                "base_future_norm": float(torch.norm(base_future.reshape(-1), p=2).item()),
                "final_shift_norm": float(torch.norm((corrected_future - base_future).reshape(-1), p=2).item()),
                **(dict(getattr(model, "_last_dynamic_coupling_info_persist", {}) or {})),
                **trigger_info,
                **joint_pair_info,
            }
            logs.append(reflection_row)
            queue_online_acceptance(reflection_row)
            if bool(getattr(cfg, "print_collection_rows", False)):
                print(
                    "[collect-row] "
                    + json.dumps(
                        {
                            "seq": int(seq_idx),
                            "sub": int(sub_idx),
                            "task": subtask,
                            "success": bool(retry_success),
                            "steps": int(retry_steps),
                            "trace": trace_path,
                            "joint_reason": reflection_row.get("joint_pair_reason"),
                            "repair_reason": reflection_row.get("joint_repair_advantage_reason"),
                            "repair_passed": reflection_row.get("joint_repair_advantage_passed"),
                            "repair_delta": reflection_row.get("joint_repair_advantage_pred"),
                            "coupling_used": reflection_row.get("dynamic_coupling_used"),
                            "coupling_applied": reflection_row.get("dynamic_coupling_applied"),
                            "coupling_reason": reflection_row.get("dynamic_coupling_reason"),
                            "coupling_target_key": reflection_row.get("dynamic_coupling_target_key"),
                            "coupling_committed_norm": reflection_row.get("dynamic_coupling_committed_norm"),
                            "coupling_dneed_norm": reflection_row.get("dynamic_coupling_dneed_norm"),
                            "coupling_shift_norm": reflection_row.get("dynamic_coupling_final_shift_norm"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            state_row = make_state_memory_row(
                f"R_seq{seq_idx}_joint{sub_idx}",
                seq_idx,
                sub_idx,
                subtask,
                bool(retry_effective_success),
                int(retry_steps),
                "recent_state",
            )
            update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)
            if record:
                rollout_video.draw_outcome(retry_effective_success)
            if retry_effective_success:
                success_counter += 1
                continue
            return success_counter
        mismatch_type = infer_mismatch_type(subtask, False, int(cfg.ep_len), cfg.ep_len)
        prompt = build_postexec_reflection_prompt(
            subtask,
            next_task,
            False,
            int(cfg.ep_len),
            mismatch_type,
            recent_summary,
            key_summary,
            important_summary,
            rule_text,
        )
        from policy_evaluation.oracle_hypothesis_rollout import parse_memory_reflection_output

        qwen = qwen_getter()
        if qwen is None:
            decision = make_rule_based_reflection_decision(
                subtask,
                False,
                int(cfg.ep_len),
                int(cfg.ep_len),
                rule_text,
            )
        else:
            decision = parse_memory_reflection_output(qwen.generate_reflection(prompt, max_new_tokens=192))
        reflection_row = make_reflection_row(
            seq_idx,
            sub_idx,
            subtask,
            False,
            int(cfg.ep_len),
            base_lang_text,
            decision,
            mismatch_type,
            recent_summary,
            key_summary,
            important_summary,
            rule_text,
            baseline_success=False,
            baseline_steps=int(cfg.ep_len),
        )
        retry_lang_text = build_augmented_instruction(base_lang_text, reflection_row, important_summary)
        adapter_info = {"future_adapter_enabled": bool(cfg.future_adapter_ckpt), "future_adapter_used": False}
        corrected_future = None
        adapter_bundle = adapter_getter()
        if adapter_bundle is not None and bool(adapter_bundle.get("token_level", False)):
            from policy_evaluation.oracle_hypothesis_rollout import defi_future_feature

            decision_obs = env.get_obs()
            original_future = _call_preserving_rng(defi_future_feature, model, decision_obs, retry_lang_text).detach().to(model.device)
            original_future_np = original_future.detach().cpu().numpy().astype(np.float32)
            memory_features, memory_sims, memory_ids, memory_mask = retrieve_adapter_memory_topk(
                subtask,
                original_future_np,
                adapter_bundle["memory_arrays"],
                int(adapter_bundle["topk"]),
            )
            if memory_ids:
                qwen_weight_map = {memory_id: 1.0 / float(len(memory_ids)) for memory_id in memory_ids}
                similarity_map = {memory_id: float(memory_sims[idx]) for idx, memory_id in enumerate(memory_ids)}
                qwen_weights, sims, mask = pad_weight_inputs(
                    qwen_weight_map,
                    similarity_map,
                    memory_ids,
                    int(adapter_bundle["topk"]),
                )
                future_pooled = mean_pool_feature(original_future_np)
                device = next(model.parameters()).device
                with torch.no_grad():
                    _, memory_state, cal_aux = adapter_bundle["calibrator"](
                        torch.from_numpy(future_pooled).unsqueeze(0).to(device),
                        torch.from_numpy(memory_features).unsqueeze(0).to(device),
                        qwen_weights.unsqueeze(0).to(device),
                        sims.unsqueeze(0).to(device),
                        mask.unsqueeze(0).to(device),
                    )
                    corrected, _, adapter_aux = adapter_bundle["adapter"](
                        original_future.unsqueeze(0).to(device),
                        memory_state,
                    )
                corrected_future = corrected.squeeze(0).detach().to(model.device)
                adapter_info.update(
                    {
                        "future_adapter_used": True,
                        "adapter_memory_ids": list(memory_ids),
                        "adapter_memory_scores": [float(x) for x in memory_sims[: len(memory_ids)].tolist()],
                        "adapter_weight_mean": float(cal_aux["aggregated_memory"].detach().mean().item()),
                        "adapter_gate_mean": float(adapter_aux["gate"].detach().mean().item()),
                        "adapter_delta_norm": float(torch.norm((corrected_future - original_future).reshape(-1), p=2).item()),
                    }
                )
            else:
                adapter_info["future_adapter_skip_reason"] = "no_memory_candidates"
        elif adapter_bundle is not None:
            adapter_info["future_adapter_skip_reason"] = "adapter_checkpoint_not_token_level"
        retry_success, retry_steps = rollout_with_lang_text(
            env,
            model,
            task_checker,
            cfg,
            subtask,
            lang_embeddings,
            val_annotations,
            retry_lang_text,
            corrected_future,
            joint_pair_action_intent if 'joint_pair_action_intent' in locals() else None,
            subtask_index=sub_idx,
            retry_mode=True,
        )
        reflection_row["success"] = bool(retry_success)
        reflection_row["steps"] = int(retry_steps)
        reflection_row["used_reflection_retry"] = True
        reflection_row["language_augmented"] = True
        reflection_row["lang_text"] = retry_lang_text
        reflection_row.update(adapter_info)
        if not joint_pair_info:
            init_joint_action_intent = None
            init_joint_info = {}
            try:
                init_joint_action_intent, init_joint_info = generate_initial_joint_action_intent(
                    cfg,
                    env,
                    model,
                    obs,
                    retry_lang_text,
                    subtask,
                    reflection_row.get("subtask_index", 0),
                    base_future,
                )
            except Exception as exc:
                init_joint_info = {
                    "joint_hypothesis_init_used": False,
                    "joint_hypothesis_init_reason": f"precompute_error:{exc}",
                }
            joint_pair_target_future, joint_pair_action_intent, joint_pair_info = select_joint_pair_target_future(
                cfg,
                base_future,
                subtask,
                reflection_row.get("subtask_index", 0),
                current_state=_joint_pair_state_from_raw(step_start_raw),
                next_task=next_task,
                initial_action_intent=init_joint_action_intent,
            )
            if joint_pair_info:
                joint_pair_info = {**init_joint_info, **joint_pair_info}
            if joint_pair_target_future is not None and corrected_future is not None:
                corrected_future = joint_pair_target_future
        reflection_row.update(joint_pair_info)
        logs.append(reflection_row)
        queue_online_acceptance(reflection_row)

        state_row = make_state_memory_row(
            f"R_seq{seq_idx}_repair{sub_idx}",
            seq_idx,
            sub_idx,
            subtask,
            bool(retry_success),
            int(retry_steps),
            "recent_state",
        )
        update_runtime_state_memory(recent_memory, runtime_key_state_memory, state_row, cfg.memory_recent_k, cfg.memory_runtime_key_max)

        if record:
            rollout_video.draw_outcome(retry_success)
        if retry_success:
            success_counter += 1
            continue
        return success_counter

    if bool(getattr(cfg, "qwen_diagnosis_only", False)) and qwen_diagnosis_pending_rows:
        # For fully successful 5-step sequences, keep only the final subtask trace.
        # Failure cases are flushed immediately above with all executed subtasks.
        flush_qwen_diagnosis_pending(qwen_diagnosis_pending_rows[-1:])
    flush_strong_gt_on_full_success()
    return success_counter


def evaluate_policy_memory(model, env, lang_embeddings, cfg, num_procs, procs_id, checkpoint, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    rollout_video = RolloutVideo(
        logger=logger,
        empty_cache=False,
        log_to_file=True,
        save_dir=save_dir,
        resolution_scale=1,
    )

    eval_sequences = cfg.eval_sequences if "eval_sequences" in cfg and cfg.eval_sequences is not None else get_sequences(cfg.num_sequences)
    eval_sequences = filter_eval_sequences_by_task(eval_sequences, getattr(cfg, "eval_task_filter", ""))
    num_seq_per_procs = len(eval_sequences) // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs * procs_id : num_seq_per_procs * (procs_id + 1)]
    record_sequence_index = int(getattr(cfg, "record_sequence_index", -1))
    record_video = bool(getattr(cfg, "record_video", False))

    key_memory_rows = load_key_memory_rows(Path(cfg.memory_key_path)) if cfg.memory_key_path else []
    runtime_key_state_memory = []
    progress_path = Path(save_dir) / "sequence_progress.jsonl" if save_dir is not None else None
    partial_results_path = Path(save_dir) / "partial_results.json" if save_dir is not None else None

    class StreamingJsonlLogs(list):
        def __init__(self, path):
            super().__init__()
            self._handle = None
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = path.open("w")

        def append(self, row):
            super().append(row)
            if self._handle is not None:
                self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._handle.flush()

        def close(self):
            if self._handle is not None:
                self._handle.close()
                self._handle = None

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
        horizon = max((len(sequence) for _, sequence in eval_sequences), default=5)
        payload = {
            "checkpoint": str(checkpoint),
            "num_sequences_total": len(eval_sequences),
            "num_sequences_finished": len(results),
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": count_success_upto(results, horizon) if results else [],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with partial_results_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()

    logs = StreamingJsonlLogs(Path(save_dir) / "memory_rollout_rows.jsonl" if save_dir is not None else None)
    strong_gt_logs = StreamingJsonlLogs(
        Path(save_dir) / "strong_gt_rows.jsonl"
        if save_dir is not None and bool(getattr(cfg, "collect_strong_gt", False))
        else None
    )
    online_acceptance_logs = StreamingJsonlLogs(
        Path(save_dir) / "online_acceptance_rows.jsonl"
        if save_dir is not None and bool(getattr(cfg, "collect_online_acceptance_rows", False))
        else None
    )
    if save_dir is not None and bool(getattr(cfg, "collect_joint_belief_transition", False)):
        with open_dict(cfg):
            cfg.joint_belief_transition_rows_path = str(Path(save_dir) / "joint_belief_transition_rows.jsonl")
            cfg.joint_belief_transition_trace_dir = str(Path(save_dir) / "joint_belief_transition")
            cfg._joint_belief_transition_count = 0
        print(
            "[JBT_COLLECT] enabled "
            f"rows={cfg.joint_belief_transition_rows_path} "
            f"trace_dir={cfg.joint_belief_transition_trace_dir}",
            flush=True,
        )
    if save_dir is not None and bool(getattr(cfg, "collect_sensitivity_teacher", False)):
        with open_dict(cfg):
            cfg.sensitivity_teacher_rows_path = str(Path(save_dir) / "sensitivity_teacher_rows.jsonl")
            cfg.sensitivity_teacher_trace_dir = str(Path(save_dir) / "sensitivity_teacher")
            cfg._sensitivity_teacher_count = 0
        print(
            "[SENS_TEACHER] enabled "
            f"rows={cfg.sensitivity_teacher_rows_path} "
            f"trace_dir={cfg.sensitivity_teacher_trace_dir}",
            flush=True,
        )
    qwen = None
    adapter_bundle = None

    def get_qwen():
        nonlocal qwen
        if bool(getattr(cfg, "no_qwen_reflection", False)):
            return None
        qwen_model_path = str(getattr(cfg, "qwen_model_path", "") or "").strip()
        if not qwen_model_path or qwen_model_path in {".", ".."}:
            return None
        if qwen is None:
            from policy_evaluation.oracle_hypothesis_rollout import QwenReflectionEncoder

            device = next(model.parameters()).device
            qwen = QwenReflectionEncoder(
                Path(qwen_model_path),
                device,
                lora_path=Path(cfg.qwen_lora_path) if cfg.qwen_lora_path else None,
                python_bin=Path(cfg.qwen_python_bin) if cfg.qwen_python_bin else None,
            )
        return qwen

    def get_adapter():
        nonlocal adapter_bundle
        if not cfg.future_adapter_ckpt and not getattr(cfg, "factor_conditioned_adapter_ckpt", ""):
            return None
        if adapter_bundle is None:
            if not cfg.memory_npz_path:
                raise ValueError("--future_adapter_ckpt/--factor_conditioned_adapter_ckpt requires --memory_npz_path")
            memory_arrays = load_memory_arrays(Path(cfg.memory_npz_path))
            device = next(model.parameters()).device
            calibrator = None
            adapter = None
            adapter_ckpt = {}
            if cfg.future_adapter_ckpt:
                if not cfg.calibrator_ckpt:
                    raise ValueError("--future_adapter_ckpt requires --calibrator_ckpt")
                calibrator_ckpt = torch.load(cfg.calibrator_ckpt, map_location="cpu")
                calibrator = MemoryWeightCalibrator(
                    future_dim=int(calibrator_ckpt["future_dim"]),
                    memory_dim=int(calibrator_ckpt["memory_dim"]),
                    max_memories=int(calibrator_ckpt["max_memories"]),
                    hidden_dim=int(calibrator_ckpt["hidden_dim"]),
                )
                calibrator.load_state_dict(calibrator_ckpt["model_state"])
                calibrator = calibrator.to(device).eval()

                adapter_ckpt = torch.load(cfg.future_adapter_ckpt, map_location="cpu")
                adapter = MemoryStateGuidedAdapter(
                    future_dim=int(adapter_ckpt["future_dim"]),
                    memory_state_dim=int(adapter_ckpt["memory_state_dim"]),
                    hidden_dim=int(adapter_ckpt["hidden_dim"]),
                )
                adapter.load_state_dict(adapter_ckpt["model_state"])
                adapter = adapter.to(device).eval()
            factor_adapter = None
            factor_to_idx = {}
            if getattr(cfg, "factor_conditioned_adapter_ckpt", ""):
                factor_ckpt = torch.load(cfg.factor_conditioned_adapter_ckpt, map_location="cpu")
                factors = [str(x) for x in factor_ckpt.get("factors", FACTOR_ORDER)]
                factor_adapter = FactorConditionedFutureAdapter(
                    future_dim=int(factor_ckpt["future_dim"]),
                    num_factors=len(factors),
                    factor_dim=int(factor_ckpt.get("factor_dim", 64)),
                    hidden_dim=int(factor_ckpt["hidden_dim"]),
                )
                factor_adapter.load_state_dict(factor_ckpt["model_state"])
                factor_adapter = factor_adapter.to(device).eval()
                factor_to_idx = {factor: idx for idx, factor in enumerate(factors)}
            adapter_bundle = {
                "memory_arrays": memory_arrays,
                "calibrator": calibrator,
                "adapter": adapter,
                "factor_adapter": factor_adapter,
                "factor_to_idx": factor_to_idx,
                "token_level": bool(adapter_ckpt.get("token_level", False)) or factor_adapter is not None,
                "future_token_shape": tuple(int(x) for x in adapter_ckpt.get("future_token_shape", ())),
                "topk": int(getattr(cfg, "memory_key_topk", 3)),
            }
            if adapter_ckpt:
                adapter_bundle["topk"] = int(adapter_bundle["topk"])
            print(
                "[INFO] Loaded future adapter: "
                f"token_level={adapter_bundle['token_level']}, "
                f"future_token_shape={adapter_bundle['future_token_shape']}, "
                f"factor_conditioned={factor_adapter is not None}"
            )
        return adapter_bundle

    try:
        results = []
        print(
            f"[EVAL] checkpoint={checkpoint} save_dir={save_dir} num_sequences={len(eval_sequences)} "
            f"num_procs={num_procs} proc_id={procs_id}",
            flush=True,
        )
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)
        for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
            with open_dict(cfg):
                cfg._runtime_sequence_index = int(i)
            seq_start_time = time.time()
            logs_len_before_sequence = len(logs)
            task_names = [str(task) for task in eval_sequence]
            print(
                f"[EVAL][START] seq={i} tasks={task_names}",
                flush=True,
            )
            append_progress(
                {
                    "event": "start",
                    "sequence_index": int(i),
                    "tasks": task_names,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            record = record_video and (record_sequence_index < 0 or i == record_sequence_index)
            result = evaluate_sequence_memory(
                env,
                model,
                task_oracle,
                initial_state,
                eval_sequence,
                lang_embeddings,
                val_annotations,
                cfg,
                record,
                rollout_video,
                i,
                get_qwen,
                get_adapter,
                key_memory_rows,
                runtime_key_state_memory,
                logs,
                strong_gt_logs,
                online_acceptance_logs,
            )
            results.append(result)
            duration = time.time() - seq_start_time
            last_rollout_row = logs[-1] if len(logs) > logs_len_before_sequence else {}
            progress_runtime_info = {
                key: value
                for key, value in dict(last_rollout_row).items()
                if str(key).startswith(
                    (
                        "dynamic_coupling_",
                        "joint_execution_",
                        "persistent_hypothesis_",
                    )
                )
            }
            print(
                f"[EVAL][DONE] seq={i} success_len={result}/{len(eval_sequence)} "
                f"duration_sec={duration:.2f}",
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
                    **progress_runtime_info,
                }
            )
            write_partial_results(results)
            if record:
                rollout_video._log_currentvideos_to_file(i, save_as_video=True)
                rollout_video.videos = []
                rollout_video.tags = []
                rollout_video.captions = []
            horizon = max((len(sequence) for _, sequence in cfg.eval_sequences), default=5) if cfg.eval_sequences else 5
            success_rates = count_success_upto(results, horizon)
            average_rate = sum(success_rates)
            description = " ".join([f"{idx + 1}/{horizon} : {v * 100:.1f}% |" for idx, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)

        results_dict = {checkpoint: results}
        print_and_save_variable_horizon(results_dict, cfg, log_dir=save_dir)
        if save_dir is not None:
            with (Path(save_dir) / "memory_rollout_rows.jsonl").open("w") as handle:
                for row in logs:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if bool(getattr(cfg, "collect_online_acceptance_rows", False)):
                with (Path(save_dir) / "online_acceptance_rows.jsonl").open("w") as handle:
                    for row in online_acceptance_logs:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return results
    finally:
        logs.close()
        strong_gt_logs.close()
        online_acceptance_logs.close()


def main(cfg):
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device

    log_wandb = cfg.log_wandb
    seed_everything(int(getattr(cfg, "seed", 0)), workers=True)
    if getattr(cfg, "checkpoint_path", ""):
        checkpoints = [Path(cfg.checkpoint_path)]
    else:
        checkpoints = get_all_checkpoints(Path(cfg.train_folder))
    lang_embeddings = None
    env = None
    results = {}

    print("train_folder", cfg.train_folder)
    print("[PATHS] pretrained_model_path =", getattr(cfg.model, "pretrained_model_path", None))
    print("[PATHS] text_encoder_path     =", getattr(cfg.model, "text_encoder_path", None))
    print("[PATHS] t5_model_path         =", getattr(cfg.model, "t5_model_path", None))
    print("[PATHS] language_goal_path    =", getattr(cfg.model, "language_goal_path", None))
    bad_paths = []
    for key in ["pretrained_model_path", "text_encoder_path", "t5_model_path", "language_goal_path"]:
        value = getattr(cfg.model, key, None)
        if isinstance(value, str) and value.strip() == "..":
            bad_paths.append(key)
    if bad_paths:
        raise ValueError(f"invalid model path '..' detected for: {bad_paths}")
    print("\n" + "=" * 80)
    print(f"Found {len(checkpoints)} checkpoint(s) for evaluation:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  [{i}] {str(ckpt)}")
    print("=" * 80 + "\n")

    for checkpoint in checkpoints:
        print(device)
        print(f"\n{'*' * 80}")
        print(f"Processing checkpoint: {str(checkpoint)}")
        print(f"{'*' * 80}")
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
        model.load_state_dict(state_dict["model"], strict=False)
        maybe_load_policy_action_intent_lora(model, cfg)
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
        log_dir = get_log_dir(cfg.train_folder)
        if bool(getattr(cfg, "collect_joint_belief_transition", False)):
            with open_dict(cfg):
                cfg.joint_belief_transition_rows_path = str(Path(log_dir) / "joint_belief_transition_rows.jsonl")
                cfg.joint_belief_transition_trace_dir = str(Path(log_dir) / "joint_belief_transition")
                cfg._joint_belief_transition_count = 0
            print(
                "[JBT_COLLECT_MAIN] enabled "
                f"rows={cfg.joint_belief_transition_rows_path} "
                f"trace_dir={cfg.joint_belief_transition_trace_dir}",
                flush=True,
            )
        if log_wandb:
            os.makedirs(log_dir / "wandb", exist_ok=True)
        evaluator = original_evaluate_policy if cfg.disable_memory_reflection else evaluate_policy_memory
        results[checkpoint] = evaluator(
            model,
            env,
            lang_embeddings,
            cfg,
            acc.num_processes,
            acc.process_index,
            checkpoint,
            save_dir=Path(log_dir),
        )
        avg_reward = torch.tensor(results[checkpoint]).float().mean().to(device)
        acc.wait_for_everyone()
        avg_reward = acc.gather_for_metrics(avg_reward).mean()
        if acc.is_main_process:
            print("average success rate ", avg_reward)

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
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--t5_model_path", type=str, default="")
    parser.add_argument("--language_goal_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    parser.add_argument("--eval_sequences_path", type=str, default="")
    parser.add_argument("--num_sequences", type=int, default=-1)
    parser.add_argument("--eval_task_filter", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ep_len", type=int, default=-1)
    parser.add_argument("--disable_memory_reflection", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--record_sequence_index", type=int, default=-1)
    parser.add_argument("--memory_key_path", type=str, default="")
    parser.add_argument("--memory_recent_k", type=int, default=4)
    parser.add_argument("--memory_key_topk", type=int, default=3)
    parser.add_argument("--memory_runtime_key_max", type=int, default=5000)
    parser.add_argument("--qwen_model_path", type=str, default="")
    parser.add_argument("--qwen_lora_path", type=str, default="")
    parser.add_argument("--policy_action_intent_lora_ckpt", type=str, default="")
    parser.add_argument("--qwen_python_bin", type=str, default="")
    parser.add_argument("--qwen_diagnosis_only", action="store_true")
    parser.add_argument("--qwen_diagnosis_max_new_tokens", type=int, default=128)
    parser.add_argument("--qwen_diagnosis_save_target_proxy", action="store_true")
    parser.add_argument("--memory_npz_path", type=str, default="")
    parser.add_argument("--calibrator_ckpt", type=str, default="")
    parser.add_argument("--future_adapter_ckpt", type=str, default="")
    parser.add_argument("--factor_conditioned_adapter_ckpt", type=str, default="")
    parser.add_argument("--factor_conditioned_allowed_factors", type=str, default="goal_completion,drawer_slider_progress")
    parser.add_argument("--counterfactual_refinement", action="store_true")
    parser.add_argument("--future_trace_dir", type=str, default="")
    parser.add_argument("--future_trace_limit", type=int, default=0)
    parser.add_argument("--save_trace_observation", action="store_true")
    parser.add_argument("--action_sensitivity_probe", action="store_true")
    parser.add_argument("--action_sensitivity_eps", type=str, default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--factor_mask_npz", type=str, default="")
    parser.add_argument("--factor_mask_mode", choices=["global", "correct", "wrong", "random", "program"], default="global")
    parser.add_argument("--factor_mask_random_seed", type=int, default=0)
    parser.add_argument("--program_conditioned_mask_ckpt", type=str, default="")
    parser.add_argument("--program_conditioned_mask_density", type=float, default=0.01)
    parser.add_argument("--joint_future_editor_ckpt", type=str, default="")
    parser.add_argument("--joint_future_editor_gate_scale", type=float, default=0.05)
    parser.add_argument("--joint_future_editor_mask_density", type=float, default=0.02)
    parser.add_argument("--joint_future_editor_hard_mask", action="store_true")
    parser.add_argument("--joint_future_editor_fixed_gate", action="store_true")
    parser.add_argument("--joint_future_editor_fixed_gate_value", type=float, default=1.0)
    parser.add_argument("--joint_editor_risk_gate", action="store_true")
    parser.add_argument("--joint_editor_risk_threshold", type=float, default=0.5)
    parser.add_argument("--joint_editor_trigger_ckpt", type=str, default="")
    parser.add_argument("--joint_editor_trigger_threshold", type=float, default=0.5)
    parser.add_argument("--joint_editor_force_non_none", action="store_true")
    parser.add_argument("--joint_editor_min_failure_memory_votes", type=int, default=2)
    parser.add_argument(
        "--joint_editor_failure_tasks",
        type=str,
        default="push_into_drawer,push_pink_block_right,stack_block,push_blue_block_right,push_red_block_right,push_pink_block_left,lift_blue_block_slider",
    )
    parser.add_argument("--risk_gated_intervention", action="store_true")
    parser.add_argument("--risk_gate_min_gamma", type=float, default=0.08)
    parser.add_argument("--risk_gate_min_memory_support", type=float, default=0.75)
    parser.add_argument("--risk_gate_min_gate", type=float, default=0.08)
    parser.add_argument("--risk_gate_max_gate", type=float, default=0.12)
    parser.add_argument("--risk_detector_ckpt", type=str, default="")
    parser.add_argument("--risk_detector_threshold", type=float, default=0.5)
    parser.add_argument("--counterfactual_verifier_ckpt", type=str, default="")
    parser.add_argument("--counterfactual_verifier_threshold", type=float, default=0.8)
    parser.add_argument("--counterfactual_verifier_temperature", type=float, default=1.0)
    parser.add_argument("--counterfactual_verifier_min_prob", type=float, default=0.0)
    parser.add_argument("--counterfactual_verifier_max_prob", type=float, default=1.0)
    parser.add_argument("--semantic_future_beam", action="store_true")
    parser.add_argument("--semantic_beam_patch_density", type=float, default=0.01)
    parser.add_argument("--semantic_beam_min_margin", type=float, default=1e-4)
    parser.add_argument("--safe_intervention_selector", action="store_true")
    parser.add_argument("--safe_selector_max_adapter_delta_norm", type=float, default=300.0)
    parser.add_argument("--safe_selector_max_adapter_gate_mean", type=float, default=0.08)
    parser.add_argument("--safe_selector_min_memory_support", type=float, default=0.75)
    parser.add_argument("--safe_selector_min_proposal_cos_gain", type=float, default=0.0)
    parser.add_argument("--immune_repulsion", action="store_true")
    parser.add_argument("--immune_only", action="store_true")
    parser.add_argument("--immune_memory_npz", type=str, default="")
    parser.add_argument("--immune_gate", type=float, default=0.02)
    parser.add_argument("--immune_topk_failure", type=int, default=5)
    parser.add_argument("--immune_alpha_repel", type=float, default=1.0)
    parser.add_argument("--immune_beta_attract", type=float, default=1.0)
    parser.add_argument("--immune_same_task_only", action="store_true")
    parser.add_argument("--immune_repair_pair_only", action="store_true")
    parser.add_argument("--immune_local_repair_field", action="store_true")
    parser.add_argument("--immune_locality_temperature", type=float, default=0.05)
    parser.add_argument("--immune_consistency_temperature", type=float, default=0.25)
    parser.add_argument("--immune_cross_task_weight", type=float, default=0.25)
    parser.add_argument("--immune_local_repair_field_ckpt", type=str, default="")
    parser.add_argument("--immune_learned_weight_temperature", type=float, default=1.0)
    parser.add_argument("--immune_q_filter_ckpt", type=str, default="")
    parser.add_argument("--immune_q_filter_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--immune_q_filter_min_gain", type=float, default=0.0)
    parser.add_argument("--immune_q_filter_temperature", type=float, default=0.1)
    parser.add_argument("--immune_q_filter_candidate_scale", type=float, default=1.0)
    parser.add_argument("--immune_q_filter_allow_fallback", action="store_true")
    parser.add_argument("--immune_factor_mask_npz", type=str, default="")
    parser.add_argument("--immune_factor_mask_mode", choices=["correct", "wrong", "random", "global"], default="correct")
    parser.add_argument("--token_immune_adapter_ckpt", type=str, default="")
    parser.add_argument("--token_immune_adapter_weight", type=float, default=1.0)
    parser.add_argument("--causal_intervention_adapter_ckpt", type=str, default="")
    parser.add_argument("--causal_intervention_gate", type=float, default=1.0)
    parser.add_argument("--future_manifold_navigator_ckpt", type=str, default="")
    parser.add_argument("--future_manifold_nav_weight", type=float, default=1.0)
    parser.add_argument("--future_manifold_risk_gate", action="store_true")
    parser.add_argument("--future_manifold_risk_threshold", type=float, default=0.0)
    parser.add_argument("--future_manifold_mask_mode", type=str, default="none", choices=["none", "topk_element", "topk_channel", "topk_time"])
    parser.add_argument("--future_manifold_mask_density", type=float, default=1.0)
    parser.add_argument("--future_success_verifier_ckpt", type=str, default="")
    parser.add_argument("--future_success_verifier_min_risk", type=float, default=0.10)
    parser.add_argument("--future_success_verifier_min_gate", type=float, default=0.0)
    parser.add_argument("--future_success_verifier_max_gate", type=float, default=0.005)
    parser.add_argument("--relation_probe_ckpt", type=str, default="")
    parser.add_argument("--relation_probe_trigger", action="store_true")
    parser.add_argument("--relation_risk_threshold", type=float, default=0.65)
    parser.add_argument("--relation_progress_threshold", type=float, default=0.6)
    parser.add_argument("--immune_geometry_gate", action="store_true")
    parser.add_argument("--immune_geometry_min_margin", type=float, default=0.0)
    parser.add_argument("--immune_geometry_max_margin", type=float, default=0.02)
    parser.add_argument("--immune_geometry_min_gate", type=float, default=0.0)
    parser.add_argument("--immune_geometry_max_gate", type=float, default=0.005)
    parser.add_argument("--immune_field_confidence_gate", action="store_true")
    parser.add_argument("--immune_field_gate_risk_alpha", type=float, default=1.0)
    parser.add_argument("--immune_field_gate_consistency_beta", type=float, default=1.0)
    parser.add_argument("--immune_field_gate_min_consistency", type=float, default=0.25)
    parser.add_argument("--immune_field_gate_temperature", type=float, default=0.25)
    parser.add_argument("--future_energy_ckpt", type=str, default="")
    parser.add_argument("--future_energy_descent", action="store_true")
    parser.add_argument("--future_energy_threshold", type=float, default=0.0)
    parser.add_argument("--future_energy_step_size", type=float, default=0.001)
    parser.add_argument("--future_energy_mask_density", type=float, default=0.01)
    parser.add_argument("--future_energy_max_shift_norm", type=float, default=1.0)
    parser.add_argument("--future_energy_mask_mode", choices=["topk", "factor"], default="topk")
    parser.add_argument("--future_energy_factor_mask_npz", type=str, default="")
    parser.add_argument("--future_energy_factor", type=str, default="auto")
    parser.add_argument("--future_energy_risk_gate", type=float, default=1.0)
    parser.add_argument("--future_energy_normalize_grad", action="store_true")
    parser.add_argument("--channel_causal_patch", action="store_true")
    parser.add_argument("--channel_patch_memory_jsonl", type=str, default="")
    parser.add_argument("--channel_patch_trace_dir", type=str, default="")
    parser.add_argument("--channel_patch_future_key", type=str, default="base_future")
    parser.add_argument("--channel_patch_mode", choices=["best", "selector", "random", "global", "complement"], default="best")
    parser.add_argument("--channel_patch_selector_ckpt", type=str, default="")
    parser.add_argument("--channel_patch_editor_ckpt", type=str, default="")
    parser.add_argument("--channel_patch_q_critic_ckpt", type=str, default="")
    parser.add_argument("--channel_patch_risk_threshold", type=float, default=-1.0)
    parser.add_argument("--channel_patch_topk_groups", type=int, default=1)
    parser.add_argument("--channel_patch_group_size", type=int, default=32)
    parser.add_argument("--channel_patch_gate", type=float, default=1.0)
    parser.add_argument("--channel_patch_k_success", type=int, default=5)
    parser.add_argument("--channel_patch_k_failure", type=int, default=5)
    parser.add_argument("--channel_patch_same_task_only", action="store_true")
    parser.add_argument("--channel_patch_random_seed", type=int, default=0)
    parser.add_argument("--joint_pair_memory_npz", type=str, default="")
    parser.add_argument("--joint_energy_rerank_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_topk", type=int, default=8)
    parser.add_argument("--joint_pair_same_task_only", action="store_true")
    parser.add_argument("--joint_pair_target_future_key", type=str, default="target_proxy_future")
    parser.add_argument("--joint_pair_without_trigger", action="store_true")
    parser.add_argument("--joint_pair_min_energy_margin", type=float, default=0.0)
    parser.add_argument("--joint_pair_max_energy", type=float, default=1e9)
    parser.add_argument("--joint_pair_min_similarity", type=float, default=-1.0)
    parser.add_argument("--joint_pair_refine_steps", type=int, default=0)
    parser.add_argument("--joint_pair_refine_future_lr", type=float, default=1e-2)
    parser.add_argument("--joint_pair_refine_action_lr", type=float, default=1e-2)
    parser.add_argument("--joint_pair_refine_grad_clip", type=float, default=1.0)
    parser.add_argument("--joint_pair_refine_future_max_delta", type=float, default=5.0)
    parser.add_argument("--joint_pair_refine_action_max_delta", type=float, default=5.0)
    parser.add_argument("--joint_pair_refine_min_delta_energy", type=float, default=0.0)
    parser.add_argument("--joint_pair_refine_backtrack_factor", type=float, default=0.5)
    parser.add_argument("--joint_pair_refine_min_step_scale", type=float, default=0.03125)
    parser.add_argument("--joint_pair_refine_max_backtracks", type=int, default=5)
    parser.add_argument("--joint_pair_refine_early_stop_patience", type=int, default=2)
    parser.add_argument("--joint_pair_refine_early_stop_min_improve", type=float, default=1e-3)
    parser.add_argument("--joint_pair_refine_early_stop_grad_norm", type=float, default=1e-3)
    parser.add_argument("--joint_pair_refine_lambda_exp", type=float, default=1.0)
    parser.add_argument("--joint_pair_refine_lambda_latent", type=float, default=1.0)
    parser.add_argument("--joint_pair_refine_lambda_physical", type=float, default=1.0)
    parser.add_argument("--joint_pair_refine_normalize_terms", action="store_true")
    parser.add_argument("--joint_pair_refine_lambda_dynamics", type=float, default=0.0)
    parser.add_argument("--joint_pair_refine_lambda_future_anchor", type=float, default=0.1)
    parser.add_argument("--joint_pair_refine_lambda_action_anchor", type=float, default=0.1)
    parser.add_argument("--joint_pair_refine_action_mix", type=float, default=1.0)
    parser.add_argument("--joint_pair_refine_fallback", action="store_true")
    parser.add_argument("--joint_pair_push_sensitive_action_mix", type=float, default=0.35)
    parser.add_argument("--joint_pair_push_sensitive_max_action_shift", type=float, default=2.0)
    parser.add_argument("--joint_pair_action_generator_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_action_generator_mix", type=float, default=0.5)
    parser.add_argument("--joint_pair_direct_generate_only", action="store_true")
    parser.add_argument("--joint_pair_no_memory", action="store_true")
    parser.add_argument("--joint_pair_retry_only", action="store_true")
    parser.add_argument("--joint_pair_dynamics_probe_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_lambda_dynamics", type=float, default=0.0)
    parser.add_argument("--joint_pair_repair_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_repair_critic_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_summary_future_adapter_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_repair_advantage_gate", action="store_true")
    parser.add_argument("--joint_pair_repair_advantage_threshold", type=float, default=0.0)
    parser.add_argument("--joint_pair_repair_advantage_logit_threshold", type=float, default=0.0)
    parser.add_argument("--joint_pair_repair_future_mix", type=float, default=0.5)
    parser.add_argument("--joint_pair_repair_action_mix", type=float, default=0.5)
    parser.add_argument("--joint_pair_repair_with_coupling_prior", action="store_true")
    parser.add_argument("--joint_pair_repair_manifold_min_improve", type=float, default=0.0)
    parser.add_argument("--joint_pair_manifold_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_cross_stage_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_cross_stage_topk", type=int, default=4)
    parser.add_argument("--joint_pair_repair_manifold_max_dist", type=float, default=1e9)
    parser.add_argument("--joint_pair_repair_manifold_min_success_gain", type=float, default=0.0)
    parser.add_argument("--joint_pair_max_future_shift", type=float, default=0.0)
    parser.add_argument("--joint_pair_value_ckpt", type=str, default="")
    parser.add_argument("--joint_pair_lambda_value", type=float, default=0.0)
    parser.add_argument("--joint_pair_lambda_cross_stage", type=float, default=0.0)
    parser.add_argument("--joint_pair_verify_acceptance", action="store_true")
    parser.add_argument("--joint_pair_verify_exp_max", type=float, default=1e9)
    parser.add_argument("--joint_pair_verify_latent_max", type=float, default=1e9)
    parser.add_argument("--joint_pair_verify_physical_max", type=float, default=1e9)
    parser.add_argument("--joint_pair_verify_require_improve", action="store_true")
    parser.add_argument("--joint_pair_verify_min_improve", type=float, default=0.0)
    parser.add_argument("--no_qwen_reflection", action="store_true")
    parser.add_argument("--print_collection_rows", action="store_true")
    parser.add_argument("--collect_strong_gt", action="store_true")
    parser.add_argument("--print_strong_gt_rows", action="store_true")
    parser.add_argument("--collect_action_trace", action="store_true")
    parser.add_argument("--action_trace_dir", type=str, default="")
    parser.add_argument("--collect_joint_belief_transition", action="store_true")
    parser.add_argument("--collect_online_acceptance_rows", action="store_true")
    parser.add_argument("--dynamic_coupling_operator_ckpt", type=str, default="")
    parser.add_argument("--dynamic_coupling_online", action="store_true")
    parser.add_argument("--dynamic_coupling_first_pass", action="store_true")
    parser.add_argument("--dynamic_coupling_generate_action_intent", action="store_true")
    parser.add_argument("--dynamic_coupling_apply_action_intent_to_policy", action="store_true")
    parser.add_argument("--dynamic_coupling_persistent_hypothesis", action="store_true")
    parser.add_argument("--dynamic_coupling_hypothesis_future_update_mix", type=float, default=1.0)
    parser.add_argument("--dynamic_coupling_hypothesis_action_update_mix", type=float, default=1.0)
    parser.add_argument("--dynamic_coupling_need_gain", type=float, default=0.35)
    parser.add_argument("--dynamic_coupling_min_scale", type=float, default=0.5)
    parser.add_argument("--dynamic_coupling_max_scale", type=float, default=1.25)
    parser.add_argument("--dynamic_coupling_keep_actions", type=int, default=64)
    parser.add_argument("--dynamic_coupling_min_committed", type=float, default=0.5)
    parser.add_argument("--dynamic_coupling_max_future_shift_norm", type=float, default=1.5)
    parser.add_argument("--dynamic_coupling_max_future_shift_ratio", type=float, default=0.35)
    parser.add_argument("--dynamic_coupling_future_mix", type=float, default=0.15)
    parser.add_argument("--dynamic_coupling_auto_shrink", action="store_true")
    parser.add_argument("--dynamic_coupling_gate_every_k", type=int, default=1)
    parser.add_argument("--dynamic_coupling_gate_start_step", type=int, default=0)
    parser.add_argument("--dynamic_coupling_gate_with_trigger", action="store_true")
    parser.add_argument("--dynamic_coupling_trigger_threshold", type=float, default=0.5)
    parser.add_argument("--dynamic_coupling_patch_remaining_actions", action="store_true")
    parser.add_argument("--dynamic_coupling_patch_mode", type=str, default="replace", choices=["replace", "delta", "mix"])
    parser.add_argument("--dynamic_coupling_patch_mix", type=float, default=1.0)
    parser.add_argument("--dynamic_coupling_delta_action_repair_ckpt", type=str, default="")
    parser.add_argument("--dynamic_coupling_delta_action_repair_mix", type=float, default=1.0)
    parser.add_argument("--dynamic_coupling_chunk_effect_ckpt", type=str, default="")
    parser.add_argument("--dynamic_coupling_chunk_keep_threshold", type=float, default=0.35)
    parser.add_argument("--dynamic_coupling_chunk_replan_threshold", type=float, default=0.75)
    parser.add_argument("--dynamic_coupling_chunk_success_threshold", type=float, default=0.5)
    parser.add_argument("--dynamic_coupling_chunk_low_compat_threshold", type=float, default=0.0)
    parser.add_argument("--dynamic_coupling_disable_replan", action="store_true")
    parser.add_argument("--dynamic_coupling_suffix_trigger_success_below", type=float, default=1.0)
    parser.add_argument("--dynamic_coupling_suffix_trigger_residual_above", type=float, default=0.0)
    parser.add_argument("--dynamic_coupling_suffix_trigger_compat_below", type=float, default=1.0)
    parser.add_argument("--joint_execution_update", action="store_true")
    parser.add_argument("--joint_execution_update_ckpt", type=str, default="")
    parser.add_argument("--joint_belief_transition", action="store_true")
    parser.add_argument("--joint_belief_transition_ckpt", type=str, default="")
    parser.add_argument("--joint_belief_transition_direct_suffix", action="store_true")
    parser.add_argument("--joint_belief_transition_sparse_suffix", action="store_true")
    parser.add_argument("--joint_belief_transition_min_learned_gate", type=float, default=0.08)
    parser.add_argument("--joint_belief_transition_max_success_prob", type=float, default=0.25)
    parser.add_argument("--joint_belief_clean_controller", action="store_true")
    parser.add_argument("--joint_belief_clean_min_confidence", type=float, default=0.0)
    parser.add_argument("--joint_belief_clean_patch_min_prob", type=float, default=0.6)
    parser.add_argument("--joint_belief_clean_patch_margin", type=float, default=0.1)
    parser.add_argument("--joint_belief_clean_replan_min_prob", type=float, default=0.7)
    parser.add_argument("--joint_belief_clean_h_update_scale", type=float, default=0.0)
    parser.add_argument("--joint_belief_summary_future_adapter_update", action="store_true")
    parser.add_argument("--joint_belief_summary_future_adapter_delta_key", choices=["future_delta", "innovation"], default="future_delta")
    parser.add_argument("--joint_belief_summary_future_adapter_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_belief_summary_future_adapter_mix", type=float, default=1.0)
    parser.add_argument("--joint_belief_summary_future_adapter_max_shift_norm", type=float, default=0.0)
    parser.add_argument("--joint_belief_patch_min_interval_steps", type=int, default=1)
    parser.add_argument("--joint_belief_patch_max_per_subtask", type=int, default=0)
    parser.add_argument("--suffix_trigger_mlp_ckpt", type=str, default="")
    parser.add_argument("--suffix_trigger_mlp_threshold", type=float, default=0.5)
    parser.add_argument("--counterfactual_suffix_transport", action="store_true")
    parser.add_argument("--counterfactual_suffix_transport_sigma", type=float, default=0.5)
    parser.add_argument("--counterfactual_suffix_transport_edit_steps", type=int, default=1)
    parser.add_argument("--counterfactual_suffix_transport_step_size", type=float, default=0.05)
    parser.add_argument("--counterfactual_suffix_transport_max_abs_delta", type=float, default=0.03)
    parser.add_argument("--linear_suffix_sensitivity_transport", action="store_true")
    parser.add_argument("--linear_suffix_sensitivity_rank", type=int, default=8)
    parser.add_argument("--linear_suffix_sensitivity_eps", type=float, default=0.01)
    parser.add_argument("--linear_suffix_sensitivity_step_size", type=float, default=1.0)
    parser.add_argument("--linear_suffix_sensitivity_max_abs_delta", type=float, default=0.03)
    parser.add_argument("--linear_suffix_sensitivity_min_step_score", type=float, default=0.0)
    parser.add_argument("--linear_suffix_sensitivity_selective_step_threshold", type=float, default=0.0)
    parser.add_argument("--linear_suffix_sensitivity_max_patch_fraction", type=float, default=1.0)
    parser.add_argument("--linear_suffix_sensitivity_replan_fraction", type=float, default=1.01)
    parser.add_argument("--linear_suffix_sensitivity_replan_delta_h_norm", type=float, default=0.0)
    parser.add_argument("--linear_suffix_sensitivity_smooth_weight", type=float, default=0.0)
    parser.add_argument("--linear_suffix_sensitivity_continuity_weight", type=float, default=0.0)
    parser.add_argument("--linear_suffix_sensitivity_mpc_iters", type=int, default=1)
    parser.add_argument("--directional_sensitivity_tcn_transport", action="store_true")
    parser.add_argument("--directional_sensitivity_tcn_ckpt", type=str, default="")
    parser.add_argument("--directional_sensitivity_tcn_max_abs_delta", type=float, default=0.015)
    parser.add_argument("--directional_sensitivity_tcn_min_delta_norm", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_step_threshold", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_top_k", type=int, default=0)
    parser.add_argument("--directional_sensitivity_tcn_top_k_floor", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_top_k_min_margin", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_force_top_k_patch", type=int, default=0)
    parser.add_argument("--directional_sensitivity_tcn_max_patch_fraction", type=float, default=1.0)
    parser.add_argument("--directional_sensitivity_tcn_max_affected_segments", type=int, default=0)
    parser.add_argument("--directional_sensitivity_tcn_replan_delta_h_norm", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_smooth_weight", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_mpc_iters", type=int, default=1)
    parser.add_argument("--directional_sensitivity_tcn_structured_mask", action="store_true")
    parser.add_argument("--directional_sensitivity_tcn_validity_threshold", type=float, default=0.5)
    parser.add_argument("--directional_sensitivity_tcn_uncertainty_replan_threshold", type=float, default=0.0)
    parser.add_argument("--directional_sensitivity_tcn_selection_lambda", type=float, default=0.5)
    parser.add_argument("--directional_sensitivity_tcn_selection_beta", type=float, default=0.15)
    parser.add_argument("--direct_action_correction_tcn_transport", action="store_true")
    parser.add_argument("--direct_action_correction_tcn_ckpt", type=str, default="")
    parser.add_argument("--collect_sensitivity_teacher", action="store_true")
    parser.add_argument("--sensitivity_teacher_mode", choices=["jvp", "paired_counterfactual"], default="jvp")
    parser.add_argument("--sensitivity_teacher_step_size", type=float, default=1.0)
    parser.add_argument("--sensitivity_teacher_max_abs_delta", type=float, default=0.0)
    parser.add_argument("--sensitivity_teacher_joint_delta_to_action_scale", type=float, default=1.0)
    parser.add_argument(
        "--sensitivity_teacher_condition_ablation",
        choices=["joint", "future_only", "action_only"],
        default="joint",
    )
    parser.add_argument(
        "--joint_execution_update_ablation_mode",
        choices=["rssm", "instant"],
        default="rssm",
        help="Ablate belief revision: rssm uses learned prior/posterior; instant uses target-current summary only.",
    )
    parser.add_argument(
        "--suffix_mask_ablation_mode",
        choices=[
            "off",
            "no_patch",
            "gt_selective",
            "tcn_selective",
            "random_mask",
            "complement_mask",
            "whole_suffix",
            "full_replan",
        ],
        default="off",
    )
    parser.add_argument("--suffix_mask_ablation_force_writeback", action="store_true")
    parser.add_argument("--suffix_mask_ablation_teacher_step_threshold", type=float, default=0.0001)
    parser.add_argument("--suffix_mask_ablation_max_abs_delta", type=float, default=0.0)
    parser.add_argument("--suffix_mask_ablation_min_affected_fraction", type=float, default=0.0)
    parser.add_argument("--suffix_mask_ablation_max_affected_fraction", type=float, default=1.0)
    parser.add_argument("--suffix_mask_ablation_seed", type=int, default=0)
    parser.add_argument("--suffix_mask_ablation_schedule_json", type=str, default="")
    parser.add_argument(
        "--joint_execution_gate_mode",
        choices=["heuristic", "learned", "learned_or_heuristic"],
        default="heuristic",
    )
    parser.add_argument("--joint_execution_update_threshold", type=float, default=1.0)
    parser.add_argument("--joint_execution_update_gain", type=float, default=1.0)
    parser.add_argument("--joint_execution_update_min_gate", type=float, default=0.0)
    parser.add_argument("--joint_execution_update_max_gate", type=float, default=0.3)
    parser.add_argument("--joint_execution_suffix_error_threshold", type=float, default=1.0)
    parser.add_argument("--joint_execution_error_residual_scale", type=float, default=30.0)
    parser.add_argument("--joint_execution_error_success_target", type=float, default=0.05)
    parser.add_argument("--joint_execution_error_compat_target", type=float, default=-0.1)
    parser.add_argument("--joint_execution_error_success_weight", type=float, default=2.0)
    parser.add_argument("--joint_execution_error_compat_weight", type=float, default=1.0)
    parser.add_argument("--debug_action_before_env_step", action="store_true")
    parser.add_argument("--debug_action_steps", type=int, default=5)
    parser.add_argument("--disable_retry_after_first_pass", action="store_true")
    parser.add_argument("--retry_requires_suffix_patch", action="store_true")
    parser.add_argument("--retry_rollback_mode", choices=["subtask_start", "none"], default="subtask_start")
    parser.add_argument("--retry_max_attempts", type=int, default=1)
    parser.add_argument("--retry_baseline_fallback_on_failure", action="store_true")

    args = parser.parse_args()
    print(f"[CLI] collect_joint_belief_transition={bool(args.collect_joint_belief_transition)}", flush=True)

    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    _apply_checkpoint_model_compat_overrides(cfg, args.checkpoint_path)
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
    if args.num_sequences > 0:
        cfg.num_sequences = int(args.num_sequences)
    if args.ep_len > 0:
        cfg.ep_len = int(args.ep_len)
    with open_dict(cfg):
        cfg.checkpoint_path = args.checkpoint_path
        cfg.seed = int(args.seed)
        cfg.eval_sequences = None
        cfg.eval_task_filter = args.eval_task_filter
        cfg.disable_memory_reflection = bool(args.disable_memory_reflection)
        cfg.record_video = bool(args.record_video)
        cfg.record_sequence_index = int(args.record_sequence_index)
        cfg.memory_key_path = args.memory_key_path
        cfg.memory_recent_k = int(args.memory_recent_k)
        cfg.memory_key_topk = int(args.memory_key_topk)
        cfg.memory_runtime_key_max = int(args.memory_runtime_key_max)
        cfg.qwen_model_path = args.qwen_model_path
        cfg.qwen_lora_path = args.qwen_lora_path
        cfg.policy_action_intent_lora_ckpt = args.policy_action_intent_lora_ckpt
        cfg.qwen_python_bin = args.qwen_python_bin
        cfg.qwen_diagnosis_only = bool(args.qwen_diagnosis_only)
        cfg.qwen_diagnosis_max_new_tokens = int(args.qwen_diagnosis_max_new_tokens)
        cfg.qwen_diagnosis_save_target_proxy = bool(args.qwen_diagnosis_save_target_proxy)
        cfg.memory_npz_path = args.memory_npz_path
        cfg.calibrator_ckpt = args.calibrator_ckpt
        cfg.future_adapter_ckpt = args.future_adapter_ckpt
        cfg.factor_conditioned_adapter_ckpt = args.factor_conditioned_adapter_ckpt
        cfg.factor_conditioned_allowed_factors = args.factor_conditioned_allowed_factors
        cfg.counterfactual_refinement = bool(args.counterfactual_refinement)
        cfg.future_trace_dir = args.future_trace_dir
        cfg.future_trace_limit = int(args.future_trace_limit)
        cfg.save_trace_observation = bool(args.save_trace_observation)
        cfg.action_sensitivity_probe = bool(args.action_sensitivity_probe)
        cfg.action_sensitivity_eps = args.action_sensitivity_eps
        cfg.factor_mask_npz = args.factor_mask_npz
        cfg.factor_mask_mode = args.factor_mask_mode
        cfg.factor_mask_random_seed = int(args.factor_mask_random_seed)
        cfg.program_conditioned_mask_ckpt = args.program_conditioned_mask_ckpt
        cfg.program_conditioned_mask_density = float(args.program_conditioned_mask_density)
        cfg.joint_future_editor_ckpt = args.joint_future_editor_ckpt
        cfg.joint_future_editor_gate_scale = float(args.joint_future_editor_gate_scale)
        cfg.joint_future_editor_mask_density = float(args.joint_future_editor_mask_density)
        cfg.joint_future_editor_hard_mask = bool(args.joint_future_editor_hard_mask)
        cfg.joint_future_editor_fixed_gate = bool(args.joint_future_editor_fixed_gate)
        cfg.joint_future_editor_fixed_gate_value = float(args.joint_future_editor_fixed_gate_value)
        cfg.joint_editor_risk_gate = bool(args.joint_editor_risk_gate)
        cfg.joint_editor_risk_threshold = float(args.joint_editor_risk_threshold)
        cfg.joint_editor_trigger_ckpt = args.joint_editor_trigger_ckpt
        cfg.joint_editor_trigger_threshold = float(args.joint_editor_trigger_threshold)
        cfg.joint_editor_force_non_none = bool(args.joint_editor_force_non_none)
        cfg.joint_editor_min_failure_memory_votes = int(args.joint_editor_min_failure_memory_votes)
        cfg.joint_editor_failure_tasks = args.joint_editor_failure_tasks
        cfg.risk_gated_intervention = bool(args.risk_gated_intervention)
        cfg.risk_gate_min_gamma = float(args.risk_gate_min_gamma)
        cfg.risk_gate_min_memory_support = float(args.risk_gate_min_memory_support)
        cfg.risk_gate_min_gate = float(args.risk_gate_min_gate)
        cfg.risk_gate_max_gate = float(args.risk_gate_max_gate)
        cfg.risk_detector_ckpt = args.risk_detector_ckpt
        cfg.risk_detector_threshold = float(args.risk_detector_threshold)
        cfg.counterfactual_verifier_ckpt = args.counterfactual_verifier_ckpt
        cfg.counterfactual_verifier_threshold = float(args.counterfactual_verifier_threshold)
        cfg.counterfactual_verifier_temperature = float(args.counterfactual_verifier_temperature)
        cfg.counterfactual_verifier_min_prob = float(args.counterfactual_verifier_min_prob)
        cfg.counterfactual_verifier_max_prob = float(args.counterfactual_verifier_max_prob)
        cfg.semantic_future_beam = bool(args.semantic_future_beam)
        cfg.semantic_beam_patch_density = float(args.semantic_beam_patch_density)
        cfg.semantic_beam_min_margin = float(args.semantic_beam_min_margin)
        cfg.safe_intervention_selector = bool(args.safe_intervention_selector)
        cfg.safe_selector_max_adapter_delta_norm = float(args.safe_selector_max_adapter_delta_norm)
        cfg.safe_selector_max_adapter_gate_mean = float(args.safe_selector_max_adapter_gate_mean)
        cfg.safe_selector_min_memory_support = float(args.safe_selector_min_memory_support)
        cfg.safe_selector_min_proposal_cos_gain = float(args.safe_selector_min_proposal_cos_gain)
        cfg.immune_repulsion = bool(args.immune_repulsion)
        cfg.immune_only = bool(args.immune_only)
        cfg.immune_memory_npz = args.immune_memory_npz
        cfg.immune_gate = float(args.immune_gate)
        cfg.immune_topk_failure = int(args.immune_topk_failure)
        cfg.immune_alpha_repel = float(args.immune_alpha_repel)
        cfg.immune_beta_attract = float(args.immune_beta_attract)
        cfg.immune_same_task_only = bool(args.immune_same_task_only)
        cfg.immune_repair_pair_only = bool(args.immune_repair_pair_only)
        cfg.immune_local_repair_field = bool(args.immune_local_repair_field)
        cfg.immune_locality_temperature = float(args.immune_locality_temperature)
        cfg.immune_consistency_temperature = float(args.immune_consistency_temperature)
        cfg.immune_cross_task_weight = float(args.immune_cross_task_weight)
        cfg.immune_local_repair_field_ckpt = args.immune_local_repair_field_ckpt
        cfg.immune_learned_weight_temperature = float(args.immune_learned_weight_temperature)
        cfg.immune_q_filter_ckpt = args.immune_q_filter_ckpt
        cfg.immune_q_filter_mode = args.immune_q_filter_mode
        cfg.immune_q_filter_min_gain = float(args.immune_q_filter_min_gain)
        cfg.immune_q_filter_temperature = float(args.immune_q_filter_temperature)
        cfg.immune_q_filter_candidate_scale = float(args.immune_q_filter_candidate_scale)
        cfg.immune_q_filter_allow_fallback = bool(args.immune_q_filter_allow_fallback)
        cfg.immune_factor_mask_npz = args.immune_factor_mask_npz
        cfg.immune_factor_mask_mode = args.immune_factor_mask_mode
        cfg.token_immune_adapter_ckpt = args.token_immune_adapter_ckpt
        cfg.token_immune_adapter_weight = float(args.token_immune_adapter_weight)
        cfg.causal_intervention_adapter_ckpt = args.causal_intervention_adapter_ckpt
        cfg.causal_intervention_gate = float(args.causal_intervention_gate)
        cfg.future_manifold_navigator_ckpt = args.future_manifold_navigator_ckpt
        cfg.future_manifold_nav_weight = float(args.future_manifold_nav_weight)
        cfg.future_manifold_risk_gate = bool(args.future_manifold_risk_gate)
        cfg.future_manifold_risk_threshold = float(args.future_manifold_risk_threshold)
        cfg.future_manifold_mask_mode = args.future_manifold_mask_mode
        cfg.future_manifold_mask_density = float(args.future_manifold_mask_density)
        cfg.future_success_verifier_ckpt = args.future_success_verifier_ckpt
        cfg.future_success_verifier_min_risk = float(args.future_success_verifier_min_risk)
        cfg.future_success_verifier_min_gate = float(args.future_success_verifier_min_gate)
        cfg.future_success_verifier_max_gate = float(args.future_success_verifier_max_gate)
        cfg.relation_probe_ckpt = args.relation_probe_ckpt
        cfg.relation_probe_trigger = bool(args.relation_probe_trigger)
        cfg.relation_risk_threshold = float(args.relation_risk_threshold)
        cfg.relation_progress_threshold = float(args.relation_progress_threshold)
        cfg.immune_geometry_gate = bool(args.immune_geometry_gate)
        cfg.immune_geometry_min_margin = float(args.immune_geometry_min_margin)
        cfg.immune_geometry_max_margin = float(args.immune_geometry_max_margin)
        cfg.immune_geometry_min_gate = float(args.immune_geometry_min_gate)
        cfg.immune_geometry_max_gate = float(args.immune_geometry_max_gate)
        cfg.immune_field_confidence_gate = bool(args.immune_field_confidence_gate)
        cfg.immune_field_gate_risk_alpha = float(args.immune_field_gate_risk_alpha)
        cfg.immune_field_gate_consistency_beta = float(args.immune_field_gate_consistency_beta)
        cfg.immune_field_gate_min_consistency = float(args.immune_field_gate_min_consistency)
        cfg.immune_field_gate_temperature = float(args.immune_field_gate_temperature)
        cfg.future_energy_ckpt = args.future_energy_ckpt
        cfg.future_energy_descent = bool(args.future_energy_descent)
        cfg.future_energy_threshold = float(args.future_energy_threshold)
        cfg.future_energy_step_size = float(args.future_energy_step_size)
        cfg.future_energy_mask_density = float(args.future_energy_mask_density)
        cfg.future_energy_max_shift_norm = float(args.future_energy_max_shift_norm)
        cfg.future_energy_mask_mode = args.future_energy_mask_mode
        cfg.future_energy_factor_mask_npz = args.future_energy_factor_mask_npz
        cfg.future_energy_factor = args.future_energy_factor
        cfg.future_energy_risk_gate = float(args.future_energy_risk_gate)
        cfg.future_energy_normalize_grad = bool(args.future_energy_normalize_grad)
        cfg.channel_causal_patch = bool(args.channel_causal_patch)
        cfg.channel_patch_memory_jsonl = args.channel_patch_memory_jsonl
        cfg.channel_patch_trace_dir = args.channel_patch_trace_dir
        cfg.channel_patch_future_key = args.channel_patch_future_key
        cfg.channel_patch_mode = args.channel_patch_mode
        cfg.channel_patch_selector_ckpt = args.channel_patch_selector_ckpt
        cfg.channel_patch_editor_ckpt = args.channel_patch_editor_ckpt
        cfg.channel_patch_q_critic_ckpt = args.channel_patch_q_critic_ckpt
        cfg.channel_patch_risk_threshold = float(args.channel_patch_risk_threshold)
        cfg.channel_patch_topk_groups = int(args.channel_patch_topk_groups)
        cfg.channel_patch_group_size = int(args.channel_patch_group_size)
        cfg.channel_patch_gate = float(args.channel_patch_gate)
        cfg.channel_patch_k_success = int(args.channel_patch_k_success)
        cfg.channel_patch_k_failure = int(args.channel_patch_k_failure)
        cfg.channel_patch_same_task_only = bool(args.channel_patch_same_task_only)
        cfg.channel_patch_random_seed = int(args.channel_patch_random_seed)
        cfg.joint_pair_memory_npz = args.joint_pair_memory_npz
        cfg.joint_energy_rerank_ckpt = args.joint_energy_rerank_ckpt
        cfg.joint_pair_topk = int(args.joint_pair_topk)
        cfg.joint_pair_same_task_only = bool(args.joint_pair_same_task_only)
        cfg.joint_pair_target_future_key = args.joint_pair_target_future_key
        cfg.joint_pair_without_trigger = bool(args.joint_pair_without_trigger)
        cfg.joint_pair_min_energy_margin = float(args.joint_pair_min_energy_margin)
        cfg.joint_pair_max_energy = float(args.joint_pair_max_energy)
        cfg.joint_pair_min_similarity = float(args.joint_pair_min_similarity)
        cfg.joint_pair_refine_steps = int(args.joint_pair_refine_steps)
        cfg.joint_pair_refine_future_lr = float(args.joint_pair_refine_future_lr)
        cfg.joint_pair_refine_action_lr = float(args.joint_pair_refine_action_lr)
        cfg.joint_pair_refine_grad_clip = float(args.joint_pair_refine_grad_clip)
        cfg.joint_pair_refine_future_max_delta = float(args.joint_pair_refine_future_max_delta)
        cfg.joint_pair_refine_action_max_delta = float(args.joint_pair_refine_action_max_delta)
        cfg.joint_pair_refine_min_delta_energy = float(args.joint_pair_refine_min_delta_energy)
        cfg.joint_pair_refine_backtrack_factor = float(args.joint_pair_refine_backtrack_factor)
        cfg.joint_pair_refine_min_step_scale = float(args.joint_pair_refine_min_step_scale)
        cfg.joint_pair_refine_max_backtracks = int(args.joint_pair_refine_max_backtracks)
        cfg.joint_pair_refine_early_stop_patience = int(args.joint_pair_refine_early_stop_patience)
        cfg.joint_pair_refine_early_stop_min_improve = float(args.joint_pair_refine_early_stop_min_improve)
        cfg.joint_pair_refine_early_stop_grad_norm = float(args.joint_pair_refine_early_stop_grad_norm)
        cfg.joint_pair_refine_lambda_exp = float(args.joint_pair_refine_lambda_exp)
        cfg.joint_pair_refine_lambda_latent = float(args.joint_pair_refine_lambda_latent)
        cfg.joint_pair_refine_lambda_physical = float(args.joint_pair_refine_lambda_physical)
        cfg.joint_pair_refine_normalize_terms = bool(args.joint_pair_refine_normalize_terms)
        cfg.joint_pair_refine_lambda_dynamics = float(args.joint_pair_refine_lambda_dynamics)
        cfg.joint_pair_refine_lambda_future_anchor = float(args.joint_pair_refine_lambda_future_anchor)
        cfg.joint_pair_refine_lambda_action_anchor = float(args.joint_pair_refine_lambda_action_anchor)
        cfg.joint_pair_refine_action_mix = float(args.joint_pair_refine_action_mix)
        cfg.joint_pair_refine_fallback = bool(args.joint_pair_refine_fallback)
        cfg.joint_pair_push_sensitive_action_mix = float(args.joint_pair_push_sensitive_action_mix)
        cfg.joint_pair_push_sensitive_max_action_shift = float(args.joint_pair_push_sensitive_max_action_shift)
        cfg.joint_pair_action_generator_ckpt = args.joint_pair_action_generator_ckpt
        cfg.joint_pair_action_generator_mix = float(args.joint_pair_action_generator_mix)
        cfg.joint_pair_direct_generate_only = bool(args.joint_pair_direct_generate_only)
        cfg.joint_pair_no_memory = bool(args.joint_pair_no_memory)
        cfg.joint_pair_retry_only = bool(args.joint_pair_retry_only)
        cfg.joint_pair_dynamics_probe_ckpt = args.joint_pair_dynamics_probe_ckpt
        cfg.joint_pair_lambda_dynamics = float(args.joint_pair_lambda_dynamics)
        cfg.joint_pair_repair_ckpt = args.joint_pair_repair_ckpt
        cfg.joint_pair_repair_critic_ckpt = args.joint_pair_repair_critic_ckpt
        cfg.joint_pair_summary_future_adapter_ckpt = args.joint_pair_summary_future_adapter_ckpt
        cfg.joint_pair_repair_advantage_gate = bool(args.joint_pair_repair_advantage_gate)
        cfg.joint_pair_repair_advantage_threshold = float(args.joint_pair_repair_advantage_threshold)
        cfg.joint_pair_repair_advantage_logit_threshold = float(args.joint_pair_repair_advantage_logit_threshold)
        cfg.joint_pair_repair_future_mix = float(args.joint_pair_repair_future_mix)
        cfg.joint_pair_repair_action_mix = float(args.joint_pair_repair_action_mix)
        cfg.joint_pair_repair_with_coupling_prior = bool(args.joint_pair_repair_with_coupling_prior)
        cfg.joint_pair_repair_manifold_min_improve = float(args.joint_pair_repair_manifold_min_improve)
        cfg.joint_pair_manifold_ckpt = args.joint_pair_manifold_ckpt
        cfg.joint_pair_cross_stage_ckpt = args.joint_pair_cross_stage_ckpt
        cfg.joint_pair_cross_stage_topk = int(args.joint_pair_cross_stage_topk)
        cfg.joint_pair_repair_manifold_max_dist = float(args.joint_pair_repair_manifold_max_dist)
        cfg.joint_pair_repair_manifold_min_success_gain = float(args.joint_pair_repair_manifold_min_success_gain)
        cfg.joint_pair_max_future_shift = float(args.joint_pair_max_future_shift)
        cfg.joint_pair_value_ckpt = args.joint_pair_value_ckpt
        cfg.joint_pair_lambda_value = float(args.joint_pair_lambda_value)
        cfg.joint_pair_lambda_cross_stage = float(args.joint_pair_lambda_cross_stage)
        cfg.joint_pair_verify_acceptance = bool(args.joint_pair_verify_acceptance)
        cfg.joint_pair_verify_exp_max = float(args.joint_pair_verify_exp_max)
        cfg.joint_pair_verify_latent_max = float(args.joint_pair_verify_latent_max)
        cfg.joint_pair_verify_physical_max = float(args.joint_pair_verify_physical_max)
        cfg.joint_pair_verify_require_improve = bool(args.joint_pair_verify_require_improve)
        cfg.joint_pair_verify_min_improve = float(args.joint_pair_verify_min_improve)
        cfg.no_qwen_reflection = bool(args.no_qwen_reflection)
        cfg.print_collection_rows = bool(args.print_collection_rows)
        cfg.collect_strong_gt = bool(args.collect_strong_gt)
        cfg.print_strong_gt_rows = bool(args.print_strong_gt_rows)
        cfg.collect_action_trace = bool(args.collect_action_trace)
        cfg.action_trace_dir = args.action_trace_dir
        cfg.collect_joint_belief_transition = bool(args.collect_joint_belief_transition)
        cfg.collect_online_acceptance_rows = bool(args.collect_online_acceptance_rows)
        cfg.dynamic_coupling_operator_ckpt = args.dynamic_coupling_operator_ckpt
        cfg.dynamic_coupling_online = bool(args.dynamic_coupling_online)
        cfg.dynamic_coupling_first_pass = bool(args.dynamic_coupling_first_pass)
        cfg.dynamic_coupling_generate_action_intent = bool(args.dynamic_coupling_generate_action_intent)
        cfg.dynamic_coupling_apply_action_intent_to_policy = bool(args.dynamic_coupling_apply_action_intent_to_policy)
        cfg.dynamic_coupling_persistent_hypothesis = bool(args.dynamic_coupling_persistent_hypothesis)
        cfg.dynamic_coupling_hypothesis_future_update_mix = float(args.dynamic_coupling_hypothesis_future_update_mix)
        cfg.dynamic_coupling_hypothesis_action_update_mix = float(args.dynamic_coupling_hypothesis_action_update_mix)
        cfg.dynamic_coupling_need_gain = float(args.dynamic_coupling_need_gain)
        cfg.dynamic_coupling_min_scale = float(args.dynamic_coupling_min_scale)
        cfg.dynamic_coupling_max_scale = float(args.dynamic_coupling_max_scale)
        cfg.dynamic_coupling_keep_actions = int(args.dynamic_coupling_keep_actions)
        cfg.dynamic_coupling_min_committed = float(args.dynamic_coupling_min_committed)
        cfg.dynamic_coupling_max_future_shift_norm = float(args.dynamic_coupling_max_future_shift_norm)
        cfg.dynamic_coupling_max_future_shift_ratio = float(args.dynamic_coupling_max_future_shift_ratio)
        cfg.dynamic_coupling_future_mix = float(args.dynamic_coupling_future_mix)
        cfg.dynamic_coupling_auto_shrink = bool(args.dynamic_coupling_auto_shrink)
        cfg.dynamic_coupling_gate_every_k = int(args.dynamic_coupling_gate_every_k)
        cfg.dynamic_coupling_gate_start_step = int(args.dynamic_coupling_gate_start_step)
        cfg.dynamic_coupling_gate_with_trigger = bool(args.dynamic_coupling_gate_with_trigger)
        cfg.dynamic_coupling_trigger_threshold = float(args.dynamic_coupling_trigger_threshold)
        cfg.dynamic_coupling_patch_remaining_actions = bool(args.dynamic_coupling_patch_remaining_actions)
        cfg.dynamic_coupling_patch_mode = args.dynamic_coupling_patch_mode
        cfg.dynamic_coupling_patch_mix = float(args.dynamic_coupling_patch_mix)
        cfg.dynamic_coupling_delta_action_repair_ckpt = args.dynamic_coupling_delta_action_repair_ckpt
        cfg.dynamic_coupling_delta_action_repair_mix = float(args.dynamic_coupling_delta_action_repair_mix)
        cfg.dynamic_coupling_chunk_effect_ckpt = args.dynamic_coupling_chunk_effect_ckpt
        cfg.dynamic_coupling_chunk_keep_threshold = float(args.dynamic_coupling_chunk_keep_threshold)
        cfg.dynamic_coupling_chunk_replan_threshold = float(args.dynamic_coupling_chunk_replan_threshold)
        cfg.dynamic_coupling_chunk_success_threshold = float(args.dynamic_coupling_chunk_success_threshold)
        cfg.dynamic_coupling_chunk_low_compat_threshold = float(args.dynamic_coupling_chunk_low_compat_threshold)
        cfg.dynamic_coupling_disable_replan = bool(args.dynamic_coupling_disable_replan)
        cfg.dynamic_coupling_suffix_trigger_success_below = float(args.dynamic_coupling_suffix_trigger_success_below)
        cfg.dynamic_coupling_suffix_trigger_residual_above = float(args.dynamic_coupling_suffix_trigger_residual_above)
        cfg.dynamic_coupling_suffix_trigger_compat_below = float(args.dynamic_coupling_suffix_trigger_compat_below)
        cfg.joint_belief_transition = bool(args.joint_belief_transition)
        cfg.joint_belief_transition_ckpt = args.joint_belief_transition_ckpt
        cfg.joint_belief_transition_direct_suffix = bool(args.joint_belief_transition_direct_suffix)
        cfg.joint_belief_transition_sparse_suffix = bool(args.joint_belief_transition_sparse_suffix)
        cfg.joint_belief_transition_min_learned_gate = float(args.joint_belief_transition_min_learned_gate)
        cfg.joint_belief_transition_max_success_prob = float(args.joint_belief_transition_max_success_prob)
        cfg.joint_belief_clean_controller = bool(args.joint_belief_clean_controller)
        cfg.joint_belief_clean_min_confidence = float(args.joint_belief_clean_min_confidence)
        cfg.joint_belief_clean_patch_min_prob = float(args.joint_belief_clean_patch_min_prob)
        cfg.joint_belief_clean_patch_margin = float(args.joint_belief_clean_patch_margin)
        cfg.joint_belief_clean_replan_min_prob = float(args.joint_belief_clean_replan_min_prob)
        cfg.joint_belief_clean_h_update_scale = float(args.joint_belief_clean_h_update_scale)
        cfg.joint_belief_summary_future_adapter_update = bool(args.joint_belief_summary_future_adapter_update)
        cfg.joint_belief_summary_future_adapter_delta_key = args.joint_belief_summary_future_adapter_delta_key
        cfg.joint_belief_summary_future_adapter_delta_scale = float(args.joint_belief_summary_future_adapter_delta_scale)
        cfg.joint_belief_summary_future_adapter_mix = float(args.joint_belief_summary_future_adapter_mix)
        cfg.joint_belief_summary_future_adapter_max_shift_norm = float(args.joint_belief_summary_future_adapter_max_shift_norm)
        cfg.joint_belief_patch_min_interval_steps = int(args.joint_belief_patch_min_interval_steps)
        cfg.joint_belief_patch_max_per_subtask = int(args.joint_belief_patch_max_per_subtask)
        cfg.suffix_trigger_mlp_ckpt = args.suffix_trigger_mlp_ckpt
        cfg.suffix_trigger_mlp_threshold = float(args.suffix_trigger_mlp_threshold)
        cfg.counterfactual_suffix_transport = bool(args.counterfactual_suffix_transport)
        cfg.counterfactual_suffix_transport_sigma = float(args.counterfactual_suffix_transport_sigma)
        cfg.counterfactual_suffix_transport_edit_steps = int(args.counterfactual_suffix_transport_edit_steps)
        cfg.counterfactual_suffix_transport_step_size = float(args.counterfactual_suffix_transport_step_size)
        cfg.counterfactual_suffix_transport_max_abs_delta = float(args.counterfactual_suffix_transport_max_abs_delta)
        cfg.linear_suffix_sensitivity_transport = bool(args.linear_suffix_sensitivity_transport)
        cfg.linear_suffix_sensitivity_rank = int(args.linear_suffix_sensitivity_rank)
        cfg.linear_suffix_sensitivity_eps = float(args.linear_suffix_sensitivity_eps)
        cfg.linear_suffix_sensitivity_step_size = float(args.linear_suffix_sensitivity_step_size)
        cfg.linear_suffix_sensitivity_max_abs_delta = float(args.linear_suffix_sensitivity_max_abs_delta)
        cfg.linear_suffix_sensitivity_min_step_score = float(args.linear_suffix_sensitivity_min_step_score)
        cfg.linear_suffix_sensitivity_selective_step_threshold = float(args.linear_suffix_sensitivity_selective_step_threshold)
        cfg.linear_suffix_sensitivity_max_patch_fraction = float(args.linear_suffix_sensitivity_max_patch_fraction)
        cfg.linear_suffix_sensitivity_replan_fraction = float(args.linear_suffix_sensitivity_replan_fraction)
        cfg.linear_suffix_sensitivity_replan_delta_h_norm = float(args.linear_suffix_sensitivity_replan_delta_h_norm)
        cfg.linear_suffix_sensitivity_smooth_weight = float(args.linear_suffix_sensitivity_smooth_weight)
        cfg.linear_suffix_sensitivity_continuity_weight = float(args.linear_suffix_sensitivity_continuity_weight)
        cfg.linear_suffix_sensitivity_mpc_iters = int(args.linear_suffix_sensitivity_mpc_iters)
        cfg.collect_sensitivity_teacher = bool(args.collect_sensitivity_teacher)
        cfg.sensitivity_teacher_mode = str(args.sensitivity_teacher_mode or "jvp")
        cfg.sensitivity_teacher_step_size = float(args.sensitivity_teacher_step_size)
        cfg.sensitivity_teacher_max_abs_delta = float(args.sensitivity_teacher_max_abs_delta)
        cfg.sensitivity_teacher_joint_delta_to_action_scale = float(args.sensitivity_teacher_joint_delta_to_action_scale)
        cfg.sensitivity_teacher_condition_ablation = str(args.sensitivity_teacher_condition_ablation or "joint")
        cfg.joint_execution_update_ablation_mode = str(args.joint_execution_update_ablation_mode or "rssm")
        cfg.suffix_mask_ablation_mode = str(args.suffix_mask_ablation_mode or "off")
        cfg.suffix_mask_ablation_force_writeback = bool(args.suffix_mask_ablation_force_writeback)
        cfg.suffix_mask_ablation_teacher_step_threshold = float(args.suffix_mask_ablation_teacher_step_threshold)
        cfg.suffix_mask_ablation_max_abs_delta = float(args.suffix_mask_ablation_max_abs_delta)
        cfg.suffix_mask_ablation_min_affected_fraction = float(args.suffix_mask_ablation_min_affected_fraction)
        cfg.suffix_mask_ablation_max_affected_fraction = float(args.suffix_mask_ablation_max_affected_fraction)
        cfg.suffix_mask_ablation_seed = int(args.suffix_mask_ablation_seed)
        cfg.suffix_mask_ablation_schedule_json = str(args.suffix_mask_ablation_schedule_json or "")
        SUFFIX_MASK_ABLATION_SCHEDULE_SET = set()
        if cfg.suffix_mask_ablation_schedule_json:
            schedule_path = Path(cfg.suffix_mask_ablation_schedule_json)
            with schedule_path.open("r", encoding="utf-8") as handle:
                schedule_payload = json.load(handle)
            schedule_rows = schedule_payload.get("cases", schedule_payload) if isinstance(schedule_payload, dict) else schedule_payload
            for item in schedule_rows:
                if not isinstance(item, dict):
                    continue
                SUFFIX_MASK_ABLATION_SCHEDULE_SET.add(
                    (
                        int(item.get("sequence_index", -1)),
                        int(item.get("subtask_index", -1)),
                        int(item.get("step", -1)),
                    )
                )
        needs_jvp_teacher_cache = bool(cfg.collect_sensitivity_teacher) and str(
            getattr(cfg, "sensitivity_teacher_mode", "jvp") or "jvp"
        ) == "jvp"
        if cfg.linear_suffix_sensitivity_transport or needs_jvp_teacher_cache or cfg.suffix_mask_ablation_mode in {
            "gt_selective",
            "random_mask",
            "complement_mask",
            "whole_suffix",
        }:
            os.environ["DEFI_SUFFIX_SENSITIVITY_CACHE"] = "1"
            os.environ["DEFI_SUFFIX_SENSITIVITY_RANK"] = str(cfg.linear_suffix_sensitivity_rank)
            os.environ["DEFI_SUFFIX_SENSITIVITY_EPS"] = str(cfg.linear_suffix_sensitivity_eps)
        cfg.directional_sensitivity_tcn_transport = bool(args.directional_sensitivity_tcn_transport)
        cfg.directional_sensitivity_tcn_ckpt = str(args.directional_sensitivity_tcn_ckpt or "")
        cfg.directional_sensitivity_tcn_max_abs_delta = float(args.directional_sensitivity_tcn_max_abs_delta)
        cfg.directional_sensitivity_tcn_min_delta_norm = float(args.directional_sensitivity_tcn_min_delta_norm)
        cfg.directional_sensitivity_tcn_step_threshold = float(args.directional_sensitivity_tcn_step_threshold)
        cfg.directional_sensitivity_tcn_top_k = int(args.directional_sensitivity_tcn_top_k)
        cfg.directional_sensitivity_tcn_top_k_floor = float(args.directional_sensitivity_tcn_top_k_floor)
        cfg.directional_sensitivity_tcn_top_k_min_margin = float(args.directional_sensitivity_tcn_top_k_min_margin)
        cfg.directional_sensitivity_tcn_force_top_k_patch = bool(args.directional_sensitivity_tcn_force_top_k_patch)
        cfg.directional_sensitivity_tcn_max_patch_fraction = float(args.directional_sensitivity_tcn_max_patch_fraction)
        cfg.directional_sensitivity_tcn_max_affected_segments = int(args.directional_sensitivity_tcn_max_affected_segments)
        cfg.directional_sensitivity_tcn_replan_delta_h_norm = float(args.directional_sensitivity_tcn_replan_delta_h_norm)
        cfg.directional_sensitivity_tcn_smooth_weight = float(args.directional_sensitivity_tcn_smooth_weight)
        cfg.directional_sensitivity_tcn_mpc_iters = int(args.directional_sensitivity_tcn_mpc_iters)
        cfg.directional_sensitivity_tcn_structured_mask = bool(args.directional_sensitivity_tcn_structured_mask)
        cfg.directional_sensitivity_tcn_validity_threshold = float(args.directional_sensitivity_tcn_validity_threshold)
        cfg.directional_sensitivity_tcn_uncertainty_replan_threshold = float(
            args.directional_sensitivity_tcn_uncertainty_replan_threshold
        )
        cfg.directional_sensitivity_tcn_selection_lambda = float(args.directional_sensitivity_tcn_selection_lambda)
        cfg.directional_sensitivity_tcn_selection_beta = float(args.directional_sensitivity_tcn_selection_beta)
        cfg.direct_action_correction_tcn_transport = bool(args.direct_action_correction_tcn_transport)
        cfg.direct_action_correction_tcn_ckpt = str(args.direct_action_correction_tcn_ckpt or "")
        cfg.joint_execution_update = bool(args.joint_execution_update or args.joint_belief_transition)
        cfg.joint_execution_update_ckpt = args.joint_belief_transition_ckpt or args.joint_execution_update_ckpt
        cfg.joint_execution_gate_mode = args.joint_execution_gate_mode
        cfg.joint_execution_update_threshold = float(args.joint_execution_update_threshold)
        cfg.joint_execution_update_gain = float(args.joint_execution_update_gain)
        cfg.joint_execution_update_min_gate = float(args.joint_execution_update_min_gate)
        cfg.joint_execution_update_max_gate = float(args.joint_execution_update_max_gate)
        cfg.joint_execution_suffix_error_threshold = float(args.joint_execution_suffix_error_threshold)
        cfg.joint_execution_error_residual_scale = float(args.joint_execution_error_residual_scale)
        cfg.joint_execution_error_success_target = float(args.joint_execution_error_success_target)
        cfg.joint_execution_error_compat_target = float(args.joint_execution_error_compat_target)
        cfg.joint_execution_error_success_weight = float(args.joint_execution_error_success_weight)
        cfg.joint_execution_error_compat_weight = float(args.joint_execution_error_compat_weight)
        cfg.debug_action_before_env_step = bool(args.debug_action_before_env_step)
        cfg.debug_action_steps = int(args.debug_action_steps)
        cfg.disable_retry_after_first_pass = bool(args.disable_retry_after_first_pass)
        cfg.retry_requires_suffix_patch = bool(args.retry_requires_suffix_patch)
        cfg.retry_rollback_mode = str(args.retry_rollback_mode)
        cfg.retry_max_attempts = int(args.retry_max_attempts)
        cfg.retry_baseline_fallback_on_failure = bool(args.retry_baseline_fallback_on_failure)
        cfg._future_trace_count = 0
        cfg._channel_patch_count = 0
        cfg._action_trace_count = 0
        if args.eval_sequences_path:
            cfg.eval_sequences = load_eval_sequences(args.eval_sequences_path)
            if args.num_sequences > 0:
                cfg.eval_sequences = cfg.eval_sequences[: int(args.num_sequences)]
            if args.eval_task_filter:
                cfg.eval_sequences = filter_eval_sequences_by_task(cfg.eval_sequences, args.eval_task_filter)
            cfg.num_sequences = len(cfg.eval_sequences)
    main(cfg)
