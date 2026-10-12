#!/usr/bin/env bash
set -euo pipefail

# Conservative no-retry CALVIN rollout for the joint-belief suffix controller.
# By default z_a is kept inside the joint hypothesis and is NOT injected into
# the policy decoder. Do not add --dynamic_coupling_apply_action_intent_to_policy
# unless you explicitly want to test decoder-side action-intent conditioning.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${STEP3_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
EP_LEN="${EP_LEN:-360}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DISABLE_RETRY_AFTER_FIRST_PASS="${DISABLE_RETRY_AFTER_FIRST_PASS:-1}"
RETRY_ROLLBACK_MODE="${RETRY_ROLLBACK_MODE:-subtask_start}"
RETRY_BASELINE_FALLBACK_ON_FAILURE="${RETRY_BASELINE_FALLBACK_ON_FAILURE:-0}"
RETRY_MAX_ATTEMPTS="${RETRY_MAX_ATTEMPTS:-1}"
DYNAMIC_COUPLING_FIRST_PASS="${DYNAMIC_COUPLING_FIRST_PASS:-1}"
DYNAMIC_COUPLING_ONLINE="${DYNAMIC_COUPLING_ONLINE:-1}"
DYNAMIC_COUPLING_GENERATE_ACTION_INTENT="${DYNAMIC_COUPLING_GENERATE_ACTION_INTENT:-1}"
DYNAMIC_COUPLING_PERSISTENT_HYPOTHESIS="${DYNAMIC_COUPLING_PERSISTENT_HYPOTHESIS:-1}"
JOINT_BELIEF_TRANSITION="${JOINT_BELIEF_TRANSITION:-1}"

VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm}"
ACTION_MODEL_FOLDER="${ACTION_MODEL_FOLDER:-/mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ACTION_MODEL_FOLDER}/saved_models/step3_defi.pt}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32}"
T5_MODEL_PATH="${T5_MODEL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/t5_base}"
LANGUAGE_GOAL_PATH="${LANGUAGE_GOAL_PATH:-/mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt}"
CALVIN_ABC_DIR="${CALVIN_ABC_DIR:-/mnt/workspace/calvin/task_ABC_D}"
EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-/mnt/workspace/manipulation/DeFi/outputs/splits/outcome_router_clean_1k_train1000.json}"
JOINT_BELIEF_TRANSITION_CKPT="${JOINT_BELIEF_TRANSITION_CKPT:-/mnt/workspace/manipulation/DeFi/outputs/full_rssm_plan_compat_v2_success_balanced_650x2/rssm_plan_compatibility.pt}"
JOINT_PAIR_ACTION_GENERATOR_CKPT="${JOINT_PAIR_ACTION_GENERATOR_CKPT:-/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt}"

# Conservative suffix defaults from the Sep 3 no-retry run:
#  - patch rarely
#  - patch only with clear patch-vs-keep margin
#  - apply a very small suffix delta
JOINT_BELIEF_CLEAN_PATCH_MIN_PROB="${JOINT_BELIEF_CLEAN_PATCH_MIN_PROB:-0.55}"
JOINT_BELIEF_CLEAN_PATCH_MARGIN="${JOINT_BELIEF_CLEAN_PATCH_MARGIN:-0.10}"
JOINT_BELIEF_CLEAN_REPLAN_MIN_PROB="${JOINT_BELIEF_CLEAN_REPLAN_MIN_PROB:-0.85}"
JOINT_BELIEF_CLEAN_H_UPDATE_SCALE="${JOINT_BELIEF_CLEAN_H_UPDATE_SCALE:-0.0}"
JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_UPDATE="${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_UPDATE:-0}"
JOINT_PAIR_SUMMARY_FUTURE_ADAPTER_CKPT="${JOINT_PAIR_SUMMARY_FUTURE_ADAPTER_CKPT:-}"
JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_DELTA_KEY="${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_DELTA_KEY:-future_delta}"
JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_DELTA_SCALE="${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_DELTA_SCALE:-1.0}"
JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_MIX="${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_MIX:-1.0}"
JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_MAX_SHIFT_NORM="${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_MAX_SHIFT_NORM:-0.0}"
JOINT_BELIEF_PATCH_MIN_INTERVAL_STEPS="${JOINT_BELIEF_PATCH_MIN_INTERVAL_STEPS:-1}"
JOINT_BELIEF_PATCH_MAX_PER_SUBTASK="${JOINT_BELIEF_PATCH_MAX_PER_SUBTASK:-0}"
DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX="${DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX:-0.02}"
DYNAMIC_COUPLING_GATE_EVERY_K="${DYNAMIC_COUPLING_GATE_EVERY_K:-1}"
DYNAMIC_COUPLING_DISABLE_REPLAN="${DYNAMIC_COUPLING_DISABLE_REPLAN:-1}"
JOINT_BELIEF_CLEAN_CONTROLLER="${JOINT_BELIEF_CLEAN_CONTROLLER:-1}"

# Optional policy-sensitivity suffix transport. Disabled by default so the
# script still matches the old conservative no-retry setting unless requested.
LINEAR_SUFFIX_SENSITIVITY="${LINEAR_SUFFIX_SENSITIVITY:-0}"
LINEAR_SUFFIX_SENSITIVITY_RANK="${LINEAR_SUFFIX_SENSITIVITY_RANK:-8}"
LINEAR_SUFFIX_SENSITIVITY_EPS="${LINEAR_SUFFIX_SENSITIVITY_EPS:-0.01}"
LINEAR_SUFFIX_SENSITIVITY_STEP_SIZE="${LINEAR_SUFFIX_SENSITIVITY_STEP_SIZE:-0.03}"
LINEAR_SUFFIX_SENSITIVITY_MAX_ABS_DELTA="${LINEAR_SUFFIX_SENSITIVITY_MAX_ABS_DELTA:-0.015}"
LINEAR_SUFFIX_SENSITIVITY_MIN_STEP_SCORE="${LINEAR_SUFFIX_SENSITIVITY_MIN_STEP_SCORE:-0.0}"
LINEAR_SUFFIX_SENSITIVITY_SELECTIVE_STEP_THRESHOLD="${LINEAR_SUFFIX_SENSITIVITY_SELECTIVE_STEP_THRESHOLD:-0.020}"
LINEAR_SUFFIX_SENSITIVITY_MAX_PATCH_FRACTION="${LINEAR_SUFFIX_SENSITIVITY_MAX_PATCH_FRACTION:-0.35}"
LINEAR_SUFFIX_SENSITIVITY_REPLAN_FRACTION="${LINEAR_SUFFIX_SENSITIVITY_REPLAN_FRACTION:-0.70}"
LINEAR_SUFFIX_SENSITIVITY_REPLAN_DELTA_H_NORM="${LINEAR_SUFFIX_SENSITIVITY_REPLAN_DELTA_H_NORM:-3.0}"
LINEAR_SUFFIX_SENSITIVITY_SMOOTH_WEIGHT="${LINEAR_SUFFIX_SENSITIVITY_SMOOTH_WEIGHT:-0.10}"
LINEAR_SUFFIX_SENSITIVITY_CONTINUITY_WEIGHT="${LINEAR_SUFFIX_SENSITIVITY_CONTINUITY_WEIGHT:-0.10}"
LINEAR_SUFFIX_SENSITIVITY_MPC_ITERS="${LINEAR_SUFFIX_SENSITIVITY_MPC_ITERS:-2}"
DIRECTIONAL_SENSITIVITY_TCN="${DIRECTIONAL_SENSITIVITY_TCN:-0}"
DIRECTIONAL_SENSITIVITY_TCN_CKPT="${DIRECTIONAL_SENSITIVITY_TCN_CKPT:-/mnt/workspace/manipulation/DeFi/outputs/directional_sensitivity_tcn_v1_success_balanced_650x2/directional_sensitivity_tcn.pt}"
DIRECTIONAL_SENSITIVITY_TCN_MAX_ABS_DELTA="${DIRECTIONAL_SENSITIVITY_TCN_MAX_ABS_DELTA:-0.0015}"
DIRECTIONAL_SENSITIVITY_TCN_MIN_DELTA_NORM="${DIRECTIONAL_SENSITIVITY_TCN_MIN_DELTA_NORM:-0.0}"
DIRECTIONAL_SENSITIVITY_TCN_STEP_THRESHOLD="${DIRECTIONAL_SENSITIVITY_TCN_STEP_THRESHOLD:-0.0039}"
DIRECTIONAL_SENSITIVITY_TCN_TOP_K="${DIRECTIONAL_SENSITIVITY_TCN_TOP_K:-0}"
DIRECTIONAL_SENSITIVITY_TCN_TOP_K_FLOOR="${DIRECTIONAL_SENSITIVITY_TCN_TOP_K_FLOOR:-0.0}"
DIRECTIONAL_SENSITIVITY_TCN_TOP_K_MIN_MARGIN="${DIRECTIONAL_SENSITIVITY_TCN_TOP_K_MIN_MARGIN:-0.0}"
DIRECTIONAL_SENSITIVITY_TCN_FORCE_TOP_K_PATCH="${DIRECTIONAL_SENSITIVITY_TCN_FORCE_TOP_K_PATCH:-0}"
DIRECTIONAL_SENSITIVITY_TCN_MAX_PATCH_FRACTION="${DIRECTIONAL_SENSITIVITY_TCN_MAX_PATCH_FRACTION:-0.35}"
DIRECTIONAL_SENSITIVITY_TCN_MAX_AFFECTED_SEGMENTS="${DIRECTIONAL_SENSITIVITY_TCN_MAX_AFFECTED_SEGMENTS:-3}"
DIRECTIONAL_SENSITIVITY_TCN_REPLAN_DELTA_H_NORM="${DIRECTIONAL_SENSITIVITY_TCN_REPLAN_DELTA_H_NORM:-4.2}"
DIRECTIONAL_SENSITIVITY_TCN_SMOOTH_WEIGHT="${DIRECTIONAL_SENSITIVITY_TCN_SMOOTH_WEIGHT:-0.10}"
DIRECTIONAL_SENSITIVITY_TCN_MPC_ITERS="${DIRECTIONAL_SENSITIVITY_TCN_MPC_ITERS:-2}"
DIRECTIONAL_SENSITIVITY_TCN_STRUCTURED_MASK="${DIRECTIONAL_SENSITIVITY_TCN_STRUCTURED_MASK:-0}"
DIRECTIONAL_SENSITIVITY_TCN_VALIDITY_THRESHOLD="${DIRECTIONAL_SENSITIVITY_TCN_VALIDITY_THRESHOLD:-0.5}"
DIRECTIONAL_SENSITIVITY_TCN_UNCERTAINTY_REPLAN_THRESHOLD="${DIRECTIONAL_SENSITIVITY_TCN_UNCERTAINTY_REPLAN_THRESHOLD:-0.0}"
DIRECTIONAL_SENSITIVITY_TCN_SELECTION_LAMBDA="${DIRECTIONAL_SENSITIVITY_TCN_SELECTION_LAMBDA:-0.5}"
DIRECTIONAL_SENSITIVITY_TCN_SELECTION_BETA="${DIRECTIONAL_SENSITIVITY_TCN_SELECTION_BETA:-0.15}"
DIRECT_ACTION_CORRECTION_TCN="${DIRECT_ACTION_CORRECTION_TCN:-0}"
DIRECT_ACTION_CORRECTION_TCN_CKPT="${DIRECT_ACTION_CORRECTION_TCN_CKPT:-}"
COLLECT_SENSITIVITY_TEACHER="${COLLECT_SENSITIVITY_TEACHER:-0}"
SENSITIVITY_TEACHER_MODE="${SENSITIVITY_TEACHER_MODE:-jvp}"
SENSITIVITY_TEACHER_STEP_SIZE="${SENSITIVITY_TEACHER_STEP_SIZE:-1.0}"
SENSITIVITY_TEACHER_MAX_ABS_DELTA="${SENSITIVITY_TEACHER_MAX_ABS_DELTA:-0.0}"
SENSITIVITY_TEACHER_JOINT_DELTA_TO_ACTION_SCALE="${SENSITIVITY_TEACHER_JOINT_DELTA_TO_ACTION_SCALE:-1.0}"
SENSITIVITY_TEACHER_CONDITION_ABLATION="${SENSITIVITY_TEACHER_CONDITION_ABLATION:-joint}"
JOINT_EXECUTION_UPDATE_ABLATION_MODE="${JOINT_EXECUTION_UPDATE_ABLATION_MODE:-rssm}"
SUFFIX_MASK_ABLATION_MODE="${SUFFIX_MASK_ABLATION_MODE:-off}"
SUFFIX_MASK_ABLATION_FORCE_WRITEBACK="${SUFFIX_MASK_ABLATION_FORCE_WRITEBACK:-0}"
SUFFIX_MASK_ABLATION_TEACHER_STEP_THRESHOLD="${SUFFIX_MASK_ABLATION_TEACHER_STEP_THRESHOLD:-0.0001}"
SUFFIX_MASK_ABLATION_MAX_ABS_DELTA="${SUFFIX_MASK_ABLATION_MAX_ABS_DELTA:-0.0}"
SUFFIX_MASK_ABLATION_MIN_AFFECTED_FRACTION="${SUFFIX_MASK_ABLATION_MIN_AFFECTED_FRACTION:-0.0}"
SUFFIX_MASK_ABLATION_MAX_AFFECTED_FRACTION="${SUFFIX_MASK_ABLATION_MAX_AFFECTED_FRACTION:-1.0}"
SUFFIX_MASK_ABLATION_SEED="${SUFFIX_MASK_ABLATION_SEED:-0}"
SUFFIX_MASK_ABLATION_SCHEDULE_JSON="${SUFFIX_MASK_ABLATION_SCHEDULE_JSON:-}"

unset EGL_VISIBLE_DEVICES
unset EGL_VISIBLE_DEVICE

mkdir -p /tmp/nvidia470199
ln -sfn /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.470.199.02 /tmp/nvidia470199/libEGL_nvidia.so.0
ln -sfn /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.470.199.02 /tmp/nvidia470199/libEGL_nvidia.so
ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-glsi.so.470.199.02 /tmp/nvidia470199/libnvidia-glsi.so.470.199.02
ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.470.199.02 /tmp/nvidia470199/libnvidia-eglcore.so.470.199.02
cat > /tmp/nvidia470199/10_nvidia_470199.json <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/tmp/nvidia470199/libEGL_nvidia.so.0"
    }
}
JSON

export EGL_VISIBLE_DEVICE="${EGL_VISIBLE_DEVICE:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/tmp/nvidia470199/10_nvidia_470199.json}"
export LD_LIBRARY_PATH="/tmp/nvidia470199:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/tmp/nvidia_470199:/usr/local/nvidia/lib:/root/CoppeliaSim:${LD_LIBRARY_PATH:-}"
export DEFI_T5_SANITIZE="${DEFI_T5_SANITIZE:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${STEP3_DIR}:${PYTHONPATH:-}"

cd "${STEP3_DIR}"

ARGS=(
  policy_evaluation/calvin_evaluate_with_memory_reflection.py
  --video_model_path "${VIDEO_MODEL_PATH}"
  --action_model_folder "${ACTION_MODEL_FOLDER}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --clip_model_path "${CLIP_MODEL_PATH}"
  --t5_model_path "${T5_MODEL_PATH}"
  --language_goal_path "${LANGUAGE_GOAL_PATH}"
  --calvin_abc_dir "${CALVIN_ABC_DIR}"
  --eval_sequences_path "${EVAL_SEQUENCES_PATH}"
  --num_sequences "${NUM_SEQUENCES}"
  --ep_len "${EP_LEN}"
  --no_qwen_reflection
  --retry_rollback_mode "${RETRY_ROLLBACK_MODE}"
  --retry_max_attempts "${RETRY_MAX_ATTEMPTS}"
  --joint_belief_clean_patch_min_prob "${JOINT_BELIEF_CLEAN_PATCH_MIN_PROB}"
  --joint_belief_clean_patch_margin "${JOINT_BELIEF_CLEAN_PATCH_MARGIN}"
  --joint_belief_clean_replan_min_prob "${JOINT_BELIEF_CLEAN_REPLAN_MIN_PROB}"
  --joint_belief_clean_h_update_scale "${JOINT_BELIEF_CLEAN_H_UPDATE_SCALE}"
  --joint_pair_summary_future_adapter_ckpt "${JOINT_PAIR_SUMMARY_FUTURE_ADAPTER_CKPT}"
  --joint_belief_summary_future_adapter_delta_key "${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_DELTA_KEY}"
  --joint_belief_summary_future_adapter_delta_scale "${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_DELTA_SCALE}"
  --joint_belief_summary_future_adapter_mix "${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_MIX}"
  --joint_belief_summary_future_adapter_max_shift_norm "${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_MAX_SHIFT_NORM}"
  --joint_belief_patch_min_interval_steps "${JOINT_BELIEF_PATCH_MIN_INTERVAL_STEPS}"
  --joint_belief_patch_max_per_subtask "${JOINT_BELIEF_PATCH_MAX_PER_SUBTASK}"
  --dynamic_coupling_delta_action_repair_mix "${DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX}"
  --dynamic_coupling_gate_every_k "${DYNAMIC_COUPLING_GATE_EVERY_K}"
  --suffix_mask_ablation_mode "${SUFFIX_MASK_ABLATION_MODE}"
  --suffix_mask_ablation_teacher_step_threshold "${SUFFIX_MASK_ABLATION_TEACHER_STEP_THRESHOLD}"
  --suffix_mask_ablation_max_abs_delta "${SUFFIX_MASK_ABLATION_MAX_ABS_DELTA}"
  --suffix_mask_ablation_min_affected_fraction "${SUFFIX_MASK_ABLATION_MIN_AFFECTED_FRACTION}"
  --suffix_mask_ablation_max_affected_fraction "${SUFFIX_MASK_ABLATION_MAX_AFFECTED_FRACTION}"
  --suffix_mask_ablation_seed "${SUFFIX_MASK_ABLATION_SEED}"
  --suffix_mask_ablation_schedule_json "${SUFFIX_MASK_ABLATION_SCHEDULE_JSON}"
  --joint_execution_update_ablation_mode "${JOINT_EXECUTION_UPDATE_ABLATION_MODE}"
)

[[ "${DISABLE_RETRY_AFTER_FIRST_PASS}" == "1" ]] && ARGS+=(--disable_retry_after_first_pass)
[[ "${RETRY_BASELINE_FALLBACK_ON_FAILURE}" == "1" ]] && ARGS+=(--retry_baseline_fallback_on_failure)
[[ "${DYNAMIC_COUPLING_FIRST_PASS}" == "1" ]] && ARGS+=(--dynamic_coupling_first_pass)
[[ "${DYNAMIC_COUPLING_ONLINE}" == "1" ]] && ARGS+=(--dynamic_coupling_online)
[[ "${DYNAMIC_COUPLING_GENERATE_ACTION_INTENT}" == "1" ]] && ARGS+=(--dynamic_coupling_generate_action_intent)
[[ "${DYNAMIC_COUPLING_PERSISTENT_HYPOTHESIS}" == "1" ]] && ARGS+=(--dynamic_coupling_persistent_hypothesis)
[[ "${JOINT_BELIEF_CLEAN_CONTROLLER}" == "1" ]] && ARGS+=(--joint_belief_clean_controller)
[[ "${JOINT_BELIEF_SUMMARY_FUTURE_ADAPTER_UPDATE}" == "1" ]] && ARGS+=(--joint_belief_summary_future_adapter_update)
[[ "${DYNAMIC_COUPLING_DISABLE_REPLAN}" == "1" ]] && ARGS+=(--dynamic_coupling_disable_replan)
[[ "${SUFFIX_MASK_ABLATION_FORCE_WRITEBACK}" == "1" ]] && ARGS+=(--suffix_mask_ablation_force_writeback)

if [[ "${JOINT_BELIEF_TRANSITION}" == "1" ]]; then
  ARGS+=(--joint_belief_transition --joint_belief_transition_ckpt "${JOINT_BELIEF_TRANSITION_CKPT}")
fi

if [[ "${DYNAMIC_COUPLING_GENERATE_ACTION_INTENT}" == "1" || "${JOINT_BELIEF_TRANSITION}" == "1" ]]; then
  ARGS+=(--joint_pair_action_generator_ckpt "${JOINT_PAIR_ACTION_GENERATOR_CKPT}")
fi

if [[ "${LINEAR_SUFFIX_SENSITIVITY}" == "1" ]]; then
  ARGS+=(
    --linear_suffix_sensitivity_transport
    --linear_suffix_sensitivity_rank "${LINEAR_SUFFIX_SENSITIVITY_RANK}"
    --linear_suffix_sensitivity_eps "${LINEAR_SUFFIX_SENSITIVITY_EPS}"
    --linear_suffix_sensitivity_step_size "${LINEAR_SUFFIX_SENSITIVITY_STEP_SIZE}"
    --linear_suffix_sensitivity_max_abs_delta "${LINEAR_SUFFIX_SENSITIVITY_MAX_ABS_DELTA}"
    --linear_suffix_sensitivity_min_step_score "${LINEAR_SUFFIX_SENSITIVITY_MIN_STEP_SCORE}"
    --linear_suffix_sensitivity_selective_step_threshold "${LINEAR_SUFFIX_SENSITIVITY_SELECTIVE_STEP_THRESHOLD}"
    --linear_suffix_sensitivity_max_patch_fraction "${LINEAR_SUFFIX_SENSITIVITY_MAX_PATCH_FRACTION}"
    --linear_suffix_sensitivity_replan_fraction "${LINEAR_SUFFIX_SENSITIVITY_REPLAN_FRACTION}"
    --linear_suffix_sensitivity_replan_delta_h_norm "${LINEAR_SUFFIX_SENSITIVITY_REPLAN_DELTA_H_NORM}"
    --linear_suffix_sensitivity_smooth_weight "${LINEAR_SUFFIX_SENSITIVITY_SMOOTH_WEIGHT}"
    --linear_suffix_sensitivity_continuity_weight "${LINEAR_SUFFIX_SENSITIVITY_CONTINUITY_WEIGHT}"
    --linear_suffix_sensitivity_mpc_iters "${LINEAR_SUFFIX_SENSITIVITY_MPC_ITERS}"
  )
fi

if [[ "${DIRECTIONAL_SENSITIVITY_TCN}" == "1" ]]; then
  ARGS+=(
    --directional_sensitivity_tcn_transport
    --directional_sensitivity_tcn_ckpt "${DIRECTIONAL_SENSITIVITY_TCN_CKPT}"
    --directional_sensitivity_tcn_max_abs_delta "${DIRECTIONAL_SENSITIVITY_TCN_MAX_ABS_DELTA}"
    --directional_sensitivity_tcn_min_delta_norm "${DIRECTIONAL_SENSITIVITY_TCN_MIN_DELTA_NORM}"
    --directional_sensitivity_tcn_step_threshold "${DIRECTIONAL_SENSITIVITY_TCN_STEP_THRESHOLD}"
    --directional_sensitivity_tcn_top_k "${DIRECTIONAL_SENSITIVITY_TCN_TOP_K}"
    --directional_sensitivity_tcn_top_k_floor "${DIRECTIONAL_SENSITIVITY_TCN_TOP_K_FLOOR}"
    --directional_sensitivity_tcn_top_k_min_margin "${DIRECTIONAL_SENSITIVITY_TCN_TOP_K_MIN_MARGIN}"
    --directional_sensitivity_tcn_force_top_k_patch "${DIRECTIONAL_SENSITIVITY_TCN_FORCE_TOP_K_PATCH}"
    --directional_sensitivity_tcn_max_patch_fraction "${DIRECTIONAL_SENSITIVITY_TCN_MAX_PATCH_FRACTION}"
    --directional_sensitivity_tcn_max_affected_segments "${DIRECTIONAL_SENSITIVITY_TCN_MAX_AFFECTED_SEGMENTS}"
    --directional_sensitivity_tcn_replan_delta_h_norm "${DIRECTIONAL_SENSITIVITY_TCN_REPLAN_DELTA_H_NORM}"
    --directional_sensitivity_tcn_smooth_weight "${DIRECTIONAL_SENSITIVITY_TCN_SMOOTH_WEIGHT}"
    --directional_sensitivity_tcn_mpc_iters "${DIRECTIONAL_SENSITIVITY_TCN_MPC_ITERS}"
    --directional_sensitivity_tcn_validity_threshold "${DIRECTIONAL_SENSITIVITY_TCN_VALIDITY_THRESHOLD}"
    --directional_sensitivity_tcn_uncertainty_replan_threshold "${DIRECTIONAL_SENSITIVITY_TCN_UNCERTAINTY_REPLAN_THRESHOLD}"
    --directional_sensitivity_tcn_selection_lambda "${DIRECTIONAL_SENSITIVITY_TCN_SELECTION_LAMBDA}"
    --directional_sensitivity_tcn_selection_beta "${DIRECTIONAL_SENSITIVITY_TCN_SELECTION_BETA}"
  )
  [[ "${DIRECTIONAL_SENSITIVITY_TCN_STRUCTURED_MASK}" == "1" ]] && ARGS+=(--directional_sensitivity_tcn_structured_mask)
fi

if [[ "${DIRECT_ACTION_CORRECTION_TCN}" == "1" ]]; then
  ARGS+=(
    --direct_action_correction_tcn_transport
    --direct_action_correction_tcn_ckpt "${DIRECT_ACTION_CORRECTION_TCN_CKPT}"
    --directional_sensitivity_tcn_max_abs_delta "${DIRECTIONAL_SENSITIVITY_TCN_MAX_ABS_DELTA}"
    --directional_sensitivity_tcn_min_delta_norm "${DIRECTIONAL_SENSITIVITY_TCN_MIN_DELTA_NORM}"
    --directional_sensitivity_tcn_step_threshold "${DIRECTIONAL_SENSITIVITY_TCN_STEP_THRESHOLD}"
    --directional_sensitivity_tcn_top_k "${DIRECTIONAL_SENSITIVITY_TCN_TOP_K}"
    --directional_sensitivity_tcn_top_k_floor "${DIRECTIONAL_SENSITIVITY_TCN_TOP_K_FLOOR}"
    --directional_sensitivity_tcn_top_k_min_margin "${DIRECTIONAL_SENSITIVITY_TCN_TOP_K_MIN_MARGIN}"
    --directional_sensitivity_tcn_force_top_k_patch "${DIRECTIONAL_SENSITIVITY_TCN_FORCE_TOP_K_PATCH}"
    --directional_sensitivity_tcn_max_patch_fraction "${DIRECTIONAL_SENSITIVITY_TCN_MAX_PATCH_FRACTION}"
    --directional_sensitivity_tcn_max_affected_segments "${DIRECTIONAL_SENSITIVITY_TCN_MAX_AFFECTED_SEGMENTS}"
    --directional_sensitivity_tcn_replan_delta_h_norm "${DIRECTIONAL_SENSITIVITY_TCN_REPLAN_DELTA_H_NORM}"
    --directional_sensitivity_tcn_smooth_weight "${DIRECTIONAL_SENSITIVITY_TCN_SMOOTH_WEIGHT}"
    --directional_sensitivity_tcn_mpc_iters "${DIRECTIONAL_SENSITIVITY_TCN_MPC_ITERS}"
    --directional_sensitivity_tcn_validity_threshold "${DIRECTIONAL_SENSITIVITY_TCN_VALIDITY_THRESHOLD}"
    --directional_sensitivity_tcn_uncertainty_replan_threshold "${DIRECTIONAL_SENSITIVITY_TCN_UNCERTAINTY_REPLAN_THRESHOLD}"
    --directional_sensitivity_tcn_selection_lambda "${DIRECTIONAL_SENSITIVITY_TCN_SELECTION_LAMBDA}"
    --directional_sensitivity_tcn_selection_beta "${DIRECTIONAL_SENSITIVITY_TCN_SELECTION_BETA}"
  )
  [[ "${DIRECTIONAL_SENSITIVITY_TCN_STRUCTURED_MASK}" == "1" ]] && ARGS+=(--directional_sensitivity_tcn_structured_mask)
fi

if [[ "${COLLECT_SENSITIVITY_TEACHER}" == "1" ]]; then
  ARGS+=(
    --collect_sensitivity_teacher
    --sensitivity_teacher_mode "${SENSITIVITY_TEACHER_MODE}"
    --sensitivity_teacher_step_size "${SENSITIVITY_TEACHER_STEP_SIZE}"
    --sensitivity_teacher_max_abs_delta "${SENSITIVITY_TEACHER_MAX_ABS_DELTA}"
    --sensitivity_teacher_joint_delta_to_action_scale "${SENSITIVITY_TEACHER_JOINT_DELTA_TO_ACTION_SCALE}"
    --sensitivity_teacher_condition_ablation "${SENSITIVITY_TEACHER_CONDITION_ABLATION}"
  )
fi

exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"
