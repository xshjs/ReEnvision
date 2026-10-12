#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/manipulation}"
REENVISION="${REENVISION:-${ROOT}/reenvision}"
PYTHON_BIN="${PYTHON_BIN:-${REENVISION}/env/defi_venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
if [[ "${DEVICE}" =~ ^[0-9]+$ ]]; then
  DEVICE="cuda:${DEVICE}"
fi
STAGE="${STAGE:-all}"

DEFI_DIR="${REENVISION}/code/DeFi/Step3_DeFI"

CALVIN_CKPT="${CALVIN_CKPT:-${REENVISION}/checkpoints/calvin/last.pt}"
CALVIN_DATA="${CALVIN_DATA:-/mnt/workspace/calvin/task_ABC_D}"

LORA_ROOT_0000="${LORA_ROOT_0000:-${ROOT}/DeFi/outputs/policy_finetune_success_expert_d_shard_0000_0099}"
LORA_ROOT_0100="${LORA_ROOT_0100:-${ROOT}/DeFi/outputs/policy_finetune_success_expert_d_shard_0100_0199}"
LORA_ROOT_0400="${LORA_ROOT_0400:-${ROOT}/DeFi/outputs/policy_finetune_success_expert_d_shard_0400_0499}"
LORA_ROOT_0500="${LORA_ROOT_0500:-${ROOT}/DeFi/outputs/policy_finetune_success_expert_d_shard_0500_0599}"

ACTION_TRACE_D300="${ACTION_TRACE_D300:-${ROOT}/BridgeVLA/eval/calvin_d300_hard_strong_gt_action_v4/action_traces}"
ACTION_TRACE_ABC700="${ACTION_TRACE_ABC700:-${ROOT}/BridgeVLA/eval/calvin_abc700_hard_strong_gt_action_v4/action_traces}"
ACTION_TRACE_ABC1500="${ACTION_TRACE_ABC1500:-${ROOT}/BridgeVLA/eval/calvin_abc_hard7_baseline_strong_gt_action_1500_v1/action_traces}"
FACTOR_DATASET="${FACTOR_DATASET:-${ROOT}/BridgeVLA/eval/combined_d300_abc700_abc1500_factorized_noqcheat_v1/combined_factor_dataset.jsonl}"

LORA_OUT="${LORA_OUT:-${ROOT}/BridgeVLA/eval/policy_lora_action_intent_v2}"
MEMORY_OUT="${MEMORY_OUT:-${ROOT}/BridgeVLA/eval/future_action_memory_d300_abc700_abc1500_state_v1}"
ENERGY_OUT="${ENERGY_OUT:-${ROOT}/BridgeVLA/eval/joint_future_action_robotics_energy_d300_abc700_abc1500_v1}"
GENERATOR_OUT="${GENERATOR_OUT:-${ROOT}/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2}"
BELIEF_OUT="${BELIEF_OUT:-${ROOT}/DeFi/outputs/full_rssm_plan_compat_v2_success_balanced_650x2}"
BELIEF_MEMORY="${BELIEF_MEMORY:-${ROOT}/DeFi/outputs/joint_belief_transition_v2_balanced_success/transition_memory_s1024_seq934_success_balanced_650x2_oversample.npz}"

run_lora() {
  cd "${DEFI_DIR}"
  "${PYTHON_BIN}" policy_evaluation/train_success_policy_lora_smoke.py \
    --roots "${LORA_ROOT_0000}" "${LORA_ROOT_0100}" "${LORA_ROOT_0400}" "${LORA_ROOT_0500}" \
    --output-dir "${LORA_OUT}" \
    --limit "${LORA_LIMIT:-1000}" \
    --steps "${LORA_STEPS:-1000}" \
    --batch-size "${LORA_BATCH_SIZE:-4}" \
    --lr "${LORA_LR:-5e-5}" \
    --lora-rank "${LORA_RANK:-4}" \
    --lora-alpha "${LORA_ALPHA:-8}" \
    --lora-dropout "${LORA_DROPOUT:-0.05}" \
    --action-intent-len "${ACTION_INTENT_LEN:-16}" \
    --use-action-intent \
    --video-model-path "${ROOT}/DeFi/ckpts/_hf_defi/step1_gfdm" \
    --checkpoint "${CALVIN_CKPT}" \
    --clip-model-path "${ROOT}/DeFi/ckpts/openai_clip_vit_base_patch32" \
    --t5-model-path "${ROOT}/DeFi/ckpts/t5_base" \
    --language-goal-path "${ROOT}/DeFi/ckpts/ViT-B-32.pt" \
    --calvin-dir "${CALVIN_DATA}" \
    --device "${DEVICE}"
}

run_memory() {
  mkdir -p "${MEMORY_OUT}"
  cd "${DEFI_DIR}"
  "${PYTHON_BIN}" policy_evaluation/build_future_action_memory.py \
    --dataset-jsonl "${FACTOR_DATASET}" \
    --action-trace-dir "${ACTION_TRACE_D300}" \
    --action-trace-dir "${ACTION_TRACE_ABC700}" \
    --action-trace-dir "${ACTION_TRACE_ABC1500}" \
    --output-npz "${MEMORY_OUT}/memory.npz" \
    --summary-json "${MEMORY_OUT}/summary.json" \
    --chunk-len 16 \
    --chunks-per-trace 3 \
    --horizon 16
}

run_energy() {
  cd "${DEFI_DIR}"
  "${PYTHON_BIN}" policy_evaluation/train_joint_future_action_robotics_energy.py \
    --memory-npz "${MEMORY_OUT}/memory.npz" \
    --output-dir "${ENERGY_OUT}" \
    --steps "${ENERGY_STEPS:-4000}" \
    --batch-size "${ENERGY_BATCH_SIZE:-128}" \
    --lr "${ENERGY_LR:-2e-4}" \
    --weight-decay 1e-4 \
    --margin 0.5 \
    --rank-weight 1.0 \
    --task-weight 0.5 \
    --future-transition-weight 0.5 \
    --state-transition-weight 0.5 \
    --contact-weight 0.25 \
    --mismatch-weight 0.5 \
    --lambda-future-dyn 1.0 \
    --lambda-state-dyn 0.5 \
    --beta-contact 0.25 \
    --device "${DEVICE}"
}

run_generator() {
  cd "${DEFI_DIR}"
  "${PYTHON_BIN}" policy_evaluation/train_joint_action_generator_mlp.py \
    --memory-npz "${MEMORY_OUT}/memory.npz" \
    --output-dir "${GENERATOR_OUT}" \
    --steps "${GENERATOR_STEPS:-4000}" \
    --batch-size "${GENERATOR_BATCH_SIZE:-256}" \
    --lr "${GENERATOR_LR:-1e-3}" \
    --weight-decay 1e-4 \
    --success-only \
    --device "${DEVICE}"
}

run_belief() {
  cd "${DEFI_DIR}"
  "${PYTHON_BIN}" policy_evaluation/train_factored_belief_action_transition.py \
    --memory-npz "${BELIEF_MEMORY}" \
    --output-dir "${BELIEF_OUT}" \
    --full-rssm-plan-head \
    --hidden-dim 1024 \
    --rssm-deter-dim 512 \
    --rssm-stoch-dim 128 \
    --kl-loss-weight 1e-3 \
    --expected-loss-weight 0.5 \
    --action-loss-weight 0.5 \
    --mode-loss-weight 0.1 \
    --confidence-loss-weight 0.05 \
    --steps "${BELIEF_STEPS:-3000}" \
    --batch-size "${BELIEF_BATCH_SIZE:-256}" \
    --lr "${BELIEF_LR:-1e-3}" \
    --device "${DEVICE}"
}

case "${STAGE}" in
  all)
    run_lora
    run_memory
    run_generator
    run_belief
    ;;
  lora) run_lora ;;
  memory) run_memory ;;
  energy) run_energy ;;
  generator) run_generator ;;
  belief) run_belief ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    echo "Use: all, lora, memory, energy, generator, belief" >&2
    exit 2
    ;;
esac
