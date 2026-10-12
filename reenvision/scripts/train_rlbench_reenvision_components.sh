#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/manipulation}"
REENVISION="${REENVISION:-${ROOT}/reenvision}"
PYTHON_BIN="${PYTHON_BIN:-${REENVISION}/env/defi_venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
if [[ "${DEVICE}" =~ ^[0-9]+$ ]]; then
  DEVICE="cuda:${DEVICE}"
fi
STAGE="${STAGE:-suffix_td}"

DEFI_DIR="${REENVISION}/code/DeFi/Step3_DeFI"
BRIDGE_RLBENCH_DIR="${REENVISION}/code/BridgeVLA/finetune/RLBench"

SUFFIX_COLLECT_DIR="${SUFFIX_COLLECT_DIR:-${ROOT}/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect}"
SUFFIX_ROWS="${SUFFIX_ROWS:-${SUFFIX_COLLECT_DIR}/suffix_transition_rows.jsonl}"
SUFFIX_MEMORY="${SUFFIX_MEMORY:-${SUFFIX_COLLECT_DIR}/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz}"
SUFFIX_SUMMARY="${SUFFIX_SUMMARY:-${SUFFIX_COLLECT_DIR}/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.json}"
SUFFIX_OUT="${SUFFIX_OUT:-${ROOT}/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/train_factored_td_balanced_taskmode_cap80_patchonly_clip005}"

run_memory() {
  cd "${BRIDGE_RLBENCH_DIR}"
  "${PYTHON_BIN}" build_bridge_suffix_transition_memory.py \
    --rows-jsonl "${SUFFIX_ROWS}" \
    --output-npz "${SUFFIX_MEMORY}" \
    --summary-json "${SUFFIX_SUMMARY}" \
    --delta-clip 0.02 \
    --balance-task-mode \
    --max-per-task-mode 80
}

run_suffix_td() {
  cd "${DEFI_DIR}"
  "${PYTHON_BIN}" policy_evaluation/train_factored_belief_action_transition.py \
    --memory-npz "${SUFFIX_MEMORY}" \
    --output-dir "${SUFFIX_OUT}" \
    --expected-transition \
    --adaptation-input post_innovation \
    --target-delta-clip 0.05 \
    --expected-loss-weight 0.5 \
    --action-loss-weight 2.0 \
    --mode-loss-weight 0.1 \
    --confidence-loss-weight 0.05 \
    --steps "${SUFFIX_STEPS:-3000}" \
    --batch-size "${SUFFIX_BATCH_SIZE:-256}" \
    --lr "${SUFFIX_LR:-1e-3}" \
    --device "${DEVICE}"
}

case "${STAGE}" in
  all)
    run_memory
    run_suffix_td
    ;;
  memory) run_memory ;;
  suffix_td) run_suffix_td ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    echo "Use: all, memory, suffix_td" >&2
    exit 2
    ;;
esac
