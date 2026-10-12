#!/usr/bin/env bash
set -euo pipefail

# Paper-style L-CALVIN suffix run.

export PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/manipulation/DeFi/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_SEQUENCES="${NUM_SEQUENCES:-100}"
export EP_LEN="${EP_LEN:-720}"
export ACTION_MODEL_FOLDER="${ACTION_MODEL_FOLDER:-/mnt/workspace/manipulation/DeFi/outputs/calvin_train/2026-10-12_20-24-17}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/workspace/manipulation/reenvision/checkpoints/calvin/last.pt}"
export CALVIN_ABC_DIR="${CALVIN_ABC_DIR:-/mnt/workspace/calvin/task_ABC_D}"
export EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-/mnt/workspace/manipulation/white_seq_num=100-seq_len=10.json}"
export JOINT_PAIR_ACTION_GENERATOR_CKPT="${JOINT_PAIR_ACTION_GENERATOR_CKPT:-/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt}"

export DYNAMIC_COUPLING_FIRST_PASS=0
export DISABLE_RETRY_AFTER_FIRST_PASS=0
export RETRY_ROLLBACK_MODE=subtask_start
export RETRY_MAX_ATTEMPTS="${RETRY_MAX_ATTEMPTS:-1}"
export LINEAR_SUFFIX_SENSITIVITY=1
export DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX="${DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX:-0.005}"
export JOINT_BELIEF_CLEAN_H_UPDATE_SCALE="${JOINT_BELIEF_CLEAN_H_UPDATE_SCALE:-0.8}"
export JOINT_BELIEF_CLEAN_PATCH_MIN_PROB="${JOINT_BELIEF_CLEAN_PATCH_MIN_PROB:-0.35}"
export JOINT_BELIEF_CLEAN_PATCH_MARGIN="${JOINT_BELIEF_CLEAN_PATCH_MARGIN:-0.00}"
export LINEAR_SUFFIX_SENSITIVITY_STEP_SIZE="${LINEAR_SUFFIX_SENSITIVITY_STEP_SIZE:-0.030}"
export LINEAR_SUFFIX_SENSITIVITY_MAX_ABS_DELTA="${LINEAR_SUFFIX_SENSITIVITY_MAX_ABS_DELTA:-0.015}"
export LINEAR_SUFFIX_SENSITIVITY_SELECTIVE_STEP_THRESHOLD="${LINEAR_SUFFIX_SENSITIVITY_SELECTIVE_STEP_THRESHOLD:-0.002}"
export LINEAR_SUFFIX_SENSITIVITY_MAX_PATCH_FRACTION="${LINEAR_SUFFIX_SENSITIVITY_MAX_PATCH_FRACTION:-0.70}"
export LINEAR_SUFFIX_SENSITIVITY_REPLAN_FRACTION="${LINEAR_SUFFIX_SENSITIVITY_REPLAN_FRACTION:-0.90}"
export LINEAR_SUFFIX_SENSITIVITY_REPLAN_DELTA_H_NORM="${LINEAR_SUFFIX_SENSITIVITY_REPLAN_DELTA_H_NORM:-0.0}"

cd /mnt/workspace/manipulation/DeFi/Step3_DeFI
bash scripts/run_calvin_joint_belief_suffix.sh \
  --policy_action_intent_lora_ckpt /mnt/workspace/manipulation/BridgeVLA/eval/policy_lora_action_intent_v2/best.pt \
  --joint_pair_direct_generate_only \
  --joint_pair_no_memory \
  --joint_belief_transition_direct_suffix \
  --joint_belief_transition_sparse_suffix
