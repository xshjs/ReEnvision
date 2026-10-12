#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the BridgeVLA RLBench suffix-TD evaluation used by the runnable
# handoff. The actual BridgeVLA checkout is expected next to this DeFI repo.

BRIDGEVLA_DIR="${BRIDGEVLA_DIR:-/mnt/workspace/manipulation/BridgeVLA}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/manipulation/DeFi/.venv/bin/python}"

cd "${BRIDGEVLA_DIR}"

export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/root/CoppeliaSim' | paste -sd ':' -)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
BRIDGEVLA_ENABLE_SUFFIX_TD="${BRIDGEVLA_ENABLE_SUFFIX_TD:-1}" \
BRIDGEVLA_SUFFIX_RETRY_ONLY="${BRIDGEVLA_SUFFIX_RETRY_ONLY:-0}" \
BRIDGEVLA_COLLECT_REPAIR_AFTER_FAILURE="${BRIDGEVLA_COLLECT_REPAIR_AFTER_FAILURE:-0}" \
BRIDGEVLA_SUFFIX_TD_CKPT="${BRIDGEVLA_SUFFIX_TD_CKPT:-/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/train_factored_td_balanced_taskmode_cap80_patchonly_clip005/factored_belief_action_transition.pt}" \
BRIDGEVLA_SUFFIX_PATCH_MIX="${BRIDGEVLA_SUFFIX_PATCH_MIX:-0.10}" \
BRIDGEVLA_SUFFIX_PATCH_MAX_ABS="${BRIDGEVLA_SUFFIX_PATCH_MAX_ABS:-0.03}" \
BRIDGEVLA_SUFFIX_PATCH_MIN_PROB="${BRIDGEVLA_SUFFIX_PATCH_MIN_PROB:-0.012}" \
BRIDGEVLA_SUFFIX_TRIGGER_ON_PROB_ONLY="${BRIDGEVLA_SUFFIX_TRIGGER_ON_PROB_ONLY:-1}" \
BRIDGEVLA_SUFFIX_PATCH_ON_REPLAN="${BRIDGEVLA_SUFFIX_PATCH_ON_REPLAN:-0}" \
BRIDGEVLA_COLLECT_SUFFIX_DATA="${BRIDGEVLA_COLLECT_SUFFIX_DATA:-1}" \
BRIDGEVLA_COLLECT_SPLIT_ZF_ZA="${BRIDGEVLA_COLLECT_SPLIT_ZF_ZA:-1}" \
PYTHON_BIN="${PYTHON_BIN}" \
MODEL_FOLDER="${MODEL_FOLDER:-/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench}" \
MODEL_NAME="${MODEL_NAME:-model_80.pth}" \
EVAL_DATAFOLDER="${EVAL_DATAFOLDER:-/mnt/workspace/manipulation/BridgeVLA/finetune/rlbench_eval_all18}" \
EVAL_EPISODES="${EVAL_EPISODES:-50}" \
TASKS="${TASKS:-close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_cups place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap}" \
LOG_NAME="${LOG_NAME:-bridgevla_suffix_td_eval_18tasks_50ep_runnable}" \
DEVICE="${DEVICE:-0}" \
bash finetune/RLBench/eval.sh
