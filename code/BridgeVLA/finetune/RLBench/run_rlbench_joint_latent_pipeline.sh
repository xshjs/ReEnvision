#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/manipulation/BridgeVLA/finetune/RLBench
source /mnt/data/manipulation/DeFi/.venv/bin/activate

export PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI:/mnt/workspace/manipulation/BridgeVLA/finetune/RLBench:${PYTHONPATH:-}
export BRIDGEVLA_PALIGEMMA_PATH=/mnt/workspace/manipulation/BridgeVLA/checkpoints/paligemma-3b-pt-224
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi
export TRANSFORMERS_CACHE=/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi
export COPPELIASIM_ROOT=/mnt/workspace/manipulation/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=xcb
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p /tmp/runtime-root
chmod 700 /tmp/runtime-root
export LIBGL_ALWAYS_SOFTWARE=1
unset DISPLAY

OUT_ROOT=${OUT_ROOT:-/mnt/workspace/manipulation/BridgeVLA/eval/rlbench_joint_latent_pipeline_v1}
TRAIN_DEMO_ZIP_DIR=${TRAIN_DEMO_ZIP_DIR:-/mnt/data/datasets/rlbench-18-tasks/data/train}
COLLECT_EPISODES=${COLLECT_EPISODES:-100}
DEVICE=${DEVICE:-cuda}
DEFI_RLBENCH_CKPT=${DEFI_RLBENCH_CKPT:-/mnt/workspace/manipulation/DeFi/outputs/rlbench_train/2026-07-16_14-29-36/saved_models/last.pt}

COLLECT_DIR=$OUT_ROOT/collect_train_rollout
LATENT_CKPT=$OUT_ROOT/bridge_task_conditioned_latent_heads.pt
MEMORY_NPZ=$OUT_ROOT/bridge_joint_memory/memory.npz
ENERGY_CKPT=$OUT_ROOT/bridge_joint_memory/joint_energy.pt
POLICY_CKPT=$OUT_ROOT/bridge_joint_latent_policy.pt

TASKS=(
  close_jar
  reach_and_drag
  insert_onto_square_peg
  meat_off_grill
  open_drawer
  place_cups
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  light_bulb_in
  slide_block_to_color_target
  place_shape_in_shape_sorter
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
)

mkdir -p "$OUT_ROOT"

echo "[1/5] Collect RLBench train rollout rows/features into $COLLECT_DIR"
xvfb-run -a --server-args='-screen 0 1024x768x24 -ac' \
python /mnt/workspace/manipulation/DeFi/Step3_DeFI/scripts/collect_rlbench_bridge_rollout.py \
  --ckpt "$DEFI_RLBENCH_CKPT" \
  --tasks "${TASKS[@]}" \
  --episodes "$COLLECT_EPISODES" \
  --max-steps 25 \
  --headless \
  --device "$DEVICE" \
  --demo-zip-dir "$TRAIN_DEMO_ZIP_DIR" \
  --video-model-path /mnt/workspace/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
  --text-encoder-path /mnt/data/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5-model-path /mnt/workspace/manipulation/DeFi/ckpts/t5_base \
  --language-goal-path /mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt \
  --action-head-type rlbench_bridge \
  --policy-output native_quat_collision \
  --output-dir "$COLLECT_DIR"

COLLECT_ROWS_JSONL="$COLLECT_DIR/rollout_rows.jsonl" python - <<'PY'
import json
import os
from pathlib import Path

rows_path = Path(os.environ["COLLECT_ROWS_JSONL"])
rows = [json.loads(line) for line in rows_path.open("r", encoding="utf-8") if line.strip()]
num_rows = len(rows)
num_success = sum(1 for row in rows if row.get("success"))
print({"collect_rows": num_rows, "collect_successes": num_success, "success_rate": (num_success / num_rows if num_rows else 0.0)}, flush=True)
if num_success <= 0:
    raise SystemExit("collect stage produced zero successful rows; aborting pipeline")
PY

echo "[2/5] Train more task-specific z_a latent heads"
python train_bridge_task_conditioned_latent_heads.py \
  --rows-jsonl "$COLLECT_DIR/rollout_rows.jsonl" \
  --output-ckpt "$LATENT_CKPT" \
  --steps 4000 \
  --batch-size 128 \
  --lr 1e-3 \
  --action-chunk-len 16 \
  --device "$DEVICE"

echo "[3/5] Build RLBench joint memory from collected train rows"
python build_bridge_joint_future_action_memory.py \
  --rows-jsonl "$COLLECT_DIR/rollout_rows.jsonl" \
  --latent-head-ckpt "$LATENT_CKPT" \
  --output-npz "$MEMORY_NPZ" \
  --device "$DEVICE"

echo "[4/5] Train joint energy E(s, z_f, z_a)"
python train_bridge_joint_energy.py \
  --memory-npz "$MEMORY_NPZ" \
  --output-ckpt "$ENERGY_CKPT" \
  --steps 4000 \
  --batch-size 128 \
  --device "$DEVICE"

echo "[5/5] Train latent policy pi(s, z_f, z_a)"
python train_bridge_joint_latent_policy.py \
  --rows-jsonl "$COLLECT_DIR/rollout_rows.jsonl" \
  --latent-head-ckpt "$LATENT_CKPT" \
  --output-ckpt "$POLICY_CKPT" \
  --steps 4000 \
  --batch-size 128 \
  --device "$DEVICE"

echo "[done] Outputs:"
echo "  collect_dir  = $COLLECT_DIR"
echo "  latent_ckpt  = $LATENT_CKPT"
echo "  memory_npz   = $MEMORY_NPZ"
echo "  energy_ckpt  = $ENERGY_CKPT"
echo "  policy_ckpt  = $POLICY_CKPT"
