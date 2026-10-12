#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/manipulation/BridgeVLA/finetune/RLBench
source /mnt/data/manipulation/DeFi/.venv/bin/activate

export BRIDGEVLA_PALIGEMMA_PATH=/mnt/workspace/manipulation/BridgeVLA/checkpoints/paligemma-3b-pt-224
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi
export TRANSFORMERS_CACHE=/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi
export COPPELIASIM_ROOT=/mnt/workspace/manipulation/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=xcb
export DISPLAY=:99
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p /tmp/runtime-root
chmod 700 /tmp/runtime-root
export LIBGL_ALWAYS_SOFTWARE=1

LOG_ROOT=/mnt/workspace/manipulation/BridgeVLA/eval/rlbench_bridge_joint_pair_all18_20ep_v1
ROLLOUT_ROOT=$LOG_ROOT/rollouts

mkdir -p "$LOG_ROOT"
mkdir -p "$ROLLOUT_ROOT"

xvfb-run --server-num=99 --server-args='-screen 0 1024x768x24 -ac' \
python eval_bridge_with_defi_sidecar.py \
  --bridge-model-path /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/model_80.pth \
  --exp-cfg-path /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/exp_cfg.yaml \
  --mvt-cfg-path /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/mvt_cfg.yaml \
  --eval-datafolder /mnt/workspace/manipulation/BridgeVLA/finetune/rlbench_eval_all18 \
  --tasks all \
  --eval-episodes 20 \
  --episode-length 25 \
  --device 0 \
  --headless \
  --log-dir "$LOG_ROOT" \
  --save-rollout-dir "$ROLLOUT_ROOT" \
  --bridge-gfdm-gidm \
  --bridge-latent-head-ckpt /mnt/workspace/manipulation/BridgeVLA/eval/bridge_latent_heads_v1/bridge_latent_heads.pt \
  --bridge-standalone-gidm-ckpt /mnt/workspace/manipulation/DeFi/ckpts/IDM/lam_idm.ckpt \
  --bridge-standalone-gidm-t5-path /mnt/workspace/manipulation/DeFi/ckpts/t5_base \
  --joint-pair-memory-npz /mnt/workspace/manipulation/BridgeVLA/eval/bridge_joint_memory_v1/memory.npz \
  --joint-energy-rerank-ckpt /mnt/workspace/manipulation/BridgeVLA/eval/bridge_joint_memory_v1/joint_energy.pt \
  --joint-pair-topk 8 \
  --joint-pair-same-task-only \
  --joint-pair-min-similarity 0.0 \
  --joint-pair-min-energy-improve -1.0 \
  --joint-pair-max-future-shift 999 \
  --joint-pair-max-motion-shift 999 \
  --joint-pair-future-alpha 0.5 \
  --joint-pair-motion-alpha 0.7 \
  --gidm-gate 1.0
