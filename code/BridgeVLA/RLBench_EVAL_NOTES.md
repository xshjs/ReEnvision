# RLBench Eval Notes

## Current known-good run

- Date: 2026-07-15
- Result: `open_drawer` score `100.0`
- Mode: `BridgeVLA official bridge head + DeFi GFDM/GIDM sidecar`
- Important: GFDM/GIDM are sidecar-only here; they do not feed back into the final action used by the official Bridge agent.

## Environment

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI
source /mnt/workspace/manipulation/DeFi/.venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export COPPELIASIM_ROOT=/mnt/workspace/manipulation/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
unset QT_QPA_PLATFORM
export LIBGL_ALWAYS_SOFTWARE=1
```

## Command

```bash
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension GLX +render' \
python scripts/eval_bridge_with_defi_sidecar.py \
  --bridge-model-path /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/model_80.pth \
  --exp-cfg-path /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/exp_cfg.yaml \
  --mvt-cfg-path /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/mvt_cfg.yaml \
  --eval-datafolder /mnt/workspace/manipulation/BridgeVLA/finetune/rlbench_eval_subset \
  --tasks open_drawer \
  --eval-episodes 3 \
  --episode-length 25 \
  --device 0 \
  --headless \
  --enable-sidecar \
  --defi-ckpt /mnt/workspace/manipulation/DeFi/outputs/rlbench_train/2026-07-15_10-22-16/saved_models/last.pt \
  --sidecar-log-dir /mnt/workspace/manipulation/BridgeVLA/eval/sidecar_open_drawer
```

## Code behavior at this point

- `scripts/eval_bridge_with_defi_sidecar.py`
  - Bridge agent still returns the final action.
  - Sidecar computes:
    - rendered view tokens
    - `GFDM -> pred_future`
    - `GIDM latent -> motion_latent`
    - official `RLBenchBridgeHead`
  - Sidecar outputs are logged only, not fed back into Bridge action execution.
- `policy_models/RLBench_gidm_mvt_policy.py`
  - `bridge_aux_scale=0.0` by default.
  - Sidecar path exists, but final action path is still the official bridge-style discrete head.
- `policy_models/module/rlbench_bridge_head.py`
  - Inputs are cast to the module dtype before `LayerNorm` / linear layers to avoid Half/Float mismatches.

## Notes

- The BridgeVLA checkpoint `/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/model_80.pth` contains `mvt1.*` weights, not a separately named `bridge_head` or `native_head`.
- When loading DeFi checkpoints into the current Bridge-compatible sidecar, incompatible keys are filtered by tensor shape.

