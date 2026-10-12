# ReEnvision Reproduction Bundle

This folder collects the code, configs, data links, checkpoints, environment links, and known local results used for the current ReEnvision paper runs.

Most large assets are symlinks to the existing workspace paths. This keeps the bundle lightweight while making the required files visible in one place. If moving to another machine, copy with symlink targets included, for example `rsync -aL reenvision/ <target>/reenvision/`, or recreate the same source paths on the new machine.

## Main Commands

CALVIN ABC-D, 1000 sequences:

```bash
NUM_SEQUENCES=1000 bash /mnt/workspace/manipulation/reenvision/scripts/run_calvin_paper_suffix.sh
```

L-CALVIN, 100 sequences, 10-step:

```bash
NUM_SEQUENCES=100 bash /mnt/workspace/manipulation/reenvision/scripts/run_lcalvin_paper_suffix.sh
```

BridgeVLA RLBench, 18 tasks x 50 episodes:

```bash
cd /mnt/workspace/manipulation/reenvision/code/BridgeVLA
set -a; source /mnt/workspace/manipulation/reenvision/configs/rlbench_92_suffix.env; set +a
bash finetune/RLBench/eval.sh
```

BridgeVLA local baseline/collection result:

```bash
cd /mnt/workspace/manipulation/reenvision/code/BridgeVLA
set -a; source /mnt/workspace/manipulation/reenvision/configs/rlbench_86_suffix_collect.env; set +a
bash finetune/RLBench/eval.sh
```

## Training Commands

Full component training commands are in `TRAINING_COMMANDS.md`.

CALVIN/L-CALVIN ReEnvision component training:

```bash
DEVICE=cuda:0 STAGE=all bash /mnt/workspace/manipulation/reenvision/scripts/train_calvin_reenvision_components.sh
```

RLBench ReEnvision component training:

```bash
DEVICE=cuda:0 STAGE=all bash /mnt/workspace/manipulation/reenvision/scripts/train_rlbench_reenvision_components.sh
```

CALVIN/DeFi quick smoke training, 1 GPU:

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
bash scripts/train_calvin.sh \
  /mnt/workspace/calvin/task_ABC_D \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/_hf_defi/step1_gfdm \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/openai_clip_vit_base_patch32 \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/ViT-B-32.pt \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/t5_base \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/IDM/lam_idm.ckpt \
  1 8 1
```

CALVIN/DeFi full training command shape:

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
bash scripts/train_calvin.sh \
  /mnt/workspace/calvin/task_ABC_D \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/_hf_defi/step1_gfdm \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/openai_clip_vit_base_patch32 \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/ViT-B-32.pt \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/t5_base \
  /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/IDM/lam_idm.ckpt \
  8 20 36
```

BridgeVLA/RLBench original training entry exists at `code/BridgeVLA/finetune/RLBench/train.sh`, but it still contains placeholder paths and the Python code has old hardcoded replay/data roots in `utils/peract_utils_rlbench.py`. For the current paper bundle, BridgeVLA is preserved primarily for evaluation with the local `model_80.pth` checkpoint.

## Expected Paper-Scale Numbers

CALVIN paper target is about 4.62 average length on 1000 sequences.

L-CALVIN paper target is about 8.4 average length on 100 sequences.

BridgeVLA/RLBench paper target is about 88.8 average success rate over 18 tasks x 50 episodes.

## Notes

The CALVIN/L-CALVIN command scripts still use absolute paths under `/mnt/workspace/manipulation` and `/mnt/workspace/calvin`. This is intentional for the current machine. When migrating, update the paths in the two script files or keep the same mount layout.

For BridgeVLA offline loading, keep the variables in `configs/rlbench_92_suffix.env`, especially `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `BRIDGEVLA_PALIGEMMA_PATH`.

The active Python environment for the commands above is `env/defi_venv`. The `env/bridgevla_eval311` link is historical and is not used by the current commands.

`code/BridgeVLA/finetune/GemBench` is kept because the RLBench agent imports it at runtime, even though GemBench is not one of the final reported environments.
