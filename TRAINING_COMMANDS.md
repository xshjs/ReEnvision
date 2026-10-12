# ReEnvision Training Commands

## Two Training Entries

All CALVIN/L-CALVIN ReEnvision small components:

```bash
DEVICE=cuda:0 STAGE=all bash /mnt/workspace/manipulation/reenvision/scripts/train_calvin_reenvision_components.sh
```

RLBench ReEnvision suffix-TD components:

```bash
DEVICE=cuda:0 STAGE=all bash /mnt/workspace/manipulation/reenvision/scripts/train_rlbench_reenvision_components.sh
```

Run one CALVIN component:

```bash
DEVICE=cuda:0 STAGE=lora bash /mnt/workspace/manipulation/reenvision/scripts/train_calvin_reenvision_components.sh
DEVICE=cuda:0 STAGE=memory bash /mnt/workspace/manipulation/reenvision/scripts/train_calvin_reenvision_components.sh
DEVICE=cuda:0 STAGE=generator bash /mnt/workspace/manipulation/reenvision/scripts/train_calvin_reenvision_components.sh
DEVICE=cuda:0 STAGE=belief bash /mnt/workspace/manipulation/reenvision/scripts/train_calvin_reenvision_components.sh
```

These are the training commands for the components used by the current ReEnvision bundle. Commands use the current machine paths and should be run from `/mnt/workspace/manipulation`.

## 1. CALVIN Diffusion Policy From Scratch

Smoke run, only to check training starts:

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

Full command shape used for the base CALVIN action model:

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

Current evaluation checkpoint:

```text
/mnt/workspace/manipulation/reenvision/checkpoints/calvin/step3_defi_run_2026-06-30_14-28-37/saved_models/step3_defi.pt
```

## 2. CALVIN Action-Intent LoRA

This trains the policy-side action-intent adapter used by CALVIN/L-CALVIN suffix inference.

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
/mnt/workspace/manipulation/reenvision/env/defi_venv/bin/python \
  policy_evaluation/train_success_policy_lora_smoke.py \
  --roots \
    /mnt/workspace/manipulation/DeFi/outputs/policy_finetune_success_expert_d_shard_0000_0099 \
    /mnt/workspace/manipulation/DeFi/outputs/policy_finetune_success_expert_d_shard_0100_0199 \
    /mnt/workspace/manipulation/DeFi/outputs/policy_finetune_success_expert_d_shard_0400_0499 \
    /mnt/workspace/manipulation/DeFi/outputs/policy_finetune_success_expert_d_shard_0500_0599 \
  --output-dir /mnt/workspace/manipulation/BridgeVLA/eval/policy_lora_action_intent_v2 \
  --limit 1000 \
  --steps 1000 \
  --batch-size 4 \
  --lr 5e-5 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --action-intent-len 16 \
  --use-action-intent \
  --video-model-path /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/_hf_defi/step1_gfdm \
  --checkpoint /mnt/workspace/manipulation/reenvision/checkpoints/calvin/step3_defi_run_2026-06-30_14-28-37/saved_models/step3_defi.pt \
  --clip-model-path /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5-model-path /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/t5_base \
  --language-goal-path /mnt/workspace/manipulation/reenvision/code/DeFi/ckpts/ViT-B-32.pt \
  --calvin-dir /mnt/workspace/calvin/task_ABC_D \
  --device cuda:0
```

Current checkpoint:

```text
/mnt/workspace/manipulation/reenvision/checkpoints/calvin/policy_lora_action_intent_v2/best.pt
```

## 3. CALVIN Generator Training Data

This builds the compact action/future training table used to train the direct action generator. The final evaluation scripts do not retrieve from this file.

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
/mnt/workspace/manipulation/reenvision/env/defi_venv/bin/python \
  policy_evaluation/build_future_action_memory.py \
  --dataset-jsonl /mnt/workspace/manipulation/BridgeVLA/eval/combined_d300_abc700_abc1500_factorized_noqcheat_v1/combined_factor_dataset.jsonl \
  --action-trace-dir /mnt/workspace/manipulation/BridgeVLA/eval/calvin_d300_hard_strong_gt_action_v4/action_traces \
  --action-trace-dir /mnt/workspace/manipulation/BridgeVLA/eval/calvin_abc700_hard_strong_gt_action_v4/action_traces \
  --action-trace-dir /mnt/workspace/manipulation/BridgeVLA/eval/calvin_abc_hard7_baseline_strong_gt_action_1500_v1/action_traces \
  --output-npz /mnt/workspace/manipulation/BridgeVLA/eval/future_action_memory_d300_abc700_abc1500_state_v1/memory.npz \
  --summary-json /mnt/workspace/manipulation/BridgeVLA/eval/future_action_memory_d300_abc700_abc1500_state_v1/summary.json \
  --chunk-len 16 \
  --chunks-per-trace 3 \
  --horizon 16
```

Current training table:

```text
/mnt/workspace/manipulation/reenvision/checkpoints/calvin/future_action_memory_d300_abc700_abc1500_state_v1/memory.npz
```

## 4. CALVIN Joint Action Generator

This trains the MLP action generator used to directly propose local action suffixes from belief/future/action context.

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
/mnt/workspace/manipulation/reenvision/env/defi_venv/bin/python \
  policy_evaluation/train_joint_action_generator_mlp.py \
  --memory-npz /mnt/workspace/manipulation/BridgeVLA/eval/future_action_memory_d300_abc700_abc1500_state_v1/memory.npz \
  --output-dir /mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2 \
  --steps 4000 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --success-only \
  --device cuda:0
```

Current checkpoint:

```text
/mnt/workspace/manipulation/reenvision/checkpoints/calvin/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt
```

## 5. CALVIN Belief/RSSM Plan Head

This trains the belief transition / plan compatibility head used to decide local suffix patching.

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
/mnt/workspace/manipulation/reenvision/env/defi_venv/bin/python \
  policy_evaluation/train_factored_belief_action_transition.py \
  --memory-npz /mnt/workspace/manipulation/DeFi/outputs/joint_belief_transition_v2_balanced_success/transition_memory_s1024_seq934_success_balanced_650x2_oversample.npz \
  --output-dir /mnt/workspace/manipulation/DeFi/outputs/full_rssm_plan_compat_v2_success_balanced_650x2 \
  --full-rssm-plan-head \
  --hidden-dim 1024 \
  --rssm-deter-dim 512 \
  --rssm-stoch-dim 128 \
  --kl-loss-weight 1e-3 \
  --expected-loss-weight 0.5 \
  --action-loss-weight 0.5 \
  --mode-loss-weight 0.1 \
  --confidence-loss-weight 0.05 \
  --steps 3000 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cuda:0
```

Current checkpoint:

```text
/mnt/workspace/manipulation/reenvision/checkpoints/calvin/full_rssm_plan_compat_v2_success_balanced_650x2/rssm_plan_compatibility.pt
```

## 6. RLBench Suffix-TD Head

The data is collected by running the BridgeVLA collection config first:

```bash
cd /mnt/workspace/manipulation/reenvision/code/BridgeVLA
set -a; source /mnt/workspace/manipulation/reenvision/configs/rlbench_86_suffix_collect.env; set +a
bash finetune/RLBench/eval.sh
```

Then build the balanced suffix-TD memory:

```bash
cd /mnt/workspace/manipulation/reenvision/code/BridgeVLA/finetune/RLBench
/mnt/workspace/manipulation/reenvision/env/defi_venv/bin/python \
  build_bridge_suffix_transition_memory.py \
  --rows-jsonl /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition_rows.jsonl \
  --output-npz /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz \
  --summary-json /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.json \
  --delta-clip 0.02 \
  --balance-task-mode \
  --max-per-task-mode 80
```

Train the RLBench suffix-TD model:

```bash
cd /mnt/workspace/manipulation/reenvision/code/DeFi/Step3_DeFI
/mnt/workspace/manipulation/reenvision/env/defi_venv/bin/python \
  policy_evaluation/train_factored_belief_action_transition.py \
  --memory-npz /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz \
  --output-dir /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/train_factored_td_balanced_taskmode_cap80_patchonly_clip005 \
  --expected-transition \
  --adaptation-input post_innovation \
  --target-delta-clip 0.05 \
  --expected-loss-weight 0.5 \
  --action-loss-weight 2.0 \
  --mode-loss-weight 0.1 \
  --confidence-loss-weight 0.05 \
  --steps 3000 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cuda:0
```

Current checkpoint:

```text
/mnt/workspace/manipulation/reenvision/checkpoints/rlbench/factored_td_patchonly_clip005/factored_belief_action_transition.pt
```

## Notes

The commands above are the current reproducible training commands for the paper-relevant components. Some data-generation stages depend on already-collected rollout/action-trace folders; those folders are included or linked in the bundle where they are still present.

The original BridgeVLA base-policy training entry is `code/BridgeVLA/finetune/RLBench/train.sh`, but it still has placeholder paths and old hardcoded replay roots. The paper bundle currently uses the existing `model_80.pth` BridgeVLA checkpoint for RLBench evaluation.
