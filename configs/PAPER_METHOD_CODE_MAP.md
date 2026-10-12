# Paper Method Code Map

This repository is currently organized around the paper reproduction paths for CALVIN, L-CALVIN, and BridgeVLA/RLBench.

## CALVIN / L-CALVIN

Entry points:

```bash
NUM_SEQUENCES=1000 bash /mnt/workspace/manipulation/run_calvin_paper_suffix.sh
NUM_SEQUENCES=100 bash /mnt/workspace/manipulation/run_lcalvin_paper_suffix.sh
```

Core runtime:

- `DeFi/Step3_DeFI/policy_evaluation/calvin_evaluate_with_memory_reflection.py`
- `DeFi/Step3_DeFI/policy_models/VPP_policy.py`
- `DeFi/Step3_DeFI/scripts/run_calvin_joint_belief_suffix.sh`

Paper-method components:

- Baseline rollout first, then one failure-recovery pass from the same subtask context.
- Joint belief transition over pooled future/action latents.
- Belief-guided suffix decision: keep / patch / replan labels, with replan disabled for the current reproduction runner.
- Local action-chunk suffix patch: only the unexecuted suffix of the current action chunk is eligible for modification.
- Linear suffix sensitivity transport is available for local action edits; RSSM-plan fallback writes a small delta action suffix when the controller selects patch.

Kept runtime imports:

- `policy_evaluation/train_*.py` files are kept in place because the CALVIN evaluator imports model classes from these modules at runtime.
- These files may contain training CLIs, but they also define runtime network classes used by the evaluator.

## BridgeVLA / RLBench

Entry points:

```bash
cd /mnt/workspace/manipulation/BridgeVLA
set -a; source /mnt/workspace/manipulation/rlbench_92_suffix_retry.env; set +a
bash finetune/RLBench/eval.sh
```

Core runtime:

- `BridgeVLA/finetune/RLBench/eval.py`
- `BridgeVLA/finetune/RLBench/eval.sh`
- `DeFi/Step3_DeFI/scripts/run_bridgevla_rlbench_suffix_td_eval.sh`
- `DeFi/Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py`

Paper-method components:

- Baseline RLBench rollout first.
- If baseline fails and the agent requests suffix-TD, run a repair pass using the same episode context.
- Suffix-TD modifies the action suffix through belief/action latent transition signals.
- Final success is baseline success or repaired success.

Offline model loading:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
BRIDGEVLA_PALIGEMMA_PATH=/mnt/workspace/manipulation/BridgeVLA/checkpoints/paligemma-3b-pt-224
```

## Archived Code

Code and scripts not used by the current paper reproduction paths were moved to:

```text
/mnt/workspace/manipulation/unused/non_paper_methods_2026-10-12
/mnt/workspace/manipulation/unused/training_scripts_2026-10-12
```

Archived items include old CALVIN video/export/ablation runners, BridgeVLA pretraining, BridgeVLA Colosseum/GemBench code, old push-buttons/debug outputs, DeFi RLBench variants, and historical memory/router/probe shell pipelines.

The archive keeps relative paths so files can be restored if needed.
