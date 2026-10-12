# ReEnvision Bundle Manifest

## Code

`code/DeFi` -> `/mnt/workspace/manipulation/DeFi`

`code/BridgeVLA` -> `/mnt/workspace/manipulation/BridgeVLA`

## Scripts

`scripts/run_calvin_paper_suffix.sh` -> CALVIN ABC-D paper-style run.

`scripts/run_lcalvin_paper_suffix.sh` -> L-CALVIN 10-step paper-style run.

## Configs

`configs/rlbench_92_suffix.env` -> BridgeVLA/RLBench suffix-TD evaluation config.

`configs/rlbench_86_suffix_collect.env` -> BridgeVLA/RLBench local baseline/collection config.

## Data

`data/calvin_task_ABC_D` -> CALVIN ABC-D dataset.

`data/rlbench_eval_all18` -> RLBench evaluation demos for 18 tasks.

`data/outcome_router_clean_1k_train1000.json` -> CALVIN 1000-sequence split.

`data/white_seq_num=100-seq_len=10.json` -> L-CALVIN 100-sequence, 10-step split.

## CALVIN Checkpoints And Assets

`checkpoints/calvin/last.pt` -> final DeFi action model checkpoint after 100-step continuation from the original Step3 checkpoint.

`checkpoints/calvin/policy_lora_action_intent_v2` -> action-intent LoRA.

`checkpoints/calvin/joint_action_generator_mlp_d300_abc700_abc1500_v2` -> joint action generator.

`checkpoints/calvin/full_rssm_plan_compat_v2_success_balanced_650x2` -> RSSM compatibility head.

The final CALVIN/L-CALVIN evaluation scripts use direct generation and do not load the old future-action memory or energy reranker.

## RLBench Checkpoints And Assets

`checkpoints/rlbench/model_80.pth` -> BridgeVLA RLBench checkpoint.

`checkpoints/rlbench/paligemma-3b-pt-224` -> local PaliGemma model folder for offline loading.

`checkpoints/rlbench/factored_td_patchonly_clip005` -> suffix-TD checkpoint folder.

## Environments

`env/defi_venv` -> active Python environment used for CALVIN/L-CALVIN training, CALVIN/L-CALVIN evaluation, and the current BridgeVLA/RLBench evaluation config.

`env/bridgevla_eval311` -> existing empty Python 3.11 venv. It is preserved only as a historical pointer and is not the active reproducibility environment.

`env/requirements_defi_venv.txt` -> exported `pip freeze` for `defi_venv`.

`env/requirements_bridgevla_eval311.txt` -> exported `pip freeze` for `bridgevla_eval311`.

## Known Local Results

`results/rlbench/bridgevla_92_44_suffix_td` -> local 92.44 RLBench suffix-TD triggered evaluation result.

`results/rlbench/bridgevla_86_22_suffix_collect` -> local 86.22 BridgeVLA collection/baseline-like result.

`results/calvin/calvin_4_742_older_joint_energy.log` -> older local CALVIN 4.742 joint-energy result, not the final paper-local chunk patch result.

`results/lcalvin/lcalvin_8_07_local` -> closest local L-CALVIN 8.07 result found so far, not the paper 8.4 result.

## Extra Method Map

`docs/PAPER_METHOD_CODE_MAP.md` records how the current PDF method maps to code paths.
