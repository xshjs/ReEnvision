# ReEnvision CALVIN Checkpoints

Final CALVIN/L-CALVIN checkpoint package contents:

- `calvin/last.pt`: final DeFi action model after 100-step continuation from the original Step3 checkpoint.
- `calvin/policy_lora_action_intent_v2`: action-intent LoRA used by CALVIN/L-CALVIN.
- `calvin/joint_action_generator_mlp_d300_abc700_abc1500_v2`: direct action suffix generator.
- `calvin/full_rssm_plan_compat_v2_success_balanced_650x2`: belief/RSSM compatibility head.

The old future-action memory and energy reranker are intentionally excluded. The final CALVIN/L-CALVIN scripts use direct generation with no memory selector.
