# CALVIN Oracle Success Conditions

Source files:

- `calvin/calvin_env/conf/tasks/new_playtable_tasks.yaml`
- `calvin/calvin_env/calvin_env/envs/tasks.py`

The oracle checks state changes between `start_info` and `end_info`. It does not check image similarity.

## Task Groups

| Tasks | Oracle Function | Success Condition |
|---|---|---|
| `push_{red,blue,pink}_block_right` | `push_object(block, +0.1, 0)` | Target block `delta_x > 0.1m` and non-robot surface contact is preserved from start to end. |
| `push_{red,blue,pink}_block_left` | `push_object(block, -0.1, 0)` | Target block `delta_x < -0.1m` and non-robot surface contact is preserved from start to end. |
| `move_slider_left` | `move_door_rel(base__slide, +0.15)` | Slider joint increases by more than `0.15`. |
| `move_slider_right` | `move_door_rel(base__slide, -0.15)` | Slider joint decreases by more than `0.15`. |
| `open_drawer` | `move_door_rel(base__drawer, +0.12)` | Drawer joint increases by more than `0.12`. |
| `close_drawer` | `move_door_rel(base__drawer, -0.12)` | Drawer joint decreases by more than `0.12`. |
| `stack_block` | `stack_objects(max_vel=1)` | Some block changes from no block-block contact to block-block contact; end contacts are only movable blocks; end linear/angular velocity components are `< 1`. |
| `unstack_block` | `unstack_objects(max_vel=1)` | Some block starts only in block-block contact and ends with no block-block contact; start linear/angular velocity components are `< 1`. |
| `push_into_drawer` | `push_object_into(any block, table/base_link, table/drawer_link)` | Any block starts with exactly one contact on table `base_link`, ends in contact with table `drawer_link`, and robot is not touching it at start/end. |
| `lift_*_block_table` | `lift_object(block, 0.05, table/base_link)` | Target block `delta_z > 0.05m`, ends in contact only with robot, and starts on table `base_link`. |
| `lift_*_block_slider` | `lift_object(block, 0.03, table/plank_link)` | Target block `delta_z > 0.03m`, ends in contact only with robot, and starts on table `plank_link`. |
| `lift_*_block_drawer` | `lift_object(block, 0.05, table/drawer_link)` | Target block `delta_z > 0.05m`, ends in contact only with robot, and starts on table `drawer_link`. |
| `place_in_slider` | `place_object(table/plank_link)` | Robot starts touching exactly one movable object, stops touching it, and object ends on table `plank_link`. |
| `place_in_drawer` | `place_object(table/drawer_link)` | Robot starts touching exactly one movable object, stops touching it, and object ends on table `drawer_link`. |
| `rotate_*_block_left` | `rotate_object(block, +60)` | Target block rotates around z by `> +60°` and `< +180°`, x/y rotation `< 30°`, translation norm `<= 0.1m`, and end has non-robot contact. |
| `rotate_*_block_right` | `rotate_object(block, -60)` | Target block rotates around z by `< -60°` and `> -180°`, x/y rotation `< 30°`, translation norm `<= 0.1m`, and end has non-robot contact. |
| `turn_on_lightbulb` | `toggle_light(lightbulb, 0, 1)` | Lightbulb logical state changes from `0` to `1`. |
| `turn_off_lightbulb` | `toggle_light(lightbulb, 1, 0)` | Lightbulb logical state changes from `1` to `0`. |
| `turn_on_led` | `toggle_light(led, 0, 1)` | LED logical state changes from `0` to `1`. |
| `turn_off_led` | `toggle_light(led, 1, 0)` | LED logical state changes from `1` to `0`. |

## Outcome Signature Upgrade

The previous symbolic contact labels are too weak:

```text
drawer_contact, block_block_contact, surface_contact, move_right
```

Use task-conditioned interaction outcomes instead:

| Old Contact Label | Better Outcome Field |
|---|---|
| `move_right = 1` | `target_delta_x > 0.1` for right push, `target_delta_x < -0.1` for left push |
| `drawer_contact = 1` | `drawer_joint_delta > 0.12` for open, `< -0.12` for close |
| `block_block_contact = 1` | new block-block contact + only block contacts + velocity `< 1` |
| `surface_contact = 1` | target-specific preserved support contact, not generic table contact |
| none | `slider_joint_delta`, `target_delta_z`, `rotation_z_deg`, `light_logical_state` |

## Validation Command

Run the explicit observed-outcome verifier against the CALVIN oracle:

```bash
cd /mnt/workspace/manipulation/DeFi
.venv/bin/python Step3_DeFI/policy_evaluation/outcome_verifier.py \
  --video-model-path /DeFI/ckpts/svd/checkpoint-100000 \
  --checkpoint /DeFI/outputs/defi_eval/saved_models/step3_defi.pt \
  --clip-model-path /DeFI/ckpts/openai_clip_vit_base_patch32 \
  --t5-model-path /DeFI/ckpts/t5_base \
  --language-goal-path /DeFI/ckpts/ViT-B-32.pt \
  --calvin-abc-dir /DeFI/ckpts/task_ABC_D \
  --output-dir outputs/outcome_verifier_vs_oracle \
  --num-sequences 5 \
  --ep-len 80
```

This writes `report.md`, `metrics.json`, and per-step `records.jsonl`.

## Outcome Dataset Command

Build a dataset of `(obs_start, skill) -> observed outcome` samples from CALVIN D annotations:

```bash
cd /mnt/workspace/manipulation/DeFi
python3 Step3_DeFI/policy_evaluation/outcome_dataset.py \
  --calvin-dir /mnt/data/calvin/task_D_D \
  --output-dir outputs/outcome_dataset_2k_mlp \
  --max-samples-per-split 1000 \
  --train-mlp \
  --epochs 50 \
  --device cpu
```

The script writes:

- `skill_stats.md`: skill distribution table for D.
- `outcomes.jsonl`: JSONL samples with `obs_start`, `skill`, and typed `outcome` fields.
- `outcomes.npz`: compact arrays for training.
- `mlp_report.md`: MAE / RMSE / R² for the lightweight MLP baseline.

The fast extractor uses `scene_obs` deltas, so push/drawer/slider/light/lift/rotate outcomes are direct state differences. Stack outcomes are geometric proxies from block poses; exact contact/velocity stack verification still requires env `info` or rollout-time extraction.

The dataset builder also supports visual features:

```bash
python3 Step3_DeFI/policy_evaluation/outcome_dataset.py \
  --calvin-dir /mnt/data/calvin/task_D_D \
  --output-dir outputs/outcome_dataset_2k_clip_static_mlp \
  --max-samples-per-split 1000 \
  --feature-source clip_static \
  --clip-model-path ckpts/openai_clip_vit_base_patch32 \
  --clip-device cuda:0 \
  --train-mlp \
  --epochs 50 \
  --device cpu
```

Supported `--feature-source` values include `state`, `clip_static`, `clip_static_gripper`, `state_clip_static`, and `state_clip_static_gripper`.

GFDM latent features are also supported, but extraction is heavier:

```bash
.venv/bin/python Step3_DeFI/policy_evaluation/outcome_dataset.py \
  --calvin-dir /mnt/data/calvin/task_D_D \
  --output-dir outputs/outcome_dataset_1k_gfdm_latent_mlp \
  --max-samples-per-split 500 \
  --feature-source gfdm_latent \
  --gfdm-video-model-path ckpts/svd/checkpoint-100000 \
  --gfdm-text-encoder-path ckpts/openai_clip_vit_base_patch32 \
  --gfdm-device cuda:0 \
  --gfdm-batch-size 2 \
  --train-mlp \
  --epochs 50 \
  --device cpu
```

Use `--feature-source state_gfdm_latent` to concatenate state and GFDM latent features.

## Predicted Outcome Verifier / Retry Command

Run both experiments:

1. `Predicted Outcome` vs `Observed Outcome` mismatch as a skill-success verifier.
2. `DeFi` vs `Outcome-Aware Retry`, where mismatch triggers retry.

```bash
cd /mnt/workspace/manipulation/DeFi
.venv/bin/python Step3_DeFI/policy_evaluation/outcome_prediction_experiments.py \
  --video-model-path /DeFI/ckpts/svd/checkpoint-100000 \
  --checkpoint /DeFI/outputs/defi_eval/saved_models/step3_defi.pt \
  --clip-model-path /DeFI/ckpts/openai_clip_vit_base_patch32 \
  --t5-model-path /DeFI/ckpts/t5_base \
  --language-goal-path /DeFI/ckpts/ViT-B-32.pt \
  --calvin-abc-dir /DeFI/ckpts/task_ABC_D \
  --calvin-dir /mnt/data/calvin/task_D_D \
  --output-dir outputs/outcome_prediction_experiments_n5 \
  --num-sequences 5 \
  --ep-len 80 \
  --retry-count 1 \
  --train-samples 1000 \
  --outcome-epochs 50 \
  --device cuda:0
```

Outputs:

- `experiment1/report.md`: verifier Accuracy / F1 / AUROC.
- `defi_no_retry/summary.json`: baseline DeFi rollout.
- `outcome_retry1/summary.json`: predicted-outcome retry rollout.
- `comparison.md`: baseline vs retry chain success rates.

## Oracle Verifier Accuracy Curve

Sweep a synthetic verifier with controlled accuracy and measure long-horizon success:

```bash
cd /mnt/workspace/manipulation/DeFi
.venv/bin/python Step3_DeFI/policy_evaluation/oracle_curve_evaluate.py \
  --video-model-path /DeFI/ckpts/svd/checkpoint-100000 \
  --checkpoint /DeFI/outputs/defi_eval/saved_models/step3_defi.pt \
  --clip-model-path /DeFI/ckpts/openai_clip_vit_base_patch32 \
  --t5-model-path /DeFI/ckpts/t5_base \
  --language-goal-path /DeFI/ckpts/ViT-B-32.pt \
  --calvin-abc-dir /DeFI/ckpts/task_ABC_D \
  --output-dir outputs/oracle_curve_n5 \
  --num-sequences 5 \
  --ep-len 80 \
  --retry-count 1 \
  --accuracies 0.6,0.7,0.8,0.9,1.0 \
  --seeds 0 \
  --device cuda:0
```

Outputs: `oracle_curve.md`, `oracle_curve.csv`, `oracle_curve.json`, and `oracle_curve.png`.

## Future Skill Probe

Compare whether future information helps recover the true skill label:

```bash
cd /mnt/workspace/manipulation/DeFi
.venv/bin/python Step3_DeFI/policy_evaluation/future_skill_probe.py \
  --calvin-dir /mnt/data/calvin/task_D_D \
  --output-dir outputs/future_skill_probe_200 \
  --max-samples-per-split 200 \
  --groups obs,obs_gt_future,gt_future_delta,obs_gfdm_future \
  --clip-model-path ckpts/openai_clip_vit_base_patch32 \
  --clip-device cuda:0 \
  --gfdm-video-model-path ckpts/svd/checkpoint-100000 \
  --gfdm-text-encoder-path ckpts/openai_clip_vit_base_patch32 \
  --gfdm-device cuda:0 \
  --gfdm-batch-size 2 \
  --epochs 50 \
  --device cpu
```

Groups:

- `obs`: current frame only.
- `obs_gt_future`: current frame plus ground-truth future frame.
- `gt_future_delta`: current frame plus feature delta to ground-truth future.
- `obs_gfdm_future`: current frame plus GFDM future latent.

This is a skill-probe proxy before wiring the latent into the full diffusion policy.
