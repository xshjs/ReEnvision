# Runnable Upload Manifest

This file lists the minimal code and checkpoint recommendations for the X5 real-robot runnable upload.

## Shared dataset folders

- `pen`: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403`
- `cups`: `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133`

## Recommended ckpts

- `pen_tape_cutter_tray` primary:
  - shared path: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt`
  - source path: `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-24_20-08-09/saved_models/epoch_002.pt`
- `pen_tape_cutter_tray` secondary:
  - shared path: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/best_val.pt`
  - source path: `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-25_19-44-39/saved_models/best_val.pt`
- `stack_cups` primary:
  - shared path: `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt`
  - source path: `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-28_21-18-37/saved_models/best_val.pt`

## Shared ckpt placement

For collaborator handoff, place the four selected checkpoints into the original shared dataset folders:

- `pen` checkpoints under `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403`
- `cups` checkpoints under `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133`

The X5 export script writes rollout-ready files under each shared folder. For the current `cups` joint-action path, use `x5_exports_best_val_joint_action`:

- `x5_exports/validation_sample_00000_raw.npy`
- `x5_exports/validation_sample_00000_ee.npy`
- `x5_exports/validation_sample_00000_joint.npy`
- `x5_exports/validation_sample_00000_summary.json`

## Upload this code set

These files are the core changes for real-robot dataset conversion, training, validation, checkpoint saving, and compatibility fixes:

- `Step3_DeFI/scripts/convert_lerobot_v30_to_defi_real_robot.py`
- `Step3_DeFI/scripts/compute_x5_ee_from_lerobot.py`
- `Step3_DeFI/scripts/convert_defi_rel_action_to_x5_ee.py`
- `Step3_DeFI/scripts/convert_defi_rel_action_to_x5_joint.py`
- `Step3_DeFI/scripts/export_x5_action_from_ckpt.py`
- `Step3_DeFI/scripts/run_bridgevla_rlbench_suffix_td_eval.sh`
- `Step3_DeFI/scripts/train_calvin.py`
- `Step3_DeFI/scripts/train_calvin_real_robot.sh`
- `Step3_DeFI/policy_conf/VPP_Calvinabc_train.yaml`
- `Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py`
- `Step3_DeFI/policy_models/VPP_policy.py`
- `Step3_DeFI/policy_models/datasets/disk_dataset.py`
- `Step3_DeFI/policy_models/edm_diffusion/gc_sampling.py`
- `Step3_DeFI/policy_models/edm_diffusion/score_wrappers.py`
- `Step3_DeFI/policy_models/m_former_univla/blocks.py`
- `Step3_DeFI/policy_models/module/Video_Former.py`
- `Step3_DeFI/policy_models/utils/x5_action_conversion.py`
- `Step3_DeFI/policy_models/utils/utils.py`
- `Step3_DeFI/README_BASELINE_RUNNABLE.md`
- `Step3_DeFI/RLBENCH_SUFFIX_MEMORY_REPRO.md`

## What these changes do

- Real-robot LeRobot v3 to DeFi conversion for X5 data.
- FK-based EE delta action generation with `future_state_offset=5`.
- Validation metrics at epoch end:
  - `val_loss`
  - `val_first_mae`
  - `val_chunk_mae`
  - `val_xyz_mae`
  - `val_rot_mae`
  - `val_gripper_mae`
- Save checkpoints:
  - every `save_every` steps
  - every epoch as `epoch_xxx.pt`
  - best validation checkpoint as `best_val.pt`
- Compatibility fixes for missing optional deps and CPU/CUDA hardcoding issues.
- Checkpoint export path from normalized DeFi output to:
  - raw normalized chunk
  - EE delta chunk
  - joint delta chunk
- RLBench suffix-memory repro documentation and code:
  - source rollout rows
  - keep/patch label construction
  - action suffix target alignment
  - non-zero delta examples
  - current proxy-supervision limitations

## Current training config behavior

- `save_every: 2000`
- `val_num_batches: 20`
- `act_seq_len: 50`
- `multistep: 50`

## Recommended X5 export commands

Use the shared checkpoints directly from the two shared dataset folders.

`pen` primary:

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

python scripts/export_x5_action_from_ckpt.py \
  --ckpt /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt \
  --root_data_dir /mnt/workspace/manipulation/datasets/defi_x5_left_pen_tape_cutter_tray_fk_ee_offset5 \
  --video_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
  --text_encoder_path /mnt/data/xiyin/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/t5_base \
  --language_goal_path /mnt/data/xiyin/manipulation/DeFi/ckpts/ViT-B-32.pt \
  --split validation \
  --sample_index 0 \
  --export_mode all \
  --raw_dataset_root /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403 \
  --episode_index 0 \
  --frame_index 0 \
  --urdf /tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf \
  --binary_gripper
```

`cups` primary:

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

python scripts/export_x5_action_from_ckpt.py \
  --config_name VPP_Calvinabc_train_joint_action \
  --action_format joint_absolute \
  --ckpt /mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt \
  --root_data_dir /mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_joint_action \
  --video_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
  --text_encoder_path /mnt/data/xiyin/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5_model_path /mnt/data/xiyin/manipulation/DeFi/ckpts/t5_base \
  --language_goal_path /mnt/data/xiyin/manipulation/DeFi/ckpts/ViT-B-32.pt \
  --split validation \
  --sample_index 0 \
  --export_mode all \
  --urdf /tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf
```

## Note on `action_dim_weights`

`policy_models/edm_diffusion/score_wrappers.py` and `policy_models/VPP_policy.py` include optional action-dimension weighting support.

Old checkpoints may show:

- `Missing keys (in model but not in checkpoint): model.action_dim_weights`

That is expected for checkpoints trained before this parameter existed.

## Suggested upload policy

Do not upload the entire current working tree blindly.

Prefer uploading only the files listed in `Upload this code set`, because the repo currently also contains many unrelated local changes and experimental files.
