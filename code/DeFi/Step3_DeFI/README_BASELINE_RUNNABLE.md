# DeFi Baseline Runnable Notes

This note is for the pure DeFi baseline path only:

- training: `Step3_DeFI/policy_conf/VPP_Calvinabc_train.yaml`
- evaluation: `Step3_DeFI/policy_evaluation/calvin_evaluate.py`

It is intended to make the repo runnable by collaborators without editing local absolute paths in source files.

## 1. Dataset layout expected by training/eval

The datamodule expects:

```text
<ROOT_DATA_DIR>/
  training/
    ep_start_end_ids.npy
    statistics.yaml                  # optional but recommended
    auto_lang_ann.npy                # or training/lang_clip_resnet50/auto_lang_ann.npy
    episode_0000000.npz              # naming pattern can vary, but must be consistent
    episode_0000001.npz
    ...
  validation/
    ep_start_end_ids.npy
    statistics.yaml                  # optional
    auto_lang_ann.npy                # or validation/lang_clip_resnet50/auto_lang_ann.npy
    episode_0000000.npz
    episode_0000001.npz
    ...
```

Each episode file should at minimum provide keys used by `ExtendedDiskDataset`:

- `rgb_static`
- `rgb_gripper`
- `robot_obs`
- `scene_obs`
- `rel_actions`

Optional keys that are consumed if present:

- `stage`
- `trans_action_indicies`
- `rot_grip_action_indicies`
- `ignore_collisions`
- `gripper_pose`
- `rlbench_target_index`

`auto_lang_ann.npy` should contain the usual CALVIN-style structure:

- `info.indx`
- `language.emb`
- `language.ann`
- optional `language.task`

## 2. Important behavior change

`HulcDataModule` no longer interactively downloads the tiny debug dataset by default.

If your dataset path is wrong, training now fails fast with a clear `FileNotFoundError`.

If you intentionally want the tiny debug dataset, set:

```bash
export DEFI_ALLOW_DEBUG_DATASET_DOWNLOAD=1
```

## 3. Pure baseline evaluation

The evaluation entrypoint now supports parameterized horizon instead of hardcoding 5-step or 10-step in source.

### 5-step, 1000 sequences

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
python policy_evaluation/calvin_evaluate.py \
  --video_model_path /path/to/step1_gfdm \
  --action_model_folder /path/to/action_model_folder \
  --clip_model_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --calvin_abc_dir /path/to/task_ABC_D \
  --eval_sequences_path /path/to/5step_sequences.json \
  --num_sequences 1000 \
  --eval_horizon 5
```

### 10-step, 100 sequences from a custom JSON

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
python policy_evaluation/calvin_evaluate.py \
  --video_model_path /path/to/step1_gfdm \
  --action_model_folder /path/to/action_model_folder \
  --clip_model_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --calvin_abc_dir /path/to/task_ABC_D \
  --default_sequences_path /path/to/custom_sequences.json \
  --default_sequence_len 10 \
  --num_sequences 100 \
  --use_default_sequences \
  --eval_horizon 10
```

## 4. Training command skeleton

For collaborators training on real robot data, the important thing is to pass dataset and model paths through config/CLI instead of modifying source.

Actual training entrypoint in this repo:

- `Step3_DeFI/scripts/train_calvin.py`
- `Step3_DeFI/scripts/train_calvin_real_robot.sh`

### LeRobot V30 real-robot conversion

For LeRobot V30 datasets, convert them into the NPZ layout above before training:

```bash
cd /path/to/DeFi/Step3_DeFI

python scripts/convert_lerobot_v30_to_defi_real_robot.py \
  --input /path/to/lerobot_v30_dataset \
  --output /path/to/defi_real_robot_dataset \
  --frame_stride 1 \
  --action_mode delta_pose_gripper \
  --arm left \
  --overwrite
```

Smoke-test conversion on a small subset:

```bash
python scripts/convert_lerobot_v30_to_defi_real_robot.py \
  --input /path/to/lerobot_v30_dataset \
  --output /path/to/defi_real_robot_smoke \
  --max_episodes 2 \
  --frame_stride 10 \
  --action_mode delta_pose_gripper \
  --arm left \
  --overwrite
```

For Sparky cup datasets, prefer `--action_mode delta_pose_gripper`. It maps `observation.images.cam_left -> rgb_static`, `observation.images.cam_right -> rgb_gripper`, writes `robot_obs[0:7]` from the selected arm TCP pose, writes `robot_obs[14]` from `gripper_left`/`gripper_right`, and writes `rel_actions = [delta_xyz / max_pos, delta_euler_xyz / max_orn, gripper]`. The default normalization is `--max_pos 0.05` meters and `--max_orn 0.50` radians, matching the existing DeFi/RLBench relative-action convention.

The legacy `--action_mode raw_slice` mode still exists for debugging, but it maps `action[:7] -> rel_actions`. For Sparky V30 data this is an absolute TCP pose, not a DeFi-compatible relative action, so do not use `raw_slice` for real-robot policy training.

For joint-space LeRobot V30 datasets where `action` and `observation.state` are both 7D joint/gripper vectors, prefer `--action_mode joint_delta`. This writes `robot_obs[:7]` from the current state, pads the remaining DeFi proprio slots with zeros, maps the chosen video keys into `rgb_static` / `rgb_gripper`, and writes `rel_actions = clip((state[t+1] - state[t]) / max_joint_delta, -1, 1)` with the last dimension converted by `--gripper_mode`.

Example for datasets with `observation.images.third_person` and `observation.images.wrist`:

```bash
python scripts/convert_lerobot_v30_to_defi_real_robot.py \
  --input /mnt/data/shared/hxw/x5_left_stack_cups_0824_1133 \
  --output /mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups \
  --static_video_key observation.images.third_person \
  --gripper_video_key observation.images.wrist \
  --action_mode joint_delta \
  --overwrite
```

Full Sparky stack-cups conversion:

```bash
python scripts/convert_lerobot_v30_to_defi_real_robot.py \
  --input /mnt/data/tacumi/lerobot_datasetV30_train_ready/sparky_ugripper/sparky_ugripper_stack_cups_v2gripper_R3_100_v30_tcp_train_ready \
  --output /mnt/workspace/manipulation/datasets/defi_lerobot_stack_cups_delta \
  --action_mode delta_pose_gripper \
  --arm left \
  --overwrite
```

Before real hardware rollout, verify the selected arm and gripper indices with `scripts/inspect_lerobot_action_semantics.py` and keep execution-side workspace/velocity limits enabled.

For X5 real-robot deployment, note that the DeFi action head predicts normalized relative actions, not raw robot commands. For the current X5 real-robot conversion path this means:

- model output dim 0:3 -> normalized EE `delta_xyz`
- model output dim 3:6 -> normalized EE `delta_euler_xyz`
- model output dim 6 -> gripper open/close signal

To convert a saved prediction array into metric EE deltas for an execution side that expects EE increments, use:

```bash
python scripts/convert_defi_rel_action_to_x5_ee.py \
  --input /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/x5_exports/validation_sample_00000_raw.npy \
  --output /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/x5_exports/validation_sample_00000_ee.npy \
  --max_pos 0.05 \
  --max_orn 0.50 \
  --binary_gripper
```

This produces `dx, dy, dz` in meters and `droll, dpitch, dyaw` in radians. If the execution side expects joint deltas instead, an additional IK/controller layer is still required.

For an execution side that expects X5 joint deltas instead of EE deltas, use:

```bash
python scripts/convert_defi_rel_action_to_x5_joint.py \
  --action_input /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/x5_exports/validation_sample_00000_raw.npy \
  --dataset_root /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403 \
  --episode_index 0 \
  --frame_index 0 \
  --urdf /tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf \
  --output /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/x5_exports/validation_sample_00000_joint.npy \
  --max_pos 0.05 \
  --max_orn 0.50 \
  --max_joint_delta 0.05 \
  --binary_gripper
```

This uses a numerical Jacobian plus damped least squares, so it is suitable as a rollout adapter or offline conversion baseline. For final online deployment, still keep robot-side joint, workspace, and velocity limits enabled.

Single-GPU example:

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
python scripts/train_calvin.py \
  --root_data_dir /path/to/real_robot_dataset \
  --video_model_path /path/to/step1_gfdm \
  --text_encoder_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --batch_size 28 \
  --max_epochs 12
```

Multi-GPU example:

```bash
cd /path/to/DeFi/Step3_DeFI

PYTHONPATH=/path/to/DeFi/Step3_DeFI \
accelerate launch --num_processes 4 \
  scripts/train_calvin.py \
  --root_data_dir /path/to/real_robot_dataset \
  --video_model_path /path/to/step1_gfdm \
  --text_encoder_path /path/to/openai_clip_vit_base_patch32 \
  --t5_model_path /path/to/t5_base \
  --language_goal_path /path/to/ViT-B-32.pt \
  --batch_size 28 \
  --max_epochs 12
```

Wrapper script example:

```bash
ROOT_DATA_DIR=/path/to/real_robot_dataset \
VIDEO_MODEL_PATH=/path/to/step1_gfdm \
TEXT_ENCODER_PATH=/path/to/openai_clip_vit_base_patch32 \
T5_MODEL_PATH=/path/to/t5_base \
LANGUAGE_GOAL_PATH=/path/to/ViT-B-32.pt \
NUM_GPUS=4 \
BATCH_SIZE=28 \
MAX_EPOCHS=12 \
PYTHON_BIN=/path/to/venv/bin/python \
bash Step3_DeFI/scripts/train_calvin_real_robot.sh
```

## 4.1 X5 real-robot ckpts and export commands

Current shared dataset folders:

- `pen`:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403`
- `cups`:
  - `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133`

Current recommended checkpoints for real-hardware bring-up:

- `pen` primary:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt`
- `pen` secondary:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/best_val.pt`
- `cups` primary:
  - `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt`
- `cups` source run:
  - `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-28_21-18-37/saved_models/best_val.pt`

Current recommendation for first real-hardware rollout:

- `pen`: start from `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt`
- `cups`: start from `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt`

Export `pen` primary ckpt to `raw + ee + joint`:

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

Export `pen` secondary ckpt:

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

python scripts/export_x5_action_from_ckpt.py \
  --ckpt /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/best_val.pt \
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

Export `cups` primary ckpt:

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

Export `cups` with the helper script:

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

bash scripts/export_x5_stack_cups_best.sh
```

The generated outputs are written under each shared dataset folder. For the current `cups` joint-action path, the helper script writes to `x5_exports_best_val_joint_action`:

- `x5_exports/validation_sample_00000_raw.npy`
- `x5_exports/validation_sample_00000_ee.npy`
- `x5_exports/validation_sample_00000_joint.npy`
- `x5_exports/validation_sample_00000_summary.json`

For real-hardware rollout, prefer `ee.npy` first. Use `joint.npy` as a compatibility or comparison path if the execution side requires joint deltas.

## 4.2 X5 real-hardware test commands

The export script is the intended bridge from a Step3 DeFi checkpoint to X5-consumable rollout files.

If the execution side consumes EE delta pose, run one of the commands below and use the generated `ee.npy`.

`pen` primary, EE-only export:

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
  --export_mode ee \
  --binary_gripper
```

`cups` primary, EE-only export:

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
  --export_mode ee \
  --urdf /tmp/arx_x5_sdk_src/arx_x5_sdk-0.1.7/arx_x5_sdk/urdf/x5_2025.urdf
```

If the execution side consumes joint deltas, run the `all` export and use the generated `joint.npy`.

`pen` primary, export `raw + ee + joint`:

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

`cups` primary, export `raw + ee + joint`:

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

Expected rollout files:

- `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/x5_exports/validation_sample_00000_ee.npy`
- `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/x5_exports/validation_sample_00000_joint.npy`
- `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/x5_exports_best_val_joint_action/validation_sample_00000_ee.npy`
- `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/x5_exports_best_val_joint_action/validation_sample_00000_joint.npy`

Known-good status as of August 29, 2026:

- `pen` `epoch_002.pt` has already been verified to export successfully
- `cups` `best_val.pt` has already been verified to export successfully
- older checkpoints may print `missing_keys: ["model.action_dim_weights"]`
- that missing-key message is expected for inference and does not block export

If your team uses a different training launcher, keep the same contract:

- dataset root passed from config/CLI
- no source edits for local paths
- no interactive fallback download

## 5. What should not stay hardcoded in a shared branch

Before pushing a collaborative branch, avoid keeping these in source:

- `/mnt/workspace/...`
- `/mnt/data/...`
- personal eval JSON paths
- personal output directories

Those should live in:

- CLI args
- environment variables
- small wrapper shell scripts committed under `scripts/`
