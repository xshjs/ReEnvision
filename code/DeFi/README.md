# Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining
### [[Paper]](https://openreview.net/pdf?id=DdrsHWobR1) [[HuggingFace]](https://huggingface.co/zbzzbz/DeFI) 

> **Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining**            
> [Wenyao Zhang*](https://zhangwenyao1.github.io/), [Bozhou Zhang*](https://zbozhou.github.io/), [Zekun Qi](https://qizekun.github.io/), [Wenjun Zeng](https://scholar.google.com/citations?user=_cUfvYQAAAAJ&hl=zh-CN), [Xin Jin](https://scholar.google.com/citations?user=byaSC-kAAAAJ&hl=zh-CN), [Li Zhang](https://lzrobots.github.io)  
> **ICLR 2026**

## Abstract
Vision-language-action (VLA) models have shown great potential in building generalist robots, but still face a dilemma–misalignment of 2D image forecasting and 3D action prediction. Besides, such a vision-action entangled training manner limits model learning from large-scale, action-free web video data. To address these issues, we propose DeFI, a novel framework that Decouples visual Forward and Inverse dynamics pretraining to exploit respective data sources, wherein video generation and action prediction are disentangled. We introduce the General Forward Dynamics Model (GFDM), pretrained on diverse human and robot videos for future prediction, and the General Inverse Dynamics Model (GIDM), trained via self-supervised learning to infer latent actions from unlabeled video transitions. These models are then integrated into a unified architecture for end-to-end fine-tuning on downstream tasks. In this manner, GFDM and GIDM first shine separately and then cooperate for mutual benefit. Extensive experiments on CALVIN ABC-D and SimplerEnv demonstrate state-of-the-art performance, with DeFI achieving an average task length of 4.51 for CALVIN, 51.2% success rate on SimplerEnvFractal benchmark and 81.3% success rate in real-world deployment, significantly outperforming prior methods.

## News
- 2026-03, we release the original version of DeFI, which includes pre-training and evaluation on the CALVIN benchmark.

## Pipeline
<div align="center">
  <img src="asset/defi.PNG"/>
</div><br/>

## TODO list
- [x] release the original version on the CALVIN benchmark
- [ ] add more benchmarks
- [ ] integrated into lerobot and starvla format

## Environment
```
conda create -n defi python==3.10
conda activate defi

pip install setuptools==57.5.0
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin
sh install.sh

cd DeFI_PATH
pip install -r requirements.txt

pip uninstall -y torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## The required checkpoints and data

<details>
<summary><b> Checkpoints </b></summary>

- [stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- ["ViT-B-32.pt" (ViT-B/32) in the CLIP](https://github.com/openai/CLIP)
- [t5-base](https://huggingface.co/google-t5/t5-base)

Download these weights and place them in the "ckpts" folder.

For the current X5 real-robot runnable path, the same base checkpoints are used together with the following recommended Step3 task checkpoints:

- `pen`: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt`
- `pen` alternative: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/best_val.pt`
- `cups`: `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt`
- `cups` source run: `/mnt/data/xiyin/manipulation/DeFi/outputs/calvin_train/2026-08-28_21-18-37/saved_models/best_val.pt`

</details>

<details>
<summary><b> Data </b></summary>

### Stage 1
For the pre-training of GFDM, we follow the data processing procedure and data format of [VPP](https://github.com/roboterax/video-prediction-policy?tab=readme-ov-file#-stage-1-training-video-model).

Example:
```
data/opensource_robotdata/bridge/
├── annotation/
│   ├── train/
│   └── val/
│       ├── 2.json      ← annotation file
│       └── ... (more json files)
└── videos/
│   ├── train/
│   └── val/
│       ├── 2/
│       │   └── rgb.mp4      ← video file
│       └── ... (more video directories)
└── latent_videos/
    ├── train/
    └── val/
        ├── 2/
        │   └── 0.pt      ← latent video file
        └── ... (more latent video directories)


```
- During training, the JSON files (annotation files) are used, which further reference the PT files (latent video files).
- During evaluation, the JSON files (annotation files) are used, which further reference the MP4 files (video files).

### Stage 2
For the pre-training of GIDM, we follow the data processing procedure and data format of [UniVLA](https://github.com/OpenDriveLab/UniVLA?tab=readme-ov-file#zero-data-preparation).

Example:
```
data/oxe/
├── DOWNLOAD_DIR/
│   ├── fractal20220817_data/
│   ├── language_table/
│   └── ...
└── CONVERSION_DIR/
    ├── fractal20220817_data/
    ├── language_table/
    └── ...
```

### Stage 3
Download the Calvin ABC-D dataset from follow [Calvin](https://github.com/mees/calvin?tab=readme-ov-file#computer--quick-start).

For the current X5 real-robot runnable path, the converted DeFi-format datasets are:

- `pen`: `/mnt/workspace/manipulation/datasets/defi_x5_left_pen_tape_cutter_tray_fk_ee_offset5`
- `cups`: `/mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_joint_action`

</details>

## Train and eval
```
### Stage 1 Scripts ###

cd Step1_GFDM

# Encode dataset videos with VAE for preprocessing
scripts/prepare_data_latent.sh

# Train FDM
scripts/train_svd.sh

# Evaluate FDM
scripts/eval_svd.sh

### Stage 2 Scripts ###

cd Step2_GIDM

# Train IDM
train.sh

### Stage 3 Scripts ###

cd Step3_DeFI

# Train DeFI
scripts/train_calvin.sh

# Evaluate DeFI
scripts/rollout_calvin.sh

```

## X5 Pen/Cups commands

The following commands are additive examples for the current X5 real-robot setup. They do not replace the original Stage 1/2/3 pipeline above.

## CALVIN Joint-Belief Suffix Rollout

### Current Paper-Style Run

CALVIN 5-step quick run:

```bash
NUM_SEQUENCES=100 bash /mnt/workspace/manipulation/run_calvin_paper_suffix.sh
```

CALVIN 5-step full run:

```bash
NUM_SEQUENCES=1000 bash /mnt/workspace/manipulation/run_calvin_paper_suffix.sh
```

### Current L-CALVIN Paper-Style Run

L-CALVIN 10-step run:

```bash
NUM_SEQUENCES=100 bash /mnt/workspace/manipulation/run_lcalvin_paper_suffix.sh
```

### Archived Ablations

Historical ablation/export/data-preparation scripts that are not part of the current paper reproduction path were moved to:

```text
/mnt/workspace/manipulation/unused/non_paper_methods_2026-10-12
```

### RLBench Suffix Memory Repro Notes

The provenance, label construction, temporal alignment assumptions, non-zero
delta examples, T/D inputs, action units, runtime patch limits, and deployment
configuration for the RLBench suffix memory are documented in:

```text
Step3_DeFI/RLBENCH_SUFFIX_MEMORY_REPRO.md
```

The runnable repo also includes the key memory builder and BridgeVLA eval wrapper:

```text
Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py
Step3_DeFI/scripts/run_bridgevla_rlbench_suffix_td_eval.sh
```

### Reproduce `factored_belief_action_with_za_balanced_v1`

This checkpoint is the CALVIN factored transition/action-adaptation model used by the joint-belief suffix controller:

```text
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_transition.pt
```

This shared copy is the default deployment path used by the runnable scripts. It was copied from the local training output:

```text
/mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_with_za_balanced_v1/train_run/factored_belief_action_transition.pt
```

It is trained from per-step rollout traces with the following factorization:

```text
h_{t+1} = T_theta(h_t, state_t, action_t, state_{t+1}, task)
A_remain^{t+1} = A_remain^t + D_psi(A_remain^t, h_t, h_{t+1}, task)
```

where `h_t` is a compact 1024-dim joint belief made from pooled future latent `z_f` and action-intent latent `z_a`.

#### 1. Collect rollout traces

This collects baseline Step3 DeFI rollout data and writes one row per executed step. The command also records the exact subprocess command in `collect_command.json`.

```bash
cd /mnt/workspace/manipulation/DeFi

PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI \
/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python \
Step3_DeFI/policy_evaluation/collect_joint_belief_transition_dataset.py \
  --num-sequences 100 \
  --ep-len 360 \
  --output-root /mnt/workspace/manipulation/DeFi/outputs/joint_belief_transition_collect_100_with_za \
  --checkpoint-path /mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder/saved_models/step3_defi.pt \
  --action-model-folder /mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder \
  --video-model-path /mnt/workspace/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm \
  --clip-model-path /mnt/workspace/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32 \
  --t5-model-path /mnt/workspace/manipulation/DeFi/ckpts/t5_base \
  --language-goal-path /mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt \
  --joint-pair-action-generator-ckpt /mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt \
  --calvin-abc-dir /mnt/workspace/calvin/task_ABC_D \
  --eval-sequences-path /mnt/workspace/manipulation/DeFi/outputs/splits/outcome_router_clean_1k_train1000.json \
  --summary-dim 1024 \
  --task-dim 128 \
  --max-rows 3000
```

The raw trace files are written under the generated CALVIN log directory:

```text
<log_dir>/joint_belief_transition_rows.jsonl
<log_dir>/joint_belief_transition/*.npz
```

Each raw trace npz contains:

```text
state_before           float32 [39]        CALVIN low-dimensional state before action
state_after            float32 [39]        CALVIN low-dimensional state after action
action                 float32 [7]         executed relative Cartesian action
h_future_before        float32 [...]       future latent before execution
h_action_before        float32 [...]       action-intent latent before execution
h_future_after         float32 [...]       future latent after execution
h_action_after         float32 [...]       action-intent latent after execution
action_remain_before   float32 [<=10, 7]   cached suffix before update
action_remain_after    float32 [<=10, 7]   cached suffix target after update
```

#### 2. Build compact memory

If collection is already finished, build the compact memory directly from its log directory:

```bash
cd /mnt/workspace/manipulation/DeFi

PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI \
/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python \
Step3_DeFI/policy_evaluation/build_joint_belief_memory_from_rollout.py \
  --log-dir /mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder/logs/2026-09-02_16-12-03 \
  --output-npz /mnt/workspace/manipulation/DeFi/outputs/joint_belief_transition_collect_100_with_za/joint_belief_transition_memory_s1024_sample3000.npz \
  --summary-json /mnt/workspace/manipulation/DeFi/outputs/joint_belief_transition_collect_100_with_za/joint_belief_transition_memory_s1024_sample3000.json \
  --summary-dim 1024 \
  --task-dim 128 \
  --max-rows 3000
```

`build_joint_belief_memory_from_rollout.py` creates:

```text
base_summary_exec      h_t, pooled [z_f, z_a], float32 [N, 1024]
target_summary_exec    h_{t+1}, pooled [z_f, z_a], float32 [N, 1024]
target_delta_exec      h_{t+1} - h_t, float32 [N, 1024]
action_chunk           A_remain^t, padded/truncated to float32 [N, 10, 7]
action_delta_target    A_remain^{t+1} - A_remain^t, float32 [N, 10, 7]
executed_action        a_t, float32 [N, 7]
state_start            state_t, float32 [N, 39]
state_end              state_{t+1}, float32 [N, 39]
task_vec               hashed task vector, float32 [N, 128]
mode_label             0 keep, 1 patch, 2 replan
success                later subtask success flag, float32 [N]
```

Action values use the same normalized CALVIN action convention as Step3 DeFI: first 3 dimensions are relative position, next 3 are relative rotation, last dimension is gripper command.

#### 3. Balance keep / patch / replan samples

The shared data bundle for `factored_belief_action_with_za_balanced_v1` is stored with the 0824 pen assets:

```text
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/
```

It contains:

```text
keep_memory_s1024_sample3000.npz                  source keep-heavy memory
delta_memory_15_11_s1024.npz                      source non-zero suffix-delta memory
transition_memory_keep_patch_replan_s1024.npz     balanced training memory
transition_memory_keep_patch_replan_s1024.json    memory construction summary
train_summary.json                                recorded training summary
train_log.jsonl                                   recorded training log
raw_trace_sample/                                 10 raw rollout rows and npz traces for audit
```

The published balanced memory was assembled from the keep-heavy baseline-shadow memory and the delta memory with non-zero action suffix targets:

```bash
cd /mnt/workspace/manipulation/DeFi

PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI \
/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python \
Step3_DeFI/policy_evaluation/balance_factored_belief_action_memory.py \
  --keep-npz /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/keep_memory_s1024_sample3000.npz \
  --delta-npz /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/delta_memory_15_11_s1024.npz \
  --output-npz /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/transition_memory_keep_patch_replan_s1024.npz \
  --summary-json /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/transition_memory_keep_patch_replan_s1024.json \
  --num-keep 1500 \
  --num-patch 750 \
  --num-replan 750 \
  --patch-max-delta-norm 8.5 \
  --seed 0
```

The resulting summary was:

```json
{
  "num_rows": 3000,
  "num_keep": 1500,
  "num_patch": 750,
  "num_replan": 750,
  "patch_max_delta_norm": 8.5,
  "delta_norm_patch_pool": [2.944275140762329, 7.82241153717041],
  "delta_norm_replan_pool": [8.765754699707031, 11.952509880065918],
  "seed": 0
}
```

#### 4. Train the factored T/D model

```bash
cd /mnt/workspace/manipulation/DeFi

PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI \
/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python \
Step3_DeFI/policy_evaluation/train_factored_belief_action_transition.py \
  --memory-npz /mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/transition_memory_keep_patch_replan_s1024.npz \
  --output-dir /mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_with_za_balanced_v1/train_run \
  --hidden-dim 1024 \
  --steps 3000 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --device cuda
```

The recorded training output was:

```json
{
  "num_train": 2711,
  "num_val": 289,
  "mode_counts": {"0": 1500, "1": 750, "2": 750},
  "best": {
    "step": 600,
    "loss": 0.001111252699047327,
    "belief_l1": 0.021070389077067375,
    "action_l1": 0.005003261845558882,
    "mode_acc": 1.0,
    "confidence_l1": 4.244964657118544e-05,
    "train_loss": 0.0008633927209302783
  }
}
```

The local training artifacts are:

```text
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/train_summary.json
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/train_log.jsonl
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/transition_memory_keep_patch_replan_s1024.npz
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/transition_memory_keep_patch_replan_s1024.json
/mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_with_za_balanced_v1/train_run/summary.json
/mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_with_za_balanced_v1/train_run/train_log.jsonl
/mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_with_za_balanced_v1/train_run/factored_belief_action_transition.pt
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_transition.pt
```

The model checkpoint is about 26 MB and the balanced memory npz is about 12 MB, so they are not committed to git by default. The shared checkpoint copy is expected at `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_transition.pt`; otherwise override `JOINT_BELIEF_TRANSITION_CKPT` in the rollout scripts.

#### 5. Minimal memory sample for sanity checking

The balanced memory has these shapes:

```text
action_chunk           float32 [3000, 10, 7]
action_delta_target    float32 [3000, 10, 7]
base_summary_exec      float32 [3000, 1024]
target_summary_exec    float32 [3000, 1024]
target_delta_exec      float32 [3000, 1024]
executed_action        float32 [3000, 7]
state_start            float32 [3000, 39]
state_end              float32 [3000, 39]
task_vec               float32 [3000, 128]
mode_label             int64   [3000]
success                float32 [3000]
```

A small raw trace sample is also included for auditing the conversion from rollout rows to compact memory:

```text
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/raw_trace_sample/
```

It contains:

```text
joint_belief_transition_rows_sample10.jsonl       10 raw row records
joint_belief_transition/row_0000000.npz ...       raw per-step trace npz files
README_raw_trace_sample.json                      field meanings and example shapes
```

A quick validation command:

```bash
cd /mnt/workspace/manipulation/DeFi

/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python - <<'PY'
import numpy as np
p = "/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_repro/transition_memory_keep_patch_replan_s1024.npz"
with np.load(p, allow_pickle=True) as d:
    for k in ["state_start", "executed_action", "base_summary_exec", "target_summary_exec", "action_chunk", "action_delta_target", "mode_label"]:
        print(k, d[k].shape, d[k].dtype, "finite=", np.isfinite(d[k]).all() if d[k].dtype.kind in "fiu" else "n/a")
    print("mode counts:", dict(zip(*np.unique(d["mode_label"], return_counts=True))))
PY
```

### Convert processed X5 data

Convert the processed `cups` LeRobot-style data into DeFi npz format. The current joint-action setup keeps `action[:7]` directly, including the continuous gripper value.

```bash
cd /mnt/workspace/manipulation/DeFi/Step3_DeFI

python scripts/convert_lerobot_v30_to_defi_real_robot.py \
  --input /mnt/data/shared/hxw/datasets/processed/x5_left_stack_cups_0824_1133 \
  --output /mnt/workspace/manipulation/datasets/defi_x5_left_stack_cups_joint_action \
  --action_mode raw_slice \
  --state_dim 15 \
  --action_dim 7 \
  --static_video_key observation.images.third_person \
  --gripper_video_key observation.images.wrist \
  --gripper_mode raw \
  --skip_broken_episodes \
  --overwrite
```

### Step 3 training

The old real-robot Step 3 training wrapper was archived under `unused/training_scripts_2026-10-12/`.

### Step 3 export for real hardware

The old real-hardware export path is not part of the current CALVIN/L-CALVIN/BridgeVLA reproduction path and has been archived with the non-paper scripts.

## X5 Real-Robot Runnable

This repository now also includes a runnable X5 real-robot path built around `Step3_DeFI`.

The original environment, checkpoint, and Stage 1/2/3 sections above remain the primary paper-oriented documentation. This section is only a compact summary for the current X5 `pen` / `cups` workflow.

### Shared dataset folders

- `pen`: `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403`
- `cups`: `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133`

### Recommended checkpoints

- `pen` first-choice:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/epoch_002.pt`
- `pen` second-choice:
  - `/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/best_val.pt`
- `cups` first-choice:
  - `/mnt/data/shared/hxw/x5_left_stack_cups_0824_1133/best_val.pt`

As of August 30, 2026, the following export paths have already been verified to run successfully:

- `pen` `epoch_002.pt`
- `cups` `best_val.pt` from the August 29, 2026 run

### What the model outputs

For the current `cups` joint-action run, the Step3 action head predicts 7D X5 joint + gripper sequences directly.

Older DeFi checkpoints can still predict normalized 7D relative actions:

- dim `0:3`: normalized EE `delta_xyz`
- dim `3:6`: normalized EE `delta_euler_xyz`
- dim `6`: gripper signal

This is not a raw X5 command. For real-robot rollout, export it into:

- `ee.npy` if the execution side consumes end-effector delta pose
- `joint.npy` if the execution side consumes joint deltas

For current X5 integration, prefer `ee.npy` first.

### Generated files

The export commands above write results under each shared folder:

- `x5_exports_best_val_joint_action/validation_sample_00000_raw.npy`
- `x5_exports_best_val_joint_action/validation_sample_00000_ee.npy`
- `x5_exports_best_val_joint_action/validation_sample_00000_joint.npy`
- `x5_exports_best_val_joint_action/validation_sample_00000_summary.json`

### Current training setup

- action chunk: `50`
- action definition: future state offset `5`
- checkpoint saving:
  - every epoch
  - every `2000` steps
  - best validation checkpoint as `best_val.pt`

For detailed Step3 runnable notes, see:

- `Step3_DeFI/README_BASELINE_RUNNABLE.md`
- `Step3_DeFI/RUNNABLE_UPLOAD_MANIFEST.md`

## BibTeX
```bibtex
@article{zhang2026disentangled,
  title={Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining},
  author={Zhang, Wenyao and Zhang, Bozhou and Qi, Zekun and Zeng, Wenjun and Jin, Xin and Zhang, Li},
  journal={arXiv preprint arXiv:2604.16391},
  year={2026}
}
```

## Acknowledgements
- [VPP](https://github.com/roboterax/video-prediction-policy)
- [UniVLA](https://github.com/OpenDriveLab/UniVLA)
- [X-VLA](https://github.com/2toinf/X-VLA)
