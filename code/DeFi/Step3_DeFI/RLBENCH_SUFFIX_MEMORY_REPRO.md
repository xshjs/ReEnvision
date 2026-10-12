# RLBench Suffix Memory Repro Notes

This document records the provenance, label construction, alignment assumptions,
and audit examples for the RLBench suffix transition memory used by the
factored belief/action transition experiments.

## Source Memories

There are two main artifact types.

Raw rollout rows:

```text
suffix_transition_rows.jsonl
suffix_transition/*.npz
```

Example source:

```text
/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition_rows.jsonl
```

Compact training memory:

```text
bridge_suffix_td_memory_*.npz
```

Example source:

```text
/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz
```

Recorded local commits at the time of inspection:

```text
BridgeVLA: 0690b6ba71459a76815381e96797d0396c51a6d9
DeFi:      dea4b92fb7b24c482e070d02464c9610b7c49ed3
```

## Keep And Delta Label Source

The labels are generated from model rollout, not directly from demonstration
oracle actions.

The memory construction script is available in this repo at:

```text
Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py
```

The original local BridgeVLA working copy path was:

```text
/mnt/workspace/manipulation/BridgeVLA/finetune/RLBench/build_bridge_suffix_transition_memory.py
```

Rules:

```text
success row -> keep, delta = 0
failure row -> find a successful row from the same task, matched by nearest step, then create a patch delta
```

So keep rows come from successful rollouts. Non-zero delta rows come from a
failure rollout paired with a successful rollout proxy.

## Actual Scripts And Commands

Collection scripts in the local BridgeVLA checkout:

```text
BridgeVLA/finetune/RLBench/eval.py
BridgeVLA/finetune/RLBench/eval.sh
```

This repo also contains a wrapper with the exact environment and parameters used
for the current BridgeVLA suffix TD evaluation:

```text
Step3_DeFI/scripts/run_bridgevla_rlbench_suffix_td_eval.sh
```

Memory construction script:

```text
Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py
```

Example memory build command:

```bash
cd /mnt/workspace/manipulation/DeFi

PYTHONPATH=/mnt/workspace/manipulation/DeFi/Step3_DeFI \
/mnt/workspace/manipulation/DeFi/.venv/bin/python \
  Step3_DeFI/policy_evaluation/build_bridge_suffix_transition_memory.py \
  --rows-jsonl /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition_rows.jsonl \
  --output-npz /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz \
  --summary-json /mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.summary.json \
  --max-per-task-mode 80 \
  --delta-clip 0.2
```

## Action Correction Label

For a failure row, the action correction target is:

```text
target_action = matched_success.action_remain_before
action_delta  = target_action - failure.action_remain_before
```

Equivalently:

```text
Delta A* = A_remain_success_matched - A_remain_failure
```

For a success row:

```text
Delta A* = 0
mode_label = keep
```

For a failure row:

```text
Delta A* is non-zero
mode_label = patch
```

## Meaning Of `action_chunk`

In the compact memory:

```text
action_chunk = action_remain_before
```

It is not the full episode action sequence. It is the short-horizon action
window remaining after the current executed action:

```text
A_remain = [a_{t+1}, ..., a_{t+K}]
```

For current RLBench artifacts the shape is:

```text
[10, 9]
```

That means 10 steps, each with a 9-dimensional BridgeVLA/RLBench action.

## Target Action Source And Assumption

The target action comes from a successful rollout row from the same task:

```text
matched_success.action_remain_before
```

It is a proxy supervision target because it satisfies:

```text
same task
nearby rollout step
successful future outcome
```

The assumption is that this successful suffix is closer to the desired local
correction than the failed suffix. This is not a strict oracle label and should
be described as success-nearest rollout proxy supervision.

## Success / Failure Filtering

The script filters by success state:

```text
success=True  -> keep
success=False -> patch, matched to a successful row
```

If no successful rows exist, the builder fails with:

```text
No successful suffix rows found
```

## Temporal Alignment

The current alignment is weak but explicit:

```text
same task + nearest step
```

Implementation:

```python
min(candidates, key=lambda r: abs(r["step"] - failure_step))
```

This is not a strict state-level nearest-neighbor match. It is task-level and
step-level proxy alignment.

## Time Correspondence Example

For a failed sample at rollout step `t=2`:

```text
o_t              = state_before
a_t              = executed_action
o_{t+1}          = state_after
h_t              = h_future_before + h_action_before
h_{t+1}          = h_future_after + h_action_after
A_remain^t       = action_remain_before
```

If this row failed, a successful row from the same task and nearest step is used:

```text
A_remain_correct = success_row.action_remain_before
Delta A* = A_remain_correct - A_remain_failure
A_remain_after = A_remain_failure + Delta A*
```

## Padding And Last Chunk Step

The builder pads or crops the matched target action to the source suffix shape:

```python
_pad_or_crop(target_data["action_remain_before"], action_remain.shape)
```

Short targets are zero-padded. Long targets are cropped.

Current compact fields:

```text
remaining_steps = action_remain.shape[0]
full_chunk_len  = action_remain.shape[0]
```

So many samples report a fixed value of 10. In the current memory, this is best
interpreted as the window length, not guaranteed true valid remaining control
steps.

## Remaining Length / Mask Caveat

The raw feature files contain masks:

```text
action_remain_before_mask
action_remain_after_mask
next_action_window_mask
```

The current compact memory builder does not preserve these masks or use them to
compute `remaining_steps`. A stricter version should:

```text
remaining_steps = sum(action_remain_before_mask)
compute loss only on valid mask positions
force padded delta positions to 0
```

This is an important caveat when auditing samples near the end of an episode or
near the end of an action window.

## Non-Zero Delta Pair Example

The following inspected memory contains non-zero delta rows:

```text
/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz
```

Memory summary:

```text
num_rows: 2368
mode_counts: keep=1403, patch=965
delta_norm percentiles: min=0, p50=0, p90=1.3963, p99=1.6190, max=1.7132
```

Example:

```text
idx=2
task=stack_blocks
mode=patch
success=0
cut_idx=2
delta_norm=1.3593
source=/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/suffix_transition/stack_blocks_ep0019_att00_step002.npz
```

Executed action:

```text
[0.4594, -0.1769, 0.9700, -0.9537, 0.3006, -0.0000, 0.0000, 0.0000]
```

First two rows of `A_remain_before`:

```text
[[ 0.2641,  0.0452, 0.9031, -0.0001, 1.0000, -0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.4014, -0.3197, 0.8729, -0.8661, 0.5000, -0.0000, 0.0000, 1.0000, 0.0000]]
```

First two rows of matched target action:

```text
[[ 0.2474, -0.0140, 0.9041,  0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.2463, -0.1197, 0.8540, -0.6661, 0.7000, -0.0000, 0.0001, 1.0000, 0.2000]]
```

First two rows of `Delta A`:

```text
[[-0.0167, -0.0592,  0.0011, 0.0001, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000],
 [-0.1551,  0.2000, -0.0190, 0.2000, 0.2000, 0.0000,  0.0000, 0.0000, 0.2000]]
```

## Audit Command

Use this command to inspect shape, mode counts, and finite values:

```bash
cd /mnt/workspace/manipulation/BridgeVLA

/mnt/workspace/manipulation/DeFi/.venv/bin/python - <<'PY'
import numpy as np

p = "checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_prob_trigger006_small_patch/suffix_collect/bridge_suffix_td_memory_balanced_taskmode_cap80_clip02.npz"
with np.load(p, allow_pickle=True) as d:
    for k in [
        "state_start",
        "executed_action",
        "base_summary_exec",
        "target_summary_exec",
        "action_chunk",
        "action_delta_target",
        "mode_label",
    ]:
        finite = np.isfinite(d[k]).all() if d[k].dtype.kind in "fiu" else "n/a"
        print(k, d[k].shape, d[k].dtype, "finite=", finite)
    print("mode counts:", dict(zip(*np.unique(d["mode_label"], return_counts=True))))
    delta = d["action_delta_target"]
    norms = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    print("delta percentiles:", np.percentile(norms, [0, 50, 90, 99, 100]))
PY
```

## Current Limitation

The current non-zero correction labels are useful for debugging and controlled
ablation, but they are proxy labels:

```text
failure suffix -> matched successful suffix
```

They are not guaranteed to be the unique optimal local action correction. For a
stronger version, collect paired interventions where the same initial state is
run once with baseline and once with successful suffix patch, then use the
successful retry suffix as the target.

## X5 Adaptation Note

The RLBench suffix memory is for BridgeVLA/RLBench ablation. It is not directly
an X5 action-space dataset. For accurate X5 deployment, use the X5-specific data
conversion, real-robot training, and export scripts in the main README.

X5 deployment paths use either:

```text
joint-action cups path: action[:7] = 6 X5 joint positions + gripper
EE export path: raw model action -> normalized relative action -> EE/joint export
```

So RLBench suffix-TD results should not be interpreted as directly runnable X5
joint commands.

## T Input Construction

The factored transition model uses:

```text
T_theta input = h_t, state_start, executed_action, state_end, task_vec, meta
```

In code:

```text
Step3_DeFI/policy_evaluation/train_factored_belief_action_transition.py
BeliefTransitionMLP.forward(h_t, state_t, action_t, state_tp1, task_vec, meta)
```

Field mapping:

```text
h_t             <- base_summary_exec
state_start     <- state before executing a_t
executed_action <- action actually sent at t
state_end       <- state after executing a_t
task_vec        <- hashed task/language vector
meta            <- sequence/stage/chunk metadata from build_meta(...)
```

`h_t` is the joint belief:

```text
h_t = [z_f^t, z_a^t]
```

## z_f / z_a / State / Task Sources

`z_f`:

```text
future latent or pooled future feature from the GFDM/future branch
```

`z_a`:

```text
action-intent or motion latent from the latent-action/action-intent branch
```

Before execution:

```text
h_future_before
h_action_before
```

After execution:

```text
h_future_after
h_action_after
```

The compact memory stores:

```text
base_summary_exec   = [h_future_before, h_action_before]
target_summary_exec = [h_future_after, h_action_after]
```

States:

```text
state_start = state_before
state_end   = state_after
```

Task vector:

```text
task_vec = deterministic hash embedding of task name / language text
```

This is a compact task identifier for the small T/D MLP, not a learned language
encoder.

## `executed_action` Fallback

Some older delta memories do not contain `executed_action`. The balancing script
uses this compatibility fallback:

```python
if "executed_action" not in memory:
    memory["executed_action"] = memory["action_chunk"][:, 0]
```

Reason:

```text
action_chunk is the short action window aligned to the current transition.
Its first element is the best available proxy for the action at that boundary.
```

This is acceptable for old-memory compatibility, but the preferred memory should
store `executed_action` explicitly from rollout.

## Action Units And Normalization

CALVIN action convention:

```text
action shape: [7]
dim 0:3   relative Cartesian translation command
dim 3:6   relative orientation command
dim 6     gripper command
```

These values are in the normalized action space used by Step3 DeFI/CALVIN
rollout. The simulator/controller applies its own conversion and scaling.

BridgeVLA/RLBench action convention:

```text
action shape: [9]
```

The saved suffix action is the BridgeVLA/RLBench policy/controller action
representation. It is not converted to X5 joint units.

The correction target is computed in the same saved action scale as
`action_chunk`:

```text
action_delta_target = action_remain_after - action_remain_before
```

## Raw Prediction Versus Actual Command

The memory stores rollout-side action windows from the policy/controller. During
training, the model predicts a delta in that same action scale.

During deployment/evaluation, the predicted suffix delta is bounded before it is
applied:

```text
delta <- clip(delta, -BRIDGEVLA_SUFFIX_PATCH_MAX_ABS, BRIDGEVLA_SUFFIX_PATCH_MAX_ABS)
patched_action = original_action + BRIDGEVLA_SUFFIX_PATCH_MIX * delta
```

Current BridgeVLA wrapper defaults:

```text
BRIDGEVLA_SUFFIX_PATCH_MIX=0.10
BRIDGEVLA_SUFFIX_PATCH_MAX_ABS=0.03
```

Current conservative CALVIN suffix runner defaults:

```text
DYNAMIC_COUPLING_DELTA_ACTION_REPAIR_MIX=0.02
```

Gripper handling:

```text
CALVIN: gripper stays in normalized DeFI/CALVIN action convention.
BridgeVLA/RLBench: gripper-related dimensions stay in BridgeVLA action convention.
X5 cups joint-action path: current training uses continuous/raw gripper from processed data.
```

## Mode Labels And Data Split

Mode labels:

```text
0 = keep
1 = patch
2 = replan
```

For the balanced CALVIN memory:

```text
keep rows come from keep-heavy source memory
patch/replan rows come from delta memory
```

The balancing script uses delta norm to split patch versus replan:

```text
patch  if ||Delta A|| <= patch_max_delta_norm
replan if ||Delta A|| >  patch_max_delta_norm
```

The published value was:

```text
patch_max_delta_norm = 8.5
```

This is a heuristic proxy threshold chosen to separate moderate suffix changes
from very large changes in the collected delta memory. It is not a ground-truth
semantic label.

Training/validation split:

```text
train_factored_belief_action_transition.py uses split_by_sequence(...)
```

The intended split is by `sequence_index`, not by random row only. However, the
balanced memory is assembled from multiple source memories. If two source files
reuse the same sequence IDs, sequence ID alone may be ambiguous. A stricter split
should use:

```text
(source_label, sequence_index)
```

or split before resampling.

## Are keep / patch / replan True Labels?

They are proxy labels, not human/oracle mode labels.

Current interpretation:

```text
keep   = successful or zero-delta row
patch  = non-zero but moderate delta row
replan = large delta row
```

For runtime-collected rows, if a controller decision is present, the builder can
use it. Otherwise it falls back to delta-norm proxy labels.

## Actual Training And Deployment Version

CALVIN joint-belief suffix checkpoint:

```text
/mnt/data/shared/hxw/x5_left_pen_tape_cutter_tray_0824_1403/factored_belief_action_transition.pt
```

This shared copy is the default runnable/deployment path. It was copied from the local training output:

```text
/mnt/workspace/manipulation/DeFi/outputs/factored_belief_action_with_za_balanced_v1/train_run/factored_belief_action_transition.pt
```

Training command:

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

Current committed DeFi repro commit:

```text
a878a57731c0044d5107eb44f5e47c480220ef89
```

BridgeVLA suffix TD wrapper:

```text
Step3_DeFI/scripts/run_bridgevla_rlbench_suffix_td_eval.sh
```

## Runtime Update Frequency And Belief Recurrence

CALVIN suffix runner:

```text
Step3_DeFI/scripts/run_calvin_joint_belief_suffix.sh
```

Important flags:

```text
--dynamic_coupling_first_pass
--dynamic_coupling_online
--dynamic_coupling_generate_action_intent
--dynamic_coupling_persistent_hypothesis
--joint_belief_transition
--joint_belief_clean_controller
--dynamic_coupling_disable_replan
```

Meaning:

```text
update frequency: every rollout/environment step after transition info is available
belief: persistent hypothesis is maintained across steps
replan: disabled in conservative default runner
suffix patch: small delta patch only when learned patch probability passes threshold
```

Conservative CALVIN defaults:

```text
patch_prob >= 0.55
patch_prob >= keep_prob + 0.10
patch mix = 0.02
```

BridgeVLA wrapper defaults:

```text
BRIDGEVLA_SUFFIX_PATCH_MIN_PROB=0.012
BRIDGEVLA_SUFFIX_TRIGGER_ON_PROB_ONLY=1
BRIDGEVLA_SUFFIX_PATCH_ON_REPLAN=0
BRIDGEVLA_SUFFIX_PATCH_MIX=0.10
BRIDGEVLA_SUFFIX_PATCH_MAX_ABS=0.03
```

## Train/Deployment Feature Preprocessing Consistency

The intended consistency is:

```text
train h_t/h_{t+1}: same pooled z_f/z_a shape as runtime
train action_chunk: same policy action space as runtime
train task_vec: same deterministic hash function as runtime
train state: same low-dimensional state source as runtime
```

Known caveats:

```text
older memories may use action_chunk[:,0] as executed_action fallback
some compact memories do not preserve suffix masks
patch/replan labels are proxy labels
X5 deployment requires separate action export/conversion and is not the same action space as RLBench suffix memory
```

For accurate X5 adaptation, use the X5-specific dataset conversion, training,
and export paths in the main README rather than the RLBench suffix memory.
