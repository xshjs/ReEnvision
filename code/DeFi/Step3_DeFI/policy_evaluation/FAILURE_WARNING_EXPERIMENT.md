# Failure Prediction / Early Warning Experiment

This experiment tests whether contact-future mismatch is an earlier and more stable failure signal than visual-future mismatch.

## Current DeFi Output Status

The completed `outputs/defi_eval` run only saved aggregate task success counts and logs. It does not contain per-step rollout trajectories, predicted future visual states, predicted future contact states, or observed simulator contacts. Therefore, the numeric validation requires a small new rollout pass that records per-step prediction/observation pairs.

## Task Split

Contact-rich tasks:

- `push_red_block_right`
- `push_blue_block_right`
- `push_pink_block_right`
- `push_into_drawer`
- `stack_block`

Controls:

- `turn_on_lightbulb`
- `turn_off_lightbulb`
- `turn_on_led`
- `turn_off_led`
- `open_drawer`
- `close_drawer`

## Trajectory JSONL Schema

Each line is one subtask trajectory:

```json
{
  "trajectory_id": "seq0003_subtask02",
  "task": "push_red_block_right",
  "success": false,
  "steps": [
    {
      "t": 0,
      "visual": {
        "pred": [0.1, 0.2],
        "obs": [0.1, 0.3]
      },
      "contact": {
        "pred": [
          {
            "body_a": "block_red",
            "link_a": "base",
            "body_b": "table",
            "link_b": "base_link",
            "position": [0.0, 0.0, 0.0],
            "force": [1.0, 0.0, 0.0]
          }
        ],
        "obs": []
      }
    }
  ]
}
```

The analysis script also accepts precomputed fields:

```json
{"t": 0, "visual_mismatch": 0.13, "contact_mismatch": 1.8}
```

## Mismatch Definitions

Visual mismatch is computed from `visual.pred` and `visual.obs` using cosine distance by default. These vectors should be CLIP, LPIPS, image embedding, or latent features; do not use raw pixel L2 as the primary signal.

Contact mismatch uses a composite score:

```text
contact_mismatch =
  λ1 * contact_existence_error
+ λ2 * link_identity_error
+ λ3 * contact_point_distance
+ λ4 * force_direction_error
+ λ5 * force_magnitude_error
+ λ6 * persistence_break_error
```

## Analysis Command

```bash
cd /mnt/workspace/manipulation/DeFi
python3 Step3_DeFI/policy_evaluation/failure_warning_analysis.py \
  --trajectories outputs/failure_warning/trajectories.jsonl \
  --output-dir outputs/failure_warning/contact_rich \
  --tasks contact_rich
```

Run the control set separately:

```bash
python3 Step3_DeFI/policy_evaluation/failure_warning_analysis.py \
  --trajectories outputs/failure_warning/trajectories.jsonl \
  --output-dir outputs/failure_warning/non_contact_rich \
  --tasks non_contact_rich
```

## Reported Metrics

The script writes:

- `metrics.json`: machine-readable metrics.
- `report.md`: summary tables.

It reports:

- Average and median warning lead time.
- Failure detection rate under thresholds calibrated from successful trajectories.
- Success false-alarm rate.
- AUROC, AUPRC, and best F1 for visual mismatch, contact mismatch, and visual+contact.

## Expected Claim

The claim is supported if, especially on contact-rich tasks:

- `contact_mismatch` has larger warning lead time than `visual_mismatch`.
- `contact_mismatch` has higher AUROC/AUPRC/F1 than `visual_mismatch`.
- `visual_plus_contact` is best or tied best.

The non-contact-rich control tasks should show a smaller or absent contact advantage.

## Lightweight Contact Outcome PoC

For a minimal self-supervised verifier experiment, each subtask trajectory can be treated as one sample:

```text
(obs_before_skill, skill_name) -> symbolic_contact_signature_after_skill
```

The lightweight script predicts multi-label outcomes such as `drawer_contact`, `block_block_contact`, `surface_contact`, and `move_right`. It uses cached visual features from the trajectory JSONL plus a learned skill embedding and a small MLP; it does not predict contact points, forces, or torques.

```bash
cd /mnt/workspace/manipulation/DeFi
python3 Step3_DeFI/policy_evaluation/contact_outcome_poc.py \
  --trajectories outputs/failure_warning/run_egl_balanced_s3f3/trajectories.jsonl \
  --output-dir outputs/contact_outcome_poc \
  --feature-source visual_pred \
  --contact-aggregation last
```

The script writes:

- `metrics.json`: macro and per-label Accuracy / F1 / AUROC.
- `report.md`: compact comparison against random, majority, and skill-prior baselines.
- `predictions.jsonl`: per-skill predicted contact outcome scores.

Use group cross-validation by `sequence_index` / trajectory id to avoid leaking skills from the same rollout into both train and test folds. If object displacement is not present in the JSONL, `move_right` is inferred from right-push skill names by default; use `--move-right-label-source none` to disable that label.

## Oracle Retry vs Expected Contact Retry

To directly compare privileged oracle retry with learned expected-contact retry, pass the contact-predictor training JSONL to `contact_retry_evaluate.py`:

```bash
cd /mnt/workspace/manipulation/DeFi
python3 Step3_DeFI/policy_evaluation/contact_retry_evaluate.py \
  --video-model-path /path/to/step1_gfdm \
  --checkpoint /path/to/defi.ckpt \
  --clip-model-path /path/to/clip \
  --calvin-abc-dir /path/to/calvin_abc \
  --output-dir outputs/oracle_vs_predicted_contact_retry \
  --num-sequences 20 \
  --retry-counts 1 \
  --expected-contact-trajectories outputs/failure_warning/run_egl_balanced_s3f3/trajectories.jsonl
```

This produces matched variants in `comparison.md`:

- `oracle_retry1`: retry decision uses the simulator task oracle.
- `predicted_contact_retry1`: retry decision uses the learned expected contact signature; the oracle is used only for final success scoring.

The target comparison is the chain success rate / retry success gap, e.g. `oracle_retry1` SR `0.80` vs `predicted_contact_retry1` SR `0.75+`.
