# Future Feature Ablation

This experiment tests whether DeFi's predicted future feature matters for long-horizon success.

## Modes

| Mode | Meaning |
|---|---|
| `gfdm` | Original DeFi: uses GFDM frame-5 feature. |
| `current` | No Future: replaces future feature with current frame feature. |
| `zero` | No Future: replaces future feature with zeros. |

The switch is implemented in `VPP_Policy.future_feature_mode`.

## Commands

```bash
cd /mnt/workspace/manipulation/DeFi

for mode in gfdm current zero; do
  .venv/bin/python Step3_DeFI/policy_evaluation/contact_retry_evaluate.py \
    --video-model-path /DeFI/ckpts/svd/checkpoint-100000 \
    --checkpoint /DeFI/outputs/defi_eval/saved_models/step3_defi.pt \
    --clip-model-path /DeFI/ckpts/openai_clip_vit_base_patch32 \
    --t5-model-path /DeFI/ckpts/t5_base \
    --language-goal-path /DeFI/ckpts/ViT-B-32.pt \
    --calvin-abc-dir /DeFI/ckpts/task_ABC_D \
    --output-dir outputs/future_ablation_n5/${mode} \
    --num-sequences 5 \
    --ep-len 80 \
    --retry-counts 0 \
    --future-feature-mode ${mode} \
    --device cuda:0
done
```

## n=5 Result

See `outputs/future_ablation_n5/summary.md`.

| Variant | Avg Seq Len | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 |
|---|---:|---:|---:|---:|---:|---:|
| Original GFDM future | 1.000 | 0.800 | 0.200 | 0.000 | 0.000 | 0.000 |
| No future = current | 1.000 | 0.600 | 0.200 | 0.200 | 0.000 | 0.000 |
| No future = zero | 1.000 | 0.800 | 0.200 | 0.000 | 0.000 | 0.000 |

This small run does not show a clear benefit from the GFDM future feature.

## Oracle Future

`Oracle Future` is not a causal online rollout setting unless privileged future observations are supplied. Valid ways to run it are:

1. **Offline imitation/action-prediction ablation**: use demonstration frame `t+k` as the future feature and evaluate action prediction loss.
2. **Privileged simulator ablation**: clone the simulator state, roll forward with ground-truth/demo actions or a known successful plan to get `obs_{t+k}`, restore the original state, then act using that future feature.

Without one of these, using GT future in closed-loop evaluation leaks information that the policy would not have at inference time.
