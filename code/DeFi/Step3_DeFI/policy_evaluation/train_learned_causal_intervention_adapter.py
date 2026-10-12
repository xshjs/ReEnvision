#!/usr/bin/env python3
"""Train causal intervention adapter with learned factor-time masks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_causal_intervention_adapter import (
    FACTORS,
    SEGMENTS,
    CausalInterventionAdapter,
    batch_to_tensors,
    choose_node,
    iter_jsonl,
    load_future,
    nearest,
    normalize_rows,
)


def build_examples(rows: list[dict[str, Any]], futures: list[np.ndarray], causal_rows: dict[int, dict[str, Any]], k: int, same_task: bool):
    pooled = np.stack([f.mean(axis=0).astype(np.float32) for f in futures])
    pooled_norm = normalize_rows(pooled)
    tasks = [str(row.get("task") or "") for row in rows]
    success_ids = [i for i, row in enumerate(rows) if bool(row.get("success"))]
    failure_ids = [i for i, row in enumerate(rows) if not bool(row.get("success"))]
    examples = []
    for idx in failure_ids:
        causal = causal_rows.get(idx)
        if not causal:
            continue
        segment, factor, node_score = choose_node(causal)
        if factor not in FACTORS or segment not in SEGMENTS or node_score <= 0:
            continue
        succ_candidates = [j for j in success_ids if (not same_task or tasks[j] == tasks[idx])] or success_ids
        fail_candidates = [j for j in failure_ids if j != idx and (not same_task or tasks[j] == tasks[idx])]
        fail_candidates = fail_candidates or [j for j in failure_ids if j != idx]
        if not succ_candidates or not fail_candidates:
            continue
        succ_ids, succ_w = nearest(idx, pooled_norm, succ_candidates, k)
        fail_ids, fail_w = nearest(idx, pooled_norm, fail_candidates, k)
        succ_center = np.tensordot(succ_w, np.stack([futures[j] for j in succ_ids]), axes=(0, 0)).astype(np.float32)
        fail_center = np.tensordot(fail_w, np.stack([futures[j] for j in fail_ids]), axes=(0, 0)).astype(np.float32)
        target_delta = (succ_center - fail_center).astype(np.float32)
        examples.append(
            {
                "idx": idx,
                "task": tasks[idx],
                "segment": segment,
                "factor": factor,
                "node_score": float(node_score),
                "base": futures[idx],
                "success": succ_center,
                "failure": fail_center,
                "mask": np.ones_like(futures[idx], dtype=np.float32),
                "target_delta": target_delta,
            }
        )
    return examples


class LearnedCausalMasks(nn.Module):
    def __init__(self, shape: tuple[int, int], init_logit: float = -5.0):
        super().__init__()
        self.shape = shape
        self.token_logits = nn.Parameter(torch.full((len(SEGMENTS), len(FACTORS), shape[0]), init_logit))
        self.channel_logits = nn.Parameter(torch.full((len(SEGMENTS), len(FACTORS), shape[1]), init_logit))

    def forward(self, segment_idx: torch.Tensor, factor_idx: torch.Tensor) -> torch.Tensor:
        t = self.token_logits[segment_idx, factor_idx]
        c = self.channel_logits[segment_idx, factor_idx]
        return torch.sigmoid(t[:, :, None] + c[:, None, :])

    def all_masks(self) -> torch.Tensor:
        return torch.sigmoid(self.token_logits[:, :, :, None] + self.channel_logits[:, :, None, :])


def overlap_loss(masks: torch.Tensor) -> torch.Tensor:
    flat = masks.reshape(masks.shape[0] * masks.shape[1], -1)
    if flat.shape[0] < 2:
        return flat.new_tensor(0.0)
    norm = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    sim = norm @ norm.T
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye).mean()


def evaluate(model, learned_masks, examples, args, device):
    model.eval()
    learned_masks.eval()
    if args.eval_max > 0:
        examples = examples[: args.eval_max]
    cosines = []
    preserve = []
    gates = []
    margin_gain = []
    mask_mean = []
    with torch.no_grad():
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            base, success, failure, _, target_delta, factor_idx, segment_idx, node_score = batch_to_tensors(batch, device)
            pred_delta, gate = model(base, success, failure, factor_idx, segment_idx, node_score)
            mask = learned_masks(segment_idx, factor_idx)
            masked_pred = mask * pred_delta
            cos = F.cosine_similarity(masked_pred.reshape(base.shape[0], -1), (mask * target_delta).reshape(base.shape[0], -1), dim=1)
            cosines.extend(cos.detach().cpu().tolist())
            preserve.append(float(((1.0 - mask) * pred_delta).pow(2).mean().item()))
            gates.append(float(gate.mean().item()))
            mask_mean.append(float(mask.mean().item()))
            edited = base + args.rollout_gate * masked_pred
            bs = F.cosine_similarity(base.reshape(base.shape[0], -1), success.reshape(base.shape[0], -1), dim=1)
            es = F.cosine_similarity(edited.reshape(base.shape[0], -1), success.reshape(base.shape[0], -1), dim=1)
            bf = F.cosine_similarity(base.reshape(base.shape[0], -1), failure.reshape(base.shape[0], -1), dim=1)
            ef = F.cosine_similarity(edited.reshape(base.shape[0], -1), failure.reshape(base.shape[0], -1), dim=1)
            margin_gain.extend(((es - ef) - (bs - bf)).detach().cpu().tolist())
    return {
        "delta_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "preserve_mse": float(np.mean(preserve)) if preserve else 0.0,
        "gate_mean": float(np.mean(gates)) if gates else 0.0,
        "mask_mean": float(np.mean(mask_mean)) if mask_mean else 0.0,
        "margin_gain": float(np.mean(margin_gain)) if margin_gain else 0.0,
        "positive_margin_gain_rate": float(np.mean(np.asarray(margin_gain) > 0)) if margin_gain else 0.0,
        "num_eval": len(examples),
    }


def export_masks(learned_masks: LearnedCausalMasks, path: Path, density: float) -> dict[str, Any]:
    soft = learned_masks.all_masks().detach().cpu().numpy()
    out: dict[str, Any] = {}
    summary: dict[str, Any] = {"soft_mean": {}, "hard_density": {}, "hard_overlap": {}}
    total = soft.shape[-2] * soft.shape[-1]
    k = max(1, int(round(total * density)))
    hard_masks = []
    for si, segment in enumerate(SEGMENTS):
        for fi, factor in enumerate(FACTORS):
            name = f"mask_{segment}_{factor}"
            flat = soft[si, fi].reshape(-1)
            threshold = np.partition(flat, -k)[-k]
            hard = (soft[si, fi] >= threshold).astype(np.float32)
            out[name] = hard
            hard_masks.append((name, hard > 0))
            summary["soft_mean"][name] = float(soft[si, fi].mean())
            summary["hard_density"][name] = float(hard.mean())
    for i, (a_name, a) in enumerate(hard_masks):
        for b_name, b in hard_masks[i + 1 :]:
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            summary["hard_overlap"][f"{a_name}__{b_name}"] = float(inter / max(union, 1))
    np.savez_compressed(path, **out, segments=np.asarray(SEGMENTS), factors=np.asarray(FACTORS))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--causal-rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", default="original_base_future")
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cond-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mask-lr", type=float, default=1e-2)
    parser.add_argument("--mask-init-logit", type=float, default=-5.0)
    parser.add_argument("--lambda-direction", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=10.0)
    parser.add_argument("--lambda-small", type=float, default=0.01)
    parser.add_argument("--lambda-sparse", type=float, default=0.05)
    parser.add_argument("--lambda-overlap", type=float, default=0.1)
    parser.add_argument("--lambda-margin", type=float, default=1.0)
    parser.add_argument("--margin-scale", type=float, default=1.0)
    parser.add_argument("--lambda-mask-floor", type=float, default=0.1)
    parser.add_argument("--lambda-gate-floor", type=float, default=0.1)
    parser.add_argument("--target-mask-mean", type=float, default=0.002)
    parser.add_argument("--target-gate-mean", type=float, default=0.08)
    parser.add_argument("--rollout-gate", type=float, default=0.02)
    parser.add_argument("--export-density", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-max", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    futures: list[np.ndarray] = []
    for row in iter_jsonl(args.dataset_jsonl):
        future = load_future(row.get("trace_path", ""), args.future_key)
        if future is None:
            continue
        rows.append(row)
        futures.append(future)
    causal_rows = {int(row["idx"]): row for row in iter_jsonl(args.causal_rows_jsonl)}
    examples = build_examples(rows, futures, causal_rows, args.k, args.same_task_only)
    if len(examples) < 8:
        raise RuntimeError(f"too few examples: {len(examples)}")
    random.shuffle(examples)
    split = max(1, int(round(len(examples) * (1 - args.val_ratio))))
    train_examples = examples[:split]
    val_examples = examples[split:] or examples[:1]

    shape = futures[0].shape
    model = CausalInterventionAdapter(shape[-1], args.hidden_dim, args.cond_dim).to(device)
    learned_masks = LearnedCausalMasks(shape, init_logit=args.mask_init_logit).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": learned_masks.parameters(), "lr": args.mask_lr},
        ],
        weight_decay=1e-4,
    )

    best = None
    best_state = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        learned_masks.train()
        batch = random.choices(train_examples, k=args.batch_size)
        base, success, failure, _, target_delta, factor_idx, segment_idx, node_score = batch_to_tensors(batch, device)
        pred_delta, gate = model(base, success, failure, factor_idx, segment_idx, node_score)
        mask = learned_masks(segment_idx, factor_idx)
        masked_pred = mask * pred_delta
        masked_target = mask * target_delta
        delta_loss = F.smooth_l1_loss(masked_pred, masked_target)
        direction_loss = 1.0 - F.cosine_similarity(masked_pred.reshape(base.shape[0], -1), masked_target.reshape(base.shape[0], -1), dim=1).mean()
        preserve_loss = ((1.0 - mask) * pred_delta).pow(2).mean()
        small_loss = pred_delta.pow(2).mean()
        edited = base + args.rollout_gate * masked_pred
        bs = F.cosine_similarity(base.reshape(base.shape[0], -1), success.reshape(base.shape[0], -1), dim=1)
        es = F.cosine_similarity(edited.reshape(base.shape[0], -1), success.reshape(base.shape[0], -1), dim=1)
        bf = F.cosine_similarity(base.reshape(base.shape[0], -1), failure.reshape(base.shape[0], -1), dim=1)
        ef = F.cosine_similarity(edited.reshape(base.shape[0], -1), failure.reshape(base.shape[0], -1), dim=1)
        margin_gain = ((es - ef) - (bs - bf)).mean()
        margin_loss = -margin_gain * float(args.margin_scale)
        mask_floor_loss = F.relu(mask.new_tensor(float(args.target_mask_mean)) - mask.mean())
        gate_floor_loss = F.relu(gate.new_tensor(float(args.target_gate_mean)) - gate.mean())
        all_masks = learned_masks.all_masks()
        sparse_loss = all_masks.mean()
        ov_loss = overlap_loss(all_masks)
        loss = (
            delta_loss
            + args.lambda_direction * direction_loss
            + args.lambda_preserve * preserve_loss
            + args.lambda_small * small_loss
            + args.lambda_sparse * sparse_loss
            + args.lambda_overlap * ov_loss
            + args.lambda_margin * margin_loss
            + args.lambda_mask_floor * mask_floor_loss
            + args.lambda_gate_floor * gate_floor_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(learned_masks.parameters()), 5.0)
        opt.step()
        if step == 1 or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(model, learned_masks, val_examples, args, device)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "direction_loss": float(direction_loss.detach().cpu()),
                "preserve_loss": float(preserve_loss.detach().cpu()),
                "small_loss": float(small_loss.detach().cpu()),
                "sparse_loss": float(sparse_loss.detach().cpu()),
                "overlap_loss": float(ov_loss.detach().cpu()),
                "margin_loss": float(margin_loss.detach().cpu()),
                "train_margin_gain": float(margin_gain.detach().cpu()),
                "mask_floor_loss": float(mask_floor_loss.detach().cpu()),
                "gate_floor_loss": float(gate_floor_loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            score = metrics["margin_gain"] + 0.1 * metrics["positive_margin_gain_rate"] - metrics["preserve_mse"] - 0.01 * metrics["mask_mean"]
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "masks": {k: v.detach().cpu() for k, v in learned_masks.state_dict().items()},
                }
            print(json.dumps(record), flush=True)

    assert best is not None and best_state is not None
    model.load_state_dict(best_state["model"])
    learned_masks.load_state_dict(best_state["masks"])
    ckpt = args.output_dir / "learned_causal_intervention_adapter.pt"
    mask_path = args.output_dir / "learned_causal_masks.npz"
    mask_summary = export_masks(learned_masks, mask_path, args.export_density)
    torch.save(
        {
            "model_state": best_state["model"],
            "mask_state": best_state["masks"],
            "future_shape": tuple(shape),
            "factors": FACTORS,
            "segments": SEGMENTS,
            "args": vars(args),
            "best": best,
        },
        ckpt,
    )
    summary = {
        "num_rows": len(rows),
        "num_examples": len(examples),
        "num_train": len(train_examples),
        "num_val": len(val_examples),
        "factor_counts": {factor: sum(e["factor"] == factor for e in examples) for factor in FACTORS},
        "segment_counts": {seg: sum(e["segment"] == seg for e in examples) for seg in SEGMENTS},
        "best": best,
        "checkpoint": str(ckpt),
        "mask_npz": str(mask_path),
        "mask_summary": mask_summary,
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
