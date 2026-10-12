#!/usr/bin/env python3
"""Train a shared factor-time causal intervention adapter."""

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


FACTORS = ["contact", "object", "motion", "goal"]
SEGMENTS = ["early", "mid", "late"]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        with np.load(str(path)) as data:
            if key not in data:
                return None
            x = data[key].astype(np.float32)
    except Exception:
        return None
    return x if x.ndim == 2 else None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_masks(path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    zeros = np.zeros(shape, dtype=np.float32)
    contact = data["mask_contact"].astype(np.float32) if "mask_contact" in data else zeros
    goal = data["mask_goal_completion"].astype(np.float32) if "mask_goal_completion" in data else zeros
    obj_parts = [data[k].astype(np.float32) for k in ("mask_object_displacement", "mask_object_identity") if k in data]
    mot_parts = [data[k].astype(np.float32) for k in ("mask_drawer_slider_progress", "mask_object_displacement") if k in data]
    masks = {
        "contact": contact,
        "goal": goal,
        "object": np.maximum.reduce(obj_parts) if obj_parts else zeros,
        "motion": np.maximum.reduce(mot_parts) if mot_parts else zeros,
    }
    for name, mask in masks.items():
        if mask.shape != shape:
            raise ValueError(f"{name} mask shape {mask.shape} != {shape}")
        masks[name] = (mask > 0).astype(np.float32)
    return masks


def time_mask(segment: str, shape: tuple[int, int]) -> np.ndarray:
    t, _ = shape
    one = max(1, t // 3)
    mask = np.zeros(shape, dtype=np.float32)
    if segment == "early":
        mask[:one] = 1.0
    elif segment == "mid":
        mask[one : max(one + 1, 2 * one)] = 1.0
    elif segment == "late":
        mask[max(0, 2 * one) :] = 1.0
    else:
        raise ValueError(segment)
    return mask


def choose_node(causal: dict[str, Any]) -> tuple[str, str, float]:
    best = ("none", "none", 0.0)
    for segment in SEGMENTS:
        for factor in FACTORS:
            score = float(causal.get(f"{segment}_{factor}_risk", 0.0))
            if score > best[2]:
                best = (segment, factor, score)
    return best


def nearest(idx: int, pooled_norm: np.ndarray, candidates: list[int], k: int) -> tuple[list[int], np.ndarray]:
    sims = (pooled_norm[idx][None, :] @ pooled_norm[candidates].T).reshape(-1)
    order = np.argsort(-sims)[: max(1, min(k, len(candidates)))]
    chosen = [candidates[int(i)] for i in order]
    weights = np.maximum(sims[order], 0.0).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones(len(chosen), dtype=np.float32)
    weights /= max(float(weights.sum()), 1e-8)
    return chosen, weights


def build_examples(
    rows: list[dict[str, Any]],
    futures: list[np.ndarray],
    causal_rows: dict[int, dict[str, Any]],
    masks: dict[str, np.ndarray],
    k: int,
    same_task: bool,
) -> list[dict[str, Any]]:
    pooled = np.stack([f.mean(axis=0).astype(np.float32) for f in futures])
    pooled_norm = normalize_rows(pooled)
    tasks = [str(row.get("task") or "") for row in rows]
    success_ids = [i for i, row in enumerate(rows) if bool(row.get("success"))]
    failure_ids = [i for i, row in enumerate(rows) if not bool(row.get("success"))]
    examples = []
    shape = futures[0].shape
    time_masks = {seg: time_mask(seg, shape) for seg in SEGMENTS}
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
        mask = masks[factor] * time_masks[segment]
        if float(mask.sum()) <= 0:
            continue
        target_delta = (succ_center - fail_center) * mask
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
                "mask": mask.astype(np.float32),
                "target_delta": target_delta.astype(np.float32),
            }
        )
    return examples


class CausalInterventionAdapter(nn.Module):
    def __init__(self, future_dim: int, hidden_dim: int = 512, cond_dim: int = 32) -> None:
        super().__init__()
        self.factor_emb = nn.Embedding(len(FACTORS), cond_dim)
        self.segment_emb = nn.Embedding(len(SEGMENTS), cond_dim)
        in_dim = future_dim * 4 + cond_dim * 2 + 1
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        base: torch.Tensor,
        success: torch.Tensor,
        failure: torch.Tensor,
        factor_idx: torch.Tensor,
        segment_idx: torch.Tensor,
        node_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction = success - failure
        repel = base - failure
        attract = success - base
        factor = self.factor_emb(factor_idx).unsqueeze(1).expand(-1, base.shape[1], -1)
        segment = self.segment_emb(segment_idx).unsqueeze(1).expand(-1, base.shape[1], -1)
        score = node_score[:, None, None].expand(-1, base.shape[1], 1)
        x = torch.cat([base, direction, repel, attract, factor, segment, score], dim=-1)
        delta = self.net(x)
        gate = self.gate(x)
        return gate * delta, gate


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1), dim=1).mean()


def batch_to_tensors(batch: list[dict[str, Any]], device: torch.device):
    base = torch.from_numpy(np.stack([b["base"] for b in batch])).to(device)
    success = torch.from_numpy(np.stack([b["success"] for b in batch])).to(device)
    failure = torch.from_numpy(np.stack([b["failure"] for b in batch])).to(device)
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch])).to(device)
    target_delta = torch.from_numpy(np.stack([b["target_delta"] for b in batch])).to(device)
    factor_idx = torch.tensor([FACTORS.index(b["factor"]) for b in batch], dtype=torch.long, device=device)
    segment_idx = torch.tensor([SEGMENTS.index(b["segment"]) for b in batch], dtype=torch.long, device=device)
    node_score = torch.tensor([b["node_score"] for b in batch], dtype=torch.float32, device=device)
    return base, success, failure, mask, target_delta, factor_idx, segment_idx, node_score


def evaluate(model, examples, args, device):
    model.eval()
    if args.eval_max > 0:
        examples = examples[: args.eval_max]
    losses = []
    cosines = []
    preserve = []
    gates = []
    success_gain = []
    failure_gain = []
    margin_gain = []
    with torch.no_grad():
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            base, success, failure, mask, target_delta, factor_idx, segment_idx, node_score = batch_to_tensors(batch, device)
            pred_delta, gate = model(base, success, failure, factor_idx, segment_idx, node_score)
            masked_pred = mask * pred_delta
            loss = F.smooth_l1_loss(masked_pred, target_delta)
            direction = cosine_loss(masked_pred, target_delta)
            losses.append(float(loss.item()))
            cosines.append(float((1.0 - direction).item()))
            preserve.append(float(((1.0 - mask) * pred_delta).pow(2).mean().item()))
            gates.append(float(gate.mean().item()))
            edited = base + args.rollout_gate * masked_pred
            bs = F.cosine_similarity(base.reshape(base.shape[0], -1), success.reshape(base.shape[0], -1), dim=1)
            es = F.cosine_similarity(edited.reshape(base.shape[0], -1), success.reshape(base.shape[0], -1), dim=1)
            bf = F.cosine_similarity(base.reshape(base.shape[0], -1), failure.reshape(base.shape[0], -1), dim=1)
            ef = F.cosine_similarity(edited.reshape(base.shape[0], -1), failure.reshape(base.shape[0], -1), dim=1)
            success_gain.extend((es - bs).detach().cpu().tolist())
            failure_gain.extend((bf - ef).detach().cpu().tolist())
            margin_gain.extend(((es - ef) - (bs - bf)).detach().cpu().tolist())
    return {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "delta_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "preserve_mse": float(np.mean(preserve)) if preserve else 0.0,
        "gate_mean": float(np.mean(gates)) if gates else 0.0,
        "success_cos_gain": float(np.mean(success_gain)) if success_gain else 0.0,
        "failure_repulsion_gain": float(np.mean(failure_gain)) if failure_gain else 0.0,
        "margin_gain": float(np.mean(margin_gain)) if margin_gain else 0.0,
        "positive_margin_gain_rate": float(np.mean(np.asarray(margin_gain) > 0)) if margin_gain else 0.0,
        "num_eval": len(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--causal-rows-jsonl", type=Path, required=True)
    parser.add_argument("--factor-mask-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", default="original_base_future")
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cond-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-direction", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=10.0)
    parser.add_argument("--lambda-small", type=float, default=0.01)
    parser.add_argument("--rollout-gate", type=float, default=0.02)
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
    if not futures:
        raise RuntimeError("no futures")
    masks = load_masks(args.factor_mask_npz, futures[0].shape)
    examples = build_examples(rows, futures, causal_rows, masks, args.k, args.same_task_only)
    if len(examples) < 8:
        raise RuntimeError(f"too few examples: {len(examples)}")
    random.shuffle(examples)
    split = max(1, int(round(len(examples) * (1.0 - args.val_ratio))))
    train_examples = examples[:split]
    val_examples = examples[split:] or examples[:1]

    model = CausalInterventionAdapter(futures[0].shape[-1], args.hidden_dim, args.cond_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.choices(train_examples, k=args.batch_size)
        base, success, failure, mask, target_delta, factor_idx, segment_idx, node_score = batch_to_tensors(batch, device)
        pred_delta, gate = model(base, success, failure, factor_idx, segment_idx, node_score)
        masked_pred = mask * pred_delta
        delta_loss = F.smooth_l1_loss(masked_pred, target_delta)
        direction_loss = cosine_loss(masked_pred, target_delta)
        preserve_loss = ((1.0 - mask) * pred_delta).pow(2).mean()
        small_loss = pred_delta.pow(2).mean()
        loss = delta_loss + args.lambda_direction * direction_loss + args.lambda_preserve * preserve_loss + args.lambda_small * small_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(model, val_examples, args, device)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "direction_loss": float(direction_loss.detach().cpu()),
                "preserve_loss": float(preserve_loss.detach().cpu()),
                "small_loss": float(small_loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            score = metrics["margin_gain"] + 0.1 * metrics["positive_margin_gain_rate"] - metrics["preserve_mse"]
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(record), flush=True)

    assert best is not None and best_state is not None
    ckpt = args.output_dir / "causal_intervention_adapter.pt"
    torch.save(
        {
            "model_state": best_state,
            "future_shape": tuple(futures[0].shape),
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
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
