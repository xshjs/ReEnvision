from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


FACTORS = [
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
]


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rows(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        factor = str(row.get("factor_label", ""))
        trace_path = Path(str(row.get("target_proxy_trace_path", "") or ""))
        if factor not in FACTORS:
            continue
        if not trace_path.exists():
            continue
        rows.append(row)
        if max_examples > 0 and len(rows) >= max_examples:
            break
    return rows


def load_trace_delta(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(str(row["target_proxy_trace_path"]))
    base = torch.from_numpy(data["base_future"]).to(device=device, dtype=torch.float32)
    proposal = torch.from_numpy(data["proposal_future"]).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(data["target_proxy_future"]).to(device=device, dtype=torch.float32)
    proposal_delta = proposal - base
    target_delta = target - base
    return proposal_delta, target_delta


def load_init_masks(path: Path | None, shape: tuple[int, int]) -> dict[str, torch.Tensor]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = np.load(str(path))
    out = {}
    for factor in FACTORS:
        key = factor if factor in bundle else f"mask_{factor}"
        if key in bundle:
            mask = np.asarray(bundle[key], dtype=np.float32)
            if mask.shape == shape:
                out[factor] = torch.from_numpy(np.clip(mask, 1e-4, 1.0 - 1e-4))
    return out


def init_logits(
    num_factors: int,
    token_dim: int,
    channel_dim: int,
    init_masks: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    token_logits = torch.nn.Parameter(torch.full((num_factors, token_dim), -4.0, device=device))
    channel_logits = torch.nn.Parameter(torch.full((num_factors, channel_dim), -4.0, device=device))
    if not init_masks:
        return token_logits, channel_logits

    with torch.no_grad():
        for idx, factor in enumerate(FACTORS):
            if factor not in init_masks:
                continue
            mask = init_masks[factor].to(device)
            token_mean = mask.mean(dim=1).clamp(1e-4, 1.0 - 1e-4)
            channel_mean = mask.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
            token_logits[idx].copy_(torch.logit(token_mean) * 0.5)
            channel_logits[idx].copy_(torch.logit(channel_mean) * 0.5)
    return token_logits, channel_logits


def make_masks(token_logits: torch.Tensor, channel_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(token_logits[:, :, None] + channel_logits[:, None, :])


def overlap_loss(masks: torch.Tensor) -> torch.Tensor:
    loss = masks.new_tensor(0.0)
    pairs = 0
    for i in range(masks.shape[0]):
        for j in range(i + 1, masks.shape[0]):
            loss = loss + (masks[i] * masks[j]).mean()
            pairs += 1
    return loss / max(pairs, 1)


def cosine_direction_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    pred = pred_delta.reshape(1, -1)
    target = target_delta.reshape(1, -1)
    return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()


def normalized_mse(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    denom = target_delta.pow(2).mean().detach().clamp_min(1e-6)
    return F.smooth_l1_loss(pred_delta, target_delta) / denom.sqrt()


def summarize_masks(masks_np: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {"density": {}, "mean": {}, "overlap": {}}
    for factor, mask in masks_np.items():
        out["density"][factor] = float((mask > 0.5).mean())
        out["mean"][factor] = float(mask.mean())
    for i, left in enumerate(FACTORS):
        for right in FACTORS[i + 1 :]:
            a = masks_np[left] > 0.5
            b = masks_np[right] > 0.5
            union = np.logical_or(a, b).sum()
            inter = np.logical_and(a, b).sum()
            out["overlap"][f"{left}__{right}"] = float(inter / max(union, 1))
    return out


def top_density_masks(soft_masks: np.ndarray, density: float) -> dict[str, np.ndarray]:
    density = float(np.clip(density, 1e-6, 1.0))
    out = {}
    flat_k = max(1, int(round(soft_masks.shape[1] * soft_masks.shape[2] * density)))
    for idx, factor in enumerate(FACTORS):
        flat = soft_masks[idx].reshape(-1)
        threshold = np.partition(flat, -flat_k)[-flat_k]
        hard = (soft_masks[idx] >= threshold).astype(np.float32)
        out[factor] = hard
    return out


def disjoint_top_density_masks(soft_masks: np.ndarray, density: float) -> dict[str, np.ndarray]:
    density = float(np.clip(density, 1e-6, 1.0))
    total = soft_masks.shape[1] * soft_masks.shape[2]
    per_factor_k = max(1, int(round(total * density)))
    owners = np.argmax(soft_masks, axis=0)
    out = {}
    for idx, factor in enumerate(FACTORS):
        score = soft_masks[idx].copy()
        score[owners != idx] = -np.inf
        flat = score.reshape(-1)
        finite = np.isfinite(flat)
        hard = np.zeros_like(flat, dtype=np.float32)
        if finite.any():
            k = min(per_factor_k, int(finite.sum()))
            chosen = np.argpartition(flat, -k)[-k:]
            hard[chosen] = 1.0
        out[factor] = hard.reshape(soft_masks.shape[1], soft_masks.shape[2])
    return out


def sample_batch(rows: list[dict[str, Any]], by_factor: dict[str, list[dict[str, Any]]], batch_size: int, balanced: bool) -> list[dict[str, Any]]:
    if not balanced:
        return random.choices(rows, k=max(1, batch_size))
    batch = []
    factors = [factor for factor in FACTORS if by_factor.get(factor)]
    for _ in range(max(1, batch_size)):
        factor = random.choice(factors)
        batch.append(random.choice(by_factor[factor]))
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weakly-supervised factor masks from CALVIN future traces.")
    parser.add_argument("--factor-supervision-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-mask-npz", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--lambda-sparse", type=float, default=0.02)
    parser.add_argument("--lambda-overlap", type=float, default=0.2)
    parser.add_argument("--lambda-mse", type=float, default=0.05)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--export-density", type=float, default=0.01)
    parser.add_argument("--balanced-factor-sampling", action="store_true")
    parser.add_argument("--disjoint-export", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    rows = load_rows(args.factor_supervision_jsonl, max_examples=args.max_examples)
    if not rows:
        raise ValueError("no usable factor rows with target_proxy_trace_path")
    factor_to_idx = {factor: idx for idx, factor in enumerate(FACTORS)}
    factor_counts = Counter(str(row["factor_label"]) for row in rows)

    first_delta, _ = load_trace_delta(rows[0], device=torch.device("cpu"))
    if first_delta.ndim != 2:
        raise ValueError(f"expected 2D future trace, got shape={tuple(first_delta.shape)}")
    token_dim, channel_dim = first_delta.shape
    init_masks = load_init_masks(args.init_mask_npz, (token_dim, channel_dim))
    token_logits, channel_logits = init_logits(len(FACTORS), token_dim, channel_dim, init_masks, device)
    init_mask_stack = None
    if init_masks:
        init_mask_stack = torch.stack(
            [init_masks[factor].to(device) if factor in init_masks else torch.zeros((token_dim, channel_dim), device=device) for factor in FACTORS],
            dim=0,
        )
    optimizer = torch.optim.AdamW([token_logits, channel_logits], lr=args.lr, weight_decay=1e-4)
    by_factor = defaultdict(list)
    for row in rows:
        by_factor[str(row["factor_label"])].append(row)

    history = []
    for step in range(args.steps):
        batch = sample_batch(rows, by_factor, args.batch_size, bool(args.balanced_factor_sampling))
        masks = make_masks(token_logits, channel_logits)
        loss = masks.new_tensor(0.0)
        dir_losses = []
        mse_losses = []
        for row in batch:
            factor_idx = factor_to_idx[str(row["factor_label"])]
            proposal_delta, target_delta = load_trace_delta(row, device)
            pred_delta = masks[factor_idx] * proposal_delta
            dir_loss = cosine_direction_loss(pred_delta, target_delta)
            mse_loss = normalized_mse(pred_delta, target_delta)
            loss = loss + dir_loss + args.lambda_mse * mse_loss
            dir_losses.append(float(dir_loss.detach().cpu()))
            mse_losses.append(float(mse_loss.detach().cpu()))
        loss = loss / len(batch)
        sparse = masks.mean()
        overlap = overlap_loss(masks)
        init_loss = masks.new_tensor(0.0)
        if init_mask_stack is not None and args.lambda_init > 0:
            init_loss = F.binary_cross_entropy(masks.clamp(1e-6, 1 - 1e-6), init_mask_stack)
        total = loss + args.lambda_sparse * sparse + args.lambda_overlap * overlap + args.lambda_init * init_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([token_logits, channel_logits], 5.0)
        optimizer.step()

        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            record = {
                "step": step + 1,
                "loss": float(total.detach().cpu()),
                "direction_loss": float(np.mean(dir_losses)),
                "mse_loss": float(np.mean(mse_losses)),
                "sparse": float(sparse.detach().cpu()),
                "overlap": float(overlap.detach().cpu()),
                "init_loss": float(init_loss.detach().cpu()),
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    with torch.no_grad():
        soft_masks = make_masks(token_logits, channel_logits).detach().cpu().numpy().astype(np.float32)
    hard_masks_np = (
        disjoint_top_density_masks(soft_masks, args.export_density)
        if args.disjoint_export
        else top_density_masks(soft_masks, args.export_density)
    )
    soft_masks_np = {f"{factor}_soft": soft_masks[idx] for idx, factor in enumerate(FACTORS)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "learned_factor_masks.npz",
        **hard_masks_np,
        **soft_masks_np,
        factors=np.asarray(FACTORS),
        token_logits=token_logits.detach().cpu().numpy().astype(np.float32),
        channel_logits=channel_logits.detach().cpu().numpy().astype(np.float32),
    )

    summary = {
        "num_examples": len(rows),
        "factor_counts": dict(factor_counts.most_common()),
        "future_shape": [token_dim, channel_dim],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_sparse": args.lambda_sparse,
        "lambda_overlap": args.lambda_overlap,
        "lambda_mse": args.lambda_mse,
        "lambda_init": args.lambda_init,
        "export_density": args.export_density,
        "balanced_factor_sampling": bool(args.balanced_factor_sampling),
        "disjoint_export": bool(args.disjoint_export),
        "loaded_init_masks": sorted(init_masks.keys()),
        "init_mask_npz": str(args.init_mask_npz) if args.init_mask_npz else "",
        "history": history,
        "mask_summary": summarize_masks(hard_masks_np),
        "soft_mask_summary": summarize_masks({factor: soft_masks[idx] for idx, factor in enumerate(FACTORS)}),
        "output_npz": str(args.output_dir / "learned_factor_masks.npz"),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
