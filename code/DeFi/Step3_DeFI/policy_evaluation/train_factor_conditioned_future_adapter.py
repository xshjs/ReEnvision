from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
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


def load_rows(path: Path, require_improve: bool, max_examples: int) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        factor = str(row.get("failure_factor") or row.get("applied_factor_mask") or "")
        trace_path = Path(str(row.get("trace_path", "") or ""))
        if factor not in FACTORS:
            continue
        if require_improve and not bool(row.get("label_improve", 0)):
            continue
        if not trace_path.exists():
            continue
        rows.append(row)
        if max_examples > 0 and len(rows) >= max_examples:
            break
    return rows


def load_factor_masks(path: Path, shape: tuple[int, int]) -> dict[str, torch.Tensor]:
    bundle = np.load(str(path))
    masks = {}
    for factor in FACTORS:
        for key in (factor, f"mask_{factor}"):
            if key not in bundle:
                continue
            mask = np.asarray(bundle[key], dtype=np.float32)
            if mask.shape != shape:
                raise ValueError(f"mask shape mismatch for {factor}: {mask.shape} vs {shape}")
            masks[factor] = torch.from_numpy(mask)
            break
    missing = [factor for factor in FACTORS if factor not in masks]
    if missing:
        raise ValueError(f"missing factor masks: {missing}")
    return masks


def load_trace(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trace = np.load(str(row["trace_path"]))
    base = torch.from_numpy(trace["base_future"]).to(device=device, dtype=torch.float32)
    proposal = torch.from_numpy(trace["proposal_future"]).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(trace["target_proxy_future"]).to(device=device, dtype=torch.float32)
    return base, proposal, target


class FactorConditionedFutureAdapter(nn.Module):
    def __init__(self, future_dim: int, num_factors: int, factor_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.future_dim = int(future_dim)
        self.factor_dim = int(factor_dim)
        self.hidden_dim = int(hidden_dim)
        self.factor_embedding = nn.Embedding(num_factors, factor_dim)
        in_dim = future_dim * 2 + factor_dim
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
            nn.Linear(hidden_dim, future_dim),
            nn.Sigmoid(),
        )

    def forward(self, base_future: torch.Tensor, proposal_future: torch.Tensor, factor_idx: torch.Tensor):
        if base_future.ndim != 3:
            raise ValueError("base_future must be [B, T, D]")
        proposal_delta = proposal_future - base_future
        factor = self.factor_embedding(factor_idx).unsqueeze(1).expand(-1, base_future.shape[1], -1)
        x = torch.cat([base_future, proposal_delta, factor], dim=-1)
        delta = self.net(x)
        gate = self.gate(x)
        return base_future + gate * delta, {"delta": delta, "gate": gate}


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1), dim=1).mean()


def sample_batch(rows: list[dict[str, Any]], by_factor: dict[str, list[dict[str, Any]]], batch_size: int, balanced: bool):
    if not balanced:
        return random.choices(rows, k=batch_size)
    factors = [factor for factor in FACTORS if by_factor.get(factor)]
    return [random.choice(by_factor[random.choice(factors)]) for _ in range(batch_size)]


def evaluate(model, rows, masks, factor_to_idx, device, max_eval: int):
    model.eval()
    rows = rows[:max_eval] if max_eval > 0 else rows
    losses, cosines, preserve_vals, gate_vals = [], [], [], []
    with torch.no_grad():
        for row in rows:
            factor = str(row.get("failure_factor") or row.get("applied_factor_mask"))
            base, proposal, target = load_trace(row, device)
            base = base.unsqueeze(0)
            proposal = proposal.unsqueeze(0)
            target = target.unsqueeze(0)
            factor_idx = torch.tensor([factor_to_idx[factor]], device=device)
            mask = masks[factor].to(device).unsqueeze(0)
            corrected, aux = model(base, proposal, factor_idx)
            pred_delta = corrected - base
            target_delta = target - base
            target_loss = F.smooth_l1_loss(mask * pred_delta, mask * target_delta)
            preserve = ((1.0 - mask) * pred_delta).pow(2).mean()
            direction = cosine_loss(mask * pred_delta, mask * target_delta)
            losses.append(float((target_loss + direction).item()))
            cosines.append(float((1.0 - direction).item()))
            preserve_vals.append(float(preserve.item()))
            gate_vals.append(float(aux["gate"].mean().item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "masked_delta_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "preserve_mse": float(np.mean(preserve_vals)) if preserve_vals else 0.0,
        "gate_mean": float(np.mean(gate_vals)) if gate_vals else 0.0,
        "num_eval": int(len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a factor-conditioned token future adapter from CALVIN future traces.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--factor-mask-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--factor-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-direction", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=10.0)
    parser.add_argument("--lambda-norm", type=float, default=0.01)
    parser.add_argument("--require-improve", action="store_true")
    parser.add_argument("--balanced-factor-sampling", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-max", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset_jsonl, bool(args.require_improve), int(args.max_examples))
    if not rows:
        raise ValueError("no usable rows")
    random.shuffle(rows)
    split = max(1, int(round(len(rows) * (1.0 - args.val_ratio))))
    train_rows, val_rows = rows[:split], rows[split:] or rows[:1]

    base0, _, _ = load_trace(rows[0], torch.device("cpu"))
    if base0.ndim != 2:
        raise ValueError(f"expected [T,D] future, got {tuple(base0.shape)}")
    masks = load_factor_masks(args.factor_mask_npz, tuple(base0.shape))
    factor_to_idx = {factor: idx for idx, factor in enumerate(FACTORS)}
    by_factor = defaultdict(list)
    for row in train_rows:
        by_factor[str(row.get("failure_factor") or row.get("applied_factor_mask"))].append(row)

    model = FactorConditionedFutureAdapter(
        future_dim=int(base0.shape[-1]),
        num_factors=len(FACTORS),
        factor_dim=args.factor_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = None
    best_state = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_rows = sample_batch(train_rows, by_factor, args.batch_size, bool(args.balanced_factor_sampling))
        bases, proposals, targets, factor_indices, batch_masks = [], [], [], [], []
        for row in batch_rows:
            factor = str(row.get("failure_factor") or row.get("applied_factor_mask"))
            base, proposal, target = load_trace(row, device)
            bases.append(base)
            proposals.append(proposal)
            targets.append(target)
            factor_indices.append(factor_to_idx[factor])
            batch_masks.append(masks[factor].to(device))
        base = torch.stack(bases, dim=0)
        proposal = torch.stack(proposals, dim=0)
        target = torch.stack(targets, dim=0)
        mask = torch.stack(batch_masks, dim=0)
        factor_idx = torch.tensor(factor_indices, device=device)

        corrected, aux = model(base, proposal, factor_idx)
        pred_delta = corrected - base
        target_delta = target - base
        delta_loss = F.smooth_l1_loss(mask * pred_delta, mask * target_delta)
        direction_loss = cosine_loss(mask * pred_delta, mask * target_delta)
        preserve_loss = ((1.0 - mask) * pred_delta).pow(2).mean()
        norm_loss = pred_delta.pow(2).mean()
        loss = delta_loss + args.lambda_direction * direction_loss + args.lambda_preserve * preserve_loss + args.lambda_norm * norm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(model, val_rows, masks, factor_to_idx, device, args.eval_max)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "direction_loss": float(direction_loss.detach().cpu()),
                "preserve_loss": float(preserve_loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            score = metrics["masked_delta_cosine"] - metrics["preserve_mse"]
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(record), flush=True)

    assert best is not None and best_state is not None
    ckpt_path = args.output_dir / "factor_conditioned_future_adapter.pt"
    torch.save(
        {
            "model_state": best_state,
            "future_dim": int(base0.shape[-1]),
            "future_token_shape": tuple(int(x) for x in base0.shape),
            "factor_dim": int(args.factor_dim),
            "hidden_dim": int(args.hidden_dim),
            "factors": FACTORS,
            "factor_mask_npz": str(args.factor_mask_npz),
            "token_level": True,
        },
        ckpt_path,
    )
    summary = {
        "num_rows": int(len(rows)),
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "factor_counts": dict(Counter(str(row.get("failure_factor") or row.get("applied_factor_mask")) for row in rows)),
        "best": best,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
