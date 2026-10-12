from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FACTORS = [
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
]

DEFAULT_TIME_WINDOWS = {
    "contact": (0.20, 0.55),
    "object_displacement": (0.30, 0.75),
    "object_identity": (0.10, 0.45),
    "drawer_slider_progress": (0.35, 0.85),
    "goal_completion": (0.60, 1.00),
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def factor_from_row(row: dict[str, Any]) -> str:
    for key in ("factor_label", "wrong_component", "failure_factor", "selected_failure_factor", "applied_factor_mask"):
        factor = str(row.get(key, "") or "")
        if factor in FACTORS:
            return factor
    return ""


def trace_path_from_row(row: dict[str, Any]) -> Path:
    for key in ("target_proxy_trace_path", "trace_path", "collection_trace_path"):
        value = str(row.get(key, "") or "")
        if value:
            return Path(value)
    return Path("")


def load_rows(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        factor = factor_from_row(row)
        trace_path = trace_path_from_row(row)
        if factor not in FACTORS or not trace_path.exists():
            continue
        row = dict(row)
        row["_factor"] = factor
        row["_trace_path"] = str(trace_path)
        rows.append(row)
        if max_examples > 0 and len(rows) >= max_examples:
            break
    return rows


def load_trace(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(str(row["_trace_path"]))
    base_key = "base_future" if "base_future" in data.files else "original_base_future"
    proposal_key = "proposal_future" if "proposal_future" in data.files else "final_future"
    target_key = "target_proxy_future" if "target_proxy_future" in data.files else "final_future"
    base = torch.from_numpy(np.asarray(data[base_key], dtype=np.float32)).to(device)
    proposal = torch.from_numpy(np.asarray(data[proposal_key], dtype=np.float32)).to(device)
    target = torch.from_numpy(np.asarray(data[target_key], dtype=np.float32)).to(device)
    return base, proposal, target


def row_time_window(row: dict[str, Any], factor: str) -> tuple[float, float]:
    value = row.get("time_window")
    if isinstance(value, list) and len(value) == 2:
        return float(value[0]), float(value[1])
    start = row.get("qwen_diagnosis_time_start")
    end = row.get("qwen_diagnosis_time_end")
    if start is not None and end is not None:
        return float(start), float(end)
    return DEFAULT_TIME_WINDOWS.get(factor, (0.0, 1.0))


class ProgramConditionedMaskHead(nn.Module):
    def __init__(self, token_dim: int, channel_dim: int, num_factors: int, hidden_dim: int = 256, factor_dim: int = 32):
        super().__init__()
        self.token_dim = int(token_dim)
        self.channel_dim = int(channel_dim)
        self.factor_embed = nn.Embedding(num_factors, factor_dim)
        self.future_proj = nn.Sequential(
            nn.Linear(channel_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.program_proj = nn.Sequential(
            nn.Linear(factor_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.token_head = nn.Linear(hidden_dim, token_dim)
        self.channel_head = nn.Linear(hidden_dim, channel_dim)

    def summarize_future(self, future: torch.Tensor) -> torch.Tensor:
        mean = future.mean(dim=1)
        std = future.std(dim=1, unbiased=False)
        maxv = future.max(dim=1).values
        minv = future.min(dim=1).values
        return torch.cat([mean, std, maxv, minv], dim=-1)

    def forward(self, future: torch.Tensor, factor_idx: torch.Tensor, time_window: torch.Tensor) -> torch.Tensor:
        future_ctx = self.future_proj(self.summarize_future(future))
        factor_ctx = self.factor_embed(factor_idx)
        program_ctx = self.program_proj(torch.cat([factor_ctx, time_window], dim=-1))
        hidden = future_ctx + program_ctx
        token_logits = self.token_head(hidden)
        channel_logits = self.channel_head(hidden)
        return torch.sigmoid(token_logits[:, :, None] + channel_logits[:, None, :])


def cosine_direction_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred_delta.flatten(1), target_delta.flatten(1), dim=1).mean()


def normalized_mse(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    denom = target_delta.pow(2).mean().detach().clamp_min(1e-6).sqrt()
    return F.smooth_l1_loss(pred_delta, target_delta) / denom


def time_prior_mask(token_dim: int, channel_dim: int, windows: torch.Tensor, device: torch.device) -> torch.Tensor:
    pos = torch.linspace(0.0, 1.0, token_dim, device=device)[None, :, None]
    start = windows[:, 0:1, None]
    end = windows[:, 1:2, None]
    mask = ((pos >= start) & (pos <= end)).float()
    return mask.expand(-1, -1, channel_dim)


def load_teacher_masks(path: Path, token_dim: int, channel_dim: int) -> dict[str, np.ndarray]:
    if path is None or not str(path):
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        for factor in FACTORS:
            key = f"mask_{factor}"
            if key not in data:
                continue
            mask = np.asarray(data[key], dtype=np.float32)
            if mask.shape != (token_dim, channel_dim):
                raise ValueError(f"{key} has shape {mask.shape}, expected {(token_dim, channel_dim)}")
            out[factor] = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return out


def sample_batch(rows: list[dict[str, Any]], by_factor: dict[str, list[dict[str, Any]]], batch_size: int, balanced: bool):
    if not balanced:
        return random.choices(rows, k=batch_size)
    factors = [factor for factor in FACTORS if by_factor.get(factor)]
    return [random.choice(by_factor[random.choice(factors)]) for _ in range(batch_size)]


def top_density(mask: np.ndarray, density: float) -> np.ndarray:
    flat = mask.reshape(-1)
    k = max(1, int(round(flat.size * float(np.clip(density, 1e-6, 1.0)))))
    threshold = np.partition(flat, -k)[-k]
    return (mask >= threshold).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train q/program-conditioned dense future mask head.")
    parser.add_argument("--factor-supervision-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lambda-mse", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.02)
    parser.add_argument("--lambda-time-prior", type=float, default=0.05)
    parser.add_argument("--teacher-mask-npz", type=Path, default=None)
    parser.add_argument("--lambda-teacher-mask", type=float, default=1.0)
    parser.add_argument("--min-mask-mean", type=float, default=0.0)
    parser.add_argument("--lambda-mask-floor", type=float, default=0.0)
    parser.add_argument("--export-density", type=float, default=0.01)
    parser.add_argument("--balanced-factor-sampling", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    rows = load_rows(args.factor_supervision_jsonl, args.max_examples)
    if not rows:
        raise ValueError("no usable rows with factor labels and future traces")
    base0, _, _ = load_trace(rows[0], torch.device("cpu"))
    if base0.ndim != 2:
        raise ValueError(f"expected 2D future, got {tuple(base0.shape)}")
    token_dim, channel_dim = tuple(int(x) for x in base0.shape)
    factor_to_idx = {factor: idx for idx, factor in enumerate(FACTORS)}
    by_factor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_factor[row["_factor"]].append(row)
    teacher_masks = load_teacher_masks(args.teacher_mask_npz, token_dim, channel_dim) if args.teacher_mask_npz else {}

    model = ProgramConditionedMaskHead(token_dim, channel_dim, len(FACTORS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for step in range(args.steps):
        batch_rows = sample_batch(rows, by_factor, args.batch_size, args.balanced_factor_sampling)
        bases, proposal_deltas, target_deltas, factor_ids, windows = [], [], [], [], []
        for row in batch_rows:
            base, proposal, target = load_trace(row, device)
            bases.append(base)
            proposal_deltas.append(proposal - base)
            target_deltas.append(target - base)
            factor_ids.append(factor_to_idx[row["_factor"]])
            windows.append(row_time_window(row, row["_factor"]))
        base_t = torch.stack(bases, dim=0)
        proposal_delta_t = torch.stack(proposal_deltas, dim=0)
        target_delta_t = torch.stack(target_deltas, dim=0)
        factor_t = torch.tensor(factor_ids, device=device, dtype=torch.long)
        window_t = torch.tensor(windows, device=device, dtype=torch.float32).clamp(0.0, 1.0)

        mask = model(base_t, factor_t, window_t)
        pred_delta = mask * proposal_delta_t
        target_norm = target_delta_t.flatten(1).norm(dim=1)
        has_delta = target_norm > 1e-6
        if bool(has_delta.any()):
            dir_loss = cosine_direction_loss(pred_delta[has_delta], target_delta_t[has_delta])
            mse_loss = normalized_mse(pred_delta[has_delta], target_delta_t[has_delta])
        else:
            dir_loss = mask.new_tensor(0.0)
            mse_loss = mask.new_tensor(0.0)
        sparse_loss = mask.mean()
        time_prior = time_prior_mask(token_dim, channel_dim, window_t, device)
        time_loss = (mask * (1.0 - time_prior)).mean()
        teacher_loss = mask.new_tensor(0.0)
        if teacher_masks:
            teacher_np = []
            for row, factor in zip(batch_rows, [r["_factor"] for r in batch_rows]):
                base_teacher = teacher_masks.get(factor)
                if base_teacher is None:
                    base_teacher = np.ones((token_dim, channel_dim), dtype=np.float32) * float(args.min_mask_mean)
                teacher_np.append(base_teacher)
            teacher = torch.from_numpy(np.stack(teacher_np, axis=0)).to(device=device, dtype=mask.dtype)
            teacher = torch.maximum(teacher, time_prior * float(max(args.min_mask_mean, 0.0)))
            teacher_loss = F.binary_cross_entropy(mask.clamp(1e-6, 1.0 - 1e-6), teacher.clamp(0.0, 1.0))
        floor_loss = F.relu(mask.new_tensor(float(args.min_mask_mean)) - mask.mean())
        loss = (
            dir_loss
            + args.lambda_mse * mse_loss
            + args.lambda_teacher_mask * teacher_loss
            + args.lambda_sparse * sparse_loss
            + args.lambda_time_prior * time_loss
            + args.lambda_mask_floor * floor_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            record = {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "direction_loss": float(dir_loss.detach().cpu()),
                "mse_loss": float(mse_loss.detach().cpu()),
                "teacher_mask_loss": float(teacher_loss.detach().cpu()),
                "sparse": float(sparse_loss.detach().cpu()),
                "time_prior_loss": float(time_loss.detach().cpu()),
                "mask_floor_loss": float(floor_loss.detach().cpu()),
                "mask_mean": float(mask.detach().mean().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "program_conditioned_mask_head.pt"
    torch.save(
        {
            "model_type": "program_conditioned_mask_head",
            "model_state": model.state_dict(),
            "factors": FACTORS,
            "factor_to_idx": factor_to_idx,
            "token_dim": token_dim,
            "channel_dim": channel_dim,
            "hidden_dim": int(args.hidden_dim),
        },
        ckpt_path,
    )

    static_masks = {}
    with torch.no_grad():
        for factor in FACTORS:
            if not by_factor.get(factor):
                continue
            chosen = by_factor[factor][: min(32, len(by_factor[factor]))]
            masks = []
            for row in chosen:
                base, _, _ = load_trace(row, device)
                window = torch.tensor([row_time_window(row, factor)], device=device, dtype=torch.float32)
                factor_id = torch.tensor([factor_to_idx[factor]], device=device, dtype=torch.long)
                masks.append(model(base.unsqueeze(0), factor_id, window).squeeze(0).detach().cpu().numpy())
            static_masks[f"mask_{factor}"] = top_density(np.mean(masks, axis=0), args.export_density)
            static_masks[f"{factor}_soft"] = np.mean(masks, axis=0).astype(np.float32)
    npz_path = args.output_dir / "program_conditioned_static_export.npz"
    np.savez_compressed(npz_path, **static_masks, factors=np.asarray(FACTORS))

    summary = {
        "num_examples": len(rows),
        "factor_counts": dict(Counter(row["_factor"] for row in rows).most_common()),
        "future_shape": [token_dim, channel_dim],
        "steps": int(args.steps),
        "history": history,
        "checkpoint_path": str(ckpt_path),
        "static_export_npz": str(npz_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
