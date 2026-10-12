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

from policy_evaluation.train_program_conditioned_mask_head import (
    DEFAULT_TIME_WINDOWS,
    FACTORS as EDIT_FACTORS,
    ProgramConditionedMaskHead,
)
from policy_evaluation.train_unified_future_encoder import UnifiedFutureEncoder, task_hash


FACTORS = ["none"] + EDIT_FACTORS


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def factor_from_row(row: dict[str, Any]) -> str:
    for key in ("factor_label", "wrong_component", "failure_factor", "selected_failure_factor", "applied_factor_mask"):
        factor = str(row.get(key, "") or "")
        if factor in EDIT_FACTORS:
            return factor
    return ""


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        with np.load(str(path)) as data:
            if key not in data.files:
                return None
            arr = np.asarray(data[key], dtype=np.float32)
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def trace_path(row: dict[str, Any]) -> str:
    for key in ("target_proxy_trace_path", "trace_path", "collection_trace_path"):
        value = str(row.get(key, "") or "")
        if value:
            return value
    return ""


def row_time_window(row: dict[str, Any], factor: str) -> tuple[float, float]:
    value = row.get("time_window")
    if isinstance(value, list) and len(value) == 2:
        return float(value[0]), float(value[1])
    return DEFAULT_TIME_WINDOWS.get(factor, (0.0, 1.0))


def load_rows(path: Path, base_key: str, target_key: str, task_dim: int, max_examples: int) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        factor = factor_from_row(row)
        path_s = trace_path(row)
        if not factor or not path_s or not Path(path_s).exists():
            continue
        base = load_future(path_s, base_key)
        target = load_future(path_s, target_key)
        if base is None or target is None:
            continue
        task = str(row.get("task", "unknown"))
        rows.append(
            {
                "base": base,
                "target": target,
                "factor": factor,
                "task": task,
                "task_vec": task_hash(task, task_dim),
                "subtask": float(row.get("subtask_index", 0) or 0) / 5.0,
                "time_window": row_time_window(row, factor),
                "sequence_index": int(row.get("sequence_index", len(rows)) or 0),
            }
        )
        if max_examples > 0 and len(rows) >= max_examples:
            break
    if not rows:
        raise ValueError("no usable rows")
    return rows


class JointFutureEditor(nn.Module):
    def __init__(
        self,
        future_dim: int,
        z_dim: int,
        task_dim: int,
        num_factors: int,
        hidden_dim: int = 512,
        factor_dim: int = 64,
        token_dim: int = 256,
    ):
        super().__init__()
        self.factor_embed = nn.Embedding(num_factors, factor_dim)
        self.token_proj = nn.Sequential(nn.LayerNorm(future_dim), nn.Linear(future_dim, token_dim), nn.GELU())
        in_dim = future_dim * 4 + z_dim + task_dim + factor_dim + 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim + token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    @staticmethod
    def summarize(future: torch.Tensor) -> torch.Tensor:
        mean = future.mean(dim=1)
        std = future.std(dim=1, unbiased=False)
        first = future[:, 0]
        last = future[:, -1]
        return torch.cat([mean, std, first, last - first], dim=-1)

    def forward(self, base: torch.Tensor, z: torch.Tensor, task_vec: torch.Tensor, factor_idx: torch.Tensor, time_window: torch.Tensor):
        factor = self.factor_embed(factor_idx)
        ctx = torch.cat([self.summarize(base), z, task_vec, factor, time_window], dim=-1)
        h = self.net(ctx)
        token_h = self.token_proj(base)
        h_tokens = h[:, None, :].expand(-1, base.shape[1], -1)
        token_delta = self.delta(torch.cat([h_tokens, token_h], dim=-1))
        gate = self.gate(h).view(-1, 1, 1)
        return token_delta, gate


def load_encoder(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = UnifiedFutureEncoder(
        future_dim=int(ckpt["future_dim"]),
        task_dim=int(ckpt["task_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_factors=len(ckpt["factors"]),
        encoder_type=str(ckpt.get("encoder_type", "pooled")),
        token_dim=int(ckpt.get("token_dim", 256)),
        num_layers=int(ckpt.get("num_layers", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_mask_head(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = ProgramConditionedMaskHead(
        token_dim=int(ckpt["token_dim"]),
        channel_dim=int(ckpt["channel_dim"]),
        num_factors=len(ckpt["factors"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def split(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({r["sequence_index"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train, valid = [], []
    for i, r in enumerate(rows):
        (valid if r["sequence_index"] in val else train).append(i)
    return train, valid


def sample_indices(rows, train_idx, batch_size):
    by_factor = defaultdict(list)
    for i in train_idx:
        by_factor[rows[i]["factor"]].append(i)
    factors = [k for k, v in by_factor.items() if v]
    return [random.choice(by_factor[random.choice(factors)]) for _ in range(batch_size)]


def make_batch(rows, idxs, device, factor_to_idx):
    chosen = [rows[i] for i in idxs]
    base = torch.from_numpy(np.stack([r["base"] for r in chosen])).to(device)
    target = torch.from_numpy(np.stack([r["target"] for r in chosen])).to(device)
    task_vec = torch.from_numpy(np.stack([r["task_vec"] for r in chosen])).to(device)
    sub = torch.tensor([r["subtask"] for r in chosen], device=device, dtype=torch.float32)
    factor = torch.tensor([factor_to_idx[r["factor"]] for r in chosen], device=device, dtype=torch.long)
    windows = torch.tensor([r["time_window"] for r in chosen], device=device, dtype=torch.float32).clamp(0.0, 1.0)
    return base, target, task_vec, sub, factor, windows


def cosine_loss(x, y):
    return 1.0 - F.cosine_similarity(x.flatten(1), y.flatten(1), dim=1).mean()


def evaluate(editor, encoder, mask_head, rows, idxs, device, factor_to_idx, max_eval, gate_scale, fixed_gate, fixed_gate_value):
    editor.eval()
    idxs = idxs[:max_eval] if max_eval > 0 else idxs
    vals = []
    with torch.no_grad():
        for start in range(0, len(idxs), 32):
            base, target, task_vec, sub, factor, windows = make_batch(rows, idxs[start : start + 32], device, factor_to_idx)
            z = encoder(base, task_vec, sub)["z"]
            mask = mask_head(base, factor, windows)
            delta, gate = editor(base, z, task_vec, factor, windows)
            if fixed_gate:
                gate = torch.full_like(gate, float(fixed_gate_value))
            edit = gate_scale * gate * mask * delta
            edited = base + edit
            target_delta = target - base
            comp = base + gate_scale * gate * (1.0 - mask) * delta
            vals.append(
                {
                    "target_cos_gain": float((F.cosine_similarity(edited.flatten(1), target.flatten(1), dim=1) - F.cosine_similarity(base.flatten(1), target.flatten(1), dim=1)).mean().item()),
                    "direction_cos": float((1.0 - cosine_loss(mask * delta, target_delta)).item()),
                    "preserve_mse": float((((1.0 - mask) * edit) ** 2).mean().item()),
                    "suff_vs_complement_margin": float((F.cosine_similarity(edited.flatten(1), target.flatten(1), dim=1) - F.cosine_similarity(comp.flatten(1), target.flatten(1), dim=1)).mean().item()),
                    "gate": float(gate.mean().item()),
                    "mask_mean": float(mask.mean().item()),
                }
            )
    out = {}
    for k in vals[0]:
        out[k] = float(np.mean([v[k] for v in vals]))
    out["num_eval"] = len(idxs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint z+q+M future editor.")
    parser.add_argument("--factor-supervision-jsonl", type=Path, required=True)
    parser.add_argument("--future-encoder-ckpt", type=Path, required=True)
    parser.add_argument("--mask-head-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-key", type=str, default="base_future")
    parser.add_argument("--target-key", type=str, default="target_proxy_future")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--editor-token-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-scale", type=float, default=0.01)
    parser.add_argument("--lambda-direction", type=float, default=1.0)
    parser.add_argument("--lambda-masked-target", dest="lambda_masked_target", type=float, default=5.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--lambda-sufficiency", type=float, default=1.0)
    parser.add_argument("--fixed-gate", action="store_true")
    parser.add_argument("--fixed-gate-value", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-max", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows = load_rows(args.factor_supervision_jsonl, args.base_key, args.target_key, args.task_dim, args.max_examples)
    train_idx, val_idx = split(rows, args.val_ratio, args.seed)
    encoder, enc_ckpt = load_encoder(args.future_encoder_ckpt, device)
    mask_head, mask_ckpt = load_mask_head(args.mask_head_ckpt, device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in mask_head.parameters():
        p.requires_grad_(False)
    factor_to_idx = {f: i for i, f in enumerate(mask_ckpt["factors"])}
    editor = JointFutureEditor(
        future_dim=int(enc_ckpt["future_dim"]),
        z_dim=int(enc_ckpt["z_dim"]),
        task_dim=args.task_dim,
        num_factors=len(mask_ckpt["factors"]),
        hidden_dim=args.hidden_dim,
        token_dim=args.editor_token_dim,
    ).to(device)
    opt = torch.optim.AdamW(editor.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        editor.train()
        idxs = sample_indices(rows, train_idx, args.batch_size)
        base, target, task_vec, sub, factor, windows = make_batch(rows, idxs, device, factor_to_idx)
        with torch.no_grad():
            z = encoder(base, task_vec, sub)["z"]
            mask = mask_head(base, factor, windows)
        delta, gate = editor(base, z, task_vec, factor, windows)
        if bool(args.fixed_gate):
            gate = torch.full_like(gate, float(args.fixed_gate_value))
        edit = args.gate_scale * gate * mask * delta
        edited = base + edit
        target_delta = target - base
        comp = base + args.gate_scale * gate * (1.0 - mask) * delta
        target_loss = F.smooth_l1_loss(edited, target)
        masked_target_loss = F.smooth_l1_loss(mask * (edited - target), mask * (target - base))
        direction_loss = cosine_loss(mask * delta, target_delta)
        preserve_loss = (((1.0 - mask) * edit) ** 2).mean()
        sufficiency_loss = F.relu(
            F.cosine_similarity(comp.flatten(1), target.flatten(1), dim=1)
            - F.cosine_similarity(edited.flatten(1), target.flatten(1), dim=1)
            + 1e-4
        ).mean()
        loss = (
            target_loss
            + args.lambda_masked_target * masked_target_loss
            + args.lambda_direction * direction_loss
            + args.lambda_preserve * preserve_loss
            + args.lambda_sufficiency * sufficiency_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(editor.parameters(), 5.0)
        opt.step()
        if step == 1 or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(
                editor,
                encoder,
                mask_head,
                rows,
                val_idx,
                device,
                factor_to_idx,
                args.eval_max,
                args.gate_scale,
                bool(args.fixed_gate),
                float(args.fixed_gate_value),
            )
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "target_loss": float(target_loss.detach().cpu()),
                "masked_target_loss": float(masked_target_loss.detach().cpu()),
                "direction_loss": float(direction_loss.detach().cpu()),
                "preserve_loss": float(preserve_loss.detach().cpu()),
                "sufficiency_loss": float(sufficiency_loss.detach().cpu()),
                **metrics,
            }
            history.append(rec)
            score = rec["target_cos_gain"] + rec["suff_vs_complement_margin"] + rec["direction_cos"]
            if best is None or score > best["score"]:
                best = {**rec, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in editor.state_dict().items()}
            print(json.dumps(rec), flush=True)

    assert best_state is not None and best is not None
    ckpt_path = args.output_dir / "joint_future_editor.pt"
    torch.save(
        {
            "model_type": "joint_future_editor",
            "model_state": best_state,
            "future_dim": int(enc_ckpt["future_dim"]),
            "z_dim": int(enc_ckpt["z_dim"]),
            "task_dim": int(args.task_dim),
            "hidden_dim": int(args.hidden_dim),
            "editor_token_dim": int(args.editor_token_dim),
            "factors": list(mask_ckpt["factors"]),
            "factor_to_idx": factor_to_idx,
            "future_encoder_ckpt": str(args.future_encoder_ckpt),
            "mask_head_ckpt": str(args.mask_head_ckpt),
            "gate_scale": float(args.gate_scale),
        },
        ckpt_path,
    )
    summary = {
        "num_examples": len(rows),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "factor_counts": dict(Counter(r["factor"] for r in rows).most_common()),
        "best": best,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
