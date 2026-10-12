from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from policy_evaluation.train_chunk_effect_phi_mlp import split_by_sequence
    from policy_evaluation.train_directional_sensitivity_tcn import FiLMResidualBlock, _finite_rows, _standardize
    from policy_evaluation.train_joint_execution_update_mlp import build_meta
except ModuleNotFoundError:
    from train_chunk_effect_phi_mlp import split_by_sequence
    from train_directional_sensitivity_tcn import FiLMResidualBlock, _finite_rows, _standardize
    from train_joint_execution_update_mlp import build_meta


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


class DirectActionCorrectionTCN(nn.Module):
    """Direct baseline: (o_t, a_t, o_{t+1}, A_remain, task) -> delta A, without h=(z_f,z_a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        task_dim: int,
        chunk_len: int,
        hidden_dim: int = 128,
        cond_dim: int = 256,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.05,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        self.chunk_len = int(chunk_len)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.output_scale = float(output_scale)
        cond_in = state_dim * 3 + action_dim + task_dim + 3
        self.cond_encoder = nn.Sequential(
            nn.LayerNorm(cond_in),
            nn.Linear(cond_in, cond_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        self.in_proj = nn.Conv1d(action_dim + 1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, cond_dim, kernel_size, dropout) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv1d(hidden_dim, action_dim, kernel_size=1)
        self.validity_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.uncertainty_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def _features(
        self,
        state_start: torch.Tensor,
        state_end: torch.Tensor,
        executed_action: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _ = action_chunk.shape
        if steps != self.chunk_len:
            raise ValueError(f"expected chunk_len={self.chunk_len}, got {steps}")
        state_delta = state_end - state_start
        cond = self.cond_encoder(torch.cat([state_start, state_end, state_delta, executed_action, task_vec, meta], dim=-1))
        t = torch.linspace(-1.0, 1.0, steps, device=action_chunk.device, dtype=action_chunk.dtype)
        t = t.view(1, steps, 1).expand(batch, -1, -1)
        x = torch.cat([action_chunk, t], dim=-1).transpose(1, 2)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        return x

    def forward_with_heads(
        self,
        state_start: torch.Tensor,
        state_end: torch.Tensor,
        executed_action: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self._features(state_start, state_end, executed_action, action_chunk, task_vec, meta)
        return {
            "delta": self.out_proj(x).transpose(1, 2) * self.output_scale,
            "invalidity_logit": self.validity_proj(x).transpose(1, 2).squeeze(-1),
            "uncertainty_logit": self.uncertainty_proj(x).transpose(1, 2).squeeze(-1),
        }

    def forward(
        self,
        state_start: torch.Tensor,
        state_end: torch.Tensor,
        executed_action: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_heads(state_start, state_end, executed_action, action_chunk, task_vec, meta)["delta"]


def _make_bank(memory: dict[str, np.ndarray], target_delta_clip: float) -> dict[str, np.ndarray]:
    required = ["state_start", "state_end", "executed_action", "action_chunk", "action_delta_target", "task_vec"]
    missing = [key for key in required if key not in memory]
    if missing:
        raise ValueError(f"memory missing keys: {missing}")
    state_start = np.asarray(memory["state_start"], dtype=np.float32)
    state_end = np.asarray(memory["state_end"], dtype=np.float32)
    executed = np.asarray(memory["executed_action"], dtype=np.float32)
    action = np.asarray(memory["action_chunk"], dtype=np.float32)
    target = np.asarray(memory["action_delta_target"], dtype=np.float32)
    task = np.asarray(memory["task_vec"], dtype=np.float32)
    meta = build_meta(memory).astype(np.float32)
    if target_delta_clip > 0:
        target = np.clip(target, -float(target_delta_clip), float(target_delta_clip)).astype(np.float32)
    mask = _finite_rows(state_start, state_end, executed, action, target, task, meta)
    if not mask.any():
        raise ValueError("no finite rows after filtering")
    return {
        "state_start": state_start[mask],
        "state_end": state_end[mask],
        "executed_action": executed[mask],
        "action": action[mask],
        "target_delta_action": target[mask],
        "task": task[mask],
        "meta": meta[mask],
        "mode": np.asarray(memory.get("mode_label", np.zeros((len(action),), dtype=np.int64)), dtype=np.int64)[mask],
        "sequence_index": np.asarray(memory.get("sequence_index", np.arange(len(action))), dtype=np.int64)[mask],
    }


def _split_stratified_by_mode(mode: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for value in np.unique(mode):
        ids = np.flatnonzero(mode == value)
        rng.shuffle(ids)
        n_val = max(1, int(round(ids.size * val_ratio))) if ids.size > 1 else 0
        val_parts.append(ids[:n_val])
        train_parts.append(ids[n_val:])
    train_idx = np.concatenate([part for part in train_parts if part.size > 0])
    val_idx = np.concatenate([part for part in val_parts if part.size > 0])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _batch(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(bank[key][ids]).to(device=device, dtype=torch.float32)
        for key in ["state_start", "state_end", "executed_action", "action", "target_delta_action", "task", "meta"]
    }


def _head_losses(
    heads: dict[str, torch.Tensor],
    target: torch.Tensor,
    affected_threshold: float,
    validity_loss_weight: float,
    uncertainty_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_step_norm = torch.norm(target, dim=-1)
    affected = (target_step_norm > affected_threshold).float()
    pos = affected.sum()
    neg = affected.numel() - pos
    pos_weight = (neg / torch.clamp(pos, min=1.0)).clamp(1.0, 20.0)
    validity_loss = F.binary_cross_entropy_with_logits(
        heads["invalidity_logit"],
        affected,
        pos_weight=pos_weight.detach(),
    )
    boundary = torch.exp(-torch.abs(target_step_norm - affected_threshold) / max(float(affected_threshold), 1e-8))
    uncertainty_loss = F.l1_loss(torch.sigmoid(heads["uncertainty_logit"]), boundary)
    loss = float(validity_loss_weight) * validity_loss + float(uncertainty_loss_weight) * uncertainty_loss
    with torch.no_grad():
        pred_mask = torch.sigmoid(heads["invalidity_logit"]) > 0.5
        tgt_mask = affected > 0.5
        tp = torch.logical_and(pred_mask, tgt_mask).sum().float()
        fp = torch.logical_and(pred_mask, ~tgt_mask).sum().float()
        fn = torch.logical_and(~pred_mask, tgt_mask).sum().float()
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    return loss, {
        "validity_loss": float(validity_loss.detach().item()),
        "uncertainty_loss": float(uncertainty_loss.detach().item()),
        "train_head_precision": float(precision.item()),
        "train_head_recall": float(recall.item()),
        "train_head_f1": float(f1.item()),
    }


def _delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    affected_threshold: float,
    affected_weight: float,
    norm_loss_weight: float,
    zero_loss_weight: float,
) -> torch.Tensor:
    step_norm = torch.norm(target, dim=-1)
    affected = step_norm > affected_threshold
    weights = torch.ones_like(step_norm)
    if affected_weight > 0:
        weights = weights + affected.float() * float(affected_weight)
    mse = ((pred - target).pow(2).mean(dim=-1) * weights).mean()
    pred_norm = torch.norm(pred, dim=-1)
    norm_loss = F.mse_loss(pred_norm, step_norm)
    zero_loss = pred_norm[~affected].pow(2).mean() if zero_loss_weight > 0 and (~affected).any() else pred_norm.new_tensor(0.0)
    return mse + float(norm_loss_weight) * norm_loss + float(zero_loss_weight) * zero_loss


def _evaluate(
    model: DirectActionCorrectionTCN,
    bank: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    affected_threshold: float,
) -> dict[str, float]:
    model.eval()
    total = max(1, len(ids))
    sums = {"loss": 0.0, "delta_l1": 0.0, "delta_cos": 0.0, "pred_delta_norm": 0.0, "target_delta_norm": 0.0}
    tp = fp = fn = tn = 0
    htp = hfp = hfn = htn = 0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            bid = ids[start : start + batch_size]
            b = _batch(bank, bid, device)
            heads = model.forward_with_heads(b["state_start"], b["state_end"], b["executed_action"], b["action"], b["task"], b["meta"])
            pred = heads["delta"]
            target = b["target_delta_action"]
            pred_f = pred.reshape(pred.shape[0], -1)
            target_f = target.reshape(target.shape[0], -1)
            sums["loss"] += float(F.mse_loss(pred, target).item()) * len(bid)
            sums["delta_l1"] += float(torch.mean(torch.abs(pred - target)).item()) * len(bid)
            sums["delta_cos"] += float(F.cosine_similarity(pred_f, target_f, dim=-1).mean().item()) * len(bid)
            sums["pred_delta_norm"] += float(torch.norm(pred_f, dim=-1).mean().item()) * len(bid)
            sums["target_delta_norm"] += float(torch.norm(target_f, dim=-1).mean().item()) * len(bid)
            pred_step = torch.norm(pred, dim=-1) > affected_threshold
            target_step = torch.norm(target, dim=-1) > affected_threshold
            tp += int(torch.logical_and(pred_step, target_step).sum().item())
            fp += int(torch.logical_and(pred_step, ~target_step).sum().item())
            fn += int(torch.logical_and(~pred_step, target_step).sum().item())
            tn += int(torch.logical_and(~pred_step, ~target_step).sum().item())
            head_step = torch.sigmoid(heads["invalidity_logit"]) > 0.5
            htp += int(torch.logical_and(head_step, target_step).sum().item())
            hfp += int(torch.logical_and(head_step, ~target_step).sum().item())
            hfn += int(torch.logical_and(~head_step, target_step).sum().item())
            htn += int(torch.logical_and(~head_step, ~target_step).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    head_precision = htp / max(1, htp + hfp)
    head_recall = htp / max(1, htp + hfn)
    head_f1 = 2.0 * head_precision * head_recall / max(1e-12, head_precision + head_recall)
    return {
        **{k: v / total for k, v in sums.items()},
        "affected_precision": float(precision),
        "affected_recall": float(recall),
        "affected_f1": float(f1),
        "affected_tp": int(tp),
        "affected_fp": int(fp),
        "affected_fn": int(fn),
        "affected_tn": int(tn),
        "validity_precision": float(head_precision),
        "validity_recall": float(head_recall),
        "validity_f1": float(head_f1),
        "validity_tp": int(htp),
        "validity_fp": int(hfp),
        "validity_fn": int(hfn),
        "validity_tn": int(htn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct observation-to-suffix correction baseline without h=(z_f,z_a).")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["stratified_mode", "sequence"], default="stratified_mode")
    parser.add_argument("--target-delta-clip", type=float, default=0.05)
    parser.add_argument("--affected-threshold", type=float, default=1e-4)
    parser.add_argument("--affected-weight", type=float, default=0.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--zero-loss-weight", type=float, default=0.0)
    parser.add_argument("--validity-loss-weight", type=float, default=0.0)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=["delta_l1", "affected_f1", "validity_f1"], default="delta_l1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    memory = _load_npz(args.memory_npz)
    bank = _make_bank(memory, args.target_delta_clip)
    n = len(bank["action"])
    if args.split_mode == "sequence":
        train_idx, val_idx = split_by_sequence(bank["sequence_index"], args.val_ratio, args.seed)
    else:
        train_idx, val_idx = _split_stratified_by_mode(bank["mode"], args.val_ratio, args.seed)
    if len(train_idx) == 0 or len(val_idx) == 0:
        perm = rng.permutation(n)
        n_val = max(1, int(n * args.val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    bank["state_start"], start_mean, start_std = _standardize(bank["state_start"][train_idx], bank["state_start"])
    bank["state_end"], end_mean, end_std = _standardize(bank["state_end"][train_idx], bank["state_end"])
    bank["executed_action"], exec_mean, exec_std = _standardize(bank["executed_action"][train_idx], bank["executed_action"])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = DirectActionCorrectionTCN(
        state_dim=bank["state_start"].shape[-1],
        action_dim=bank["action"].shape[-1],
        task_dim=bank["task"].shape[-1],
        chunk_len=bank["action"].shape[1],
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        output_scale=args.output_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    best: dict[str, float] | None = None
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=len(train_idx) < args.batch_size)
        b = _batch(bank, ids, device)
        heads = model.forward_with_heads(b["state_start"], b["state_end"], b["executed_action"], b["action"], b["task"], b["meta"])
        loss = _delta_loss(
            heads["delta"],
            b["target_delta_action"],
            args.affected_threshold,
            args.affected_weight,
            args.norm_loss_weight,
            args.zero_loss_weight,
        )
        head_metrics = {}
        if args.validity_loss_weight > 0.0 or args.uncertainty_loss_weight > 0.0:
            aux_loss, head_metrics = _head_losses(
                heads,
                b["target_delta_action"],
                args.affected_threshold,
                args.validity_loss_weight,
                args.uncertainty_loss_weight,
            )
            loss = loss + aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            metrics = _evaluate(model, bank, val_idx, device, args.eval_batch_size, args.affected_threshold)
            metrics["step"] = int(step)
            metrics["train_loss"] = float(loss.item())
            metrics.update(head_metrics)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            metric = str(args.best_metric)
            better = best is None or (metrics[metric] > best[metric] if metric.endswith("_f1") else metrics[metric] < best[metric])
            if better:
                best = dict(metrics)
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt_path = args.output_dir / "direct_action_correction_tcn.pt"
    torch.save(
        {
            "runtime_type": "direct_action_correction_tcn",
            "model_state": best_state if best_state is not None else model.state_dict(),
            "state_dim": int(bank["state_start"].shape[-1]),
            "action_dim": int(bank["action"].shape[-1]),
            "task_dim": int(bank["task"].shape[-1]),
            "chunk_len": int(bank["action"].shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "cond_dim": int(args.cond_dim),
            "num_blocks": int(args.num_blocks),
            "kernel_size": int(args.kernel_size),
            "dropout": float(args.dropout),
            "output_scale": float(args.output_scale),
            "target_delta_clip": float(args.target_delta_clip),
            "affected_threshold": float(args.affected_threshold),
            "affected_weight": float(args.affected_weight),
            "norm_loss_weight": float(args.norm_loss_weight),
            "zero_loss_weight": float(args.zero_loss_weight),
            "validity_loss_weight": float(args.validity_loss_weight),
            "uncertainty_loss_weight": float(args.uncertainty_loss_weight),
            "has_validity_head": True,
            "state_start_mean": start_mean,
            "state_start_std": start_std,
            "state_end_mean": end_mean,
            "state_end_std": end_std,
            "executed_action_mean": exec_mean,
            "executed_action_std": exec_std,
        },
        ckpt_path,
    )
    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "split_mode": str(args.split_mode),
        "train_mode_counts": {str(k): int(v) for k, v in zip(*np.unique(bank["mode"][train_idx], return_counts=True))},
        "val_mode_counts": {str(k): int(v) for k, v in zip(*np.unique(bank["mode"][val_idx], return_counts=True))},
        "steps": int(args.steps),
        "best_metric": str(args.best_metric),
        "has_validity_head": True,
        "best": best,
        "output_ckpt": str(ckpt_path),
        "supervision": "direct_o_t_a_t_o_t1_to_teacher_suffix_delta_without_h",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
