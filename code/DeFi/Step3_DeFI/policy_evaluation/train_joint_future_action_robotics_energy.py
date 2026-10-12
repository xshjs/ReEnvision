from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from policy_evaluation.train_joint_future_action_energy import (
        build_pair_index,
        build_success_pools,
        hashed_task,
        load_memory,
        split_by_sequence,
    )
    from policy_evaluation.train_state_action_dynamics_probe import StateActionDynamicsProbe
except ModuleNotFoundError:
    from train_joint_future_action_energy import (
        build_pair_index,
        build_success_pools,
        hashed_task,
        load_memory,
        split_by_sequence,
    )
    from train_state_action_dynamics_probe import StateActionDynamicsProbe


def make_feature_bank(memory: dict[str, np.ndarray], task_dim: int) -> dict[str, np.ndarray]:
    required = [
        "base_summary_exec",
        "target_summary_exec",
        "target_delta_exec",
        "action_chunk",
        "success",
        "task",
        "subtask_index",
        "state_start",
        "state_transition",
    ]
    missing = [k for k in required if k not in memory]
    if missing:
        raise ValueError(f"robotics energy requires memory keys: {missing}")
    task_vec = np.stack([hashed_task(str(t), task_dim) for t in memory["task"].tolist()]).astype(np.float32)
    subtask = (np.asarray(memory["subtask_index"], dtype=np.float32).reshape(-1, 1) / 5.0).astype(np.float32)
    factor = np.asarray(memory["factor"], dtype=object) if "factor" in memory else np.asarray(["none"] * len(task_vec), dtype=object)
    contact_bad = np.asarray([1.0 if str(x) == "contact" else 0.0 for x in factor.tolist()], dtype=np.float32)
    return {
        "base_summary": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "target_summary": np.asarray(memory["target_summary_exec"], dtype=np.float32),
        "target_delta": np.asarray(memory["target_delta_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "task_vec": task_vec,
        "subtask": subtask,
        "success": np.asarray(memory["success"], dtype=np.float32).reshape(-1),
        "state_start": np.asarray(memory["state_start"], dtype=np.float32),
        "state_transition": np.asarray(memory["state_transition"], dtype=np.float32),
        "contact_bad": contact_bad,
        "task": np.asarray(memory["task"], dtype=object),
    }


def compute_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


class JointFutureActionRoboticsEnergy(nn.Module):
    def __init__(
        self,
        future_dim: int,
        action_dim: int,
        state_dim: int,
        task_dim: int,
        hidden_dim: int,
        z_dim: int,
        dropout: float,
        state_manifold_dim: int = 0,
    ):
        super().__init__()
        self.future_dim = int(future_dim)
        self.state_dim = int(state_dim)
        self.state_manifold_dim = int(state_manifold_dim)
        future_in = future_dim * 3 + task_dim + 1
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(future_in),
            nn.Linear(future_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim + task_dim + 1),
            nn.Linear(state_dim + task_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.action_pool = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, z_dim),
            nn.GELU(),
            nn.LayerNorm(z_dim),
        )
        self.sa_fuse = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )
        self.future_delta_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )
        if self.state_manifold_dim > 0:
            self.state_transition_head = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.state_manifold_dim),
                nn.LayerNorm(self.state_manifold_dim),
            )
        else:
            self.state_transition_head = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )
        self.task_energy_head = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.success_head = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contact_head = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_future(
        self,
        base_summary: torch.Tensor,
        target_summary: torch.Tensor,
        target_delta: torch.Tensor,
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([base_summary, target_summary, target_delta, task_vec, subtask], dim=-1)
        return self.future_encoder(x)

    def encode_state_action(
        self,
        current_state: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_feat = self.state_encoder(torch.cat([current_state, task_vec, subtask], dim=-1))
        h = self.action_encoder(action_chunk.transpose(1, 2))
        action_feat = self.action_pool(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1))
        return state_feat, self.sa_fuse(torch.cat([state_feat, action_feat], dim=-1))

    def forward(
        self,
        base_summary: torch.Tensor,
        target_summary: torch.Tensor,
        target_delta: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
        current_state: torch.Tensor,
        target_state_transition: torch.Tensor | None = None,
        state_transition_mean: torch.Tensor | None = None,
        state_transition_std: torch.Tensor | None = None,
        lambda_future_dyn: float = 1.0,
        lambda_state_dyn: float = 1.0,
        beta_contact: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        z_f = self.encode_future(base_summary, target_summary, target_delta, task_vec, subtask)
        z_s, z_sa = self.encode_state_action(current_state, action_chunk, task_vec, subtask)
        fused = torch.cat([z_f, z_sa], dim=-1)
        pred_future_delta = self.future_delta_head(z_sa)
        pred_state_transition = self.state_transition_head(z_sa)
        task_energy = self.task_energy_head(fused).squeeze(-1)
        success_logit = self.success_head(fused).squeeze(-1)
        contact_bad_logit = self.contact_head(torch.cat([z_s, z_sa], dim=-1)).squeeze(-1)

        future_dyn_energy = F.smooth_l1_loss(pred_future_delta, target_delta, reduction="none").mean(dim=-1)
        if target_state_transition is not None:
            pred_state_norm = pred_state_transition
            tgt_state_norm = target_state_transition
            if state_transition_mean is not None and state_transition_std is not None:
                pred_state_norm = (pred_state_transition - state_transition_mean) / state_transition_std
                tgt_state_norm = (target_state_transition - state_transition_mean) / state_transition_std
            state_dyn_energy = F.smooth_l1_loss(pred_state_norm, tgt_state_norm, reduction="none").mean(dim=-1)
        else:
            state_dyn_energy = torch.zeros_like(task_energy)
        contact_energy = torch.sigmoid(contact_bad_logit)
        total_energy = task_energy + float(lambda_future_dyn) * future_dyn_energy + float(lambda_state_dyn) * state_dyn_energy + float(beta_contact) * contact_energy
        return {
            "z_f": z_f,
            "z_sa": z_sa,
            "pred_future_delta": pred_future_delta,
            "pred_state_transition": pred_state_transition,
            "task_energy": task_energy,
            "future_dyn_energy": future_dyn_energy,
            "state_dyn_energy": state_dyn_energy,
            "contact_bad_logit": contact_bad_logit,
            "contact_energy": contact_energy,
            "success_logit": success_logit,
            "energy": total_energy,
        }


def gather_batch(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_summary": torch.from_numpy(bank["target_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "action_chunk": torch.from_numpy(bank["action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "subtask": torch.from_numpy(bank["subtask"][ids]).to(device),
        "success": torch.from_numpy(bank["success"][ids]).to(device),
        "state_start": torch.from_numpy(bank["state_start"][ids]).to(device),
        "state_transition": torch.from_numpy(bank["state_transition"][ids]).to(device),
        "contact_bad": torch.from_numpy(bank["contact_bad"][ids]).to(device),
    }


def load_state_manifold_bundle(path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model = StateActionDynamicsProbe(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        z_dim=int(ckpt["z_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
        transition_latent_dim=int(ckpt.get("transition_latent_dim", 0) or 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    latent_dim = int(ckpt.get("transition_latent_dim", 0) or 0)
    if latent_dim <= 0:
        raise ValueError("state manifold ckpt must have transition_latent_dim > 0")
    return {
        "model": model,
        "latent_dim": latent_dim,
    }


def gather_mismatch_future_batch(
    bank: dict[str, np.ndarray],
    memory: dict[str, np.ndarray],
    pos_idx: np.ndarray,
    success_by_task: dict[str, list[int]],
    success_all: list[int],
    same_task_only: bool,
    rng: random.Random,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    mismatch_ids = []
    for idx in pos_idx.tolist():
        task = str(memory["task"][idx])
        pool = [j for j in success_by_task.get(task, []) if int(j) != int(idx)] if same_task_only else []
        if not pool:
            pool = [j for j in success_by_task.get(task, []) if int(j) != int(idx)]
        if not pool:
            pool = [j for j in success_all if int(j) != int(idx)]
        if not pool:
            pool = [int(idx)]
        mismatch_ids.append(int(rng.choice(pool)))
    mismatch_ids = np.asarray(mismatch_ids, dtype=np.int64)
    batch = gather_batch(bank, pos_idx, device)
    for key in ("target_summary", "target_delta", "success", "contact_bad"):
        batch[key] = torch.from_numpy(bank[key][mismatch_ids]).to(device)
    return batch


def compute_metrics(
    model: JointFutureActionRoboticsEnergy,
    bank: dict[str, np.ndarray],
    memory: dict[str, np.ndarray],
    eval_idx: np.ndarray,
    eval_pairs: list[tuple[int, int]],
    success_by_task: dict[str, list[int]],
    success_all: list[int],
    same_task_only: bool,
    state_transition_mean: torch.Tensor,
    state_transition_std: torch.Tensor,
    state_manifold: dict[str, Any] | None,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    model.eval()
    metrics = {
        "energy": [],
        "success_prob": [],
        "y": [],
        "future_cos": [],
        "state_cos": [],
        "contact_pred": [],
        "contact_y": [],
    }
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = gather_batch(bank, ids, device)
            target_state_for_energy = batch["state_transition"]
            target_state_mean = state_transition_mean
            target_state_std = state_transition_std
            if state_manifold is not None:
                target_state_for_energy = state_manifold["model"].transition_encoder(batch["state_transition"]).detach()
                target_state_mean = None
                target_state_std = None
            out = model(
                batch["base_summary"],
                batch["target_summary"],
                batch["target_delta"],
                batch["action_chunk"],
                batch["task_vec"],
                batch["subtask"],
                batch["state_start"],
                target_state_for_energy,
                target_state_mean,
                target_state_std,
                args.lambda_future_dyn,
                args.lambda_state_dyn,
                args.beta_contact,
            )
            metrics["energy"].append(out["energy"].cpu())
            metrics["success_prob"].append(torch.sigmoid(out["success_logit"]).cpu())
            metrics["y"].append(batch["success"].cpu())
            metrics["contact_pred"].append((torch.sigmoid(out["contact_bad_logit"]) >= 0.5).float().cpu())
            metrics["contact_y"].append(batch["contact_bad"].cpu())
            metrics["future_cos"].append(F.cosine_similarity(out["pred_future_delta"], batch["target_delta"], dim=-1).cpu())
            if state_manifold is not None:
                target_latent = state_manifold["model"].transition_encoder(batch["state_transition"]).detach()
                metrics["state_cos"].append(F.cosine_similarity(out["pred_state_transition"], target_latent, dim=-1).cpu())
            else:
                pred_state = (out["pred_state_transition"] - state_transition_mean) / state_transition_std
                true_state = (batch["state_transition"] - state_transition_mean) / state_transition_std
                metrics["state_cos"].append(F.cosine_similarity(pred_state, true_state, dim=-1).cpu())
    energy = torch.cat(metrics["energy"]).numpy()
    prob = torch.cat(metrics["success_prob"]).numpy()
    y = torch.cat(metrics["y"]).numpy()
    contact_pred = torch.cat(metrics["contact_pred"]).numpy()
    contact_y = torch.cat(metrics["contact_y"]).numpy()
    success_pred = (prob >= 0.5).astype(np.float32)
    success_mask = y > 0.5
    fail_mask = ~success_mask
    auc = 0.0
    if success_mask.any() and fail_mask.any():
        pos = energy[fail_mask]
        neg = energy[success_mask]
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    pair_acc_vals = []
    pair_margin_vals = []
    energy_map = {int(idx): float(energy[i]) for i, idx in enumerate(eval_idx.tolist())}
    for pos_idx, neg_idx in eval_pairs:
        if pos_idx in energy_map and neg_idx in energy_map:
            e_pos = energy_map[pos_idx]
            e_neg = energy_map[neg_idx]
            pair_acc_vals.append(float(e_pos < e_neg))
            pair_margin_vals.append(float(e_neg - e_pos))
    contact_tp = int(((contact_pred == 1) & (contact_y == 1)).sum())
    contact_fp = int(((contact_pred == 1) & (contact_y == 0)).sum())
    contact_fn = int(((contact_pred == 0) & (contact_y == 1)).sum())
    contact_prec = contact_tp / max(contact_tp + contact_fp, 1)
    contact_rec = contact_tp / max(contact_tp + contact_fn, 1)
    contact_f1 = 2 * contact_prec * contact_rec / max(contact_prec + contact_rec, 1e-8)
    return {
        "success_acc": float((success_pred == y).mean()),
        "success_f1": float(2 * (((success_pred == 1) & (y == 1)).sum() / max((success_pred == 1).sum() + (y == 1).sum(), 1e-8))),
        "energy_auc_fail": float(auc),
        "pair_acc": float(np.mean(pair_acc_vals)) if pair_acc_vals else 0.0,
        "pair_margin": float(np.mean(pair_margin_vals)) if pair_margin_vals else 0.0,
        "future_transition_cos": float(torch.cat(metrics["future_cos"]).mean().item()),
        "state_transition_cos": float(torch.cat(metrics["state_cos"]).mean().item()),
        "contact_acc": float((contact_pred == contact_y).mean()),
        "contact_f1": float(contact_f1),
        "energy_success_mean": float(energy[success_mask].mean()) if success_mask.any() else 0.0,
        "energy_fail_mean": float(energy[fail_mask].mean()) if fail_mask.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified robotics energy E_task + lambda E_dynamics + beta E_contact.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=0.5)
    parser.add_argument("--future-transition-weight", type=float, default=0.5)
    parser.add_argument("--state-transition-weight", type=float, default=0.5)
    parser.add_argument("--contact-weight", type=float, default=0.25)
    parser.add_argument("--mismatch-weight", type=float, default=0.5)
    parser.add_argument("--lambda-future-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-state-dyn", type=float, default=0.5)
    parser.add_argument("--beta-contact", type=float, default=0.25)
    parser.add_argument("--state-manifold-ckpt", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory = load_memory(args.memory_npz)
    bank = make_feature_bank(memory, args.task_dim)
    train_idx, val_idx = split_by_sequence(memory, args.val_ratio, args.seed)
    train_pairs = build_pair_index(memory, train_idx, args.same_task_only, args.seed)
    val_pairs = build_pair_index(memory, val_idx, args.same_task_only, args.seed + 1)
    success_by_task, success_all = build_success_pools(memory, train_idx)
    val_success_by_task, val_success_all = build_success_pools(memory, val_idx)

    state_transition_mean_np, state_transition_std_np = compute_stats(bank["state_transition"][train_idx])
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    state_transition_mean = torch.from_numpy(state_transition_mean_np).to(device)
    state_transition_std = torch.from_numpy(state_transition_std_np).to(device)
    state_manifold = load_state_manifold_bundle(args.state_manifold_ckpt, device) if args.state_manifold_ckpt is not None else None

    model = JointFutureActionRoboticsEnergy(
        future_dim=bank["base_summary"].shape[-1],
        action_dim=bank["action_chunk"].shape[-1],
        state_dim=bank["state_start"].shape[-1],
        task_dim=args.task_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        dropout=args.dropout,
        state_manifold_dim=(state_manifold["latent_dim"] if state_manifold is not None else 0),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"score": -1e18, "metrics": None}
    train_history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_pairs = [train_pairs[rng.randrange(len(train_pairs))] for _ in range(args.batch_size)]
        pos_ids = np.asarray([p[0] for p in batch_pairs], dtype=np.int64)
        neg_ids = np.asarray([p[1] for p in batch_pairs], dtype=np.int64)
        pos = gather_batch(bank, pos_ids, device)
        neg = gather_batch(bank, neg_ids, device)
        mismatch = gather_mismatch_future_batch(bank, memory, pos_ids, success_by_task, success_all, args.same_task_only, rng, device)

        pos_state_for_energy = pos["state_transition"]
        neg_state_for_energy = neg["state_transition"]
        mismatch_state_for_energy = mismatch["state_transition"]
        pos_state_mean = neg_state_mean = mismatch_state_mean = state_transition_mean
        pos_state_std = neg_state_std = mismatch_state_std = state_transition_std
        if state_manifold is not None:
            with torch.no_grad():
                pos_state_for_energy = state_manifold["model"].transition_encoder(pos["state_transition"]).detach()
                neg_state_for_energy = state_manifold["model"].transition_encoder(neg["state_transition"]).detach()
                mismatch_state_for_energy = state_manifold["model"].transition_encoder(mismatch["state_transition"]).detach()
            pos_state_mean = neg_state_mean = mismatch_state_mean = None
            pos_state_std = neg_state_std = mismatch_state_std = None

        out_pos = model(
            pos["base_summary"], pos["target_summary"], pos["target_delta"], pos["action_chunk"], pos["task_vec"], pos["subtask"],
            pos["state_start"], pos_state_for_energy, pos_state_mean, pos_state_std,
            args.lambda_future_dyn, args.lambda_state_dyn, args.beta_contact,
        )
        out_neg = model(
            neg["base_summary"], neg["target_summary"], neg["target_delta"], neg["action_chunk"], neg["task_vec"], neg["subtask"],
            neg["state_start"], neg_state_for_energy, neg_state_mean, neg_state_std,
            args.lambda_future_dyn, args.lambda_state_dyn, args.beta_contact,
        )
        out_mismatch = model(
            mismatch["base_summary"], mismatch["target_summary"], mismatch["target_delta"], mismatch["action_chunk"], mismatch["task_vec"], mismatch["subtask"],
            mismatch["state_start"], mismatch_state_for_energy, mismatch_state_mean, mismatch_state_std,
            args.lambda_future_dyn, args.lambda_state_dyn, args.beta_contact,
        )

        rank_loss = F.relu(args.margin + out_pos["energy"] - out_neg["energy"]).mean()
        task_rows = {k: torch.cat([pos[k], neg[k]], dim=0) for k in pos}
        task_logits = torch.cat([out_pos["success_logit"], out_neg["success_logit"]], dim=0)
        task_loss = F.binary_cross_entropy_with_logits(task_logits, task_rows["success"])
        future_transition_loss = F.smooth_l1_loss(out_pos["pred_future_delta"], pos["target_delta"])
        if state_manifold is not None:
            with torch.no_grad():
                true_state_latent = state_manifold["model"].transition_encoder(pos["state_transition"]).detach()
            state_transition_loss = 1.0 - F.cosine_similarity(out_pos["pred_state_transition"], true_state_latent, dim=-1).mean()
        else:
            pred_state_norm = (out_pos["pred_state_transition"] - state_transition_mean) / state_transition_std
            true_state_norm = (pos["state_transition"] - state_transition_mean) / state_transition_std
            state_transition_loss = F.smooth_l1_loss(pred_state_norm, true_state_norm)
        contact_rows = torch.cat([pos["contact_bad"], neg["contact_bad"]], dim=0)
        contact_logits = torch.cat([out_pos["contact_bad_logit"], out_neg["contact_bad_logit"]], dim=0)
        contact_loss = F.binary_cross_entropy_with_logits(contact_logits, contact_rows)
        mismatch_loss = F.relu(args.margin + out_pos["energy"] - out_mismatch["energy"]).mean()

        loss = (
            args.rank_weight * rank_loss
            + args.task_weight * task_loss
            + args.future_transition_weight * future_transition_loss
            + args.state_transition_weight * state_transition_loss
            + args.contact_weight * contact_loss
            + args.mismatch_weight * mismatch_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            metrics = compute_metrics(
                model, bank, memory, val_idx, val_pairs, val_success_by_task, val_success_all, args.same_task_only,
                state_transition_mean, state_transition_std, state_manifold, args, device, args.eval_batch_size, args.seed + step,
            )
            record = {
                "step": int(step),
                "loss": float(loss.detach().item()),
                "rank_loss": float(rank_loss.detach().item()),
                "task_loss": float(task_loss.detach().item()),
                "future_transition_loss": float(future_transition_loss.detach().item()),
                "state_transition_loss": float(state_transition_loss.detach().item()),
                "contact_loss": float(contact_loss.detach().item()),
                "mismatch_loss": float(mismatch_loss.detach().item()),
                **metrics,
            }
            train_history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            score = metrics["pair_acc"] + metrics["success_f1"] + metrics["future_transition_cos"] + metrics["state_transition_cos"] + metrics["contact_f1"]
            if score > best["score"]:
                best = {"score": float(score), "metrics": record}
                torch.save(
                    {
                        "model_type": "robotics_energy",
                        "future_dim": int(bank["base_summary"].shape[-1]),
                        "action_dim": int(bank["action_chunk"].shape[-1]),
                        "state_dim": int(bank["state_start"].shape[-1]),
                        "task_dim": int(args.task_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "z_dim": int(args.z_dim),
                        "dropout": float(args.dropout),
                        "state_transition_mean": state_transition_mean_np,
                        "state_transition_std": state_transition_std_np,
                        "lambda_future_dyn": float(args.lambda_future_dyn),
                        "lambda_state_dyn": float(args.lambda_state_dyn),
                        "beta_contact": float(args.beta_contact),
                        "state_manifold_ckpt": str(args.state_manifold_ckpt) if args.state_manifold_ckpt is not None else None,
                        "state_manifold_dim": int(state_manifold["latent_dim"]) if state_manifold is not None else 0,
                        "model_state": model.state_dict(),
                        "best_metrics": record,
                    },
                    args.output_dir / "joint_future_action_robotics_energy.pt",
                )

    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "steps": int(args.steps),
        "best": best["metrics"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
