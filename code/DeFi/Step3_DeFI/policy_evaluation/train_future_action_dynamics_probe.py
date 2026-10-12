from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_joint_future_action_energy import hashed_task, load_memory, split_by_sequence


def build_success_pools(memory: dict[str, np.ndarray], indices: np.ndarray) -> tuple[dict[str, list[int]], list[int]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    all_ids: list[int] = []
    for idx in indices.tolist():
        task = str(memory["task"][idx])
        by_task[task].append(int(idx))
        all_ids.append(int(idx))
    return by_task, all_ids


def make_feature_bank(memory: dict[str, np.ndarray], task_dim: int) -> dict[str, np.ndarray]:
    task_vec = np.stack([hashed_task(str(t), task_dim) for t in memory["task"].tolist()]).astype(np.float32)
    subtask = (np.asarray(memory["subtask_index"], dtype=np.float32).reshape(-1, 1) / 5.0).astype(np.float32)
    return {
        "base_summary": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "target_summary": np.asarray(memory["target_summary_exec"], dtype=np.float32),
        "target_delta": np.asarray(memory["target_delta_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "task_vec": task_vec,
        "subtask": subtask,
        "success": np.asarray(memory["success"], dtype=np.float32),
        "task": np.asarray(memory["task"]),
    }


def compute_delta_stats(target_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = target_delta.mean(axis=0).astype(np.float32)
    std = target_delta.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class FutureActionDynamicsProbe(nn.Module):
    def __init__(self, action_dim: int, future_dim: int, task_dim: int, hidden_dim: int, z_dim: int, dropout: float):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(future_dim + task_dim + 1),
            nn.Linear(future_dim + task_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )
        self.compat_head = nn.Sequential(
            nn.LayerNorm(z_dim + future_dim),
            nn.Linear(z_dim + future_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.success_head = nn.Linear(z_dim, 1)

    def encode(self, base_summary: torch.Tensor, action_chunk: torch.Tensor, task_vec: torch.Tensor, subtask: torch.Tensor) -> torch.Tensor:
        h = self.action_encoder(action_chunk.transpose(1, 2))
        a_mean = h.mean(dim=-1)
        a_max = h.amax(dim=-1)
        s = self.state_encoder(torch.cat([base_summary, task_vec, subtask], dim=-1))
        return self.fuse(torch.cat([a_mean, a_max, s], dim=-1))

    def forward(
        self,
        base_summary: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
        target_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encode(base_summary, action_chunk, task_vec, subtask)
        pred_delta = self.delta_head(z)
        out = {
            "z": z,
            "pred_delta": pred_delta,
            "pred_target_summary": base_summary + pred_delta,
            "success_logit": self.success_head(z).squeeze(-1),
        }
        if target_delta is not None:
            out["compat_logit"] = self.compat_head(torch.cat([z, target_delta], dim=-1)).squeeze(-1)
        return out


def gather_batch(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_summary": torch.from_numpy(bank["target_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "action_chunk": torch.from_numpy(bank["action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "subtask": torch.from_numpy(bank["subtask"][ids]).to(device),
        "success": torch.from_numpy(bank["success"][ids]).to(device),
    }


def normalize_delta(delta: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (delta - mean) / std


def make_train_batch(
    bank: dict[str, np.ndarray],
    memory: dict[str, np.ndarray],
    ids: np.ndarray,
    by_task: dict[str, list[int]],
    all_ids: list[int],
    same_task_negative: bool,
    rng: random.Random,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = gather_batch(bank, ids, device)
    neg_ids = []
    for idx in ids.tolist():
        task = str(memory["task"][idx])
        pool = [j for j in by_task.get(task, []) if int(j) != int(idx)] if same_task_negative else []
        if not pool:
            pool = [j for j in all_ids if int(j) != int(idx)]
        if not pool:
            pool = [int(idx)]
        neg_ids.append(int(rng.choice(pool)))
    neg_ids_np = np.asarray(neg_ids, dtype=np.int64)
    batch["neg_target_delta"] = torch.from_numpy(bank["target_delta"][neg_ids_np]).to(device)
    batch["neg_target_summary"] = torch.from_numpy(bank["target_summary"][neg_ids_np]).to(device)
    return batch


def compute_metrics(
    model: FutureActionDynamicsProbe,
    bank: dict[str, np.ndarray],
    eval_idx: np.ndarray,
    by_task: dict[str, list[int]],
    all_ids: list[int],
    same_task_negative: bool,
    device: torch.device,
    batch_size: int,
    seed: int,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
) -> dict[str, Any]:
    rng = random.Random(seed)
    model.eval()
    delta_losses = []
    delta_cos = []
    compat_acc = []
    success_acc = []
    success_tp = success_fp = success_fn = success_tn = 0
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = make_train_batch(bank, {"task": bank["task"]}, ids, by_task, all_ids, same_task_negative, rng, device)
            out = model(batch["base_summary"], batch["action_chunk"], batch["task_vec"], batch["subtask"], batch["target_delta"])
            pred_delta_norm = normalize_delta(out["pred_delta"], delta_mean, delta_std)
            pos_delta_norm = normalize_delta(batch["target_delta"], delta_mean, delta_std)
            neg_delta_norm = normalize_delta(batch["neg_target_delta"], delta_mean, delta_std)
            delta_losses.append(F.smooth_l1_loss(pred_delta_norm, pos_delta_norm).cpu())
            delta_cos.append(F.cosine_similarity(pred_delta_norm, pos_delta_norm, dim=-1).mean().cpu())

            pos_logit = out["compat_logit"]
            neg_logit = model(
                batch["base_summary"],
                batch["action_chunk"],
                batch["task_vec"],
                batch["subtask"],
                batch["neg_target_delta"],
            )["compat_logit"]
            compat_margin = F.cosine_similarity(pred_delta_norm, pos_delta_norm, dim=-1) - F.cosine_similarity(pred_delta_norm, neg_delta_norm, dim=-1)
            compat_acc.append(((compat_margin > 0).float().mean()).cpu())

            pred_success = (torch.sigmoid(out["success_logit"]) >= 0.5).float()
            gt_success = (batch["success"] > 0.5).float()
            success_acc.append((pred_success == gt_success).float().mean().cpu())
            success_tp += int(((pred_success == 1) & (gt_success == 1)).sum().item())
            success_fp += int(((pred_success == 1) & (gt_success == 0)).sum().item())
            success_fn += int(((pred_success == 0) & (gt_success == 1)).sum().item())
            success_tn += int(((pred_success == 0) & (gt_success == 0)).sum().item())
    precision = success_tp / max(success_tp + success_fp, 1)
    recall = success_tp / max(success_tp + success_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "transition_loss": float(torch.stack(delta_losses).mean().item()) if delta_losses else 0.0,
        "transition_cos": float(torch.stack(delta_cos).mean().item()) if delta_cos else 0.0,
        "compat_acc": float(torch.stack(compat_acc).mean().item()) if compat_acc else 0.0,
        "success_acc": float(torch.stack(success_acc).mean().item()) if success_acc else 0.0,
        "success_precision": float(precision),
        "success_recall": float(recall),
        "success_f1": float(f1),
        "tp": int(success_tp),
        "fp": int(success_fp),
        "fn": int(success_fn),
        "tn": int(success_tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train D(s, z_a)->z_f transition head and R_dynamics(s, z_a, z_f) compatibility probe.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--compat-weight", type=float, default=1.0)
    parser.add_argument("--success-weight", type=float, default=0.25)
    parser.add_argument("--compat-margin", type=float, default=0.2)
    parser.add_argument("--same-task-negative", action="store_true")
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
    train_by_task, train_all = build_success_pools(memory, train_idx)
    val_by_task, val_all = build_success_pools(memory, val_idx)
    delta_mean_np, delta_std_np = compute_delta_stats(bank["target_delta"][train_idx])

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    delta_mean = torch.from_numpy(delta_mean_np).to(device)
    delta_std = torch.from_numpy(delta_std_np).to(device)
    model = FutureActionDynamicsProbe(
        action_dim=int(bank["action_chunk"].shape[-1]),
        future_dim=int(bank["target_delta"].shape[-1]),
        task_dim=int(args.task_dim),
        hidden_dim=int(args.hidden_dim),
        z_dim=int(args.z_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        ids = np.asarray(rng.choices(train_idx.tolist(), k=min(len(train_idx), args.batch_size)), dtype=np.int64)
        batch = make_train_batch(bank, memory, ids, train_by_task, train_all, bool(args.same_task_negative), rng, device)
        out = model(batch["base_summary"], batch["action_chunk"], batch["task_vec"], batch["subtask"], batch["target_delta"])
        pred_delta_norm = normalize_delta(out["pred_delta"], delta_mean, delta_std)
        pos_delta_norm = normalize_delta(batch["target_delta"], delta_mean, delta_std)
        neg_delta_norm = normalize_delta(batch["neg_target_delta"], delta_mean, delta_std)

        pos_sim = F.cosine_similarity(pred_delta_norm, pos_delta_norm, dim=-1)
        neg_sim = F.cosine_similarity(pred_delta_norm, neg_delta_norm, dim=-1)

        transition_loss = F.smooth_l1_loss(pred_delta_norm, pos_delta_norm)
        compat_loss = F.relu(float(args.compat_margin) - (pos_sim - neg_sim)).mean()
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        loss = (
            float(args.transition_weight) * transition_loss
            + float(args.compat_weight) * compat_loss
            + float(args.success_weight) * success_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            metrics = compute_metrics(
                model,
                bank,
                val_idx,
                val_by_task,
                val_all,
                bool(args.same_task_negative),
                device,
                int(args.eval_batch_size),
                args.seed + step,
                delta_mean,
                delta_std,
            )
            rec = {
                "step": int(step),
                "loss": float(loss.item()),
                "transition_loss_train": float(transition_loss.item()),
                "compat_loss_train": float(compat_loss.item()),
                "success_loss_train": float(success_loss.item()),
                **metrics,
            }
            history.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            score = metrics["transition_cos"] + metrics["compat_acc"] + metrics["success_f1"]
            if best is None or score > best["score"]:
                best = {"score": float(score), "metrics": rec}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "action_dim": int(bank["action_chunk"].shape[-1]),
                        "future_dim": int(bank["target_delta"].shape[-1]),
                        "task_dim": int(args.task_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "z_dim": int(args.z_dim),
                        "dropout": float(args.dropout),
                        "args": vars(args),
                        "delta_mean": delta_mean_np,
                        "delta_std": delta_std_np,
                        "best_metrics": rec,
                    },
                    args.output_dir / "future_action_dynamics_probe.pt",
                )

    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "steps": int(args.steps),
        "best": best["metrics"] if best is not None else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if history:
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
