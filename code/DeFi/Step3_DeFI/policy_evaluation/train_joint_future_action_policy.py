from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hashed_task(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def load_memory(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        memory = {k: data[k] for k in data.files}
    required = [
        "base_summary_exec",
        "target_summary_exec",
        "target_delta_exec",
        "action_chunk",
        "success",
        "task",
        "sequence_index",
        "subtask_index",
    ]
    missing = [k for k in required if k not in memory]
    if missing:
        raise ValueError(f"memory npz missing keys: {missing}")
    return memory


def split_by_sequence(memory: dict[str, np.ndarray], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = sorted({int(x) for x in memory["sequence_index"].tolist()})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_set = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, seq in enumerate(memory["sequence_index"].tolist()):
        (val_idx if int(seq) in val_set else train_idx).append(idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_feature_bank(memory: dict[str, np.ndarray], task_dim: int) -> dict[str, np.ndarray]:
    task_vec = np.stack([hashed_task(str(t), task_dim) for t in memory["task"].tolist()]).astype(np.float32)
    subtask = (np.asarray(memory["subtask_index"], dtype=np.float32).reshape(-1, 1) / 5.0).astype(np.float32)
    success = np.asarray(memory["success"], dtype=np.float32).reshape(-1)
    return {
        "base_summary": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "target_summary": np.asarray(memory["target_summary_exec"], dtype=np.float32),
        "target_delta": np.asarray(memory["target_delta_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "task_vec": task_vec,
        "subtask": subtask,
        "success": success,
        "task": np.asarray(memory["task"], dtype=object),
    }


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_summary": torch.from_numpy(bank["target_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "action_chunk": torch.from_numpy(bank["action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "subtask": torch.from_numpy(bank["subtask"][ids]).to(device),
        "success": torch.from_numpy(bank["success"][ids]).to(device),
    }


def build_pair_ids(bank: dict[str, np.ndarray], indices: np.ndarray, same_task_only: bool, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    success_by_task: dict[str, list[int]] = defaultdict(list)
    fail_by_task: dict[str, list[int]] = defaultdict(list)
    success_all: list[int] = []
    fail_all: list[int] = []
    for idx in indices.tolist():
        task = str(bank["task"][idx])
        if bank["success"][idx] > 0.5:
            success_by_task[task].append(idx)
            success_all.append(idx)
        else:
            fail_by_task[task].append(idx)
            fail_all.append(idx)
    pairs: list[tuple[int, int]] = []
    for task in sorted(set(list(success_by_task.keys()) + list(fail_by_task.keys()))):
        pos_pool = success_by_task.get(task, [])
        neg_pool = fail_by_task.get(task, [])
        if not pos_pool or not neg_pool:
            if same_task_only:
                continue
            pos_pool = pos_pool or success_all
            neg_pool = neg_pool or fail_all
        if not pos_pool or not neg_pool:
            continue
        for neg_idx in neg_pool:
            pairs.append((rng.choice(pos_pool), neg_idx))
    if not pairs:
        raise ValueError("no pair ids built")
    return pairs


class JointFutureActionPolicy(nn.Module):
    def __init__(self, future_dim: int, action_dim: int, task_dim: int, hidden_dim: int, z_dim: int, dropout: float):
        super().__init__()
        future_in = future_dim * 3 + task_dim + 1
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(future_in),
            nn.Linear(future_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.action_pool = nn.Linear(hidden_dim * 2, z_dim)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.action_decoder = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * 16),
        )
        self.success_head = nn.Sequential(
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_future(self, base_summary: torch.Tensor, target_summary: torch.Tensor, target_delta: torch.Tensor, task_vec: torch.Tensor, subtask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_summary, target_summary, target_delta, task_vec, subtask], dim=-1)
        return self.future_encoder(x)

    def encode_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        h = self.action_encoder(action_chunk.transpose(1, 2))
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.action_pool(pooled)

    def fused(self, z_f: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_f, z_a], dim=-1)

    def forward(self, base_summary: torch.Tensor, target_summary: torch.Tensor, target_delta: torch.Tensor, action_chunk: torch.Tensor, task_vec: torch.Tensor, subtask: torch.Tensor) -> dict[str, torch.Tensor]:
        z_f = self.encode_future(base_summary, target_summary, target_delta, task_vec, subtask)
        z_a = self.encode_action(action_chunk)
        fused = self.fused(z_f, z_a)
        pred_action = self.action_decoder(fused).reshape(action_chunk.shape[0], 16, action_chunk.shape[-1])
        return {
            "z_f": z_f,
            "z_a": z_a,
            "energy": self.energy_head(fused).squeeze(-1),
            "success_logit": self.success_head(fused).squeeze(-1),
            "pred_action": pred_action,
        }


def evaluate(model: JointFutureActionPolicy, bank: dict[str, np.ndarray], eval_idx: np.ndarray, eval_pairs: list[tuple[int, int]], device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    all_energy, all_prob, all_y, action_losses = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = gather(bank, ids, device)
            out = model(
                batch["base_summary"],
                batch["target_summary"],
                batch["target_delta"],
                batch["action_chunk"],
                batch["task_vec"],
                batch["subtask"],
            )
            all_energy.append(out["energy"].cpu())
            all_prob.append(torch.sigmoid(out["success_logit"]).cpu())
            all_y.append(batch["success"].cpu())
            action_losses.append(F.smooth_l1_loss(out["pred_action"], batch["action_chunk"]).cpu())
    energy = torch.cat(all_energy).numpy()
    prob = torch.cat(all_prob).numpy()
    y = torch.cat(all_y).numpy()
    pred = (prob >= 0.5).astype(np.float32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    success_mask = y > 0.5
    fail_mask = ~success_mask
    auc = 0.0
    if success_mask.any() and fail_mask.any():
        pos = energy[fail_mask]
        neg = energy[success_mask]
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    pair_acc, pair_margin = [], []
    energy_map = {int(idx): float(energy[i]) for i, idx in enumerate(eval_idx.tolist())}
    for pos_idx, neg_idx in eval_pairs:
        if pos_idx in energy_map and neg_idx in energy_map:
            e_pos = energy_map[pos_idx]
            e_neg = energy_map[neg_idx]
            pair_acc.append(float(e_pos < e_neg))
            pair_margin.append(float(e_neg - e_pos))
    return {
        "success_acc": float((pred == y).mean()),
        "success_f1": float(f1),
        "energy_auc_fail": float(auc),
        "pair_acc": float(np.mean(pair_acc)) if pair_acc else 0.0,
        "pair_margin": float(np.mean(pair_margin)) if pair_margin else 0.0,
        "action_l1": float(torch.stack(action_losses).mean().item()) if action_losses else 0.0,
        "energy_success_mean": float(energy[success_mask].mean()) if success_mask.any() else 0.0,
        "energy_fail_mean": float(energy[fail_mask].mean()) if fail_mask.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal unified joint future-action model with energy and policy heads.")
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
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--success-weight", type=float, default=0.5)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    memory = load_memory(args.memory_npz)
    bank = build_feature_bank(memory, args.task_dim)
    train_idx, val_idx = split_by_sequence(memory, args.val_ratio, args.seed)
    train_pairs = build_pair_ids(bank, train_idx, args.same_task_only, args.seed)
    val_pairs = build_pair_ids(bank, val_idx, args.same_task_only, args.seed + 1)

    future_dim = int(bank["base_summary"].shape[1])
    action_dim = int(bank["action_chunk"].shape[2])
    model = JointFutureActionPolicy(future_dim, action_dim, args.task_dim, args.hidden_dim, args.z_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best = None
    best_state = None
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        pair_ids = rng.integers(0, len(train_pairs), size=min(args.batch_size, len(train_pairs)))
        pos_idx = np.asarray([train_pairs[i][0] for i in pair_ids], dtype=np.int64)
        neg_idx = np.asarray([train_pairs[i][1] for i in pair_ids], dtype=np.int64)
        pos = gather(bank, pos_idx, device)
        neg = gather(bank, neg_idx, device)

        pos_out = model(pos["base_summary"], pos["target_summary"], pos["target_delta"], pos["action_chunk"], pos["task_vec"], pos["subtask"])
        neg_out = model(neg["base_summary"], neg["target_summary"], neg["target_delta"], neg["action_chunk"], neg["task_vec"], neg["subtask"])

        rank_loss = F.relu(args.margin + pos_out["energy"] - neg_out["energy"]).mean()
        success_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pos_out["success_logit"], pos["success"])
            + F.binary_cross_entropy_with_logits(neg_out["success_logit"], neg["success"])
        )
        policy_loss = F.smooth_l1_loss(pos_out["pred_action"], pos["action_chunk"])
        align_loss = (1.0 - F.cosine_similarity(pos_out["z_f"], pos_out["z_a"], dim=-1)).mean()
        loss = (
            args.energy_weight * rank_loss
            + args.success_weight * success_loss
            + args.policy_weight * policy_loss
            + args.align_weight * align_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == args.steps:
            metrics = evaluate(model, bank, val_idx, val_pairs, device, args.eval_batch_size)
            metrics.update(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "rank_loss": float(rank_loss.item()),
                    "success_loss": float(success_loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "align_loss": float(align_loss.item()),
                }
            )
            history.append(metrics)
            score = metrics["success_f1"] + metrics["pair_acc"] - metrics["action_l1"]
            if best is None or score > best["score"]:
                best = {**metrics, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best is not None and best_state is not None
    ckpt_path = args.output_dir / "joint_future_action_policy.pt"
    torch.save(
        {
            "model_state": best_state,
            "future_dim": future_dim,
            "action_dim": action_dim,
            "task_dim": int(args.task_dim),
            "hidden_dim": int(args.hidden_dim),
            "z_dim": int(args.z_dim),
            "dropout": float(args.dropout),
        },
        ckpt_path,
    )
    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_train_pairs": int(len(train_pairs)),
        "num_val_pairs": int(len(val_pairs)),
        "same_task_only": bool(args.same_task_only),
        "best": best,
        "history": history,
        "output_ckpt": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
