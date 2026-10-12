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


def build_pair_index(memory: dict[str, np.ndarray], indices: np.ndarray, same_task_only: bool, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    success_by_task: dict[str, list[int]] = defaultdict(list)
    failure_by_task: dict[str, list[int]] = defaultdict(list)
    success_all: list[int] = []
    failure_all: list[int] = []
    for idx in indices.tolist():
        task = str(memory["task"][idx])
        success = bool(memory["success"][idx] > 0.5)
        if success:
            success_by_task[task].append(idx)
            success_all.append(idx)
        else:
            failure_by_task[task].append(idx)
            failure_all.append(idx)
    pairs: list[tuple[int, int]] = []
    task_names = sorted(set(list(success_by_task.keys()) + list(failure_by_task.keys())))
    for task in task_names:
        pos_pool = success_by_task.get(task, [])
        neg_pool = failure_by_task.get(task, [])
        if not pos_pool or not neg_pool:
            if same_task_only:
                continue
            pos_pool = pos_pool or success_all
            neg_pool = neg_pool or failure_all
        if not pos_pool or not neg_pool:
            continue
        for neg_idx in neg_pool:
            pos_idx = rng.choice(pos_pool)
            pairs.append((pos_idx, neg_idx))
    if not pairs:
        raise ValueError("no success/failure pairs built from memory")
    return pairs


def build_success_pools(memory: dict[str, np.ndarray], indices: np.ndarray) -> tuple[dict[str, list[int]], list[int]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    all_success: list[int] = []
    for idx in indices.tolist():
        if bool(memory["success"][idx] > 0.5):
            task = str(memory["task"][idx])
            by_task[task].append(idx)
            all_success.append(idx)
    return by_task, all_success


class JointFutureActionEnergy(nn.Module):
    def __init__(self, action_dim: int, future_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(future_dim * 3 + task_dim + 1),
            nn.Linear(future_dim * 3 + task_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        base_summary: torch.Tensor,
        target_summary: torch.Tensor,
        target_delta: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.action_encoder(action_chunk.transpose(1, 2))
        a_mean = h.mean(dim=-1)
        a_max = h.amax(dim=-1)
        f = self.future_encoder(torch.cat([base_summary, target_summary, target_delta, task_vec, subtask], dim=-1))
        return self.energy_head(torch.cat([a_mean, a_max, f], dim=-1)).squeeze(-1)


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
    }


def gather_batch(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_summary": torch.from_numpy(bank["target_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "action_chunk": torch.from_numpy(bank["action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "subtask": torch.from_numpy(bank["subtask"][ids]).to(device),
    }


def gather_mismatched_action_batch(
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
    batch["action_chunk"] = torch.from_numpy(bank["action_chunk"][mismatch_ids]).to(device)
    return batch


def compute_metrics(
    model: JointFutureActionEnergy,
    bank: dict[str, np.ndarray],
    eval_idx: np.ndarray,
    eval_pairs: list[tuple[int, int]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    energies = np.zeros(len(eval_idx), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = gather_batch(bank, ids, device)
            energy = model(**batch).detach().cpu().numpy()
            energies[start : start + len(ids)] = energy
    success = bank["success"][eval_idx] > 0.5
    fail = ~success
    auc = 0.0
    if success.any() and fail.any():
        pos = energies[fail]
        neg = energies[success]
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    pair_acc_vals = []
    pair_margin_vals = []
    eval_pos = np.asarray([p[0] for p in eval_pairs], dtype=np.int64)
    eval_neg = np.asarray([p[1] for p in eval_pairs], dtype=np.int64)
    idx_to_energy = {int(idx): float(energies[i]) for i, idx in enumerate(eval_idx.tolist())}
    for pos_idx, neg_idx in zip(eval_pos.tolist(), eval_neg.tolist()):
        if pos_idx in idx_to_energy and neg_idx in idx_to_energy:
            e_pos = idx_to_energy[pos_idx]
            e_neg = idx_to_energy[neg_idx]
            pair_acc_vals.append(float(e_pos < e_neg))
            pair_margin_vals.append(float(e_neg - e_pos))
    order = np.argsort(-energies)
    top = {}
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(round(len(order) * frac)))
        ids = order[:k]
        top[f"top_{int(frac*100)}pct_fail_rate"] = float(fail[ids].mean())
    return {
        "energy_success_mean": float(energies[success].mean()) if success.any() else 0.0,
        "energy_fail_mean": float(energies[fail].mean()) if fail.any() else 0.0,
        "auc_fail_energy": auc,
        "pair_acc": float(np.mean(pair_acc_vals)) if pair_acc_vals else 0.0,
        "pair_margin": float(np.mean(pair_margin_vals)) if pair_margin_vals else 0.0,
        **top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint future-action energy E(s, z_f, z_a) from pair memory.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lambda-energy-reg", type=float, default=0.05)
    parser.add_argument("--lambda-fail-margin", type=float, default=0.05)
    parser.add_argument("--fail-margin", type=float, default=1.0)
    parser.add_argument("--lambda-hard-negative", type=float, default=0.5)
    parser.add_argument("--hard-negative-margin", type=float, default=0.5)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    memory = load_memory(args.memory_npz)
    bank = make_feature_bank(memory, args.task_dim)
    train_idx, val_idx = split_by_sequence(memory, args.val_ratio, args.seed)
    train_pairs = build_pair_index(memory, train_idx, args.same_task_only, args.seed)
    try:
        val_pairs = build_pair_index(memory, val_idx, args.same_task_only, args.seed + 17)
    except ValueError:
        val_pairs = build_pair_index(memory, val_idx, False, args.seed + 17)
    train_success_by_task, train_success_all = build_success_pools(memory, train_idx)

    model = JointFutureActionEnergy(
        action_dim=int(bank["action_chunk"].shape[-1]),
        future_dim=int(bank["base_summary"].shape[-1]),
        task_dim=int(args.task_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history = []
    best = None
    best_state = None

    pair_arr = np.asarray(train_pairs, dtype=np.int64)
    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.integers(0, len(pair_arr), size=max(1, args.batch_size))
        pos_idx = pair_arr[ids, 0]
        neg_idx = pair_arr[ids, 1]
        pos_batch = gather_batch(bank, pos_idx, device)
        neg_batch = gather_batch(bank, neg_idx, device)
        mismatch_batch = gather_mismatched_action_batch(
            bank,
            memory,
            pos_idx,
            train_success_by_task,
            train_success_all,
            bool(args.same_task_only),
            random.Random(args.seed + step),
            device,
        )
        e_pos = model(**pos_batch)
        e_neg = model(**neg_batch)
        e_mismatch = model(**mismatch_batch)
        rank_loss = F.relu(float(args.margin) + e_pos - e_neg).mean()
        hard_negative_loss = F.relu(float(args.hard_negative_margin) + e_pos - e_mismatch).mean()
        compact_loss = 0.5 * (e_pos.pow(2).mean() + F.relu(float(args.fail_margin) - e_neg).pow(2).mean())
        fail_margin_loss = F.relu(float(args.fail_margin) - e_neg).mean()
        loss = (
            rank_loss
            + float(args.lambda_hard_negative) * hard_negative_loss
            + float(args.lambda_energy_reg) * compact_loss
            + float(args.lambda_fail_margin) * fail_margin_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
                "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
                "compact_loss": float(compact_loss.detach().cpu()),
                "fail_margin_loss": float(fail_margin_loss.detach().cpu()),
                **compute_metrics(model, bank, val_idx, val_pairs, device, args.eval_batch_size),
            }
            history.append(record)
            score = record["auc_fail_energy"] + record["pair_acc"] + 0.1 * record["top_10pct_fail_rate"]
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.output_dir / "joint_future_action_energy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "action_dim": int(bank["action_chunk"].shape[-1]),
            "future_dim": int(bank["base_summary"].shape[-1]),
            "task_dim": int(args.task_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "margin": float(args.margin),
            "fail_margin": float(args.fail_margin),
            "hard_negative_margin": float(args.hard_negative_margin),
            "lambda_hard_negative": float(args.lambda_hard_negative),
        },
        ckpt_path,
    )

    summary = {
        "memory_npz": str(args.memory_npz),
        "output_dir": str(args.output_dir),
        "num_items": int(len(bank["success"])),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_train_pairs": int(len(train_pairs)),
        "num_val_pairs": int(len(val_pairs)),
        "best": best,
        "history": history,
        "ckpt": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
