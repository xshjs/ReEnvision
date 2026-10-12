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
    from policy_evaluation.train_joint_future_action_energy import hashed_task, load_memory, split_by_sequence
except ModuleNotFoundError:
    from train_joint_future_action_energy import hashed_task, load_memory, split_by_sequence


class DeltaActionRepairMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        future_dim: int,
        action_dim: int,
        chunk_len: int,
        task_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.future_dim = int(future_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.task_dim = int(task_dim)
        in_dim = state_dim + future_dim * 3 + chunk_len * action_dim + task_dim + 1
        out_dim = chunk_len * action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        state_start: torch.Tensor,
        base_summary: torch.Tensor,
        target_summary: torch.Tensor,
        target_delta: torch.Tensor,
        current_action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                state_start,
                base_summary,
                target_summary,
                target_delta,
                current_action_chunk.reshape(current_action_chunk.shape[0], -1),
                task_vec,
                subtask,
            ],
            dim=-1,
        )
        y = self.net(x)
        return y.reshape(x.shape[0], self.chunk_len, self.action_dim)


def build_success_index(memory: dict[str, np.ndarray], ids: np.ndarray) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = {}
    for idx in ids.tolist():
        if bool(memory["success"][idx] > 0.5):
            key = (str(memory["task"][idx]), int(memory["subtask_index"][idx]))
            out.setdefault(key, []).append(int(idx))
    return out


def _nearest_success_target(
    memory: dict[str, np.ndarray],
    idx: int,
    pool: list[int],
    same_index_ok: bool,
) -> int | None:
    if not pool:
        return None
    base_target = np.asarray(memory["target_summary_exec"][idx], dtype=np.float32).reshape(-1)
    base_state = np.asarray(memory["state_start"][idx], dtype=np.float32).reshape(-1)
    best = None
    best_score = None
    for cand in pool:
        if not same_index_ok and int(cand) == int(idx):
            continue
        cand_target = np.asarray(memory["target_summary_exec"][cand], dtype=np.float32).reshape(-1)
        cand_state = np.asarray(memory["state_start"][cand], dtype=np.float32).reshape(-1)
        score = float(np.linalg.norm(base_target - cand_target) + 0.25 * np.linalg.norm(base_state - cand_state))
        if best is None or score < best_score:
            best = int(cand)
            best_score = score
    return best


def make_feature_bank(memory: dict[str, np.ndarray], task_dim: int, source_indices: np.ndarray, target_indices: np.ndarray) -> dict[str, np.ndarray]:
    task_vec = np.stack([hashed_task(str(memory["task"][i]), task_dim) for i in source_indices.tolist()]).astype(np.float32)
    subtask = (np.asarray(memory["subtask_index"][source_indices], dtype=np.float32).reshape(-1, 1) / 5.0).astype(np.float32)
    current_action = np.asarray(memory["action_chunk"][source_indices], dtype=np.float32)
    target_action = np.asarray(memory["action_chunk"][target_indices], dtype=np.float32)
    delta_action = (target_action - current_action).astype(np.float32)
    return {
        "state_start": np.asarray(memory["state_start"][source_indices], dtype=np.float32),
        "base_summary": np.asarray(memory["base_summary_exec"][source_indices], dtype=np.float32),
        "target_summary": np.asarray(memory["target_summary_exec"][source_indices], dtype=np.float32),
        "target_delta": np.asarray(memory["target_delta_exec"][source_indices], dtype=np.float32),
        "current_action_chunk": current_action,
        "target_action_chunk": target_action,
        "delta_action_chunk": delta_action,
        "task_vec": task_vec,
        "subtask": subtask,
        "sequence_index": np.asarray(memory["sequence_index"][source_indices], dtype=np.int32),
        "task": np.asarray(memory["task"][source_indices], dtype=object),
    }


def build_dataset(
    memory: dict[str, np.ndarray],
    indices: np.ndarray,
    task_dim: int,
    source_mode: str,
) -> dict[str, np.ndarray]:
    success_index = build_success_index(memory, indices)
    source_ids: list[int] = []
    target_ids: list[int] = []
    for idx in indices.tolist():
        is_success = bool(memory["success"][idx] > 0.5)
        if source_mode == "failure_only" and is_success:
            continue
        if source_mode == "success_only" and not is_success:
            continue
        key = (str(memory["task"][idx]), int(memory["subtask_index"][idx]))
        target = _nearest_success_target(memory, int(idx), success_index.get(key, []), same_index_ok=is_success)
        if target is None:
            continue
        source_ids.append(int(idx))
        target_ids.append(int(target))
    if not source_ids:
        raise ValueError("no delta-action repair pairs built from memory")
    return make_feature_bank(
        memory,
        task_dim=int(task_dim),
        source_indices=np.asarray(source_ids, dtype=np.int64),
        target_indices=np.asarray(target_ids, dtype=np.int64),
    )


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_start": torch.from_numpy(bank["state_start"][ids]).to(device),
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_summary": torch.from_numpy(bank["target_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "current_action_chunk": torch.from_numpy(bank["current_action_chunk"][ids]).to(device),
        "delta_action_chunk": torch.from_numpy(bank["delta_action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "subtask": torch.from_numpy(bank["subtask"][ids]).to(device),
    }


def make_pair_sampling_probs(bank: dict[str, np.ndarray], mode: str) -> np.ndarray | None:
    if str(mode) == "none":
        return None
    if str(mode) != "task_subtask":
        raise ValueError(f"unknown balance sampling mode: {mode}")
    keys = [
        (str(task), int(subtask))
        for task, subtask in zip(bank["task"].tolist(), np.asarray(bank["subtask"]).reshape(-1).tolist(), strict=True)
    ]
    counts: dict[tuple[str, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray([1.0 / float(counts[key]) for key in keys], dtype=np.float64)
    weights /= weights.sum()
    return weights


def evaluate(model: DeltaActionRepairMLP, bank: dict[str, np.ndarray], eval_idx: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    losses = []
    delta_l2s = []
    repaired_l1s = []
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = gather(bank, ids, device)
            pred_delta = model(
                batch["state_start"],
                batch["base_summary"],
                batch["target_summary"],
                batch["target_delta"],
                batch["current_action_chunk"],
                batch["task_vec"],
                batch["subtask"],
            )
            losses.append(F.smooth_l1_loss(pred_delta, batch["delta_action_chunk"]).cpu())
            delta_l2s.append(torch.norm((pred_delta - batch["delta_action_chunk"]).reshape(pred_delta.shape[0], -1), dim=-1).mean().cpu())
            repaired = batch["current_action_chunk"] + pred_delta
            target = batch["current_action_chunk"] + batch["delta_action_chunk"]
            repaired_l1s.append(F.smooth_l1_loss(repaired, target).cpu())
    return {
        "delta_l1": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "delta_l2": float(torch.stack(delta_l2s).mean().item()) if delta_l2s else 0.0,
        "repaired_action_l1": float(torch.stack(repaired_l1s).mean().item()) if repaired_l1s else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a delta-action repair head for online chunk correction.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--source-mode", choices=["failure_only", "all", "success_only"], default="failure_only")
    parser.add_argument("--balance-sampling", choices=["none", "task_subtask"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory = load_memory(args.memory_npz)
    if "state_start" not in memory:
        raise ValueError("delta-action repair memory requires state_start in memory npz")
    train_mem_idx, val_mem_idx = split_by_sequence(memory, args.val_ratio, args.seed)
    train_bank = build_dataset(memory, train_mem_idx, int(args.task_dim), str(args.source_mode))
    val_bank = build_dataset(memory, val_mem_idx, int(args.task_dim), str(args.source_mode))

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = DeltaActionRepairMLP(
        state_dim=int(train_bank["state_start"].shape[-1]),
        future_dim=int(train_bank["base_summary"].shape[-1]),
        action_dim=int(train_bank["current_action_chunk"].shape[-1]),
        chunk_len=int(train_bank["current_action_chunk"].shape[1]),
        task_dim=int(args.task_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = np.arange(len(train_bank["sequence_index"]), dtype=np.int64)
    val_idx = np.arange(len(val_bank["sequence_index"]), dtype=np.int64)
    train_sampling_probs = make_pair_sampling_probs(train_bank, args.balance_sampling)
    best_loss = float("inf")
    best_metrics = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        if train_sampling_probs is not None:
            batch_ids = np.random.choice(train_idx, size=int(args.batch_size), replace=True, p=train_sampling_probs)
        elif len(train_idx) >= args.batch_size:
            batch_ids = np.asarray(random.sample(train_idx.tolist(), args.batch_size), dtype=np.int64)
        else:
            batch_ids = np.asarray([random.choice(train_idx.tolist()) for _ in range(args.batch_size)], dtype=np.int64)
        batch = gather(train_bank, batch_ids, device)
        pred_delta = model(
            batch["state_start"],
            batch["base_summary"],
            batch["target_summary"],
            batch["target_delta"],
            batch["current_action_chunk"],
            batch["task_vec"],
            batch["subtask"],
        )
        loss = F.smooth_l1_loss(pred_delta, batch["delta_action_chunk"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(model, val_bank, val_idx, device, args.eval_batch_size)
            record = {"step": int(step), "loss": float(loss.detach().item()), **metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if metrics["delta_l1"] < best_loss:
                best_loss = float(metrics["delta_l1"])
                best_metrics = dict(record)
                torch.save(
                    {
                        "model_type": "delta_action_repair_mlp",
                        "state_dim": int(train_bank["state_start"].shape[-1]),
                        "future_dim": int(train_bank["base_summary"].shape[-1]),
                        "action_dim": int(train_bank["current_action_chunk"].shape[-1]),
                        "chunk_len": int(train_bank["current_action_chunk"].shape[1]),
                        "task_dim": int(args.task_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "source_mode": str(args.source_mode),
                        "model_state": model.state_dict(),
                        "best_metrics": best_metrics,
                    },
                    args.output_dir / "delta_action_repair_mlp.pt",
                )

    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train_pairs": int(len(train_idx)),
        "num_val_pairs": int(len(val_idx)),
        "steps": int(args.steps),
        "source_mode": str(args.source_mode),
        "balance_sampling": str(args.balance_sampling),
        "best": best_metrics,
        "output_ckpt": str(args.output_dir / "delta_action_repair_mlp.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
