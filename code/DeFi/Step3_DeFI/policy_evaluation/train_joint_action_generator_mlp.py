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


class JointActionGeneratorMLP(nn.Module):
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
        in_dim = state_dim + future_dim * 3 + task_dim + 1
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
        task_vec: torch.Tensor,
        subtask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_start, base_summary, target_summary, target_delta, task_vec, subtask], dim=-1)
        y = self.net(x)
        return y.reshape(x.shape[0], self.chunk_len, self.action_dim)


def make_feature_bank(memory: dict[str, np.ndarray], task_dim: int) -> dict[str, np.ndarray]:
    required = [
        "state_start",
        "base_summary_exec",
        "target_summary_exec",
        "target_delta_exec",
        "action_chunk",
        "task",
        "subtask_index",
        "success",
        "sequence_index",
    ]
    missing = [k for k in required if k not in memory]
    if missing:
        raise ValueError(f"action generator memory missing keys: {missing}")
    task_vec = np.stack([hashed_task(str(t), task_dim) for t in memory["task"].tolist()]).astype(np.float32)
    subtask = (np.asarray(memory["subtask_index"], dtype=np.float32).reshape(-1, 1) / 5.0).astype(np.float32)
    return {
        "state_start": np.asarray(memory["state_start"], dtype=np.float32),
        "base_summary": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "target_summary": np.asarray(memory["target_summary_exec"], dtype=np.float32),
        "target_delta": np.asarray(memory["target_delta_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "task_vec": task_vec,
        "subtask": subtask,
        "success": np.asarray(memory["success"], dtype=np.float32).reshape(-1),
        "sequence_index": np.asarray(memory["sequence_index"], dtype=np.int32),
        "task": np.asarray(memory["task"], dtype=object),
    }


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_start": torch.from_numpy(bank["state_start"][ids]).to(device),
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_summary": torch.from_numpy(bank["target_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "action_chunk": torch.from_numpy(bank["action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "subtask": torch.from_numpy(bank["subtask"][ids]).to(device),
        "success": torch.from_numpy(bank["success"][ids]).to(device),
    }


def evaluate(model: JointActionGeneratorMLP, bank: dict[str, np.ndarray], eval_idx: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    losses = []
    l2s = []
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = gather(bank, ids, device)
            pred = model(
                batch["state_start"],
                batch["base_summary"],
                batch["target_summary"],
                batch["target_delta"],
                batch["task_vec"],
                batch["subtask"],
            )
            losses.append(F.smooth_l1_loss(pred, batch["action_chunk"]).cpu())
            l2s.append(torch.norm((pred - batch["action_chunk"]).reshape(pred.shape[0], -1), dim=-1).mean().cpu())
    return {
        "action_l1": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "action_l2": float(torch.stack(l2s).mean().item()) if l2s else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP generator for z_a / action_chunk from (s, z_f, task).")
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
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory = load_memory(args.memory_npz)
    bank = make_feature_bank(memory, args.task_dim)
    if args.success_only:
        keep = np.where(bank["success"] > 0.5)[0]
        if keep.size <= 0:
            raise ValueError("no successful rows in memory for success-only action generator training")
        for key, value in list(bank.items()):
            bank[key] = value[keep]
        memory = {k: np.asarray(v)[keep] for k, v in memory.items()}
    train_idx, val_idx = split_by_sequence(memory, args.val_ratio, args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = JointActionGeneratorMLP(
        state_dim=int(bank["state_start"].shape[-1]),
        future_dim=int(bank["base_summary"].shape[-1]),
        action_dim=int(bank["action_chunk"].shape[-1]),
        chunk_len=int(bank["action_chunk"].shape[1]),
        task_dim=int(args.task_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_metrics = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        if len(train_idx) >= args.batch_size:
            batch_ids = np.asarray(random.sample(train_idx.tolist(), args.batch_size), dtype=np.int64)
        else:
            batch_ids = np.asarray([random.choice(train_idx.tolist()) for _ in range(args.batch_size)], dtype=np.int64)
        batch = gather(bank, batch_ids, device)
        pred = model(
            batch["state_start"],
            batch["base_summary"],
            batch["target_summary"],
            batch["target_delta"],
            batch["task_vec"],
            batch["subtask"],
        )
        loss = F.smooth_l1_loss(pred, batch["action_chunk"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(model, bank, val_idx, device, args.eval_batch_size)
            record = {"step": int(step), "loss": float(loss.detach().item()), **metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if metrics["action_l1"] < best_loss:
                best_loss = float(metrics["action_l1"])
                best_metrics = dict(record)
                torch.save(
                    {
                        "model_type": "action_generator_mlp",
                        "state_dim": int(bank["state_start"].shape[-1]),
                        "future_dim": int(bank["base_summary"].shape[-1]),
                        "action_dim": int(bank["action_chunk"].shape[-1]),
                        "chunk_len": int(bank["action_chunk"].shape[1]),
                        "task_dim": int(args.task_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "success_only": bool(args.success_only),
                        "model_state": model.state_dict(),
                        "best_metrics": best_metrics,
                    },
                    args.output_dir / "joint_action_generator_mlp.pt",
                )

    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "steps": int(args.steps),
        "success_only": bool(args.success_only),
        "best": best_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
