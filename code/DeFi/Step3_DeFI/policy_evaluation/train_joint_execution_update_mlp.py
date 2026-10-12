from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from policy_evaluation.train_chunk_effect_phi_mlp import split_by_sequence
except ModuleNotFoundError:
    from train_chunk_effect_phi_mlp import split_by_sequence


class JointExecutionUpdateMLP(nn.Module):
    """Proxy joint belief transition: U(h_t, s_t, a_t, s_{t+1}) -> delta h and suffix delta.

    In the collected CALVIN data we do not have supervised z_f/z_a transitions directly, so
    base/target summaries stand in for the long-horizon belief and action_chunk supervises
    the short-horizon suffix correction.
    """
    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        action_dim: int,
        chunk_len: int,
        task_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.task_dim = int(task_dim)
        meta_dim = 3
        in_dim = state_dim * 2 + summary_dim * 2 + chunk_len * action_dim + task_dim + meta_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_delta_head = nn.Linear(hidden_dim, summary_dim)
        self.action_delta_head = nn.Linear(hidden_dim, chunk_len * action_dim)
        self.error_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_start: torch.Tensor,
        state_end: torch.Tensor,
        base_summary: torch.Tensor,
        target_summary: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                state_start,
                state_end,
                base_summary,
                target_summary,
                action_chunk.reshape(action_chunk.shape[0], -1),
                task_vec,
                meta,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        return {
            "future_delta": self.future_delta_head(h),
            "action_delta": self.action_delta_head(h).reshape(x.shape[0], self.chunk_len, self.action_dim),
            "error": F.softplus(self.error_head(h).squeeze(-1)),
            "gate_logit": self.gate_head(h).squeeze(-1),
            "success_logit": self.success_head(h).squeeze(-1),
        }


# Naming alias for the paper formulation. Kept as an alias so old checkpoints/scripts
# using JointExecutionUpdateMLP remain load-compatible.
JointBeliefTransitionMLP = JointExecutionUpdateMLP


def load_memory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    required = [
        "state_start",
        "state_end",
        "base_summary_exec",
        "target_summary_exec",
        "target_delta_exec",
        "action_chunk",
        "success",
        "task_vec",
        "sequence_index",
        "subtask_index",
        "cut_idx",
        "remaining_steps",
        "full_chunk_len",
    ]
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"memory missing keys: {missing}")
    return out


def build_meta(memory: dict[str, np.ndarray]) -> np.ndarray:
    subtask = np.asarray(memory["subtask_index"], dtype=np.float32).reshape(-1, 1) / 5.0
    cut = np.asarray(memory["cut_idx"], dtype=np.float32).reshape(-1, 1)
    full = np.asarray(memory["full_chunk_len"], dtype=np.float32).reshape(-1, 1)
    remain = np.asarray(memory["remaining_steps"], dtype=np.float32).reshape(-1, 1)
    cut_norm = cut / np.maximum(full - 1.0, 1.0)
    remain_norm = remain / np.maximum(full, 1.0)
    return np.concatenate([subtask, cut_norm, remain_norm], axis=-1).astype(np.float32)


def build_action_delta_targets(memory: dict[str, np.ndarray]) -> np.ndarray:
    if "action_delta_target" in memory:
        return np.asarray(memory["action_delta_target"], dtype=np.float32)
    action = np.asarray(memory["action_chunk"], dtype=np.float32)
    success = np.asarray(memory["success"], dtype=np.float32).reshape(-1)
    task = np.asarray(memory.get("task", np.asarray(["unknown"] * len(success), dtype=object)), dtype=object)
    subtask = np.asarray(memory["subtask_index"], dtype=np.int32).reshape(-1)
    out = np.zeros_like(action, dtype=np.float32)
    for i in range(action.shape[0]):
        if success[i] > 0.5:
            continue
        candidates = np.where((success > 0.5) & (task == task[i]) & (subtask == subtask[i]))[0]
        if candidates.size == 0:
            continue
        target_delta = np.asarray(memory["target_delta_exec"][i], dtype=np.float32).reshape(-1)
        cand_delta = np.asarray(memory["target_delta_exec"][candidates], dtype=np.float32)
        d = np.linalg.norm(cand_delta - target_delta.reshape(1, -1), axis=1)
        j = int(candidates[int(np.argmin(d))])
        out[i] = action[j] - action[i]
    return out


def make_bank(memory: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_start = np.asarray(memory["state_start"], dtype=np.float32)
    state_end = np.asarray(memory["state_end"], dtype=np.float32)
    base = np.asarray(memory["base_summary_exec"], dtype=np.float32)
    target = np.asarray(memory["target_summary_exec"], dtype=np.float32)
    target_delta = np.asarray(memory["target_delta_exec"], dtype=np.float32)
    action = np.asarray(memory["action_chunk"], dtype=np.float32)
    success = np.asarray(memory["success"], dtype=np.float32).reshape(-1)
    state_motion = np.linalg.norm(state_end - state_start, axis=1)
    target_motion = np.linalg.norm(target_delta, axis=1)
    error_target = (
        np.linalg.norm(target_delta, axis=1) / max(float(np.median(target_motion[target_motion > 0])) if np.any(target_motion > 0) else 1.0, 1e-6)
        + (1.0 - success)
        + state_motion / max(float(np.median(state_motion[state_motion > 0])) if np.any(state_motion > 0) else 1.0, 1e-6)
    ).astype(np.float32)
    action_delta = build_action_delta_targets(memory)
    action_delta_norm = np.linalg.norm(action_delta.reshape(action_delta.shape[0], -1), axis=1)
    norm_scale = max(float(np.percentile(action_delta_norm[action_delta_norm > 0], 90)) if np.any(action_delta_norm > 0) else 1.0, 1e-6)
    gate_target = np.clip(action_delta_norm / norm_scale, 0.0, 1.0).astype(np.float32)
    return {
        "state_start": state_start,
        "state_end": state_end,
        "base_summary": base,
        "target_summary": target,
        "future_delta": target_delta,
        "action_chunk": action,
        "action_delta": action_delta,
        "error": error_target,
        "gate": gate_target,
        "success": success,
        "task_vec": np.asarray(memory["task_vec"], dtype=np.float32),
        "meta": build_meta(memory),
        "sequence_index": np.asarray(memory["sequence_index"], dtype=np.int32).reshape(-1),
    }


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value[ids]).to(device) for key, value in bank.items() if value.dtype.kind in "f"}


def eval_model(model: JointExecutionUpdateMLP, bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device, batch_size: int, gate_loss_weight: float) -> dict[str, Any]:
    model.eval()
    vals: dict[str, list[torch.Tensor]] = {"loss": [], "future_l1": [], "action_l1": [], "error_l1": [], "gate_l1": [], "success_acc": []}
    with torch.no_grad():
        for start in range(0, ids.size, batch_size):
            batch = gather(bank, ids[start : start + batch_size], device)
            out = model(
                batch["state_start"],
                batch["state_end"],
                batch["base_summary"],
                batch["target_summary"],
                batch["action_chunk"],
                batch["task_vec"],
                batch["meta"],
            )
            future_loss = F.smooth_l1_loss(out["future_delta"], batch["future_delta"])
            action_loss = F.smooth_l1_loss(out["action_delta"], batch["action_delta"])
            error_loss = F.smooth_l1_loss(out["error"], batch["error"])
            gate_loss = F.binary_cross_entropy_with_logits(out["gate_logit"], batch["gate"])
            success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
            loss = future_loss + 0.5 * action_loss + 0.1 * error_loss + float(gate_loss_weight) * gate_loss + 0.1 * success_loss
            vals["loss"].append(loss.cpu())
            vals["future_l1"].append(F.l1_loss(out["future_delta"], batch["future_delta"]).cpu())
            vals["action_l1"].append(F.l1_loss(out["action_delta"], batch["action_delta"]).cpu())
            vals["error_l1"].append(F.l1_loss(out["error"], batch["error"]).cpu())
            vals["gate_l1"].append(F.l1_loss(out["gate_logit"].sigmoid(), batch["gate"]).cpu())
            vals["success_acc"].append(((out["success_logit"].sigmoid() >= 0.5) == (batch["success"] > 0.5)).float().mean().cpu())
    return {key: float(torch.stack(value).mean().item()) for key, value in vals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a unified Joint Execution Update proxy U(h,s,a,s') -> (delta_h,e).")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    memory = load_memory(args.memory_npz)
    bank = make_bank(memory)
    for key, value in bank.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite values in {key}")

    train_ids, val_ids = split_by_sequence(bank["sequence_index"], args.val_ratio, args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = JointExecutionUpdateMLP(
        state_dim=bank["state_start"].shape[-1],
        summary_dim=bank["base_summary"].shape[-1],
        action_dim=bank["action_chunk"].shape[-1],
        chunk_len=bank["action_chunk"].shape[-2],
        task_dim=bank["task_vec"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best = None
    best_path = args.output_dir / "joint_execution_update_mlp.pt"
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.choice(train_ids, size=min(args.batch_size, train_ids.size), replace=train_ids.size < args.batch_size)
        batch = gather(bank, ids, device)
        out = model(
            batch["state_start"],
            batch["state_end"],
            batch["base_summary"],
            batch["target_summary"],
            batch["action_chunk"],
            batch["task_vec"],
            batch["meta"],
        )
        future_loss = F.smooth_l1_loss(out["future_delta"], batch["future_delta"])
        action_loss = F.smooth_l1_loss(out["action_delta"], batch["action_delta"])
        error_loss = F.smooth_l1_loss(out["error"], batch["error"])
        gate_loss = F.binary_cross_entropy_with_logits(out["gate_logit"], batch["gate"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        loss = future_loss + 0.5 * action_loss + 0.1 * error_loss + float(args.gate_loss_weight) * gate_loss + 0.1 * success_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            metrics = eval_model(model, bank, val_ids, device, args.eval_batch_size, args.gate_loss_weight)
            record = {"step": step, "train_loss": float(loss.item()), **metrics}
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if best is None or record["loss"] < best["loss"]:
                best = record
                torch.save(
                    {
                        "model_type": "joint_execution_update_mlp",
                        "state_dim": int(bank["state_start"].shape[-1]),
                        "summary_dim": int(bank["base_summary"].shape[-1]),
                        "action_dim": int(bank["action_chunk"].shape[-1]),
                        "chunk_len": int(bank["action_chunk"].shape[-2]),
                        "task_dim": int(bank["task_vec"].shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "has_learned_gate": True,
                        "gate_loss_weight": float(args.gate_loss_weight),
                        "model_state": model.state_dict(),
                        "best_metrics": best,
                    },
                    best_path,
                )

    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(train_ids.size),
        "num_val": int(val_ids.size),
        "steps": int(args.steps),
        "best": best,
        "output_ckpt": str(best_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
