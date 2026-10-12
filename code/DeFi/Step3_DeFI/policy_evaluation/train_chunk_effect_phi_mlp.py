from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_by_sequence(seq: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_ratio)))
    val_seq = set(int(x) for x in uniq[:n_val].tolist())
    val = np.asarray([i for i, s in enumerate(seq.tolist()) if int(s) in val_seq], dtype=np.int64)
    train = np.asarray([i for i, s in enumerate(seq.tolist()) if int(s) not in val_seq], dtype=np.int64)
    if train.size == 0 or val.size == 0:
        ids = np.arange(seq.shape[0])
        rng.shuffle(ids)
        n_val = max(1, int(round(ids.size * val_ratio)))
        return ids[n_val:], ids[:n_val]
    return train, val


class ChunkEffectPhiMLP(nn.Module):
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
        in_dim = state_dim + summary_dim + chunk_len * action_dim + task_dim + meta_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, summary_dim)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_start: torch.Tensor,
        base_summary: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                state_start,
                base_summary,
                action_chunk.reshape(action_chunk.shape[0], -1),
                task_vec,
                meta,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        return {
            "delta": self.delta_head(h),
            "success_logit": self.success_head(h).squeeze(-1),
        }


def load_memory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    required = [
        "state_start",
        "base_summary_exec",
        "target_delta_exec",
        "action_chunk",
        "task_vec",
        "success",
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


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_start": torch.from_numpy(bank["state_start"][ids]).to(device),
        "base_summary": torch.from_numpy(bank["base_summary"][ids]).to(device),
        "target_delta": torch.from_numpy(bank["target_delta"][ids]).to(device),
        "action_chunk": torch.from_numpy(bank["action_chunk"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "meta": torch.from_numpy(bank["meta"][ids]).to(device),
        "success": torch.from_numpy(bank["success"][ids]).to(device),
    }


def evaluate(model: ChunkEffectPhiMLP, bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    losses = []
    delta_l1 = []
    delta_l2 = []
    cos = []
    succ_ok = []
    with torch.no_grad():
        for start in range(0, ids.size, batch_size):
            batch = gather(bank, ids[start : start + batch_size], device)
            out = model(batch["state_start"], batch["base_summary"], batch["action_chunk"], batch["task_vec"], batch["meta"])
            pred = out["delta"]
            target = batch["target_delta"]
            reg = F.smooth_l1_loss(pred, target)
            bce = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
            losses.append((reg + 0.1 * bce).cpu())
            delta_l1.append(F.l1_loss(pred, target).cpu())
            delta_l2.append(torch.norm(pred - target, dim=-1).mean().cpu())
            cos.append(F.cosine_similarity(pred, target, dim=-1).mean().cpu())
            succ_ok.append(((out["success_logit"].sigmoid() >= 0.5) == (batch["success"] > 0.5)).float().mean().cpu())
    return {
        "loss": float(torch.stack(losses).mean().item()),
        "delta_l1": float(torch.stack(delta_l1).mean().item()),
        "delta_l2": float(torch.stack(delta_l2).mean().item()),
        "delta_cos": float(torch.stack(cos).mean().item()),
        "success_acc": float(torch.stack(succ_ok).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pooled chunk-effect predictor P_phi for fast coupling.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--best-metric", choices=["loss", "delta_l1", "delta_l2", "neg_delta_cos"], default="delta_l1")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    memory = load_memory(args.memory_npz)
    bank = {
        "state_start": np.asarray(memory["state_start"], dtype=np.float32),
        "base_summary": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "target_delta": np.asarray(memory["target_delta_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "task_vec": np.asarray(memory["task_vec"], dtype=np.float32),
        "meta": build_meta(memory),
        "success": np.asarray(memory["success"], dtype=np.float32).reshape(-1),
        "sequence_index": np.asarray(memory["sequence_index"], dtype=np.int32).reshape(-1),
    }
    for key, value in bank.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite values in {key}")

    train_ids, val_ids = split_by_sequence(bank["sequence_index"], args.val_ratio, args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = ChunkEffectPhiMLP(
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best = None
    best_path = args.output_dir / "chunk_effect_phi_mlp.pt"
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.choice(train_ids, size=min(args.batch_size, train_ids.size), replace=train_ids.size < args.batch_size)
        batch = gather(bank, ids, device)
        out = model(batch["state_start"], batch["base_summary"], batch["action_chunk"], batch["task_vec"], batch["meta"])
        reg_loss = F.smooth_l1_loss(out["delta"], batch["target_delta"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        loss = reg_loss + 0.1 * success_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            metrics = evaluate(model, bank, val_ids, device, args.eval_batch_size)
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_delta_l1": float(F.l1_loss(out["delta"], batch["target_delta"]).item()),
                **metrics,
            }
            record["neg_delta_cos"] = -float(record["delta_cos"])
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if best is None or record[args.best_metric] < best[args.best_metric]:
                best = record
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "state_dim": int(bank["state_start"].shape[-1]),
                        "summary_dim": int(bank["base_summary"].shape[-1]),
                        "action_dim": int(bank["action_chunk"].shape[-1]),
                        "chunk_len": int(bank["action_chunk"].shape[-2]),
                        "task_dim": int(bank["task_vec"].shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "best_metric": args.best_metric,
                        "best": best,
                        "memory_npz": str(args.memory_npz),
                    },
                    best_path,
                )

    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(train_ids.size),
        "num_val": int(val_ids.size),
        "steps": int(args.steps),
        "best_metric": args.best_metric,
        "best": best,
        "output_ckpt": str(best_path),
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
