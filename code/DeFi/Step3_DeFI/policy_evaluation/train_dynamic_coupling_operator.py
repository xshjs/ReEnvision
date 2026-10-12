from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row.get("sequence_index", -1)) for row in rows if int(row.get("sequence_index", -1)) >= 0})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row.get("sequence_index", -1)) in val else train_idx).append(idx)
    return train_idx, val_idx


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def stable_task_hash(task: str, task_dim: int) -> np.ndarray:
    vec = np.zeros((int(task_dim),), dtype=np.float32)
    key = hashlib.md5(str(task).encode("utf-8")).hexdigest()
    hid = int(key[:8], 16) % int(task_dim)
    vec[hid] = 1.0
    return vec


def make_arrays(rows: list[dict[str, Any]], task_dim: int) -> dict[str, np.ndarray]:
    tasks = sorted({str(row.get("task", "unknown")) for row in rows})
    max_stage = max(float(row.get("subtask_index", 0)) for row in rows)
    max_stage = max(max_stage, 1.0)

    state = np.stack([np.asarray(row["state_t"], dtype=np.float32) for row in rows], axis=0)
    base = np.stack([np.asarray(row["base_t"], dtype=np.float32) for row in rows], axis=0)
    target = np.stack([np.asarray(row["target_t"], dtype=np.float32) for row in rows], axis=0)
    progress = np.stack([np.asarray(row["future_progress"], dtype=np.float32) for row in rows], axis=0)
    hist_actions = np.stack([np.asarray(row["history_actions"], dtype=np.float32) for row in rows], axis=0)
    hist_mask = np.stack([np.asarray(row["history_mask"], dtype=np.float32) for row in rows], axis=0)

    task_vec = np.stack([stable_task_hash(str(row.get("task", "unknown")), task_dim) for row in rows], axis=0)
    stage = np.asarray([float(row.get("subtask_index", 0)) / max_stage for row in rows], dtype=np.float32).reshape(-1, 1)
    return {
        "state": state.astype(np.float32),
        "base": base.astype(np.float32),
        "target": target.astype(np.float32),
        "progress": progress.astype(np.float32),
        "pending_effect": np.stack([np.asarray(row.get("pending_effect", row["future_progress"]), dtype=np.float32) for row in rows], axis=0).astype(np.float32),
        "observed_effect": np.stack([np.asarray(row.get("observed_effect", row["future_progress"]), dtype=np.float32) for row in rows], axis=0).astype(np.float32),
        "residual_to_target": np.stack([np.asarray(row.get("residual_to_target", row["future_progress"]), dtype=np.float32) for row in rows], axis=0).astype(np.float32),
        "residual_before": np.stack([np.asarray(row.get("residual_before", row["future_progress"]), dtype=np.float32) for row in rows], axis=0).astype(np.float32),
        "hist_actions": hist_actions.astype(np.float32),
        "hist_mask": hist_mask.astype(np.float32),
        "task_vec": task_vec.astype(np.float32),
        "stage": stage.astype(np.float32),
        "tasks": tasks,
    }


class DynamicCouplingOperator(nn.Module):
    def __init__(
        self,
        state_dim: int,
        future_dim: int,
        progress_dim: int,
        action_shape: tuple[int, int],
        task_dim: int,
        hidden_dim: int,
        history_len: int,
        dropout: float,
    ):
        super().__init__()
        self.future_dim = int(future_dim)
        self.progress_dim = int(progress_dim)
        self.history_len = int(history_len)
        action_flat_dim = int(action_shape[0] * action_shape[1])
        context_dim = int(state_dim + future_dim * 2 + task_dim + 1)
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(action_flat_dim),
            nn.Linear(action_flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.alpha_head = nn.Linear(hidden_dim * 2, 1)
        self.k_head = nn.Linear(hidden_dim * 2, progress_dim)

    def forward(
        self,
        state_t: torch.Tensor,
        base_t: torch.Tensor,
        target_t: torch.Tensor,
        hist_actions: torch.Tensor,
        hist_mask: torch.Tensor,
        task_vec: torch.Tensor,
        stage: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(torch.cat([state_t, base_t, target_t, task_vec, stage], dim=-1))
        bsz = hist_actions.shape[0]
        flat_actions = hist_actions.reshape(bsz, self.history_len, -1)
        act_h = self.action_proj(flat_actions.reshape(bsz * self.history_len, -1)).reshape(bsz, self.history_len, -1)
        ctx_expand = ctx[:, None, :].expand(-1, self.history_len, -1)
        pair_h = torch.cat([ctx_expand, act_h], dim=-1)
        alpha_logits = self.alpha_head(pair_h).squeeze(-1)
        alpha_logits = alpha_logits.masked_fill(hist_mask <= 0.0, -1e9)
        alpha = torch.softmax(alpha_logits, dim=-1)
        alpha = alpha * hist_mask
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        per_delay = self.k_head(pair_h)
        committed = (alpha[..., None] * per_delay).sum(dim=1)
        return {
            "committed": committed,
            "alpha": alpha,
            "per_delay": per_delay,
        }


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device):
    return {
        "state": torch.from_numpy(bank["state"][ids]).to(device),
        "base": torch.from_numpy(bank["base"][ids]).to(device),
        "target": torch.from_numpy(bank["target"][ids]).to(device),
        "progress": torch.from_numpy(bank["progress"][ids]).to(device),
        "pending_effect": torch.from_numpy(bank["pending_effect"][ids]).to(device),
        "observed_effect": torch.from_numpy(bank["observed_effect"][ids]).to(device),
        "residual_to_target": torch.from_numpy(bank["residual_to_target"][ids]).to(device),
        "residual_before": torch.from_numpy(bank["residual_before"][ids]).to(device),
        "hist_actions": torch.from_numpy(bank["hist_actions"][ids]).to(device),
        "hist_mask": torch.from_numpy(bank["hist_mask"][ids]).to(device),
        "task_vec": torch.from_numpy(bank["task_vec"][ids]).to(device),
        "stage": torch.from_numpy(bank["stage"][ids]).to(device),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size <= 1:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(c) else c)


def _select_target(batch: dict[str, torch.Tensor], target_key: str) -> torch.Tensor:
    if target_key not in batch:
        raise KeyError(f"unknown target_key: {target_key}")
    return batch[target_key]


def evaluate(
    model: DynamicCouplingOperator,
    bank: dict[str, np.ndarray],
    eval_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
    target_key: str,
):
    model.eval()
    losses = []
    cosines = []
    pred_norms = []
    target_norms = []
    alphas = []
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = gather(bank, ids, device)
            out = model(
                batch["state"],
                batch["base"],
                batch["target"],
                batch["hist_actions"],
                batch["hist_mask"],
                batch["task_vec"],
                batch["stage"],
            )
            pred = out["committed"]
            target = _select_target(batch, target_key)
            losses.append(F.smooth_l1_loss(pred, target).cpu())
            cosines.append(F.cosine_similarity(pred, target, dim=-1).mean().cpu())
            pred_norms.append(torch.norm(pred, dim=-1).mean().cpu())
            target_norms.append(torch.norm(target, dim=-1).mean().cpu())
            alphas.append(out["alpha"].cpu())
    alpha_cat = torch.cat(alphas, dim=0) if alphas else torch.zeros((0, 1))
    alpha_mean = alpha_cat.mean(dim=0).tolist() if alpha_cat.numel() > 0 else []
    alpha_entropy = float((-(alpha_cat.clamp_min(1e-8) * alpha_cat.clamp_min(1e-8).log()).sum(dim=-1).mean().item())) if alpha_cat.numel() > 0 else 0.0
    return {
        "loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "committed_future_cos": float(torch.stack(cosines).mean().item()) if cosines else 0.0,
        "pred_norm": float(torch.stack(pred_norms).mean().item()) if pred_norms else 0.0,
        "target_norm": float(torch.stack(target_norms).mean().item()) if target_norms else 0.0,
        "alpha_entropy": alpha_entropy,
        "alpha_mean": [float(x) for x in alpha_mean],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic coupling operator Γ_t for committed future progress.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target-key",
        choices=["progress", "pending_effect", "observed_effect", "residual_to_target", "residual_before"],
        default="progress",
        help="Which decomposition target to supervise committed effect against.",
    )
    args = parser.parse_args()

    rows = load_rows(args.rows_jsonl)
    bank = make_arrays(rows, int(args.task_dim))
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    model = DynamicCouplingOperator(
        state_dim=int(bank["state"].shape[-1]),
        future_dim=int(bank["base"].shape[-1]),
        progress_dim=int(bank[str(args.target_key)].shape[-1]),
        action_shape=tuple(int(x) for x in bank["hist_actions"].shape[-2:]),
        task_dim=int(args.task_dim),
        hidden_dim=int(args.hidden_dim),
        history_len=int(bank["hist_actions"].shape[1]),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    train_idx_np = np.asarray(train_idx, dtype=np.int64)
    for step in range(1, args.steps + 1):
        batch_ids = np.random.choice(train_idx_np, size=min(int(args.batch_size), len(train_idx_np)), replace=len(train_idx_np) < int(args.batch_size))
        batch = gather(bank, batch_ids, device)
        out = model(
            batch["state"],
            batch["base"],
            batch["target"],
            batch["hist_actions"],
            batch["hist_mask"],
            batch["task_vec"],
            batch["stage"],
        )
        pred = out["committed"]
        target = _select_target(batch, str(args.target_key))
        loss = F.smooth_l1_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(
                model,
                bank,
                np.asarray(val_idx, dtype=np.int64),
                device,
                int(args.eval_batch_size),
                str(args.target_key),
            )
            record = {"step": int(step), "train_loss": float(loss.detach().item()), **metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if best is None or metrics["loss"] < best["loss"]:
                best = dict(record)
                torch.save(
                    {
                        "model_type": "dynamic_coupling_operator",
                        "state_dim": int(bank["state"].shape[-1]),
                        "future_dim": int(bank["base"].shape[-1]),
                        "progress_dim": int(bank[str(args.target_key)].shape[-1]),
                        "target_key": str(args.target_key),
                        "action_shape": [int(x) for x in bank["hist_actions"].shape[-2:]],
                        "history_len": int(bank["hist_actions"].shape[1]),
                        "task_dim": int(args.task_dim),
                        "task_hash_mode": "md5_mod",
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "model_state": model.state_dict(),
                        "best_metrics": best,
                    },
                    args.output_dir / "dynamic_coupling_operator.pt",
                )

    summary = {
        "rows_jsonl": str(args.rows_jsonl),
        "num_rows": int(len(rows)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "steps": int(args.steps),
        "target_key": str(args.target_key),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "dynamic_coupling_operator.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
