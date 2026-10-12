import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hashed_task(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    for tok in str(text).replace("_", " ").lower().split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


class BridgeJointEnergy(nn.Module):
    def __init__(self, latent_dim: int, task_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 3 + task_dim),
            nn.Linear(latent_dim * 3 + task_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_latent, future_latent, motion_latent, task_vec):
        return self.net(torch.cat([state_latent, future_latent, motion_latent, task_vec], dim=-1)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-npz", required=True)
    parser.add_argument("--output-ckpt", required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with np.load(args.memory_npz, allow_pickle=True) as data:
        memory = {k: data[k] for k in data.files}
    required = ["state_latent", "future_latent", "motion_latent", "success", "task"]
    missing = [k for k in required if k not in memory]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    state = np.asarray(memory["state_latent"], dtype=np.float32)
    future = np.asarray(memory["future_latent"], dtype=np.float32)
    motion = np.asarray(memory["motion_latent"], dtype=np.float32)
    success = np.asarray(memory["success"], dtype=np.float32) > 0.5
    tasks = [str(x) for x in memory["task"].tolist()]
    task_vec = np.stack([hashed_task(x, args.task_dim) for x in tasks], axis=0).astype(np.float32)
    latent_dim = int(state.shape[1])

    pos_ids = np.nonzero(success)[0].tolist()
    neg_ids = np.nonzero(~success)[0].tolist()
    if not pos_ids or not neg_ids:
        raise RuntimeError("need both success and failure rows")

    success_by_task = {}
    for idx in pos_ids:
        success_by_task.setdefault(tasks[idx], []).append(idx)

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = BridgeJointEnergy(latent_dim=latent_dim, task_dim=args.task_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    state_t = torch.from_numpy(state).to(device)
    future_t = torch.from_numpy(future).to(device)
    motion_t = torch.from_numpy(motion).to(device)
    task_t = torch.from_numpy(task_vec).to(device)
    label_t = torch.from_numpy(success.astype(np.float32)).to(device)

    best = None
    best_loss = None
    for step in range(1, args.steps + 1):
        pair_batch = min(args.batch_size, len(pos_ids), len(neg_ids))
        pos = np.asarray(rng.choices(pos_ids, k=pair_batch), dtype=np.int64)
        neg = np.asarray(rng.choices(neg_ids, k=pair_batch), dtype=np.int64)
        mismatch = []
        for idx in pos.tolist():
            pool = [j for j in success_by_task.get(tasks[idx], []) if j != idx]
            if not pool:
                pool = [j for j in pos_ids if j != idx]
            mismatch.append(rng.choice(pool) if pool else idx)
        mismatch = np.asarray(mismatch, dtype=np.int64)

        e_pos = model(state_t[pos], future_t[pos], motion_t[pos], task_t[pos])
        e_neg = model(state_t[neg], future_t[neg], motion_t[neg], task_t[neg])
        e_mis = model(state_t[pos], future_t[pos], motion_t[mismatch], task_t[pos])

        rank_loss = F.relu(args.margin + e_pos - e_neg).mean()
        hard_neg_loss = F.relu(0.25 + e_pos - e_mis).mean()

        batch_ids = np.concatenate([pos, neg], axis=0)
        logits = -model(state_t[batch_ids], future_t[batch_ids], motion_t[batch_ids], task_t[batch_ids])
        cls_loss = F.binary_cross_entropy_with_logits(logits, label_t[batch_ids])
        loss = rank_loss + hard_neg_loss + 0.5 * cls_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if step == 1 or step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                energy = model(state_t, future_t, motion_t, task_t)
                pred = (-energy > 0).float()
                acc = float((pred == label_t).float().mean().item())
                print(json.dumps({
                    "step": step,
                    "loss": loss_val,
                    "rank_loss": float(rank_loss.item()),
                    "hard_negative_loss": float(hard_neg_loss.item()),
                    "success_acc": acc,
                    "energy_success_mean": float(energy[label_t > 0.5].mean().item()),
                    "energy_fail_mean": float(energy[label_t <= 0.5].mean().item()),
                }, ensure_ascii=False), flush=True)

    out = Path(args.output_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best if best is not None else model.state_dict(),
            "latent_dim": latent_dim,
            "task_dim": int(args.task_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "memory_npz": str(args.memory_npz),
        },
        out,
    )
    print(json.dumps({"saved_ckpt": str(out), "best_loss": best_loss}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
