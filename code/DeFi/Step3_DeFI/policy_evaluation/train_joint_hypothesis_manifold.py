from __future__ import annotations

import argparse
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
    seqs = sorted(
        {
            int(row.get("sequence_index_neg", -1))
            for row in rows
            if int(row.get("sequence_index_neg", -1)) >= 0
        }
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val_set = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        seq = int(row.get("sequence_index_neg", -1))
        (val_idx if seq in val_set else train_idx).append(idx)
    if not train_idx:
        train_idx = list(range(max(0, len(rows) - 1)))
    if not val_idx:
        val_idx = [len(rows) - 1]
    return train_idx, val_idx


def _arr(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float32).reshape(-1)


def make_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted({str(row.get("task", "unknown")) for row in rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}

    state_neg = np.stack([_arr(r, "state_neg") for r in rows], axis=0)
    future_neg = np.stack([_arr(r, "future_neg") for r in rows], axis=0)
    action_neg = np.stack([_arr(r, "action_neg") for r in rows], axis=0)
    state_pos = np.stack([_arr(r, "state_pos") for r in rows], axis=0)
    future_pos = np.stack([_arr(r, "future_pos") for r in rows], axis=0)
    action_pos = np.stack([_arr(r, "action_pos") for r in rows], axis=0)

    task_ids = np.asarray([task_to_id[str(r.get("task", "unknown"))] for r in rows], dtype=np.int64)
    task_onehot = np.eye(len(tasks), dtype=np.float32)[task_ids]
    stages = np.asarray([float(r.get("stage", 0)) for r in rows], dtype=np.float32).reshape(-1, 1)
    stages = stages / max(float(stages.max()), 1.0)
    weights = np.asarray([float(r.get("return_gap", 1.0)) for r in rows], dtype=np.float32)
    weights = np.maximum(weights, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)

    h_neg = np.concatenate([state_neg, future_neg, action_neg, task_onehot, stages], axis=1).astype(np.float32)
    h_pos = np.concatenate([state_pos, future_pos, action_pos, task_onehot, stages], axis=1).astype(np.float32)
    return {
        "h_neg": h_neg,
        "h_pos": h_pos,
        "weights": weights.astype(np.float32),
        "state_dim": state_neg.shape[1],
        "future_dim": future_neg.shape[1],
        "action_dim": action_neg.shape[1],
        "num_tasks": len(tasks),
        "task_names": tasks,
        "input_dim": h_neg.shape[1],
    }


class JointHypothesisManifold(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.success_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        success_logit = self.success_head(z).squeeze(-1)
        return {"z": z, "success_logit": success_logit}


def evaluate(
    model: JointHypothesisManifold,
    h_neg: torch.Tensor,
    h_pos: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
):
    model.eval()
    with torch.no_grad():
        out_neg = model(h_neg)
        out_pos = model(h_pos)
        z_neg = out_neg["z"]
        z_pos = out_pos["z"]
        pos_logit = out_pos["success_logit"]
        neg_logit = out_neg["success_logit"]

        pair_dist = torch.norm(z_pos - z_neg, dim=-1, p=2)
        rank_gap = pos_logit - neg_logit
        rank_acc = float((rank_gap > 0).float().mean().item())
        pair_margin = float(rank_gap.mean().item())
        pair_loss = F.relu(margin - rank_gap)
        cls_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
            + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
        )
        loss = float((pair_loss * weights).mean().item() + cls_loss.item())
        success_acc = float(
            torch.cat([(torch.sigmoid(pos_logit) >= 0.5).float(), (torch.sigmoid(neg_logit) < 0.5).float()], dim=0).mean().item()
        )
        dist_mean = float(pair_dist.mean().item())
    return {
        "loss": loss,
        "pair_acc": rank_acc,
        "pair_margin": pair_margin,
        "pair_dist_mean": dist_mean,
        "success_acc": success_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train joint hypothesis manifold encoder m(h) for h=(s,z_f,z_a,task,stage).")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=0.5)
    parser.add_argument("--dist-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = list(iter_jsonl(args.pairs_jsonl))
    if not rows:
        raise ValueError(f"no rows in {args.pairs_jsonl}")
    arrays = make_arrays(rows)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    h_neg = torch.from_numpy(arrays["h_neg"]).to(device)
    h_pos = torch.from_numpy(arrays["h_pos"]).to(device)
    weights = torch.from_numpy(arrays["weights"]).to(device)

    h_neg_train, h_pos_train, w_train = h_neg[train_idx], h_pos[train_idx], weights[train_idx]
    h_neg_val, h_pos_val, w_val = h_neg[val_idx], h_pos[val_idx], weights[val_idx]

    model = JointHypothesisManifold(
        input_dim=int(arrays["input_dim"]),
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history: list[dict[str, Any]] = []

    for step in range(1, args.steps + 1):
        model.train()
        batch_size = min(int(args.batch_size), h_neg_train.shape[0])
        batch_ids = torch.randint(0, h_neg_train.shape[0], (batch_size,), device=device)
        out_neg = model(h_neg_train[batch_ids])
        out_pos = model(h_pos_train[batch_ids])

        neg_logit = out_neg["success_logit"]
        pos_logit = out_pos["success_logit"]
        z_neg = out_neg["z"]
        z_pos = out_pos["z"]
        batch_w = w_train[batch_ids]

        rank_gap = pos_logit - neg_logit
        rank_loss = (F.relu(float(args.margin) - rank_gap) * batch_w).mean()
        cls_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
            + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
        )
        # Encourage bad/good hypotheses to be separated in manifold space.
        dist = torch.norm(z_pos - z_neg, dim=-1, p=2)
        dist_loss = (1.0 / (dist + 1e-6) * batch_w).mean()

        loss = (
            float(args.rank_weight) * rank_loss
            + float(args.cls_weight) * cls_loss
            + float(args.dist_weight) * dist_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == args.steps or step == 1:
            metrics = {
                "step": int(step),
                "train_loss": float(loss.item()),
                "rank_loss": float(rank_loss.item()),
                "cls_loss": float(cls_loss.item()),
                "dist_loss": float(dist_loss.item()),
                **evaluate(model, h_neg_val, h_pos_val, w_val, float(args.margin)),
            }
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            score = metrics["pair_acc"] + metrics["pair_margin"] + 0.25 * metrics["success_acc"] + 0.1 * metrics["pair_dist_mean"]
            if best is None or score > best["score"]:
                best = {"score": float(score), "metrics": metrics}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": int(arrays["input_dim"]),
                        "hidden_dim": int(args.hidden_dim),
                        "latent_dim": int(args.latent_dim),
                        "dropout": float(args.dropout),
                        "state_dim": int(arrays["state_dim"]),
                        "future_dim": int(arrays["future_dim"]),
                        "action_dim": int(arrays["action_dim"]),
                        "num_tasks": int(arrays["num_tasks"]),
                        "task_names": list(arrays["task_names"]),
                        "best_metrics": metrics,
                    },
                    args.output_dir / "joint_hypothesis_manifold.pt",
                )

    summary = {
        "num_rows": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "input_dim": int(arrays["input_dim"]),
        "latent_dim": int(args.latent_dim),
        "best": None if best is None else best["metrics"],
        "history": history,
        "output_ckpt": str(args.output_dir / "joint_hypothesis_manifold.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
