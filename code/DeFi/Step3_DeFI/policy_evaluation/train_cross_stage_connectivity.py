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
    seqs = sorted({int(row.get("sequence_index", -1)) for row in rows if int(row.get("sequence_index", -1)) >= 0})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row.get("sequence_index", -1)) in val else train_idx).append(idx)
    return train_idx, val_idx


def _arr(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float32).reshape(-1)


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


def _build_h(
    *,
    state: np.ndarray,
    future: np.ndarray,
    action: np.ndarray,
    task: str,
    stage: float,
    task_to_id: dict[str, int],
    num_tasks: int,
    max_stage: float,
) -> np.ndarray:
    task_onehot = np.zeros((num_tasks,), dtype=np.float32)
    if task in task_to_id:
        task_onehot[task_to_id[task]] = 1.0
    stage_arr = np.asarray([float(stage) / max(max_stage, 1.0)], dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float32).reshape(-1),
            np.asarray(future, dtype=np.float32).reshape(-1),
            np.asarray(action, dtype=np.float32).reshape(-1),
            task_onehot,
            stage_arr,
        ],
        axis=0,
    ).astype(np.float32)


def make_arrays(rows: list[dict[str, Any]], task_names: list[str]) -> dict[str, np.ndarray]:
    task_to_id = {task: i for i, task in enumerate(task_names)}
    num_tasks = len(task_names)
    max_stage = max(float(row.get("stage", 0)) for row in rows)
    max_stage = max(max_stage, 1.0)

    x_cur, x_pos, x_neg, w = [], [], [], []
    for row in rows:
        stage = float(row.get("stage", 0))
        cur = _build_h(
            state=_arr(row, "current_state"),
            future=_arr(row, "current_future"),
            action=_arr(row, "current_action"),
            task=str(row.get("task", "unknown")),
            stage=stage,
            task_to_id=task_to_id,
            num_tasks=num_tasks,
            max_stage=max_stage,
        )
        pos = _build_h(
            state=_arr(row, "next_state_pos"),
            future=_arr(row, "next_future_pos"),
            action=_arr(row, "next_action_pos"),
            task=str(row.get("next_task_pos", "unknown")),
            stage=stage + 1.0,
            task_to_id=task_to_id,
            num_tasks=num_tasks,
            max_stage=max_stage,
        )
        neg = _build_h(
            state=_arr(row, "next_state_neg"),
            future=_arr(row, "next_future_neg"),
            action=_arr(row, "next_action_neg"),
            task=str(row.get("next_task_neg", "unknown")),
            stage=stage + 1.0,
            task_to_id=task_to_id,
            num_tasks=num_tasks,
            max_stage=max_stage,
        )
        x_cur.append(cur)
        x_pos.append(pos)
        x_neg.append(neg)
        w.append(max(float(row.get("distance_margin", 1.0)), 1e-3))

    weights = np.asarray(w, dtype=np.float32)
    weights = weights / max(float(weights.mean()), 1e-6)
    return {
        "x_cur": np.stack(x_cur, axis=0).astype(np.float32),
        "x_pos": np.stack(x_pos, axis=0).astype(np.float32),
        "x_neg": np.stack(x_neg, axis=0).astype(np.float32),
        "weights": weights,
    }


class ConnectivityMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = latent_dim * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cur: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(cur - nxt)
        x = torch.cat([cur, nxt, delta], dim=-1)
        return self.net(x).squeeze(-1)


def tensorize_features(arrays: dict[str, np.ndarray], ids: list[int], device: torch.device):
    return (
        torch.from_numpy(arrays["x_cur"][ids]).to(device),
        torch.from_numpy(arrays["x_pos"][ids]).to(device),
        torch.from_numpy(arrays["x_neg"][ids]).to(device),
        torch.from_numpy(arrays["weights"][ids]).to(device),
    )


def encode_with_manifold(
    manifold: JointHypothesisManifold,
    x_cur: torch.Tensor,
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zs_cur, zs_pos, zs_neg = [], [], []
    manifold.eval()
    with torch.no_grad():
        for start in range(0, x_cur.shape[0], batch_size):
            end = min(start + batch_size, x_cur.shape[0])
            zs_cur.append(manifold.encode(x_cur[start:end]).cpu())
            zs_pos.append(manifold.encode(x_pos[start:end]).cpu())
            zs_neg.append(manifold.encode(x_neg[start:end]).cpu())
    return torch.cat(zs_cur, dim=0), torch.cat(zs_pos, dim=0), torch.cat(zs_neg, dim=0)


def evaluate(model: ConnectivityMLP, x_cur: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor, w: torch.Tensor, margin: float):
    model.eval()
    with torch.no_grad():
        s_pos = model(x_cur, x_pos)
        s_neg = model(x_cur, x_neg)
        diff = s_pos - s_neg
        loss = (F.relu(margin - diff) * w).mean()
        acc = float((diff > 0).float().mean().item())
        mean_gap = float(diff.mean().item())
        median_gap = float(torch.median(diff).item())
    return {
        "loss": float(loss.item()),
        "pair_acc": acc,
        "mean_gap": mean_gap,
        "median_gap": median_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Train cross-stage connectivity scorer C_cross-stage.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--manifold-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(args.pairs_jsonl))
    if not rows:
        raise ValueError(f"no rows in {args.pairs_jsonl}")
    ckpt = torch.load(args.manifold_ckpt, map_location="cpu")
    task_names = [str(x) for x in ckpt.get("task_names", [])]
    if not task_names:
        task_names = sorted(
            {str(row.get("task", "unknown")) for row in rows}
            | {str(row.get("next_task_pos", "unknown")) for row in rows}
            | {str(row.get("next_task_neg", "unknown")) for row in rows}
        )
    arrays = make_arrays(rows, task_names)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    manifold = JointHypothesisManifold(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    manifold.load_state_dict(ckpt["model_state"])

    cur_train_feat, pos_train_feat, neg_train_feat, w_train = tensorize_features(arrays, train_idx, device)
    cur_val_feat, pos_val_feat, neg_val_feat, w_val = tensorize_features(arrays, val_idx, device)

    cur_train, pos_train, neg_train = encode_with_manifold(manifold, cur_train_feat, pos_train_feat, neg_train_feat, int(args.batch_size))
    cur_val, pos_val, neg_val = encode_with_manifold(manifold, cur_val_feat, pos_val_feat, neg_val_feat, int(args.batch_size))
    cur_train = cur_train.to(device)
    pos_train = pos_train.to(device)
    neg_train = neg_train.to(device)
    cur_val = cur_val.to(device)
    pos_val = pos_val.to(device)
    neg_val = neg_val.to(device)

    model = ConnectivityMLP(cur_train.shape[-1], int(args.hidden_dim), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_ids = torch.randint(0, cur_train.shape[0], (min(int(args.batch_size), cur_train.shape[0]),), device=device)
        s_pos = model(cur_train[batch_ids], pos_train[batch_ids])
        s_neg = model(cur_train[batch_ids], neg_train[batch_ids])
        diff = s_pos - s_neg
        loss = (F.relu(float(args.margin) - diff) * w_train[batch_ids]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == args.steps:
            metrics = {
                "step": int(step),
                "train_loss": float(loss.item()),
                **evaluate(model, cur_val, pos_val, neg_val, w_val, float(args.margin)),
            }
            history.append(metrics)
            if best is None or metrics["loss"] < best["loss"]:
                best = metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "latent_dim": int(cur_train.shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "margin": float(args.margin),
                        "manifold_ckpt": str(args.manifold_ckpt),
                    },
                    args.output_dir / "cross_stage_connectivity.pt",
                )

    summary = {
        "num_pairs": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "cross_stage_connectivity.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
