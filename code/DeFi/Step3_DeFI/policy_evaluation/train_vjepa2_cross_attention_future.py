import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionFuture(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.demo_proj = nn.Linear(dim * 3, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ff": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, current: torch.Tensor, demo_current: torch.Tensor, demo_future: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(current).unsqueeze(1)
        demo_delta = demo_future - demo_current
        memory = self.demo_proj(torch.cat([demo_current, demo_future, demo_delta], dim=-1))
        for layer in self.layers:
            attended, _ = layer["attn"](query, memory, memory, need_weights=False)
            query = layer["norm1"](query + attended)
            query = layer["norm2"](query + layer["ff"](query))
        return F.normalize(self.out(query.squeeze(1)), dim=-1)


def load_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def topk_same_task(
    query: np.ndarray,
    query_tasks: Sequence[str],
    train_start: np.ndarray,
    train_tasks: Sequence[str],
    k: int,
    exclude_self: bool = False,
) -> np.ndarray:
    by_task = {
        task: np.asarray([i for i, item in enumerate(train_tasks) if item == task], dtype=np.int64)
        for task in sorted(set(train_tasks))
    }
    out = []
    for i, (feat, task) in enumerate(zip(query, query_tasks)):
        idx = by_task.get(str(task), np.arange(len(train_start), dtype=np.int64))
        if exclude_self:
            idx = idx[idx != i]
        if len(idx) == 0:
            idx = np.arange(len(train_start), dtype=np.int64)
            if exclude_self and i < len(train_start):
                idx = idx[idx != i]
        sims = train_start[idx] @ feat
        order = idx[np.argsort(-sims)]
        if len(order) < k:
            order = np.pad(order, (0, k - len(order)), mode="edge")
        out.append(order[:k])
    return np.stack(out, axis=0).astype(np.int64)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def evaluate(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    sims = cosine(pred, target)
    l2 = np.linalg.norm(pred - target, axis=1)
    return {
        "cosine_mean": float(sims.mean()),
        "cosine_median": float(np.median(sims)),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
    }


def retrieval_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    tasks: Sequence[str],
    train_end: np.ndarray,
    train_tasks: Sequence[str],
) -> Dict[str, float]:
    top1_same_task = []
    top1_cos = []
    oracle_same_task_cos = []
    for feat, gt, task in zip(pred, target, tasks):
        idx = np.asarray([i for i, item in enumerate(train_tasks) if item == task], dtype=np.int64)
        if len(idx) == 0:
            idx = np.arange(len(train_end), dtype=np.int64)
        pred_sims = train_end[idx] @ feat
        chosen = idx[int(np.argmax(pred_sims))]
        top1_same_task.append(float(np.dot(train_end[chosen], gt)))
        top1_cos.append(float(np.dot(train_end[chosen], feat)))
        gt_sims = train_end[idx] @ gt
        oracle_same_task_cos.append(float(np.max(gt_sims)))
    return {
        "retrieved_future_to_gt_cosine_mean": float(np.mean(top1_same_task)),
        "retrieved_future_to_pred_cosine_mean": float(np.mean(top1_cos)),
        "oracle_same_task_future_to_gt_cosine_mean": float(np.mean(oracle_same_task_cos)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V-JEPA2 retrieval-conditioned cross-attention future predictor.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_features(args.features)
    train_start = data["train_start"].astype(np.float32)
    train_end = data["train_end"].astype(np.float32)
    val_start = data["val_start"].astype(np.float32)
    val_end = data["val_end"].astype(np.float32)
    train_tasks = data["train_tasks"].astype(str)
    val_tasks = data["val_tasks"].astype(str)

    train_knn = topk_same_task(train_start, train_tasks, train_start, train_tasks, args.k, exclude_self=True)
    val_knn = topk_same_task(val_start, val_tasks, train_start, train_tasks, args.k, exclude_self=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttentionFuture(train_start.shape[1], args.hidden_dim, args.heads, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = torch.from_numpy(train_start).to(device)
    y = torch.from_numpy(train_end).to(device)
    dc = torch.from_numpy(train_start[train_knn]).to(device)
    df = torch.from_numpy(train_end[train_knn]).to(device)
    xv = torch.from_numpy(val_start).to(device)
    yv = torch.from_numpy(val_end).to(device)
    dcv = torch.from_numpy(train_start[val_knn]).to(device)
    dfv = torch.from_numpy(train_end[val_knn]).to(device)

    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        pred = model(x, dc, df)
        loss = 1.0 - torch.mean(torch.sum(pred * y, dim=-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = 1.0 - torch.mean(torch.sum(model(xv, dcv, dfv) * yv, dim=-1))
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_train = model(x, dc, df).cpu().numpy()
        pred_val = model(xv, dcv, dfv).cpu().numpy()

    nn_val_idx = val_knn[:, 0]
    metrics = {
        "num_train": int(len(train_start)),
        "num_val": int(len(val_start)),
        "k": int(args.k),
        "target": "V-JEPA current + K retrieved demos -> V-JEPA GT future",
        "nearest_demo_val": evaluate(train_end[nn_val_idx], val_end),
        "current_to_future_val": evaluate(val_start, val_end),
        "cross_attention_train": evaluate(pred_train, train_end),
        "cross_attention_val": evaluate(pred_val, val_end),
    }
    metrics["cross_attention_retrieve_val"] = retrieval_metrics(pred_val, val_end, val_tasks, train_end, train_tasks)

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(train_start.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "heads": int(args.heads),
            "layers": int(args.layers),
            "k": int(args.k),
        },
        args.output_dir / "vjepa2_cross_attention_future.pt",
    )
    np.savez_compressed(args.output_dir / "predictions.npz", pred_val=pred_val, val_end=val_end, val_tasks=val_tasks)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
