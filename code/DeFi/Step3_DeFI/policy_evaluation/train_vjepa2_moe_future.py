import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


STAGE_NAMES = ["approach", "grasp", "transport", "insert", "release"]


def infer_stage(task: str) -> int:
    task = str(task)
    if "lift" in task:
        return STAGE_NAMES.index("grasp")
    if "place_in_drawer" in task or "place_in_slider" in task or "push_into_drawer" in task:
        return STAGE_NAMES.index("insert")
    if "push" in task or "move_slider" in task:
        return STAGE_NAMES.index("transport")
    if "stack" in task or "unstack" in task or "rotate" in task:
        return STAGE_NAMES.index("release")
    return STAGE_NAMES.index("approach")


class MoeFuture(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_experts: int,
        future_aware_router: bool = False,
        num_tasks: int = 0,
        num_stages: int = 0,
        condition_router: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.future_aware_router = future_aware_router
        self.condition_router = condition_router
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.demo_proj = nn.Linear(dim * 3, hidden_dim)
        if condition_router:
            self.task_embed = nn.Embedding(num_tasks, hidden_dim)
            self.stage_embed = nn.Embedding(num_stages, hidden_dim)
        else:
            self.task_embed = None
            self.stage_embed = None
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
        if future_aware_router:
            router_dim = dim * (1 + num_experts * 2)
            if condition_router:
                router_dim += hidden_dim * 2
            self.router = nn.Sequential(
                nn.LayerNorm(router_dim),
                nn.Linear(router_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts),
            )
        else:
            self.router = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def encode(self, current: torch.Tensor, demo_current: torch.Tensor, demo_future: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(current).unsqueeze(1)
        demo_delta = demo_future - demo_current
        memory = self.demo_proj(torch.cat([demo_current, demo_future, demo_delta], dim=-1))
        for layer in self.layers:
            attended, _ = layer["attn"](query, memory, memory, need_weights=False)
            query = layer["norm1"](query + attended)
            query = layer["norm2"](query + layer["ff"](query))
        return query.squeeze(1)

    def forward(
        self,
        current: torch.Tensor,
        demo_current: torch.Tensor,
        demo_future: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
    ):
        hidden = self.encode(current, demo_current, demo_future)
        futures = torch.stack([F.normalize(expert(hidden), dim=-1) for expert in self.experts], dim=1)
        if self.future_aware_router:
            current_expanded = current.unsqueeze(1).expand(-1, self.num_experts, -1)
            router_parts = [
                current,
                futures.reshape(futures.shape[0], -1),
                (futures - current_expanded).reshape(futures.shape[0], -1),
            ]
            if self.condition_router:
                if task_ids is None or stage_ids is None:
                    raise ValueError("task_ids and stage_ids are required for condition_router")
                router_parts.extend([self.task_embed(task_ids), self.stage_embed(stage_ids)])
            router_input = torch.cat(router_parts, dim=-1)
            logits = self.router(router_input)
        else:
            logits = self.router(hidden)
        return futures, logits


def load_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def topk_same_task(query: np.ndarray, query_tasks: Sequence[str], train_start: np.ndarray, train_tasks: Sequence[str], k: int, exclude_self: bool) -> np.ndarray:
    by_task = {
        task: np.asarray([idx for idx, item in enumerate(train_tasks) if item == task], dtype=np.int64)
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
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def summarize(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    sims = cosine(pred, target)
    l2 = np.linalg.norm(pred - target, axis=-1)
    return {
        "cosine_mean": float(sims.mean()),
        "cosine_median": float(np.median(sims)),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
    }


def retrieval_cos(pred: np.ndarray, target: np.ndarray, tasks: Sequence[str], train_end: np.ndarray, train_tasks: Sequence[str]) -> float:
    values = []
    for feat, gt, task in zip(pred, target, tasks):
        idx = np.asarray([i for i, item in enumerate(train_tasks) if item == task], dtype=np.int64)
        if len(idx) == 0:
            idx = np.arange(len(train_end), dtype=np.int64)
        chosen = idx[int(np.argmax(train_end[idx] @ feat))]
        values.append(float(np.dot(train_end[chosen], gt)))
    return float(np.mean(values))


def balanced_assignment(sims: torch.Tensor) -> torch.Tensor:
    batch, num_experts = sims.shape
    capacity = int(np.ceil(batch / num_experts))
    assigned = torch.full((batch,), -1, dtype=torch.long, device=sims.device)
    counts = torch.zeros(num_experts, dtype=torch.long, device=sims.device)
    confidence = sims.max(dim=1).values
    for sample_idx in torch.argsort(confidence, descending=True).tolist():
        for expert_idx in torch.argsort(sims[sample_idx], descending=True).tolist():
            if int(counts[expert_idx]) < capacity:
                assigned[sample_idx] = expert_idx
                counts[expert_idx] += 1
                break
    if bool((assigned < 0).any()):
        assigned[assigned < 0] = sims[assigned < 0].argmax(dim=1)
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V-JEPA retrieved-demo MoE future predictor.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--router-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.02)
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--balanced-oracle-assignment", action="store_true")
    parser.add_argument("--future-aware-router", action="store_true")
    parser.add_argument("--condition-router", action="store_true")
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
    task_vocab = {task: idx for idx, task in enumerate(sorted(set(train_tasks.tolist()) | set(val_tasks.tolist())))}
    train_task_ids = np.asarray([task_vocab[str(task)] for task in train_tasks], dtype=np.int64)
    val_task_ids = np.asarray([task_vocab[str(task)] for task in val_tasks], dtype=np.int64)
    train_stage_ids = np.asarray([infer_stage(str(task)) for task in train_tasks], dtype=np.int64)
    val_stage_ids = np.asarray([infer_stage(str(task)) for task in val_tasks], dtype=np.int64)

    train_knn = topk_same_task(train_start, train_tasks, train_start, train_tasks, args.k, True)
    val_knn = topk_same_task(val_start, val_tasks, train_start, train_tasks, args.k, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoeFuture(
        train_start.shape[1],
        args.hidden_dim,
        args.heads,
        args.layers,
        args.num_experts,
        args.future_aware_router,
        len(task_vocab),
        len(STAGE_NAMES),
        args.condition_router,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = torch.from_numpy(train_start).to(device)
    y = torch.from_numpy(train_end).to(device)
    dc = torch.from_numpy(train_start[train_knn]).to(device)
    df = torch.from_numpy(train_end[train_knn]).to(device)
    task_ids = torch.from_numpy(train_task_ids).to(device)
    stage_ids = torch.from_numpy(train_stage_ids).to(device)
    xv = torch.from_numpy(val_start).to(device)
    yv = torch.from_numpy(val_end).to(device)
    dcv = torch.from_numpy(train_start[val_knn]).to(device)
    dfv = torch.from_numpy(train_end[val_knn]).to(device)
    task_idsv = torch.from_numpy(val_task_ids).to(device)
    stage_idsv = torch.from_numpy(val_stage_ids).to(device)

    best_state = None
    best_val = float("inf")
    for _ in range(args.epochs):
        model.train()
        futures, logits = model(x, dc, df, task_ids, stage_ids)
        sims = torch.einsum("bmd,bd->bm", futures, y)
        if args.balanced_oracle_assignment:
            best_idx = balanced_assignment(sims.detach())
            best_sim = sims[torch.arange(len(x), device=device), best_idx]
        else:
            best_sim, best_idx = sims.max(dim=1)
        future_loss = 1.0 - best_sim.mean()
        router_loss = F.cross_entropy(logits, best_idx.detach())
        pair = torch.einsum("bmd,bnd->bmn", futures, futures)
        off_diag = pair[:, ~torch.eye(args.num_experts, dtype=torch.bool, device=device)]
        diversity_loss = off_diag.mean()
        probs = torch.softmax(logits, dim=-1).mean(dim=0)
        balance_loss = torch.sum((probs - (1.0 / args.num_experts)) ** 2)
        loss = future_loss + args.router_weight * router_loss + args.diversity_weight * diversity_loss + args.balance_weight * balance_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vfutures, vlogits = model(xv, dcv, dfv, task_idsv, stage_idsv)
            vbest = torch.einsum("bmd,bd->bm", vfutures, yv).max(dim=1).values
            val_loss = 1.0 - vbest.mean()
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_futures, train_logits = model(x, dc, df, task_ids, stage_ids)
        val_futures, val_logits = model(xv, dcv, dfv, task_idsv, stage_idsv)
        train_sims = torch.einsum("bmd,bd->bm", train_futures, y)
        val_sims = torch.einsum("bmd,bd->bm", val_futures, yv)
        train_best_idx = train_sims.argmax(dim=1)
        val_best_idx = val_sims.argmax(dim=1)
        val_router_idx = val_logits.argmax(dim=1)
        train_best = train_futures[torch.arange(len(x), device=device), train_best_idx].cpu().numpy()
        val_best = val_futures[torch.arange(len(xv), device=device), val_best_idx].cpu().numpy()
        val_router = val_futures[torch.arange(len(xv), device=device), val_router_idx].cpu().numpy()
        val_expert_counts = torch.bincount(val_best_idx.cpu(), minlength=args.num_experts).numpy().tolist()
        val_router_counts = torch.bincount(val_router_idx.cpu(), minlength=args.num_experts).numpy().tolist()

    nn_idx = val_knn[:, 0]
    metrics = {
        "num_train": int(len(train_start)),
        "num_val": int(len(val_start)),
        "k": int(args.k),
        "num_experts": int(args.num_experts),
        "balanced_oracle_assignment": bool(args.balanced_oracle_assignment),
        "future_aware_router": bool(args.future_aware_router),
        "condition_router": bool(args.condition_router),
        "nearest_demo_val": summarize(train_end[nn_idx], val_end),
        "moe_train_oracle_best": summarize(train_best, train_end),
        "moe_val_oracle_best": summarize(val_best, val_end),
        "moe_val_router_selected": summarize(val_router, val_end),
        "moe_val_router_matches_oracle_expert": float(np.mean(val_router_idx.cpu().numpy() == val_best_idx.cpu().numpy())),
        "moe_val_oracle_retrieved_future_to_gt_cosine_mean": retrieval_cos(val_best, val_end, val_tasks, train_end, train_tasks),
        "moe_val_router_retrieved_future_to_gt_cosine_mean": retrieval_cos(val_router, val_end, val_tasks, train_end, train_tasks),
        "oracle_expert_counts": val_expert_counts,
        "router_expert_counts": val_router_counts,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(train_start.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "heads": int(args.heads),
            "layers": int(args.layers),
            "k": int(args.k),
            "num_experts": int(args.num_experts),
            "future_aware_router": bool(args.future_aware_router),
            "condition_router": bool(args.condition_router),
            "task_vocab": task_vocab,
            "stage_names": STAGE_NAMES,
        },
        args.output_dir / "vjepa2_moe_future.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
