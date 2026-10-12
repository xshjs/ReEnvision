from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        data = np.load(str(path))
    except Exception:
        return None
    if key not in data.files:
        return None
    x = np.asarray(data[key], dtype=np.float32)
    return x if x.ndim == 2 else None


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class TokenImmuneAdapter(nn.Module):
    def __init__(self, future_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        in_dim = future_dim * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.success_head = nn.Sequential(
            nn.LayerNorm(future_dim),
            nn.Linear(future_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, current: torch.Tensor, failure: torch.Tensor, success: torch.Tensor):
        x = torch.cat([current, failure, success], dim=-1)
        delta = self.net(x)
        gate = self.gate(x)
        return gate * delta, {"gate": gate, "raw_delta": delta}

    def success_logits(self, future: torch.Tensor) -> torch.Tensor:
        return self.success_head(future.mean(dim=1)).squeeze(-1)


def build_pairs(
    rows: list[dict[str, Any]],
    futures: list[np.ndarray],
    same_task_success: bool,
    same_task_failure: bool,
    k_failure: int,
) -> list[dict[str, Any]]:
    pooled = np.stack([future.mean(axis=0).astype(np.float32) for future in futures], axis=0)
    pooled_norm = normalize(pooled)
    success_indices = [idx for idx, row in enumerate(rows) if bool(row.get("success", False))]
    failure_indices = [idx for idx, row in enumerate(rows) if not bool(row.get("success", False))]
    if not success_indices or not failure_indices:
        return []

    nearest_success = {}
    for fail_idx in failure_indices:
        task = str(rows[fail_idx].get("task"))
        candidates = [idx for idx in success_indices if (not same_task_success or str(rows[idx].get("task")) == task)]
        if not candidates:
            candidates = success_indices
        sims = pooled_norm[fail_idx][None, :] @ pooled_norm[candidates].T
        nearest_success[fail_idx] = candidates[int(np.argmax(sims.reshape(-1)))]

    pairs = []
    for fail_idx in failure_indices:
        task = str(rows[fail_idx].get("task"))
        fail_candidates = [
            idx for idx in failure_indices
            if idx != fail_idx and (not same_task_failure or str(rows[idx].get("task")) == task)
        ]
        if not fail_candidates:
            fail_candidates = [idx for idx in failure_indices if idx != fail_idx] or [fail_idx]
        sims = (pooled_norm[fail_idx][None, :] @ pooled_norm[fail_candidates].T).reshape(-1)
        order = np.argsort(-sims)[: max(1, k_failure)]
        neighbor_failures = [fail_candidates[int(i)] for i in order]
        neighbor_sims = sims[order].astype(np.float32)
        weights = np.maximum(neighbor_sims, 0.0)
        if float(weights.sum()) <= 1e-8:
            weights = np.ones_like(weights, dtype=np.float32)
        weights = weights / np.maximum(weights.sum(), 1e-8)
        neighbor_successes = [nearest_success[idx] for idx in neighbor_failures]
        best = nearest_success[fail_idx]
        pairs.append(
            {
                "failure_index": int(fail_idx),
                "success_index": int(best),
                "neighbor_failure_indices": [int(x) for x in neighbor_failures],
                "neighbor_success_indices": [int(x) for x in neighbor_successes],
                "neighbor_weights": [float(x) for x in weights.tolist()],
                "task": task,
                "success_task": str(rows[best].get("task")),
                "neighbor_failure_similarity": [float(x) for x in neighbor_sims.tolist()],
            }
        )
    return pairs


def weighted_center(futures: list[np.ndarray], indices: list[int], weights: list[float]) -> torch.Tensor:
    tensors = [torch.from_numpy(futures[idx]) * float(weight) for idx, weight in zip(indices, weights)]
    return torch.stack(tensors, dim=0).sum(dim=0)


def make_target_delta(
    args: argparse.Namespace,
    current: torch.Tensor,
    failure: torch.Tensor,
    success: torch.Tensor,
    target_proxy: torch.Tensor | None,
) -> torch.Tensor:
    manifold_delta = args.alpha_repel * (current - failure) + args.beta_attract * (success - current)
    if args.target_mode == "manifold":
        return manifold_delta
    if target_proxy is None:
        raise ValueError(f"target_mode={args.target_mode} requires target_proxy_future")
    proxy_delta = target_proxy - current
    if args.target_mode == "target-proxy":
        return proxy_delta
    return args.hybrid_manifold_weight * manifold_delta + args.hybrid_target_weight * proxy_delta


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1), dim=1).mean()


def evaluate(model, pairs, futures, labels, target_futures, args, device, max_eval: int):
    model.eval()
    eval_pairs = pairs[:max_eval] if max_eval > 0 else pairs
    losses, cosines, gates, success_gains, failure_gains, proxy_gains, margins = [], [], [], [], [], [], []
    cls_logits, cls_targets = [], []
    with torch.no_grad():
        eval_cls_indices = list(range(min(len(futures), max(1, max_eval))))
        for idx in eval_cls_indices:
            future = torch.from_numpy(futures[idx]).unsqueeze(0).to(device)
            cls_logits.append(float(model.success_logits(future).item()))
            cls_targets.append(float(labels[idx]))
        for item in eval_pairs:
            f_idx = item["failure_index"]
            current = torch.from_numpy(futures[f_idx]).unsqueeze(0).to(device)
            failure = weighted_center(futures, item["neighbor_failure_indices"], item["neighbor_weights"]).unsqueeze(0).to(device)
            success = weighted_center(futures, item["neighbor_success_indices"], item["neighbor_weights"]).unsqueeze(0).to(device)
            target_proxy = None
            if target_futures is not None:
                target_proxy = torch.from_numpy(target_futures[f_idx]).unsqueeze(0).to(device)
            target_delta = make_target_delta(args, current, failure, success, target_proxy)
            pred_delta, aux = model(current, failure, success)
            edited = current + args.gate * pred_delta
            edit_logit = model.success_logits(edited)
            fail_logit = model.success_logits(failure)
            succ_logit = model.success_logits(success)
            margins.append(float((edit_logit - fail_logit).mean().item()))
            margins.append(float((succ_logit - edit_logit).mean().item()))
            losses.append(float(F.smooth_l1_loss(pred_delta, target_delta).item()))
            cosines.append(float((1.0 - cosine_loss(pred_delta, target_delta)).item()))
            gates.append(float(aux["gate"].mean().item()))
            base_success = F.cosine_similarity(current.reshape(1, -1), success.reshape(1, -1)).item()
            edit_success = F.cosine_similarity(edited.reshape(1, -1), success.reshape(1, -1)).item()
            success_gains.append(float(edit_success - base_success))
            base_failure = F.cosine_similarity(current.reshape(1, -1), failure.reshape(1, -1)).item()
            edit_failure = F.cosine_similarity(edited.reshape(1, -1), failure.reshape(1, -1)).item()
            failure_gains.append(float(base_failure - edit_failure))
            if target_proxy is not None:
                base_proxy = F.cosine_similarity(current.reshape(1, -1), target_proxy.reshape(1, -1)).item()
                edit_proxy = F.cosine_similarity(edited.reshape(1, -1), target_proxy.reshape(1, -1)).item()
                proxy_gains.append(float(edit_proxy - base_proxy))
    if cls_logits:
        probs = 1.0 / (1.0 + np.exp(-np.asarray(cls_logits, dtype=np.float32)))
        targets = np.asarray(cls_targets, dtype=np.float32)
        pred = probs >= 0.5
        cls_acc = float((pred == (targets > 0.5)).mean())
    else:
        cls_acc = 0.0
    return {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "delta_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "gate_mean": float(np.mean(gates)) if gates else 0.0,
        "success_cos_gain": float(np.mean(success_gains)) if success_gains else 0.0,
        "failure_repulsion_gain": float(np.mean(failure_gains)) if failure_gains else 0.0,
        "target_proxy_cos_gain": float(np.mean(proxy_gains)) if proxy_gains else 0.0,
        "success_head_acc": cls_acc,
        "success_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "num_eval": int(len(eval_pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-level immune adapter from failure/success future pairs.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", choices=["base_future", "proposal_future", "gated_future", "final_future"], default="base_future")
    parser.add_argument("--same-task-success", action="store_true")
    parser.add_argument("--same-task-failure", action="store_true")
    parser.add_argument("--k-failure", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha-repel", type=float, default=1.0)
    parser.add_argument("--beta-attract", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=["manifold", "target-proxy", "hybrid"], default="manifold")
    parser.add_argument("--hybrid-manifold-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-target-weight", type=float, default=1.0)
    parser.add_argument("--gate", type=float, default=0.02)
    parser.add_argument("--lambda-cos", type=float, default=1.0)
    parser.add_argument("--lambda-small", type=float, default=0.01)
    parser.add_argument("--lambda-success", type=float, default=0.5)
    parser.add_argument("--lambda-ranking", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--cls-batch-size", type=int, default=32)
    parser.add_argument("--eval-max", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, futures, labels, target_futures = [], [], [], []
    for row in iter_jsonl(args.dataset_jsonl):
        path = row.get("trace_path")
        if not path or not Path(str(path)).exists():
            continue
        future = load_future(path, args.future_key)
        if future is None:
            continue
        target_future = None
        if args.target_mode in {"target-proxy", "hybrid"}:
            target_future = load_future(path, "target_proxy_future")
            if target_future is None:
                continue
        rows.append(row)
        futures.append(future)
        labels.append(1.0 if bool(row.get("success", False)) else 0.0)
        if args.target_mode in {"target-proxy", "hybrid"}:
            target_futures.append(target_future)
    target_futures_or_none = target_futures if args.target_mode in {"target-proxy", "hybrid"} else None
    pairs = build_pairs(rows, futures, bool(args.same_task_success), bool(args.same_task_failure), int(args.k_failure))
    if not pairs:
        raise ValueError("no failure/success pairs")
    random.shuffle(pairs)
    split = max(1, int(round(len(pairs) * 0.8)))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:] or pairs[:1]

    future_dim = int(futures[0].shape[-1])
    model = TokenImmuneAdapter(future_dim=future_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = None
    best_state = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.choices(train_pairs, k=max(1, args.batch_size))
        current = torch.stack([torch.from_numpy(futures[item["failure_index"]]) for item in batch], dim=0).to(device)
        failure = torch.stack(
            [weighted_center(futures, item["neighbor_failure_indices"], item["neighbor_weights"]) for item in batch],
            dim=0,
        ).to(device)
        success = torch.stack(
            [weighted_center(futures, item["neighbor_success_indices"], item["neighbor_weights"]) for item in batch],
            dim=0,
        ).to(device)
        target_proxy = None
        if target_futures_or_none is not None:
            target_proxy = torch.stack(
                [torch.from_numpy(target_futures_or_none[item["failure_index"]]) for item in batch],
                dim=0,
            ).to(device)
        target_delta = make_target_delta(args, current, failure, success, target_proxy)
        pred_delta, aux = model(current, failure, success)
        edited = current + args.gate * pred_delta
        delta_loss = F.smooth_l1_loss(pred_delta, target_delta)
        dir_loss = cosine_loss(pred_delta, target_delta)
        small_loss = pred_delta.pow(2).mean()

        cls_indices = random.choices(range(len(futures)), k=max(1, args.cls_batch_size))
        cls_future = torch.stack([torch.from_numpy(futures[idx]) for idx in cls_indices], dim=0).to(device)
        cls_target = torch.tensor([labels[idx] for idx in cls_indices], dtype=torch.float32, device=device)
        cls_logits = model.success_logits(cls_future)
        pos = max(float(sum(labels)), 1.0)
        neg = max(float(len(labels) - sum(labels)), 1.0)
        pos_weight = len(labels) / (2.0 * pos)
        neg_weight = len(labels) / (2.0 * neg)
        cls_weight = torch.where(
            cls_target > 0.5,
            torch.full_like(cls_target, float(pos_weight)),
            torch.full_like(cls_target, float(neg_weight)),
        )
        success_loss = (
            F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="none") * cls_weight
        ).mean()

        edit_logits = model.success_logits(edited)
        fail_logits = model.success_logits(failure)
        succ_logits = model.success_logits(success)
        edit_success_loss = F.binary_cross_entropy_with_logits(edit_logits, torch.ones_like(edit_logits))
        rank_fail_loss = F.relu(args.ranking_margin - (edit_logits - fail_logits)).mean()
        rank_succ_loss = F.relu(args.ranking_margin - (succ_logits - fail_logits)).mean()
        ranking_loss = 0.5 * (rank_fail_loss + rank_succ_loss)

        loss = (
            delta_loss
            + args.lambda_cos * dir_loss
            + args.lambda_small * small_loss
            + args.lambda_success * (success_loss + edit_success_loss)
            + args.lambda_ranking * ranking_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            metrics = evaluate(model, val_pairs, futures, labels, target_futures_or_none, args, device, args.eval_max)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "direction_loss": float(dir_loss.detach().cpu()),
                "small_loss": float(small_loss.detach().cpu()),
                "success_loss": float(success_loss.detach().cpu()),
                "edit_success_loss": float(edit_success_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            score = (
                metrics["delta_cosine"]
                + metrics["target_proxy_cos_gain"]
                + metrics["failure_repulsion_gain"]
                + 0.1 * metrics["success_head_acc"]
            )
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(record), flush=True)

    assert best_state is not None and best is not None
    ckpt_path = args.output_dir / "token_immune_adapter.pt"
    torch.save(
        {
            "model_state": best_state,
            "future_dim": future_dim,
            "hidden_dim": int(args.hidden_dim),
            "future_token_shape": tuple(int(x) for x in futures[0].shape),
            "future_key": args.future_key,
            "alpha_repel": float(args.alpha_repel),
            "beta_attract": float(args.beta_attract),
            "target_mode": str(args.target_mode),
            "hybrid_manifold_weight": float(args.hybrid_manifold_weight),
            "hybrid_target_weight": float(args.hybrid_target_weight),
            "gate": float(args.gate),
            "lambda_success": float(args.lambda_success),
            "lambda_ranking": float(args.lambda_ranking),
            "ranking_margin": float(args.ranking_margin),
            "token_level": True,
        },
        ckpt_path,
    )
    summary = {
        "num_rows": int(len(rows)),
        "num_pairs": int(len(pairs)),
        "train_pairs": int(len(train_pairs)),
        "val_pairs": int(len(val_pairs)),
        "same_task_success": bool(args.same_task_success),
        "same_task_failure": bool(args.same_task_failure),
        "k_failure": int(args.k_failure),
        "target_mode": str(args.target_mode),
        "hybrid_manifold_weight": float(args.hybrid_manifold_weight),
        "hybrid_target_weight": float(args.hybrid_target_weight),
        "success_count": int(sum(labels)),
        "failure_count": int(len(labels) - sum(labels)),
        "lambda_success": float(args.lambda_success),
        "lambda_ranking": float(args.lambda_ranking),
        "ranking_margin": float(args.ranking_margin),
        "best": best,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "pairs.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in pairs))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
