from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        with np.load(str(path)) as data:
            if key not in data.files:
                return None
            x = np.asarray(data[key], dtype=np.float32)
    except Exception:
        return None
    return x if x.ndim == 2 else None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def task_hash(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % dim
        vec[idx] += 1.0
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def load_rows(dataset_jsonl: Path, future_key: str, target_key: str) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray | None]]:
    rows, futures, targets = [], [], []
    for row in iter_jsonl(dataset_jsonl):
        trace = row.get("trace_path")
        if not trace or not Path(str(trace)).exists():
            continue
        future = load_future(trace, future_key)
        if future is None:
            continue
        target = load_future(trace, target_key)
        rows.append(row)
        futures.append(future)
        targets.append(target)
    if not rows:
        raise ValueError("no usable future rows")
    return rows, futures, targets


def nearest_center(
    idx: int,
    pooled_norm: np.ndarray,
    futures: list[np.ndarray],
    candidates: list[int],
    k: int,
) -> tuple[np.ndarray, list[int], list[float]]:
    sims = (pooled_norm[idx][None, :] @ pooled_norm[candidates].T).reshape(-1)
    order = np.argsort(-sims)[: max(1, min(k, len(candidates)))]
    chosen = [candidates[int(i)] for i in order]
    weights = np.maximum(sims[order], 0.0).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones(len(chosen), dtype=np.float32)
    weights /= max(float(weights.sum()), 1e-8)
    center = np.tensordot(weights, np.stack([futures[j] for j in chosen], axis=0), axes=(0, 0)).astype(np.float32)
    return center, chosen, [float(x) for x in weights.tolist()]


def build_examples(rows, futures, targets, k: int, same_task: bool, require_task_both: bool) -> list[dict[str, Any]]:
    pooled = np.stack([f.mean(axis=0).astype(np.float32) for f in futures], axis=0)
    pooled_norm = normalize_rows(pooled)
    tasks = [str(row.get("task") or "unknown") for row in rows]
    success_ids = [i for i, row in enumerate(rows) if bool(row.get("success"))]
    failure_ids = [i for i, row in enumerate(rows) if not bool(row.get("success"))]
    if not success_ids or not failure_ids:
        raise ValueError("need both success and failure futures")
    success_set = set(success_ids)
    failure_set = set(failure_ids)
    task_success = {task for i, task in enumerate(tasks) if i in success_set}
    task_failure = {task for i, task in enumerate(tasks) if i in failure_set}
    valid_tasks = task_success & task_failure

    examples = []
    for idx, row in enumerate(rows):
        task = tasks[idx]
        if require_task_both and task not in valid_tasks:
            continue
        succ_candidates = [j for j in success_ids if (not same_task or tasks[j] == task)] or success_ids
        fail_candidates = [j for j in failure_ids if j != idx and (not same_task or tasks[j] == task)]
        fail_candidates = fail_candidates or [j for j in failure_ids if j != idx] or failure_ids
        succ_center, succ_chosen, succ_w = nearest_center(idx, pooled_norm, futures, succ_candidates, k)
        fail_center, fail_chosen, fail_w = nearest_center(idx, pooled_norm, futures, fail_candidates, k)
        target = targets[idx] if targets[idx] is not None else succ_center
        success = bool(row.get("success"))
        if success:
            nav_delta = np.zeros_like(futures[idx], dtype=np.float32)
        else:
            # Navigate away from the local failure basin and toward the local success manifold.
            nav_delta = (succ_center - futures[idx]) + 0.5 * (futures[idx] - fail_center)
        examples.append(
            {
                "idx": idx,
                "task": task,
                "subtask_index": int(row.get("subtask_index", 0) or 0),
                "success": success,
                "base": futures[idx],
                "target": target.astype(np.float32),
                "success_center": succ_center,
                "failure_center": fail_center,
                "target_delta": nav_delta.astype(np.float32),
                "succ_ids": succ_chosen,
                "fail_ids": fail_chosen,
                "succ_weights": succ_w,
                "fail_weights": fail_w,
            }
        )
    return examples


class FutureManifoldNavigator(nn.Module):
    def __init__(self, future_dim: int, task_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.task_dim = int(task_dim)
        in_dim = future_dim * 4 + task_dim + 1
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Linear(hidden_dim, future_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.energy_head = nn.Sequential(
            nn.LayerNorm(future_dim + task_dim + 1),
            nn.Linear(future_dim + task_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, base, success_center, failure_center, task_vec, subtask_index):
        direction = success_center - failure_center
        attract = success_center - base
        repel = base - failure_center
        task = task_vec.unsqueeze(1).expand(-1, base.shape[1], -1)
        sub = subtask_index[:, None, None].expand(-1, base.shape[1], 1)
        x = torch.cat([base, direction, attract, repel, task, sub], dim=-1)
        h = self.encoder(x)
        raw_delta = self.delta(h)
        gate = self.gate(h)
        return gate * raw_delta, {"gate": gate, "raw_delta": raw_delta}

    def energy(self, future, task_vec, subtask_index):
        pooled = future.mean(dim=1)
        sub = subtask_index[:, None]
        x = torch.cat([pooled, task_vec, sub], dim=-1)
        return self.energy_head(x).squeeze(-1)


def batch_to_tensors(batch: list[dict[str, Any]], task_dim: int, device):
    base = torch.from_numpy(np.stack([b["base"] for b in batch])).to(device)
    target = torch.from_numpy(np.stack([b["target"] for b in batch])).to(device)
    success = torch.from_numpy(np.stack([b["success_center"] for b in batch])).to(device)
    failure = torch.from_numpy(np.stack([b["failure_center"] for b in batch])).to(device)
    target_delta = torch.from_numpy(np.stack([b["target_delta"] for b in batch])).to(device)
    task_vec = torch.from_numpy(np.stack([task_hash(b["task"], task_dim) for b in batch])).to(device)
    sub = torch.tensor([float(b["subtask_index"]) / 5.0 for b in batch], dtype=torch.float32, device=device)
    labels = torch.tensor([1.0 if b["success"] else 0.0 for b in batch], dtype=torch.float32, device=device)
    return base, target, success, failure, target_delta, task_vec, sub, labels


def cosine_loss(x, y):
    return 1.0 - F.cosine_similarity(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1), dim=1).mean()


def evaluate(model, examples, task_dim: int, rollout_gate: float, device, max_eval: int):
    model.eval()
    subset = examples[:max_eval] if max_eval > 0 else examples
    losses, cosines, gates, success_gains, failure_gains, target_gains, edit_norms = [], [], [], [], [], [], []
    e_success, e_failure, risk_correct = [], [], []
    with torch.no_grad():
        for ex in subset:
            base, target, success, failure, target_delta, task_vec, sub, label = batch_to_tensors([ex], task_dim, device)
            pred_delta, aux = model(base, success, failure, task_vec, sub)
            edited = base + rollout_gate * pred_delta
            losses.append(float(F.smooth_l1_loss(pred_delta, target_delta).item()))
            cosines.append(float((1.0 - cosine_loss(pred_delta, target_delta)).item()))
            gates.append(float(aux["gate"].mean().item()))
            edit_norms.append(float(torch.norm((rollout_gate * pred_delta).reshape(-1), p=2).item()))
            success_gains.append(float((F.cosine_similarity(edited.reshape(1, -1), success.reshape(1, -1)) - F.cosine_similarity(base.reshape(1, -1), success.reshape(1, -1))).item()))
            failure_gains.append(float((F.cosine_similarity(base.reshape(1, -1), failure.reshape(1, -1)) - F.cosine_similarity(edited.reshape(1, -1), failure.reshape(1, -1))).item()))
            target_gains.append(float((F.cosine_similarity(edited.reshape(1, -1), target.reshape(1, -1)) - F.cosine_similarity(base.reshape(1, -1), target.reshape(1, -1))).item()))
            eb = model.energy(base, task_vec, sub)
            es = model.energy(success, task_vec, sub)
            ef = model.energy(failure, task_vec, sub)
            e_success.append(float(es.item()))
            e_failure.append(float(ef.item()))
            risk_correct.append(float((ef > es).item()))
    return {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "delta_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "gate_mean": float(np.mean(gates)) if gates else 0.0,
        "edit_norm": float(np.mean(edit_norms)) if edit_norms else 0.0,
        "success_cos_gain": float(np.mean(success_gains)) if success_gains else 0.0,
        "failure_repulsion_gain": float(np.mean(failure_gains)) if failure_gains else 0.0,
        "target_proxy_cos_gain": float(np.mean(target_gains)) if target_gains else 0.0,
        "energy_success_mean": float(np.mean(e_success)) if e_success else 0.0,
        "energy_failure_mean": float(np.mean(e_failure)) if e_failure else 0.0,
        "energy_pair_acc": float(np.mean(risk_correct)) if risk_correct else 0.0,
        "num_eval": int(len(subset)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train task-conditioned future manifold navigator.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", type=str, default="original_base_future")
    parser.add_argument("--target-key", type=str, default="target_proxy_future")
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--require-task-success-failure", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--failure-batch-frac", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-direction", type=float, default=1.0)
    parser.add_argument("--lambda-preserve-success", type=float, default=10.0)
    parser.add_argument("--lambda-target", type=float, default=0.5)
    parser.add_argument("--lambda-energy", type=float, default=0.2)
    parser.add_argument("--energy-margin", type=float, default=1.0)
    parser.add_argument("--rollout-gate", type=float, default=0.02)
    parser.add_argument("--eval-max", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows, futures, targets = load_rows(args.dataset_jsonl, args.future_key, args.target_key)
    examples = build_examples(rows, futures, targets, args.k, args.same_task_only, args.require_task_success_failure)
    seqs = sorted({int(rows[ex["idx"]].get("sequence_index", ex["idx"]) or ex["idx"]) for ex in examples})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(seqs)
    val_seqs = set(seqs[: max(1, int(round(len(seqs) * args.val_ratio)))])
    train = [ex for ex in examples if int(rows[ex["idx"]].get("sequence_index", ex["idx"]) or ex["idx"]) not in val_seqs]
    val = [ex for ex in examples if int(rows[ex["idx"]].get("sequence_index", ex["idx"]) or ex["idx"]) in val_seqs]
    if not train or not val:
        train = examples
        val = examples[: max(1, min(len(examples), args.eval_max))]

    model = FutureManifoldNavigator(futures[0].shape[-1], args.task_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = None
    best_state = None
    train_failures = [ex for ex in train if not ex["success"]]
    train_successes = [ex for ex in train if ex["success"]]

    for step in range(1, args.steps + 1):
        model.train()
        batch_size = max(1, args.batch_size)
        n_fail = min(len(train_failures), max(1, int(round(batch_size * float(args.failure_batch_frac)))))
        n_success = max(0, batch_size - n_fail)
        batch = []
        if train_failures:
            batch.extend(random.choices(train_failures, k=n_fail))
        if train_successes and n_success > 0:
            batch.extend(random.choices(train_successes, k=n_success))
        if not batch:
            batch = random.choices(train, k=batch_size)
        base, target, success, failure, target_delta, task_vec, sub, label = batch_to_tensors(batch, args.task_dim, device)
        pred_delta, aux = model(base, success, failure, task_vec, sub)
        edited = base + float(args.rollout_gate) * pred_delta
        success_mask = label > 0.5
        failure_mask = ~success_mask
        if failure_mask.any():
            loss_delta = F.smooth_l1_loss(pred_delta[failure_mask], target_delta[failure_mask])
            loss_direction = cosine_loss(pred_delta[failure_mask], target_delta[failure_mask])
        else:
            loss_delta = torch.zeros((), device=device)
            loss_direction = torch.zeros((), device=device)
        if success_mask.any():
            loss_preserve = pred_delta[success_mask].pow(2).mean()
        else:
            loss_preserve = torch.zeros((), device=device)
        loss_target = F.smooth_l1_loss(edited, target)
        e_success = model.energy(success, task_vec, sub)
        e_failure = model.energy(failure, task_vec, sub)
        e_base = model.energy(base, task_vec, sub)
        e_edited = model.energy(edited, task_vec, sub)
        loss_energy_rank = F.relu(float(args.energy_margin) + e_success - e_failure).mean()
        loss_energy_nav = F.relu(e_edited - e_base).mean()
        loss = (
            loss_delta
            + float(args.lambda_direction) * loss_direction
            + float(args.lambda_preserve_success) * loss_preserve
            + float(args.lambda_target) * loss_target
            + float(args.lambda_energy) * (loss_energy_rank + loss_energy_nav)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "delta_loss": float(loss_delta.detach().cpu()),
                "direction_loss": float(loss_direction.detach().cpu()),
                "preserve_success_loss": float(loss_preserve.detach().cpu()),
                "target_loss": float(loss_target.detach().cpu()),
                "energy_rank_loss": float(loss_energy_rank.detach().cpu()),
                "energy_nav_loss": float(loss_energy_nav.detach().cpu()),
                **evaluate(model, val, args.task_dim, args.rollout_gate, device, args.eval_max),
            }
            print(json.dumps(rec), flush=True)
            history.append(rec)
            score = rec["success_cos_gain"] + rec["failure_repulsion_gain"] + rec["target_proxy_cos_gain"] + 0.01 * rec["energy_pair_acc"]
            if best is None or score > best["score"]:
                best = {**rec, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt_path = args.output_dir / "future_manifold_navigator.pt"
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "future_shape": tuple(futures[0].shape),
            "future_dim": int(futures[0].shape[-1]),
            "task_dim": int(args.task_dim),
            "hidden_dim": int(args.hidden_dim),
            "args": vars(args),
            "best": best,
        },
        ckpt_path,
    )
    summary = {
        "num_rows": len(rows),
        "num_examples": len(examples),
        "num_train": len(train),
        "num_val": len(val),
        "success_count": int(sum(1 for ex in examples if ex["success"])),
        "failure_count": int(sum(1 for ex in examples if not ex["success"])),
        "task_counts": {
            task: {
                "success": int(sum(1 for ex in examples if ex["task"] == task and ex["success"])),
                "failure": int(sum(1 for ex in examples if ex["task"] == task and not ex["success"])),
            }
            for task in sorted({ex["task"] for ex in examples})
        },
        "best": best,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
