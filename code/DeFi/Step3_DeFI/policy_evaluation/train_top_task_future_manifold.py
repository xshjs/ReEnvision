from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_TASKS = ["push_into_drawer", "place_in_slider", "push_red_block_right", "stack_block"]


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


def task_hash(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % dim
        vec[idx] += 1.0
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_examples(path: Path, tasks: set[str], future_key: str):
    rows, futures = [], []
    for row in iter_jsonl(path):
        task = str(row.get("task") or "")
        if task not in tasks:
            continue
        trace = row.get("trace_path")
        if not trace or not Path(str(trace)).exists():
            continue
        future = load_future(trace, future_key)
        if future is None:
            continue
        rows.append(row)
        futures.append(future)
    if not rows:
        raise ValueError("no usable rows for selected tasks")
    return rows, futures


def make_task_centers(rows: list[dict[str, Any]], futures: list[np.ndarray]):
    by_task_label = defaultdict(list)
    for i, row in enumerate(rows):
        by_task_label[(str(row.get("task")), bool(row.get("success")))].append(i)
    pooled = np.stack([f.mean(axis=0).astype(np.float32) for f in futures], axis=0)
    pooled_norm = normalize_rows(pooled)
    centers = {}
    for i, row in enumerate(rows):
        task = str(row.get("task"))
        opposite = not bool(row.get("success"))
        same_success = by_task_label.get((task, True), [])
        same_failure = [j for j in by_task_label.get((task, False), []) if j != i]
        if not same_success or not same_failure:
            continue

        def nearest(candidates):
            sims = (pooled_norm[i][None, :] @ pooled_norm[candidates].T).reshape(-1)
            order = np.argsort(-sims)[: min(5, len(candidates))]
            chosen = [candidates[int(k)] for k in order]
            w = np.maximum(sims[order], 0.0).astype(np.float32)
            if float(w.sum()) <= 1e-8:
                w = np.ones(len(chosen), dtype=np.float32)
            w /= max(float(w.sum()), 1e-8)
            return np.tensordot(w, np.stack([futures[j] for j in chosen]), axes=(0, 0)).astype(np.float32)

        centers[i] = {
            "success_center": nearest(same_success),
            "failure_center": nearest(same_failure),
            "has_opposite": opposite,
        }
    return centers


class TopTaskFutureManifold(nn.Module):
    def __init__(self, future_dim: int, task_dim: int = 128, z_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.task_dim = int(task_dim)
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(future_dim * 4 + task_dim + 1),
            nn.Linear(future_dim * 4 + task_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.risk_head = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nav_in_dim = future_dim * 4 + task_dim + z_dim * 3 + 3
        self.nav_net = nn.Sequential(
            nn.LayerNorm(nav_in_dim),
            nn.Linear(nav_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )
        self.nav_gate = nn.Sequential(
            nn.LayerNorm(nav_in_dim),
            nn.Linear(nav_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def summary(future: torch.Tensor) -> torch.Tensor:
        return torch.cat([future.mean(dim=1), future.std(dim=1, unbiased=False), future[:, 0], future[:, -1]], dim=-1)

    def encode(self, future: torch.Tensor, task_vec: torch.Tensor, subtask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.summary(future), task_vec, subtask[:, None]], dim=-1)
        return F.normalize(self.future_encoder(x), dim=-1)

    def risk(self, z: torch.Tensor) -> torch.Tensor:
        return self.risk_head(z).squeeze(-1)

    def navigate(self, base, success_center, failure_center, task_vec, subtask):
        z_b = self.encode(base, task_vec, subtask)
        z_s = self.encode(success_center, task_vec, subtask)
        z_f = self.encode(failure_center, task_vec, subtask)
        risk = self.risk(z_b)
        sim_s = F.cosine_similarity(z_b, z_s, dim=-1)
        sim_f = F.cosine_similarity(z_b, z_f, dim=-1)
        boundary_margin = sim_f - sim_s
        z_b_tok = z_b.unsqueeze(1).expand(-1, base.shape[1], -1)
        z_s_tok = z_s.unsqueeze(1).expand(-1, base.shape[1], -1)
        z_f_tok = z_f.unsqueeze(1).expand(-1, base.shape[1], -1)
        task_tok = task_vec.unsqueeze(1).expand(-1, base.shape[1], -1)
        sub_tok = subtask[:, None, None].expand(-1, base.shape[1], 1)
        risk_tok = risk[:, None, None].expand(-1, base.shape[1], 1)
        margin_tok = boundary_margin[:, None, None].expand(-1, base.shape[1], 1)
        direction = success_center - failure_center
        attract = success_center - base
        repel = base - failure_center
        x = torch.cat(
            [base, direction, attract, repel, task_tok, z_b_tok, z_s_tok, z_f_tok, risk_tok, margin_tok, sub_tok],
            dim=-1,
        )
        raw = self.nav_net(x)
        gate = self.nav_gate(x)
        return gate * raw, {"gate": gate, "risk": risk, "boundary_margin": boundary_margin}


def batch_tensors(batch, futures, centers, task_dim, device):
    ids = [b["id"] for b in batch]
    base = torch.from_numpy(np.stack([futures[i] for i in ids])).to(device)
    success = torch.from_numpy(np.stack([centers[i]["success_center"] for i in ids])).to(device)
    failure = torch.from_numpy(np.stack([centers[i]["failure_center"] for i in ids])).to(device)
    task_vec = torch.from_numpy(np.stack([task_hash(str(b["task"]), task_dim) for b in batch])).to(device)
    sub = torch.tensor([float(b.get("subtask_index", 0)) / 5.0 for b in batch], dtype=torch.float32, device=device)
    y_fail = torch.tensor([0.0 if b["success"] else 1.0 for b in batch], dtype=torch.float32, device=device)
    return base, success, failure, task_vec, sub, y_fail


def cosine_loss(x, y):
    return 1.0 - F.cosine_similarity(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1), dim=1).mean()


def evaluate(model, eval_items, futures, centers, task_dim, rollout_gate, device, max_eval):
    model.eval()
    items = eval_items[:max_eval] if max_eval > 0 else eval_items
    if not items:
        return {}
    rec = defaultdict(list)
    with torch.no_grad():
        for item in items:
            base, success, failure, task_vec, sub, y_fail = batch_tensors([item], futures, centers, task_dim, device)
            z_b = model.encode(base, task_vec, sub)
            z_s = model.encode(success, task_vec, sub)
            z_f = model.encode(failure, task_vec, sub)
            risk = model.risk(z_b)
            delta, aux = model.navigate(base, success, failure, task_vec, sub)
            edited = base + float(rollout_gate) * delta
            target_delta = success - base + 0.5 * (base - failure)
            rec["delta_cosine"].append(float((1.0 - cosine_loss(delta, target_delta)).item()))
            rec["gate_mean"].append(float(aux["gate"].mean().item()))
            rec["edit_norm"].append(float(torch.norm((float(rollout_gate) * delta).reshape(-1), p=2).item()))
            rec["success_cos_gain"].append(float((F.cosine_similarity(edited.reshape(1, -1), success.reshape(1, -1)) - F.cosine_similarity(base.reshape(1, -1), success.reshape(1, -1))).item()))
            rec["failure_repulsion_gain"].append(float((F.cosine_similarity(base.reshape(1, -1), failure.reshape(1, -1)) - F.cosine_similarity(edited.reshape(1, -1), failure.reshape(1, -1))).item()))
            rec["margin_z"].append(float((F.cosine_similarity(z_b, z_s) - F.cosine_similarity(z_b, z_f)).item()))
            rec["risk"].append(float(risk.item()))
            rec["y_fail"].append(float(y_fail.item()))
    risk = np.asarray(rec["risk"])
    y = np.asarray(rec["y_fail"])
    pred = risk > 0
    out = {k: float(np.mean(v)) for k, v in rec.items() if k not in ("risk", "y_fail")}
    out["risk_acc_at_0"] = float((pred == (y > 0.5)).mean())
    out["fail_rate"] = float(y.mean())
    out["num_eval"] = int(len(items))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train top-failure-task future manifold encoder + navigator.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--future-key", type=str, default="original_base_future")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rollout-gate", type=float, default=0.005)
    parser.add_argument("--lambda-contrast", type=float, default=1.0)
    parser.add_argument("--lambda-risk", type=float, default=0.5)
    parser.add_argument("--lambda-nav", type=float, default=5.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--margin", type=float, default=0.2)
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

    rows, futures = load_examples(args.dataset_jsonl, set(args.tasks), args.future_key)
    centers = make_task_centers(rows, futures)
    items = []
    for i, row in enumerate(rows):
        if i not in centers:
            continue
        items.append({"id": i, "task": str(row.get("task")), "subtask_index": int(row.get("subtask_index", 0) or 0), "success": bool(row.get("success"))})
    if not items:
        raise ValueError("no items have same-task success and failure centers")

    seqs = sorted({int(rows[x["id"]].get("sequence_index", x["id"]) or x["id"]) for x in items})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(seqs)
    val_seqs = set(seqs[: max(1, int(round(len(seqs) * args.val_ratio)))])
    train = [x for x in items if int(rows[x["id"]].get("sequence_index", x["id"]) or x["id"]) not in val_seqs]
    val = [x for x in items if int(rows[x["id"]].get("sequence_index", x["id"]) or x["id"]) in val_seqs]
    train_s = [x for x in train if x["success"]]
    train_f = [x for x in train if not x["success"]]
    if not train_s or not train_f:
        raise ValueError("train split needs both success and failure")

    model = TopTaskFutureManifold(futures[0].shape[-1], args.task_dim, args.z_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best, best_state = [], None, None

    for step in range(1, args.steps + 1):
        model.train()
        half = max(1, args.batch_size // 2)
        batch = random.choices(train_f, k=half) + random.choices(train_s, k=max(1, args.batch_size - half))
        random.shuffle(batch)
        base, success, failure, task_vec, sub, y_fail = batch_tensors(batch, futures, centers, args.task_dim, device)
        z_b = model.encode(base, task_vec, sub)
        z_s = model.encode(success, task_vec, sub)
        z_f = model.encode(failure, task_vec, sub)
        sim_s = F.cosine_similarity(z_b, z_s, dim=-1)
        sim_f = F.cosine_similarity(z_b, z_f, dim=-1)
        loss_contrast = F.relu(float(args.margin) + sim_f - sim_s).mean()
        risk = model.risk(z_b)
        loss_risk = F.binary_cross_entropy_with_logits(risk, y_fail)
        delta, _ = model.navigate(base, success, failure, task_vec, sub)
        target_delta = success - base + 0.5 * (base - failure)
        fail_mask = y_fail > 0.5
        loss_nav = cosine_loss(delta[fail_mask], target_delta[fail_mask]) + F.smooth_l1_loss(delta[fail_mask], target_delta[fail_mask])
        succ_mask = ~fail_mask
        loss_preserve = delta[succ_mask].pow(2).mean() if succ_mask.any() else torch.zeros((), device=device)
        loss = (
            float(args.lambda_contrast) * loss_contrast
            + float(args.lambda_risk) * loss_risk
            + float(args.lambda_nav) * loss_nav
            + float(args.lambda_preserve) * loss_preserve
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "contrast_loss": float(loss_contrast.detach().cpu()),
                "risk_loss": float(loss_risk.detach().cpu()),
                "nav_loss": float(loss_nav.detach().cpu()),
                "preserve_loss": float(loss_preserve.detach().cpu()),
                **evaluate(model, val, futures, centers, args.task_dim, args.rollout_gate, device, args.eval_max),
            }
            print(json.dumps(rec), flush=True)
            history.append(rec)
            score = rec.get("success_cos_gain", 0.0) + rec.get("failure_repulsion_gain", 0.0) + 0.01 * rec.get("risk_acc_at_0", 0.0)
            if best is None or score > best["score"]:
                best = {**rec, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt = args.output_dir / "top_task_future_manifold.pt"
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "future_shape": tuple(futures[0].shape),
            "future_dim": int(futures[0].shape[-1]),
            "task_dim": int(args.task_dim),
            "z_dim": int(args.z_dim),
            "hidden_dim": int(args.hidden_dim),
            "tasks": list(args.tasks),
            "args": vars(args),
            "best": best,
        },
        ckpt,
    )
    task_counts = {}
    for task in args.tasks:
        task_counts[task] = {
            "success": int(sum(1 for x in items if x["task"] == task and x["success"])),
            "failure": int(sum(1 for x in items if x["task"] == task and not x["success"])),
        }
    summary = {
        "num_rows": len(rows),
        "num_items": len(items),
        "num_train": len(train),
        "num_val": len(val),
        "task_counts": task_counts,
        "best": best,
        "checkpoint": str(ckpt),
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
