from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FACTORS = ["none", "contact", "object_displacement", "object_identity", "drawer_slider_progress", "goal_completion"]


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
            arr = np.asarray(data[key], dtype=np.float32)
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def task_hash(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "little", signed=False) % dim
        vec[idx] += 1.0
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def factor_from_row(row: dict[str, Any], success: bool) -> str:
    if success:
        return "none"
    for key in ("factor_label", "wrong_component", "failure_factor", "selected_failure_factor", "applied_factor_mask"):
        factor = str(row.get(key, "") or "")
        if factor in FACTORS and factor != "none":
            return factor
    task = str(row.get("task", "")).lower()
    if any(x in task for x in ("drawer", "slider")):
        return "drawer_slider_progress"
    if any(x in task for x in ("push", "move", "place")):
        return "object_displacement"
    if any(x in task for x in ("lift", "stack", "rotate", "block")):
        return "contact"
    if any(x in task for x in ("led", "lightbulb")):
        return "goal_completion"
    return "goal_completion"


def trace_path_from_row(row: dict[str, Any]) -> str:
    for key in ("trace_path", "target_proxy_trace_path", "collection_trace_path"):
        value = str(row.get(key, "") or "")
        if value:
            return value
    return ""


def base_key_from_row(row: dict[str, Any], default_key: str) -> str:
    return str(row.get("f_base_key", "") or default_key)


def target_key_from_row(row: dict[str, Any]) -> str:
    return str(row.get("f_final_key", "") or "target_proxy_future")


def load_rows(paths: list[Path], default_future_key: str, task_dim: int, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        for row in iter_jsonl(path):
            trace = trace_path_from_row(row)
            if not trace or not Path(trace).exists():
                continue
            base = load_future(trace, base_key_from_row(row, default_future_key))
            if base is None:
                continue
            success = bool(row.get("success", row.get("label_success", False)))
            factor = factor_from_row(row, success)
            if factor not in FACTORS:
                continue
            key = (trace, row.get("sequence_index"), row.get("subtask_index"), row.get("task"))
            if key in seen:
                continue
            seen.add(key)
            target = load_future(trace, target_key_from_row(row))
            if target is None:
                target = base
            task = str(row.get("task", "unknown"))
            rows.append(
                {
                    "future": base.astype(np.float32),
                    "target": target.astype(np.float32),
                    "success": success,
                    "factor": factor,
                    "task": task,
                    "task_vec": task_hash(task, task_dim),
                    "subtask": float(row.get("subtask_index", 0) or 0) / 5.0,
                    "sequence_index": int(row.get("sequence_index", len(rows)) or 0),
                }
            )
            if max_examples > 0 and len(rows) >= max_examples:
                return rows
    if not rows:
        raise ValueError("no usable future rows")
    return rows


def task_balanced_success_failure_rows(rows: list[dict[str, Any]], seed: int, min_failures_per_task: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_task: dict[str, dict[bool, list[dict[str, Any]]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), {True: [], False: []})[bool(row["success"])].append(row)
    out = []
    for task in sorted(by_task):
        succ = list(by_task[task][True])
        fail = list(by_task[task][False])
        if len(succ) == 0 or len(fail) < min_failures_per_task:
            continue
        k = min(len(succ), len(fail))
        out.extend(rng.sample(succ, k))
        out.extend(rng.sample(fail, k))
    rng.shuffle(out)
    if not out:
        raise ValueError("task-balanced filtering removed all rows")
    return out


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row["sequence_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row["sequence_index"]) in val else train_idx).append(idx)
    return train_idx, val_idx


class UnifiedFutureEncoder(nn.Module):
    def __init__(
        self,
        future_dim: int,
        task_dim: int,
        z_dim: int = 256,
        hidden_dim: int = 512,
        num_factors: int = 6,
        encoder_type: str = "pooled",
        token_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.encoder_type = str(encoder_type)
        self.future_dim = int(future_dim)
        self.task_dim = int(task_dim)
        if self.encoder_type == "token":
            self.token_proj = nn.Sequential(nn.LayerNorm(future_dim), nn.Linear(future_dim, token_dim), nn.GELU())
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.attn_pool = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
            in_dim = token_dim + task_dim + 1
        else:
            in_dim = future_dim * 4 + task_dim + 1
            self.token_proj = None
            self.token_encoder = None
            self.attn_pool = None
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, z_dim), nn.LayerNorm(z_dim))
        self.risk_head = nn.Linear(z_dim, 1)
        self.factor_head = nn.Linear(z_dim, num_factors)
        self.delta_head = nn.Linear(z_dim, future_dim)

    def summarize(self, future: torch.Tensor) -> torch.Tensor:
        mean = future.mean(dim=1)
        std = future.std(dim=1, unbiased=False)
        first = future[:, 0]
        last = future[:, -1]
        return torch.cat([mean, std, first, last - first], dim=-1)

    def encode(self, future: torch.Tensor, task_vec: torch.Tensor, subtask: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "token":
            tokens = self.token_proj(future)
            tokens = self.token_encoder(tokens)
            weights = torch.softmax(self.attn_pool(tokens).squeeze(-1), dim=-1)
            pooled = torch.sum(tokens * weights[:, :, None], dim=1)
            x = torch.cat([pooled, task_vec, subtask[:, None]], dim=-1)
        else:
            x = torch.cat([self.summarize(future), task_vec, subtask[:, None]], dim=-1)
        return self.net(x)

    def forward(self, future: torch.Tensor, task_vec: torch.Tensor, subtask: torch.Tensor):
        z = self.encode(future, task_vec, subtask)
        return {
            "z": z,
            "risk_logit": self.risk_head(z).squeeze(-1),
            "factor_logits": self.factor_head(z),
            "delta_channel": self.delta_head(z),
        }


def batch(rows: list[dict[str, Any]], indices: list[int], device: torch.device, factor_to_idx: dict[str, int]):
    chosen = [rows[i] for i in indices]
    future = torch.from_numpy(np.stack([r["future"] for r in chosen])).to(device)
    target = torch.from_numpy(np.stack([r["target"] for r in chosen])).to(device)
    task_vec = torch.from_numpy(np.stack([r["task_vec"] for r in chosen])).to(device)
    sub = torch.tensor([r["subtask"] for r in chosen], device=device, dtype=torch.float32)
    success = torch.tensor([1.0 if r["success"] else 0.0 for r in chosen], device=device)
    factor = torch.tensor([factor_to_idx[r["factor"]] for r in chosen], device=device, dtype=torch.long)
    return future, target, task_vec, sub, success, factor


def balanced_indices(rows, train_idx, batch_size, factor_balanced: bool):
    if factor_balanced:
        by_factor = {}
        for i in train_idx:
            by_factor.setdefault(rows[i]["factor"], []).append(i)
        factors = [factor for factor, idxs in by_factor.items() if idxs]
        return [random.choice(by_factor[random.choice(factors)]) for _ in range(batch_size)]
    succ = [i for i in train_idx if rows[i]["success"]]
    fail = [i for i in train_idx if not rows[i]["success"]]
    out = []
    for _ in range(batch_size):
        pool = fail if random.random() < 0.5 and fail else succ
        out.append(random.choice(pool if pool else train_idx))
    return out


def eval_metrics(model, rows, val_idx, device, factor_to_idx, max_eval):
    model.eval()
    idx = val_idx[:max_eval] if max_eval > 0 else val_idx
    if not idx:
        return {}
    with torch.no_grad():
        future, target, task_vec, sub, success, factor = batch(rows, idx, device, factor_to_idx)
        out = model(future, task_vec, sub)
        risk = torch.sigmoid(out["risk_logit"]).detach().cpu().numpy()
        y_fail = (1.0 - success.detach().cpu().numpy()).astype(np.int64)
        pred_fail = (risk >= 0.5).astype(np.int64)
        factor_pred = out["factor_logits"].argmax(dim=-1)
        factor_acc = float((factor_pred == factor).float().mean().item())
        non_none = factor != factor_to_idx["none"]
        factor_fail_acc = float((factor_pred[non_none] == factor[non_none]).float().mean().item()) if bool(non_none.any()) else 0.0
        pos = risk[y_fail == 1]
        neg = risk[y_fail == 0]
        auc = 0.0
        if len(pos) and len(neg):
            auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
        z = F.normalize(out["z"], dim=-1)
        succ_z = z[success > 0.5]
        fail_z = z[success < 0.5]
        margin = 0.0
        if len(succ_z) and len(fail_z):
            margin = float((succ_z.mean(dim=0) - fail_z.mean(dim=0)).norm().item())
    return {
        "risk_auc": auc,
        "risk_acc": float((pred_fail == y_fail).mean()),
        "factor_acc": factor_acc,
        "factor_failure_acc": factor_fail_acc,
        "fail_rate": float(y_fail.mean()),
        "z_success_failure_margin": margin,
        "num_eval": int(len(idx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified future encoder z=Enc(F,o,tau).")
    parser.add_argument("--dataset-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", type=str, default="original_base_future")
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-type", choices=["pooled", "token"], default="pooled")
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-factor", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.1)
    parser.add_argument("--lambda-prototype", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-max", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--task-balanced-success-failure", action="store_true")
    parser.add_argument("--min-failures-per-task", type=int, default=1)
    parser.add_argument("--factor-balanced-batches", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows = load_rows(args.dataset_jsonl, args.future_key, args.task_dim, args.max_examples)
    raw_num_rows = len(rows)
    raw_factor_counts = Counter(row["factor"] for row in rows)
    if args.task_balanced_success_failure:
        rows = task_balanced_success_failure_rows(rows, args.seed, args.min_failures_per_task)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    factor_to_idx = {factor: idx for idx, factor in enumerate(FACTORS)}
    future_dim = int(rows[0]["future"].shape[-1])
    model = UnifiedFutureEncoder(
        future_dim,
        args.task_dim,
        args.z_dim,
        args.hidden_dim,
        len(FACTORS),
        encoder_type=args.encoder_type,
        token_dim=args.token_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_success = sum(1 for i in train_idx if rows[i]["success"])
    train_fail = len(train_idx) - train_success
    pos_weight = torch.tensor([max(train_success, 1) / max(train_fail, 1)], device=device)
    history = []
    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        idx = balanced_indices(rows, train_idx, max(1, args.batch_size), bool(args.factor_balanced_batches))
        future, target, task_vec, sub, success, factor = batch(rows, idx, device, factor_to_idx)
        out = model(future, task_vec, sub)
        fail_label = 1.0 - success
        risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], fail_label, pos_weight=pos_weight)
        factor_loss = F.cross_entropy(out["factor_logits"], factor)
        target_delta = target - future
        channel_delta = target_delta.mean(dim=1)
        delta_loss = F.smooth_l1_loss(out["delta_channel"], channel_delta)
        z = F.normalize(out["z"], dim=-1)
        proto_loss = z.new_tensor(0.0)
        succ_z = z[success > 0.5]
        fail_z = z[success < 0.5]
        if len(succ_z) and len(fail_z):
            proto_loss = F.relu(1.0 - (succ_z.mean(dim=0) - fail_z.mean(dim=0)).norm())
        loss = risk_loss + args.lambda_factor * factor_loss + args.lambda_delta * delta_loss + args.lambda_prototype * proto_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            metrics = eval_metrics(model, rows, val_idx, device, factor_to_idx, args.eval_max)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "risk_loss": float(risk_loss.detach().cpu()),
                "factor_loss": float(factor_loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "prototype_loss": float(proto_loss.detach().cpu()),
                **metrics,
            }
            history.append(record)
            score = record.get("risk_auc", 0.0) + record.get("factor_failure_acc", 0.0)
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(record), flush=True)

    assert best_state is not None and best is not None
    ckpt_path = args.output_dir / "unified_future_encoder.pt"
    torch.save(
        {
            "model_type": "unified_future_encoder",
            "model_state": best_state,
            "future_dim": future_dim,
            "task_dim": int(args.task_dim),
            "z_dim": int(args.z_dim),
            "hidden_dim": int(args.hidden_dim),
            "encoder_type": args.encoder_type,
            "token_dim": int(args.token_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "factors": FACTORS,
            "factor_to_idx": factor_to_idx,
        },
        ckpt_path,
    )
    summary = {
        "dataset_jsonl": [str(p) for p in args.dataset_jsonl],
        "raw_num_examples": raw_num_rows,
        "raw_factor_counts": dict(raw_factor_counts.most_common()),
        "task_balanced_success_failure": bool(args.task_balanced_success_failure),
        "min_failures_per_task": int(args.min_failures_per_task),
        "factor_balanced_batches": bool(args.factor_balanced_batches),
        "encoder_type": args.encoder_type,
        "token_dim": int(args.token_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "num_examples": len(rows),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "factor_counts": dict(Counter(row["factor"] for row in rows).most_common()),
        "success_count": sum(1 for row in rows if row["success"]),
        "failure_count": sum(1 for row in rows if not row["success"]),
        "best": best,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
