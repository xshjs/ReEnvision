from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTCOMES = [
    "success_within_horizon",
    "success_after_horizon",
    "failure_within_horizon",
    "failure_after_horizon",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def action_trace_from_row(row: dict[str, Any], action_trace_dirs: list[Path]) -> Path | None:
    for key in ("trace_path", "collection_trace_path", "target_proxy_trace_path"):
        raw = str(row.get(key, "") or "")
        if not raw:
            continue
        future_path = Path(raw)
        action_name = future_path.name.replace("future_trace_", "action_trace_")
        for trace_dir in [future_path.parent.parent / "action_traces", *action_trace_dirs]:
            candidate = trace_dir / action_name
            if candidate.exists():
                return candidate
    return None


def future_trace_from_row(row: dict[str, Any]) -> Path | None:
    for key in ("trace_path", "collection_trace_path", "target_proxy_trace_path"):
        raw = str(row.get(key, "") or "")
        if raw and Path(raw).exists():
            return Path(raw)
    return None


def load_future_summary(path: Path, future_key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if future_key in data:
                arr = np.asarray(data[future_key], dtype=np.float32)
            elif "base_future" in data:
                arr = np.asarray(data["base_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    first = arr[0]
    last = arr[-1]
    return np.concatenate([mean, std, first, last - first], axis=0).astype(np.float32)


def outcome_from_row(success: bool, remaining_steps: int, horizon: int) -> str:
    if success:
        return "success_within_horizon" if remaining_steps <= horizon else "success_after_horizon"
    return "failure_within_horizon" if remaining_steps <= horizon else "failure_after_horizon"


def sample_chunk_starts(length: int, chunk_len: int, chunks_per_trace: int, success: bool, rng: random.Random) -> list[int]:
    if length <= chunk_len:
        return [0]
    max_start = length - chunk_len
    if chunks_per_trace <= 1:
        return [max_start if not success else rng.randint(0, max_start)]
    anchors = [0, max_start // 2, max_start]
    starts = {min(max_start, max(0, a)) for a in anchors[:chunks_per_trace]}
    if not success:
        tail_band = max(1, max_start // 4)
        starts.add(rng.randint(max(0, max_start - tail_band), max_start))
    while len(starts) < min(chunks_per_trace, max_start + 1):
        starts.add(rng.randint(0, max_start) if success else rng.randint(max(0, max_start - max(1, max_start // 3)), max_start))
    return sorted(starts)[:chunks_per_trace]


def pad_chunk(actions: np.ndarray, chunk_len: int) -> np.ndarray:
    out = np.zeros((chunk_len, actions.shape[-1]), dtype=np.float32)
    take = min(chunk_len, actions.shape[0])
    out[:take] = actions[:take]
    return out


def load_samples(
    dataset_paths: list[Path],
    action_trace_dirs: list[Path],
    chunk_len: int,
    horizon: int,
    chunks_per_trace: int,
    future_key: str,
    seed: int,
    max_rows: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    rows = []
    task_set = set()
    seen = set()
    row_count = 0
    for path in dataset_paths:
        for row in iter_jsonl(path):
            action_path = action_trace_from_row(row, action_trace_dirs)
            future_path = future_trace_from_row(row)
            if action_path is None or future_path is None:
                continue
            try:
                with np.load(action_path) as data:
                    actions = np.asarray(data["actions"], dtype=np.float32)
            except Exception:
                continue
            future_summary = load_future_summary(future_path, future_key)
            if future_summary is None or actions.ndim != 2 or actions.shape[0] <= 0:
                continue
            task = str(row.get("task", "unknown"))
            success = bool(row.get("success", False))
            starts = sample_chunk_starts(actions.shape[0], chunk_len, chunks_per_trace, success, rng)
            task_set.add(task)
            for start in starts:
                key = (str(action_path), int(start))
                if key in seen:
                    continue
                seen.add(key)
                remaining_steps = max(actions.shape[0] - start, 1)
                rows.append(
                    {
                        "actions": pad_chunk(actions[start : start + chunk_len], chunk_len),
                        "future_summary": future_summary,
                        "task": task,
                        "success": success,
                        "outcome": outcome_from_row(success, remaining_steps, horizon),
                        "sequence_index": int(row.get("sequence_index", row_count) or 0),
                    }
                )
            row_count += 1
            if max_rows > 0 and row_count >= max_rows:
                return rows, sorted(task_set)
    if not rows:
        raise ValueError("no usable action/future pairs found")
    return rows, sorted(task_set)


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    seqs = sorted({int(row["sequence_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row["sequence_index"]) in val else train_idx).append(idx)
    return train_idx, val_idx


class ActionExecutabilityProbe(nn.Module):
    def __init__(self, action_dim: int, future_dim: int, z_dim: int, hidden_dim: int, dropout: float, num_tasks: int):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(future_dim),
            nn.Linear(future_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.joint = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )
        self.exec_head = nn.Linear(z_dim, 1)
        self.outcome_head = nn.Linear(z_dim, len(OUTCOMES))
        self.task_head = nn.Linear(z_dim, num_tasks)

    def forward(self, actions: torch.Tensor, future_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.action_encoder(actions.transpose(1, 2))
        a_mean = h.mean(dim=-1)
        a_max = h.amax(dim=-1)
        f = self.future_encoder(future_summary)
        z = self.joint(torch.cat([a_mean, a_max, f], dim=-1))
        return {
            "z": z,
            "exec_logit": self.exec_head(z).squeeze(-1),
            "outcome_logits": self.outcome_head(z),
            "task_logits": self.task_head(z),
        }


def make_batch(
    rows: list[dict[str, Any]],
    indices: list[int],
    task_to_idx: dict[str, int],
    outcome_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = [rows[i] for i in indices]
    actions = torch.from_numpy(np.stack([row["actions"] for row in chosen]).astype(np.float32)).to(device)
    future_summary = torch.from_numpy(np.stack([row["future_summary"] for row in chosen]).astype(np.float32)).to(device)
    exec_target = torch.tensor([1.0 if row["success"] else 0.0 for row in chosen], dtype=torch.float32, device=device)
    outcome = torch.tensor([outcome_to_idx[row["outcome"]] for row in chosen], dtype=torch.long, device=device)
    task = torch.tensor([task_to_idx[row["task"]] for row in chosen], dtype=torch.long, device=device)
    return actions, future_summary, exec_target, outcome, task


def sample_balanced_indices(rows: list[dict[str, Any]], train_idx: list[int], batch_size: int, rng: random.Random) -> list[int]:
    by_group: dict[tuple[bool, str], list[int]] = {}
    for idx in train_idx:
        key = (bool(rows[idx]["success"]), str(rows[idx]["outcome"]))
        by_group.setdefault(key, []).append(idx)
    groups = [group for group, ids in by_group.items() if ids]
    out = []
    while len(out) < batch_size:
        group = rng.choice(groups)
        out.append(rng.choice(by_group[group]))
    return out


def binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = prob >= 0.5
    gold = labels >= 0.5
    tp = int((pred & gold).sum().item())
    fp = int((pred & ~gold).sum().item())
    fn = int((~pred & gold).sum().item())
    tn = int((~pred & ~gold).sum().item())
    return {
        "exec_acc": float((pred == gold).float().mean().item()),
        "exec_precision": tp / max(tp + fp, 1),
        "exec_recall": tp / max(tp + fn, 1),
        "exec_f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    scores = []
    for cls in range(num_classes):
        tp = int(((pred == cls) & (labels == cls)).sum().item())
        fp = int(((pred == cls) & (labels != cls)).sum().item())
        fn = int(((pred != cls) & (labels == cls)).sum().item())
        scores.append((2 * tp) / max(2 * tp + fp + fn, 1))
    return float(sum(scores) / max(len(scores), 1))


def supervised_contrastive_loss(z: torch.Tensor, task: torch.Tensor, success: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    positive = ((task[:, None] == task[None, :]) | ((success[:, None] > 0.5) == (success[None, :] > 0.5))) & (~eye)
    if not bool(positive.any().item()):
        return z.new_zeros(())
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = positive.sum(dim=1).clamp_min(1)
    loss = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / pos_count.float())
    valid = positive.any(dim=1)
    return loss[valid].mean() if bool(valid.any().item()) else z.new_zeros(())


def evaluate(
    model: ActionExecutabilityProbe,
    rows: list[dict[str, Any]],
    indices: list[int],
    task_to_idx: dict[str, int],
    outcome_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        actions, future_summary, exec_target, outcome, task = make_batch(rows, indices, task_to_idx, outcome_to_idx, device)
        out = model(actions, future_summary)
        exec_loss = F.binary_cross_entropy_with_logits(out["exec_logit"], exec_target)
        outcome_loss = F.cross_entropy(out["outcome_logits"], outcome)
        task_loss = F.cross_entropy(out["task_logits"], task)
        metrics = binary_metrics_from_logits(out["exec_logit"], exec_target)
        metrics.update(
            {
                "outcome_acc": float((out["outcome_logits"].argmax(dim=-1) == outcome).float().mean().item()),
                "outcome_macro_f1": macro_f1_from_logits(out["outcome_logits"], outcome, len(outcome_to_idx)),
                "task_acc": float((out["task_logits"].argmax(dim=-1) == task).float().mean().item()),
                "exec_loss": float(exec_loss.item()),
                "outcome_loss": float(outcome_loss.item()),
                "task_loss": float(task_loss.item()),
            }
        )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal action executability probe R_exec(context, z_a).")
    parser.add_argument("--dataset-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--action-trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--chunks-per-trace", type=int, default=3)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows, tasks = load_samples(
        dataset_paths=list(args.dataset_jsonl),
        action_trace_dirs=list(args.action_trace_dir),
        chunk_len=args.chunk_len,
        horizon=args.horizon,
        chunks_per_trace=args.chunks_per_trace,
        future_key=args.future_key,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    outcome_to_idx = {name: idx for idx, name in enumerate(OUTCOMES)}

    model = ActionExecutabilityProbe(
        action_dim=int(rows[0]["actions"].shape[-1]),
        future_dim=int(rows[0]["future_summary"].shape[-1]),
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_tasks=len(tasks),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "action_executability_probe.pt"
    history = []
    best = None
    best_score = -1.0

    for step in range(1, args.steps + 1):
        batch_idx = sample_balanced_indices(rows, train_idx, args.batch_size, rng)
        actions, future_summary, exec_target, outcome, task = make_batch(rows, batch_idx, task_to_idx, outcome_to_idx, device)
        out = model(actions, future_summary)
        exec_loss = F.binary_cross_entropy_with_logits(out["exec_logit"], exec_target)
        outcome_loss = F.cross_entropy(out["outcome_logits"], outcome)
        task_loss = F.cross_entropy(out["task_logits"], task)
        contrastive_loss = supervised_contrastive_loss(out["z"], task, exec_target, args.contrastive_temperature)
        loss = exec_loss + outcome_loss + 0.5 * task_loss + args.contrastive_weight * contrastive_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            rec = evaluate(model, rows, val_idx, task_to_idx, outcome_to_idx, device)
            rec = {"step": step, "loss": float(loss.item()), "contrastive_loss": float(contrastive_loss.item()), **rec}
            history.append(rec)
            score = rec["exec_f1"] + 0.5 * rec["outcome_macro_f1"] + 0.25 * rec["task_acc"]
            if score > best_score:
                best_score = score
                best = rec
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "tasks": tasks,
                        "outcomes": OUTCOMES,
                        "chunk_len": args.chunk_len,
                        "horizon": args.horizon,
                        "future_key": args.future_key,
                        "z_dim": args.z_dim,
                        "hidden_dim": args.hidden_dim,
                    },
                    ckpt_path,
                )

    summary = {
        "dataset_jsonl": [str(p) for p in args.dataset_jsonl],
        "action_trace_dirs": [str(p) for p in args.action_trace_dir],
        "num_rows": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "chunk_len": args.chunk_len,
        "horizon": args.horizon,
        "chunks_per_trace": args.chunks_per_trace,
        "task_counts": dict(Counter(row["task"] for row in rows).most_common()),
        "outcome_counts": dict(Counter(row["outcome"] for row in rows).most_common()),
        "success_counts": dict(Counter(bool(row["success"]) for row in rows).most_common()),
        "best": best,
        "history": history,
        "output_ckpt": str(ckpt_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
