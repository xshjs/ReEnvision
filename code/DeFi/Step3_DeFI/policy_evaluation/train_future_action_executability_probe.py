from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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


def future_trace_from_row(row: dict[str, Any]) -> Path | None:
    for key in ("trace_path", "collection_trace_path", "target_proxy_trace_path"):
        raw = str(row.get(key, "") or "")
        if raw and Path(raw).exists():
            return Path(raw)
    return None


def action_trace_from_row(row: dict[str, Any], action_trace_dirs: list[Path]) -> Path | None:
    trace = future_trace_from_row(row)
    if trace is None:
        return None
    action_name = trace.name.replace("future_trace_", "action_trace_")
    for trace_dir in [trace.parent.parent / "action_traces", *action_trace_dirs]:
        candidate = trace_dir / action_name
        if candidate.exists():
            return candidate
    return None


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "target_proxy_future" and "final_future" in data:
                arr = np.asarray(data["final_future"], dtype=np.float32)
            elif key == "base_future" and "original_base_future" in data:
                arr = np.asarray(data["base_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def future_summary(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    first = arr[0]
    last = arr[-1]
    delta = last - first
    return np.concatenate([mean, std, first, delta], axis=0).astype(np.float32)


def outcome_from_row(success: bool, remaining_steps: int, horizon: int) -> str:
    if success:
        return "success_within_horizon" if remaining_steps <= horizon else "success_after_horizon"
    return "failure_within_horizon" if remaining_steps <= horizon else "failure_after_horizon"


def chunk_starts(length: int, chunk_len: int, chunks_per_trace: int, success: bool, rng: random.Random) -> list[int]:
    if length <= chunk_len:
        return [0]
    max_start = length - chunk_len
    anchors = [0, max_start // 2, max_start]
    starts = {min(max_start, max(0, a)) for a in anchors[:chunks_per_trace]}
    if not success:
        tail_band = max(1, max_start // 4)
        starts.add(rng.randint(max(0, max_start - tail_band), max_start))
    target = min(chunks_per_trace, max_start + 1)
    if len(starts) < target:
        if success:
            candidates = list(range(0, max_start + 1))
        else:
            tail_band = max(1, max_start // 3)
            tail_candidates = list(range(max(0, max_start - tail_band), max_start + 1))
            full_candidates = list(range(0, max_start + 1))
            candidates = tail_candidates + [c for c in full_candidates if c not in tail_candidates]
        rng.shuffle(candidates)
        for candidate in candidates:
            starts.add(candidate)
            if len(starts) >= target:
                break
    return sorted(starts)[:chunks_per_trace]


def pad_chunk(actions: np.ndarray, chunk_len: int) -> np.ndarray:
    out = np.zeros((chunk_len, actions.shape[-1]), dtype=np.float32)
    take = min(chunk_len, actions.shape[0])
    out[:take] = actions[:take]
    return out


def load_rows(
    dataset_paths: list[Path],
    action_trace_dirs: list[Path],
    chunk_len: int,
    horizon: int,
    chunks_per_trace: int,
    base_future_key: str,
    target_future_key: str,
    seed: int,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    raw_count = 0
    for path in dataset_paths:
        for row in iter_jsonl(path):
            future_path = future_trace_from_row(row)
            action_path = action_trace_from_row(row, action_trace_dirs)
            if future_path is None or action_path is None:
                continue
            base = load_future(future_path, base_future_key)
            target = load_future(future_path, target_future_key)
            if base is None or target is None:
                continue
            try:
                with np.load(action_path) as data:
                    actions = np.asarray(data["actions"], dtype=np.float32)
            except Exception:
                continue
            if actions.ndim != 2 or actions.shape[0] <= 0:
                continue
            task = str(row.get("task", "unknown"))
            success = bool(row.get("success", False))
            starts = chunk_starts(actions.shape[0], chunk_len, chunks_per_trace, success, rng)
            base_sum = future_summary(base)
            target_sum = future_summary(target)
            for start in starts:
                remaining_steps = max(actions.shape[0] - start, 1)
                rows.append(
                    {
                        "task": task,
                        "success": success,
                        "outcome": outcome_from_row(success, remaining_steps, horizon),
                        "base_summary": base_sum,
                        "target_summary": target_sum,
                        "target_delta": (target_sum - base_sum).astype(np.float32),
                        "action_chunk": pad_chunk(actions[start : start + chunk_len], chunk_len),
                        "sequence_index": int(row.get("sequence_index", raw_count) or 0),
                        "row_id": len(rows),
                    }
                )
            raw_count += 1
            if max_rows > 0 and raw_count >= max_rows:
                return rows
    if not rows:
        raise ValueError("no usable rows")
    return rows


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    seqs = sorted({int(row["sequence_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row["sequence_index"]) in val else train_idx).append(idx)
    return train_idx, val_idx


def build_pairs(rows: list[dict[str, Any]], indices: list[int], negatives_per_positive: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_task: dict[str, list[int]] = defaultdict(list)
    by_task_success: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for idx in indices:
        task = str(rows[idx]["task"])
        by_task[task].append(idx)
        by_task_success[(task, bool(rows[idx]["success"]))].append(idx)

    pairs = []
    for idx in indices:
        row = rows[idx]
        pairs.append({"left": idx, "right": idx, "label": 1})
        task = str(row["task"])
        opposite = by_task_success.get((task, not bool(row["success"])), [])
        same_task = [j for j in by_task[task] if j != idx]
        pool = opposite if opposite else same_task
        if not pool:
            pool = [j for j in indices if j != idx]
        if not pool:
            continue
        chosen = [rng.choice(pool) for _ in range(negatives_per_positive)]
        for j in chosen:
            pairs.append({"left": idx, "right": j, "label": 0})
    rng.shuffle(pairs)
    if not pairs:
        raise ValueError("no pair samples built")
    return pairs


class FutureActionExecutabilityProbe(nn.Module):
    def __init__(self, action_dim: int, future_dim: int, z_dim: int, hidden_dim: int, dropout: float, num_tasks: int):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(future_dim * 3),
            nn.Linear(future_dim * 3, hidden_dim),
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

    def forward(self, action_chunk: torch.Tensor, base_summary: torch.Tensor, target_summary: torch.Tensor, target_delta: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.action_encoder(action_chunk.transpose(1, 2))
        a_mean = h.mean(dim=-1)
        a_max = h.amax(dim=-1)
        f = self.future_encoder(torch.cat([base_summary, target_summary, target_delta], dim=-1))
        z = self.joint(torch.cat([a_mean, a_max, f], dim=-1))
        return {
            "z": z,
            "exec_logit": self.exec_head(z).squeeze(-1),
            "outcome_logits": self.outcome_head(z),
            "task_logits": self.task_head(z),
        }


def make_batch(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    batch_ids: list[int],
    task_to_idx: dict[str, int],
    outcome_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = [pairs[i] for i in batch_ids]
    left_rows = [rows[p["left"]] for p in chosen]
    right_rows = [rows[p["right"]] for p in chosen]
    action_chunk = torch.from_numpy(np.stack([r["action_chunk"] for r in left_rows]).astype(np.float32)).to(device)
    base_summary = torch.from_numpy(np.stack([r["base_summary"] for r in left_rows]).astype(np.float32)).to(device)
    target_summary = torch.from_numpy(np.stack([r["target_summary"] for r in right_rows]).astype(np.float32)).to(device)
    target_delta = torch.from_numpy(np.stack([r["target_delta"] for r in right_rows]).astype(np.float32)).to(device)
    pair_label = torch.tensor([float(p["label"]) for p in chosen], dtype=torch.float32, device=device)
    outcome = torch.tensor([outcome_to_idx[r["outcome"]] for r in left_rows], dtype=torch.long, device=device)
    task = torch.tensor([task_to_idx[r["task"]] for r in left_rows], dtype=torch.long, device=device)
    return action_chunk, base_summary, target_summary, target_delta, pair_label, outcome, task


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
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


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    vals = []
    for cls in range(num_classes):
        tp = int(((pred == cls) & (labels == cls)).sum().item())
        fp = int(((pred == cls) & (labels != cls)).sum().item())
        fn = int(((pred != cls) & (labels == cls)).sum().item())
        vals.append((2 * tp) / max(2 * tp + fp + fn, 1))
    return float(sum(vals) / max(len(vals), 1))


def contrastive_loss(z: torch.Tensor, task: torch.Tensor, label: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    positive = ((task[:, None] == task[None, :]) | ((label[:, None] > 0.5) == (label[None, :] > 0.5))) & (~eye)
    if not bool(positive.any().item()):
        return z.new_zeros(())
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = positive.sum(dim=1).clamp_min(1)
    loss = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / pos_count.float())
    valid = positive.any(dim=1)
    return loss[valid].mean() if bool(valid.any().item()) else z.new_zeros(())


def evaluate(
    model: FutureActionExecutabilityProbe,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    indices: list[int],
    task_to_idx: dict[str, int],
    outcome_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        action_chunk, base_summary, target_summary, target_delta, pair_label, outcome, task = make_batch(
            rows, pairs, indices, task_to_idx, outcome_to_idx, device
        )
        out = model(action_chunk, base_summary, target_summary, target_delta)
        exec_loss = F.binary_cross_entropy_with_logits(out["exec_logit"], pair_label)
        outcome_loss = F.cross_entropy(out["outcome_logits"], outcome)
        task_loss = F.cross_entropy(out["task_logits"], task)
        rec = binary_metrics(out["exec_logit"], pair_label)
        rec.update(
            {
                "outcome_acc": float((out["outcome_logits"].argmax(dim=-1) == outcome).float().mean().item()),
                "outcome_macro_f1": macro_f1(out["outcome_logits"], outcome, len(outcome_to_idx)),
                "task_acc": float((out["task_logits"].argmax(dim=-1) == task).float().mean().item()),
                "exec_loss": float(exec_loss.item()),
                "outcome_loss": float(outcome_loss.item()),
                "task_loss": float(task_loss.item()),
            }
        )
        return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal R_exec(z_f, z_a) from future/action pairs.")
    parser.add_argument("--dataset-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--action-trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-future-key", default="base_future")
    parser.add_argument("--target-future-key", default="final_future")
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--chunks-per-trace", type=int, default=3)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
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

    rows = load_rows(
        dataset_paths=list(args.dataset_jsonl),
        action_trace_dirs=list(args.action_trace_dir),
        chunk_len=args.chunk_len,
        horizon=args.horizon,
        chunks_per_trace=args.chunks_per_trace,
        base_future_key=args.base_future_key,
        target_future_key=args.target_future_key,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    train_pairs = build_pairs(rows, train_idx, args.negatives_per_positive, args.seed)
    val_pairs = build_pairs(rows, val_idx, args.negatives_per_positive, args.seed + 17)
    tasks = sorted({str(row["task"]) for row in rows})
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    outcome_to_idx = {name: idx for idx, name in enumerate(OUTCOMES)}
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    model = FutureActionExecutabilityProbe(
        action_dim=int(rows[0]["action_chunk"].shape[-1]),
        future_dim=int(rows[0]["base_summary"].shape[-1]),
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_tasks=len(tasks),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "future_action_executability_probe.pt"
    history = []
    best = None
    best_score = -1.0

    for step in range(1, args.steps + 1):
        batch_ids = [rng.randrange(len(train_pairs)) for _ in range(args.batch_size)]
        action_chunk, base_summary, target_summary, target_delta, pair_label, outcome, task = make_batch(
            rows, train_pairs, batch_ids, task_to_idx, outcome_to_idx, device
        )
        out = model(action_chunk, base_summary, target_summary, target_delta)
        exec_loss = F.binary_cross_entropy_with_logits(out["exec_logit"], pair_label)
        outcome_loss = F.cross_entropy(out["outcome_logits"], outcome)
        task_loss = F.cross_entropy(out["task_logits"], task)
        con_loss = contrastive_loss(out["z"], task, pair_label, args.contrastive_temperature)
        loss = exec_loss + outcome_loss + 0.5 * task_loss + args.contrastive_weight * con_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            eval_ids = list(range(len(val_pairs)))
            rec = evaluate(model, rows, val_pairs, eval_ids, task_to_idx, outcome_to_idx, device)
            rec = {"step": step, "loss": float(loss.item()), "contrastive_loss": float(con_loss.item()), **rec}
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
                        "base_future_key": args.base_future_key,
                        "target_future_key": args.target_future_key,
                        "chunk_len": args.chunk_len,
                        "z_dim": args.z_dim,
                        "hidden_dim": args.hidden_dim,
                    },
                    ckpt_path,
                )

    summary = {
        "dataset_jsonl": [str(p) for p in args.dataset_jsonl],
        "action_trace_dirs": [str(p) for p in args.action_trace_dir],
        "num_rows": len(rows),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "chunk_len": args.chunk_len,
        "horizon": args.horizon,
        "negatives_per_positive": args.negatives_per_positive,
        "task_counts": dict(Counter(row["task"] for row in rows).most_common()),
        "success_counts": dict(Counter(bool(row["success"]) for row in rows).most_common()),
        "best": best,
        "history": history,
        "output_ckpt": str(ckpt_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
