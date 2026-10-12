from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FACTORS = [
    "none",
    "contact",
    "object_displacement",
    "object_identity",
    "drawer_slider_progress",
    "goal_completion",
]

FACTOR_BINARY = [
    "none",
    "failure",
]

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
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    return None


def action_trace_from_row(row: dict[str, Any], action_trace_dirs: list[Path]) -> Path | None:
    future_path = future_trace_from_row(row)
    if future_path is not None:
        action_dir = future_path.parent.parent / "action_traces"
        action_name = future_path.name.replace("future_trace_", "action_trace_")
        candidate = action_dir / action_name
        if candidate.exists():
            return candidate
        for trace_dir in action_trace_dirs:
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


def future_summary_exec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def _row_vector(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr if arr.size > 0 else None


def _delta_stats_vector(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if not isinstance(value, dict) or not bool(value.get("available", False)):
        return None
    keys = ("delta_norm", "delta_mean", "delta_std", "delta_max_abs")
    try:
        return np.asarray([float(value.get(name, 0.0) or 0.0) for name in keys], dtype=np.float32)
    except Exception:
        return None


def transition_from_row(row: dict[str, Any], future_path: Path) -> tuple[np.ndarray | None, str]:
    robot_start = _row_vector(row, "strong_gt_robot_obs_start")
    robot_end = _row_vector(row, "strong_gt_robot_obs_end")
    scene_start = _row_vector(row, "strong_gt_scene_obs_start")
    scene_end = _row_vector(row, "strong_gt_scene_obs_end")
    if robot_start is not None and robot_end is not None and scene_start is not None and scene_end is not None:
        return np.concatenate([robot_end - robot_start, scene_end - scene_start], axis=0).astype(np.float32), "strong_gt_obs_delta"

    robot_stats = _delta_stats_vector(row, "strong_gt_robot_obs_delta")
    scene_stats = _delta_stats_vector(row, "strong_gt_scene_obs_delta")
    if robot_stats is not None and scene_stats is not None:
        return np.concatenate([robot_stats, scene_stats], axis=0).astype(np.float32), "strong_gt_delta_stats"

    base_future = load_future(future_path, "base_future")
    target_future = load_future(future_path, "target_proxy_future")
    if base_future is None or target_future is None:
        return None, "missing"
    return (future_summary_exec(target_future) - future_summary_exec(base_future)).astype(np.float32), "future_delta"


def factor_from_row(row: dict[str, Any], factor_mode: str) -> str:
    if bool(row.get("success", False)):
        return "none"
    if factor_mode == "binary_fail":
        return "failure"
    for key in ("factor_label", "wrong_component", "failure_factor", "selected_failure_factor", "applied_factor_mask"):
        value = str(row.get(key, "") or "")
        if value in FACTORS:
            return value
    return "goal_completion"


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
    fractions = [0.0, 0.5, 1.0]
    starts = {min(max_start, max(0, int(round(max_start * frac)))) for frac in fractions[:chunks_per_trace]}
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
            # Prefer failure-near-tail starts, but keep the full range available so short traces
            # cannot get stuck when the tail band has too few unique positions.
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


def load_samples(
    dataset_paths: list[Path],
    action_trace_dirs: list[Path],
    chunk_len: int,
    horizon: int,
    chunks_per_trace: int,
    seed: int,
    factor_mode: str,
    progress_every_rows: int,
    skip_debug_row: int,
    debug_row_start: int,
    debug_row_end: int,
    debug_log_path: Path | None,
    max_rows: int = 0,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rng = random.Random(seed)
    rows = []
    task_set = set()
    factor_set = set(FACTOR_BINARY if factor_mode == "binary_fail" else FACTORS)
    transition_dim = None
    row_count = 0
    seen_rows = 0
    reason_counts: Counter[str] = Counter()
    start_time = time.time()

    def emit_event(payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        print(text, flush=True)
        if debug_log_path is not None:
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

    def emit_progress(force: bool = False) -> None:
        if not force and (progress_every_rows <= 0 or seen_rows % progress_every_rows != 0):
            return
        elapsed = max(time.time() - start_time, 1e-6)
        payload = {
            "stage": "load_samples",
            "seen_rows": seen_rows,
            "accepted_source_rows": row_count,
            "chunk_rows_built": len(rows),
            "elapsed_sec": round(elapsed, 2),
            "rows_per_sec": round(seen_rows / elapsed, 2),
            "reason_counts": dict(reason_counts),
        }
        emit_event(payload)

    def debug_enabled_for_row(row_idx: int) -> bool:
        if debug_row_start <= 0 and debug_row_end <= 0:
            return True
        if debug_row_start > 0 and row_idx < debug_row_start:
            return False
        if debug_row_end > 0 and row_idx > debug_row_end:
            return False
        return True

    for path in dataset_paths:
        emit_event({"stage": "load_samples_start", "dataset": str(path)})
        for row in iter_jsonl(path):
            seen_rows += 1
            if skip_debug_row > 0 and seen_rows == skip_debug_row:
                reason_counts["skip_debug_row"] += 1
                emit_event({"stage": "skip_debug_row", "row": seen_rows})
                emit_progress(force=True)
                continue
            if debug_enabled_for_row(seen_rows):
                emit_event(
                    {
                        "stage": "debug_row",
                        "row": seen_rows,
                        "trace_id": row.get("trace_id"),
                        "task": row.get("task"),
                    }
                )
            action_path = action_trace_from_row(row, action_trace_dirs)
            future_path = future_trace_from_row(row)
            if action_path is None or future_path is None:
                if future_path is None:
                    reason_counts["missing_future_trace"] += 1
                else:
                    reason_counts["missing_action_trace"] += 1
                emit_progress()
                continue
            transition, transition_source = transition_from_row(row, future_path)
            if transition is None:
                reason_counts["missing_transition"] += 1
                emit_progress()
                continue
            if transition_dim is None:
                transition_dim = int(transition.shape[-1])
            elif int(transition.shape[-1]) != transition_dim:
                reason_counts["transition_dim_mismatch"] += 1
                emit_progress()
                continue
            try:
                if debug_enabled_for_row(seen_rows):
                    emit_event(
                        {
                            "stage": "trace_load_start",
                            "row": seen_rows,
                            "path": str(action_path),
                        }
                    )
                with np.load(action_path) as data:
                    if debug_enabled_for_row(seen_rows):
                        emit_event(
                            {
                                "stage": "after_trace_load",
                                "row": seen_rows,
                                "keys": list(data.keys()),
                            }
                        )
                    actions = np.asarray(data["actions"], dtype=np.float32)
                    if debug_enabled_for_row(seen_rows):
                        emit_event(
                            {
                                "stage": "actions_loaded",
                                "row": seen_rows,
                                "shape": list(actions.shape),
                                "dtype": str(actions.dtype),
                            }
                        )
                if debug_enabled_for_row(seen_rows):
                    emit_event(
                        {
                            "stage": "trace_load_done",
                            "row": seen_rows,
                            "path": str(action_path),
                        }
                    )
            except Exception:
                reason_counts["action_npz_load_error"] += 1
                emit_progress()
                continue
            if actions.ndim != 2 or actions.shape[0] <= 0:
                reason_counts["bad_action_shape"] += 1
                emit_progress()
                continue
            task = str(row.get("task", "unknown"))
            factor = factor_from_row(row, factor_mode)
            success = bool(row.get("success", False))
            task_set.add(task)
            factor_set.add(factor)
            starts = sample_chunk_starts(actions.shape[0], chunk_len, chunks_per_trace, success, rng)
            if debug_enabled_for_row(seen_rows):
                emit_event(
                    {
                        "stage": "chunk_build_start",
                        "row": seen_rows,
                        "trace_len": int(actions.shape[0]),
                        "num_chunks": len(starts),
                    }
                )
            for chunk_idx, start in enumerate(starts):
                if debug_enabled_for_row(seen_rows):
                    emit_event(
                        {
                            "stage": "chunk",
                            "row": seen_rows,
                            "chunk_idx": int(chunk_idx),
                            "start": int(start),
                        }
                    )
                chunk = pad_chunk(actions[start : start + chunk_len], chunk_len)
                remaining_steps = max(actions.shape[0] - start, 1)
                outcome = outcome_from_row(success, remaining_steps, horizon)
                rows.append(
                    {
                        "actions": chunk,
                        "task": task,
                        "factor": factor,
                        "outcome": outcome,
                        "success": success,
                        "transition": transition,
                        "transition_source": transition_source,
                        "trace_length": int(actions.shape[0]),
                        "chunk_start": int(start),
                        "sequence_index": int(row.get("sequence_index", row_count) or 0),
                        "subtask_index": int(row.get("subtask_index", 0) or 0),
                    }
                )
            if debug_enabled_for_row(seen_rows):
                emit_event(
                    {
                        "stage": "chunk_build_done",
                        "row": seen_rows,
                    }
                )
            row_count += 1
            reason_counts["accepted_source_rows"] += 1
            emit_progress()
            if max_rows > 0 and row_count >= max_rows:
                emit_progress(force=True)
                return rows, sorted(task_set), sorted(factor_set)
    if not rows:
        emit_progress(force=True)
        raise ValueError("no usable action chunks found")
    emit_progress(force=True)
    return rows, sorted(task_set), sorted(factor_set)


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    seqs = sorted({int(row["sequence_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row["sequence_index"]) in val else train_idx).append(idx)
    return train_idx, val_idx


class ActionChunkEncoder(nn.Module):
    def __init__(self, action_dim: int, z_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        x = actions.transpose(1, 2)
        h = self.temporal(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(pooled)


class ProbeModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        z_dim: int,
        hidden_dim: int,
        dropout: float,
        num_tasks: int,
        num_factors: int,
        transition_dim: int,
    ):
        super().__init__()
        self.encoder = ActionChunkEncoder(action_dim, z_dim, hidden_dim, dropout)
        self.task_head = nn.Linear(z_dim, num_tasks)
        self.factor_head = nn.Linear(z_dim, num_factors)
        self.outcome_head = nn.Linear(z_dim, len(OUTCOMES))
        self.transition_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, transition_dim),
        )

    def forward(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(actions)
        return {
            "z": z,
            "task_logits": self.task_head(z),
            "factor_logits": self.factor_head(z),
            "outcome_logits": self.outcome_head(z),
            "transition_pred": self.transition_head(z),
        }


def make_batch(
    rows: list[dict[str, Any]],
    indices: list[int],
    task_to_idx: dict[str, int],
    factor_to_idx: dict[str, int],
    outcome_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen = [rows[i] for i in indices]
    actions = torch.from_numpy(np.stack([row["actions"] for row in chosen]).astype(np.float32)).to(device)
    task = torch.tensor([task_to_idx[row["task"]] for row in chosen], dtype=torch.long, device=device)
    factor = torch.tensor([factor_to_idx[row["factor"]] for row in chosen], dtype=torch.long, device=device)
    outcome = torch.tensor([outcome_to_idx[row["outcome"]] for row in chosen], dtype=torch.long, device=device)
    transition = torch.from_numpy(np.stack([row["transition"] for row in chosen]).astype(np.float32)).to(device)
    return actions, task, factor, outcome, transition


def sample_balanced_indices(rows: list[dict[str, Any]], train_idx: list[int], batch_size: int, rng: random.Random) -> list[int]:
    by_group: dict[tuple[str, str], list[int]] = {}
    for idx in train_idx:
        key = (rows[idx]["factor"], rows[idx]["outcome"])
        by_group.setdefault(key, []).append(idx)
    groups = [group for group, ids in by_group.items() if ids]
    out = []
    while len(out) < batch_size:
        group = rng.choice(groups)
        out.append(rng.choice(by_group[group]))
    return out


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    scores = []
    for cls in range(num_classes):
        tp = int(((pred == cls) & (labels == cls)).sum().item())
        fp = int(((pred == cls) & (labels != cls)).sum().item())
        fn = int(((pred != cls) & (labels == cls)).sum().item())
        denom = max(2 * tp + fp + fn, 1)
        scores.append((2 * tp) / denom)
    return float(sum(scores) / max(len(scores), 1))


def balanced_eval_indices(rows: list[dict[str, Any]], indices: list[int], key_name: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_key: dict[str, list[int]] = {}
    for idx in indices:
        by_key.setdefault(str(rows[idx][key_name]), []).append(idx)
    if not by_key:
        return indices
    keep = min(len(v) for v in by_key.values() if v)
    if keep <= 0:
        return indices
    out = []
    for key, vals in sorted(by_key.items()):
        out.extend(rng.sample(vals, keep))
    rng.shuffle(out)
    return out


def supervised_contrastive_loss(z: torch.Tensor, task: torch.Tensor, factor: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    positive = ((task[:, None] == task[None, :]) | (factor[:, None] == factor[None, :])) & (~eye)
    if not bool(positive.any().item()):
        return z.new_zeros(())
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = positive.sum(dim=1).clamp_min(1)
    loss = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / pos_count.float())
    valid = positive.any(dim=1)
    return loss[valid].mean() if bool(valid.any().item()) else z.new_zeros(())


def evaluate(
    model: ProbeModel,
    rows: list[dict[str, Any]],
    indices: list[int],
    task_to_idx: dict[str, int],
    factor_to_idx: dict[str, int],
    outcome_to_idx: dict[str, int],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        actions, task, factor, outcome, transition = make_batch(rows, indices, task_to_idx, factor_to_idx, outcome_to_idx, device)
        out = model(actions)
        task_loss = F.cross_entropy(out["task_logits"], task)
        factor_loss = F.cross_entropy(out["factor_logits"], factor)
        outcome_loss = F.cross_entropy(out["outcome_logits"], outcome)
        transition_loss = F.smooth_l1_loss(out["transition_pred"], transition)
        transition_cos = float(
            F.cosine_similarity(out["transition_pred"], transition, dim=-1).mean().item()
        )
        factor_bal_idx = balanced_eval_indices(rows, indices, "factor", seed)
        outcome_bal_idx = balanced_eval_indices(rows, indices, "outcome", seed)
        factor_macro = macro_f1_from_logits(out["factor_logits"], factor, len(factor_to_idx))
        outcome_macro = macro_f1_from_logits(out["outcome_logits"], outcome, len(outcome_to_idx))
        balanced = {}
        for name, subset in (("factor_bal", factor_bal_idx), ("outcome_bal", outcome_bal_idx)):
            a2, t2, f2, o2, _ = make_batch(rows, subset, task_to_idx, factor_to_idx, outcome_to_idx, device)
            out2 = model(a2)
            if name == "factor_bal":
                balanced["factor_bal_acc"] = accuracy(out2["factor_logits"], f2)
                balanced["factor_bal_macro_f1"] = macro_f1_from_logits(out2["factor_logits"], f2, len(factor_to_idx))
            else:
                balanced["outcome_bal_acc"] = accuracy(out2["outcome_logits"], o2)
                balanced["outcome_bal_macro_f1"] = macro_f1_from_logits(out2["outcome_logits"], o2, len(outcome_to_idx))
        return {
            "task_acc": accuracy(out["task_logits"], task),
            "factor_acc": accuracy(out["factor_logits"], factor),
            "outcome_acc": accuracy(out["outcome_logits"], outcome),
            "factor_macro_f1": factor_macro,
            "outcome_macro_f1": outcome_macro,
            "task_loss": float(task_loss.item()),
            "factor_loss": float(factor_loss.item()),
            "outcome_loss": float(outcome_loss.item()),
            "transition_loss": float(transition_loss.item()),
            "transition_cos": transition_cos,
            **balanced,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal action-latent probe z_a from saved action traces.")
    parser.add_argument("--dataset-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--action-trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every-rows", type=int, default=500)
    parser.add_argument("--skip-debug-row", type=int, default=0)
    parser.add_argument("--debug-row-start", type=int, default=0)
    parser.add_argument("--debug-row-end", type=int, default=0)
    parser.add_argument("--debug-log-path", type=Path, default=None)
    parser.add_argument("--factor-mode", choices=("multiclass", "binary_fail"), default="multiclass")
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(json.dumps({"stage": "main_start", "device_arg": args.device, "seed": args.seed}, ensure_ascii=False), flush=True)
    rows, tasks, factors = load_samples(
        dataset_paths=list(args.dataset_jsonl),
        action_trace_dirs=list(args.action_trace_dir),
        chunk_len=args.chunk_len,
        horizon=args.horizon,
        chunks_per_trace=args.chunks_per_trace,
        seed=args.seed,
        factor_mode=args.factor_mode,
        progress_every_rows=args.progress_every_rows,
        skip_debug_row=args.skip_debug_row,
        debug_row_start=args.debug_row_start,
        debug_row_end=args.debug_row_end,
        debug_log_path=args.debug_log_path,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {
                "stage": "load_samples_done",
                "num_rows": len(rows),
                "num_tasks": len(tasks),
                "num_factors": len(factors),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    print(
        json.dumps(
            {
                "stage": "split_done",
                "num_train": len(train_idx),
                "num_val": len(val_idx),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(json.dumps({"stage": "device_resolved", "device": str(device)}, ensure_ascii=False), flush=True)

    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    factor_to_idx = {factor: idx for idx, factor in enumerate(factors)}
    outcome_to_idx = {name: idx for idx, name in enumerate(OUTCOMES)}

    model = ProbeModel(
        action_dim=int(rows[0]["actions"].shape[-1]),
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_tasks=len(tasks),
        num_factors=len(factors),
        transition_dim=int(rows[0]["transition"].shape[-1]),
    ).to(device)
    print(
        json.dumps(
            {
                "stage": "model_init_done",
                "action_dim": int(rows[0]["actions"].shape[-1]),
                "transition_dim": int(rows[0]["transition"].shape[-1]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    history = []
    best = None
    best_score = -1.0
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "action_latent_probe.pt"

    for step in range(1, args.steps + 1):
        if step == 1:
            print(json.dumps({"stage": "first_batch_start"}, ensure_ascii=False), flush=True)
        batch_idx = sample_balanced_indices(rows, train_idx, args.batch_size, rng)
        actions, task, factor, outcome, transition = make_batch(rows, batch_idx, task_to_idx, factor_to_idx, outcome_to_idx, device)
        out = model(actions)
        task_loss = F.cross_entropy(out["task_logits"], task)
        factor_loss = F.cross_entropy(out["factor_logits"], factor)
        outcome_loss = F.cross_entropy(out["outcome_logits"], outcome)
        transition_loss = F.smooth_l1_loss(out["transition_pred"], transition)
        contrastive_loss = supervised_contrastive_loss(out["z"], task, factor, args.contrastive_temperature)
        loss = (
            float(args.task_weight) * task_loss
            + float(args.factor_weight) * factor_loss
            + float(args.outcome_weight) * outcome_loss
            + float(args.transition_weight) * transition_loss
            + float(args.contrastive_weight) * contrastive_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1:
            print(json.dumps({"stage": "first_batch_done", "loss": float(loss.item())}, ensure_ascii=False), flush=True)

        if step == 1 or step % 100 == 0 or step == args.steps:
            metrics = evaluate(model, rows, val_idx, task_to_idx, factor_to_idx, outcome_to_idx, device, args.seed)
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "contrastive_loss": float(contrastive_loss.item()),
                "transition_loss_train": float(transition_loss.item()),
                **metrics,
            }
            history.append(rec)
            score = (
                rec["transition_cos"]
                + rec["factor_bal_macro_f1"]
                + rec["outcome_bal_macro_f1"]
                + 0.5 * rec["factor_macro_f1"]
                + 0.5 * rec["outcome_macro_f1"]
                + 0.25 * rec["task_acc"]
            )
            if score > best_score:
                best_score = score
                best = rec
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "tasks": tasks,
                        "factors": factors,
                        "outcomes": OUTCOMES,
                        "chunk_len": args.chunk_len,
                        "horizon": args.horizon,
                        "z_dim": args.z_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "factor_mode": args.factor_mode,
                        "transition_dim": int(rows[0]["transition"].shape[-1]),
                        "task_weight": args.task_weight,
                        "factor_weight": args.factor_weight,
                        "outcome_weight": args.outcome_weight,
                        "transition_weight": args.transition_weight,
                        "contrastive_weight": args.contrastive_weight,
                        "contrastive_temperature": args.contrastive_temperature,
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
        "factor_mode": args.factor_mode,
        "task_weight": args.task_weight,
        "factor_weight": args.factor_weight,
        "outcome_weight": args.outcome_weight,
        "transition_weight": args.transition_weight,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_temperature": args.contrastive_temperature,
        "task_counts": dict(Counter(row["task"] for row in rows).most_common()),
        "factor_counts": dict(Counter(row["factor"] for row in rows).most_common()),
        "outcome_counts": dict(Counter(row["outcome"] for row in rows).most_common()),
        "transition_source_counts": dict(Counter(row["transition_source"] for row in rows).most_common()),
        "transition_dim": int(rows[0]["transition"].shape[-1]),
        "transition_norm_mean": float(np.mean([float(np.linalg.norm(row["transition"])) for row in rows])),
        "transition_norm_std": float(np.std([float(np.linalg.norm(row["transition"])) for row in rows])),
        "best": best,
        "history": history,
        "output_ckpt": str(ckpt_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
