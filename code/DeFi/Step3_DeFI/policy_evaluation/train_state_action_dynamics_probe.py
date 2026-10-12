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


def action_trace_from_row(row: dict[str, Any], future_trace: Path | None, action_trace_dirs: list[Path]) -> Path | None:
    for key in ("action_trace_path", "collection_action_trace_path"):
        raw = str(row.get(key, "") or "")
        if raw and Path(raw).exists():
            return Path(raw)
    if future_trace is not None:
        action_name = future_trace.name.replace("future_trace_", "action_trace_")
        for trace_dir in [future_trace.parent.parent / "action_traces", *action_trace_dirs]:
            candidate = trace_dir / action_name
            if candidate.exists():
                return candidate
    return None


def row_vec(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return arr


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
        candidates = list(range(0, max_start + 1))
        rng.shuffle(candidates)
        for c in candidates:
            starts.add(c)
            if len(starts) >= target:
                break
    return sorted(starts)[:chunks_per_trace]


def pad_chunk(actions: np.ndarray, chunk_len: int) -> np.ndarray:
    out = np.zeros((chunk_len, actions.shape[-1]), dtype=np.float32)
    take = min(chunk_len, actions.shape[0])
    out[:take] = actions[:take]
    return out


def load_samples(
    dataset_jsonl: Path,
    action_trace_dirs: list[Path],
    chunk_len: int,
    chunks_per_trace: int,
    seed: int,
    max_source_rows: int,
) -> tuple[list[dict[str, Any]], int, Counter[str]]:
    rng = random.Random(seed)
    rows = []
    seen = 0
    reasons: Counter[str] = Counter()
    for row in iter_jsonl(dataset_jsonl):
        future_trace = future_trace_from_row(row)
        action_trace = action_trace_from_row(row, future_trace, action_trace_dirs)
        if action_trace is None:
            reasons["missing_action_trace"] += 1
            continue
        robot_start = row_vec(row, "strong_gt_robot_obs_start")
        robot_end = row_vec(row, "strong_gt_robot_obs_end")
        scene_start = row_vec(row, "strong_gt_scene_obs_start")
        scene_end = row_vec(row, "strong_gt_scene_obs_end")
        if robot_start is None or robot_end is None or scene_start is None or scene_end is None:
            reasons["missing_strong_gt_obs"] += 1
            continue
        state = np.concatenate([robot_start, scene_start], axis=0).astype(np.float32)
        next_state = np.concatenate([robot_end, scene_end], axis=0).astype(np.float32)
        transition = (next_state - state).astype(np.float32)
        try:
            with np.load(action_trace) as data:
                actions = np.asarray(data["actions"], dtype=np.float32)
        except Exception:
            reasons["bad_action_trace"] += 1
            continue
        if actions.ndim != 2 or actions.shape[0] <= 0:
            reasons["empty_action_trace"] += 1
            continue
        success = bool(row.get("success", False))
        starts = chunk_starts(actions.shape[0], chunk_len, chunks_per_trace, success, rng)
        for start in starts:
            rows.append(
                {
                    "task": str(row.get("task", "unknown")),
                    "success": success,
                    "sequence_index": int(row.get("sequence_index", seen) or 0),
                    "subtask_index": int(row.get("subtask_index", 0) or 0),
                    "state": state,
                    "next_state": next_state,
                    "transition": transition,
                    "action_chunk": pad_chunk(actions[start : start + chunk_len], chunk_len),
                }
            )
        seen += 1
        reasons["accepted_source_rows"] += 1
        if max_source_rows > 0 and seen >= max_source_rows:
            break
    if not rows:
        raise ValueError("no usable rows")
    return rows, seen, reasons


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    seqs = sorted({int(row["sequence_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_set = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row["sequence_index"]) in val_set else train_idx).append(idx)
    return train_idx, val_idx


def compute_stats(rows: list[dict[str, Any]], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([rows[i]["transition"] for i in indices], axis=0).astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


def build_task_pools(rows: list[dict[str, Any]], indices: list[int]) -> tuple[dict[str, list[int]], list[int]]:
    by_task: dict[str, list[int]] = defaultdict(list)
    all_ids: list[int] = []
    for idx in indices:
        task = str(rows[idx]["task"])
        by_task[task].append(idx)
        all_ids.append(idx)
    return by_task, all_ids


class StateActionDynamicsProbe(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        z_dim: int,
        dropout: float,
        transition_latent_dim: int = 0,
    ):
        super().__init__()
        self.transition_latent_dim = int(transition_latent_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
            nn.LayerNorm(z_dim),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        if self.transition_latent_dim > 0:
            self.transition_encoder = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.transition_latent_dim),
                nn.LayerNorm(self.transition_latent_dim),
            )
            self.transition_decoder = nn.Sequential(
                nn.Linear(self.transition_latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )
            self.transition_latent_head = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.transition_latent_dim),
                nn.LayerNorm(self.transition_latent_dim),
            )
        self.success_head = nn.Linear(z_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        transition_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        s = self.state_encoder(state)
        h = self.action_encoder(action_chunk.transpose(1, 2))
        a_mean = h.mean(dim=-1)
        a_max = h.amax(dim=-1)
        z = self.fuse(torch.cat([s, a_mean, a_max], dim=-1))
        pred_transition = self.transition_head(z)
        out = {
            "z": z,
            "pred_transition": pred_transition,
            "pred_next_state": state + pred_transition,
            "success_logit": self.success_head(z).squeeze(-1),
        }
        if self.transition_latent_dim > 0:
            pred_transition_latent = self.transition_latent_head(z)
            out["pred_transition_latent"] = pred_transition_latent
            out["pred_transition_from_latent"] = self.transition_decoder(pred_transition_latent)
            if transition_target is not None:
                target_transition_latent = self.transition_encoder(transition_target)
                out["target_transition_latent"] = target_transition_latent
                out["recon_transition"] = self.transition_decoder(target_transition_latent)
        return out


def make_batch(
    rows: list[dict[str, Any]],
    ids: list[int],
    by_task: dict[str, list[int]],
    all_ids: list[int],
    same_task_negative: bool,
    rng: random.Random,
    transition_mean: torch.Tensor,
    transition_std: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    chosen = [rows[i] for i in ids]
    state = torch.from_numpy(np.stack([r["state"] for r in chosen]).astype(np.float32)).to(device)
    next_state = torch.from_numpy(np.stack([r["next_state"] for r in chosen]).astype(np.float32)).to(device)
    transition = torch.from_numpy(np.stack([r["transition"] for r in chosen]).astype(np.float32)).to(device)
    action_chunk = torch.from_numpy(np.stack([r["action_chunk"] for r in chosen]).astype(np.float32)).to(device)
    success = torch.from_numpy(np.asarray([1.0 if r["success"] else 0.0 for r in chosen], dtype=np.float32)).to(device)

    neg_ids = []
    for idx in ids:
        task = str(rows[idx]["task"])
        pool = [j for j in by_task.get(task, []) if int(j) != int(idx)] if same_task_negative else []
        if not pool:
            pool = [j for j in all_ids if int(j) != int(idx)]
        if not pool:
            pool = [int(idx)]
        neg_ids.append(int(rng.choice(pool)))
    neg_transition = torch.from_numpy(np.stack([rows[i]["transition"] for i in neg_ids]).astype(np.float32)).to(device)
    return {
        "state": state,
        "next_state": next_state,
        "transition": transition,
        "transition_norm": (transition - transition_mean) / transition_std,
        "neg_transition": neg_transition,
        "neg_transition_norm": (neg_transition - transition_mean) / transition_std,
        "action_chunk": action_chunk,
        "success": success,
    }


def compute_metrics(
    model: StateActionDynamicsProbe,
    rows: list[dict[str, Any]],
    eval_idx: list[int],
    by_task: dict[str, list[int]],
    all_ids: list[int],
    same_task_negative: bool,
    transition_mean: torch.Tensor,
    transition_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    model.eval()
    transition_losses = []
    transition_cos = []
    manifold_recon_losses = []
    manifold_latent_cos = []
    compat_acc = []
    success_acc = []
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            ids = eval_idx[start : start + batch_size]
            batch = make_batch(rows, ids, by_task, all_ids, same_task_negative, rng, transition_mean, transition_std, device)
            out = model(batch["state"], batch["action_chunk"], transition_target=batch["transition"])
            pred_transition = out.get("pred_transition_from_latent", out["pred_transition"])
            pred_norm = (pred_transition - transition_mean) / transition_std
            transition_losses.append(F.smooth_l1_loss(pred_norm, batch["transition_norm"]).cpu())
            transition_cos.append(F.cosine_similarity(pred_norm, batch["transition_norm"], dim=-1).mean().cpu())
            if "recon_transition" in out and "pred_transition_latent" in out and "target_transition_latent" in out:
                manifold_recon_losses.append(F.smooth_l1_loss(out["recon_transition"], batch["transition"]).cpu())
                manifold_latent_cos.append(
                    F.cosine_similarity(out["pred_transition_latent"], out["target_transition_latent"], dim=-1).mean().cpu()
                )
            pos_sim = F.cosine_similarity(pred_norm, batch["transition_norm"], dim=-1)
            neg_sim = F.cosine_similarity(pred_norm, batch["neg_transition_norm"], dim=-1)
            compat_acc.append(((pos_sim > neg_sim).float().mean()).cpu())

            pred_success = (torch.sigmoid(out["success_logit"]) >= 0.5).float()
            gt_success = (batch["success"] > 0.5).float()
            success_acc.append((pred_success == gt_success).float().mean().cpu())
            tp += int(((pred_success == 1) & (gt_success == 1)).sum().item())
            fp += int(((pred_success == 1) & (gt_success == 0)).sum().item())
            fn += int(((pred_success == 0) & (gt_success == 1)).sum().item())
            tn += int(((pred_success == 0) & (gt_success == 0)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "transition_loss": float(torch.stack(transition_losses).mean().item()) if transition_losses else 0.0,
        "transition_cos": float(torch.stack(transition_cos).mean().item()) if transition_cos else 0.0,
        "manifold_recon_loss": float(torch.stack(manifold_recon_losses).mean().item()) if manifold_recon_losses else 0.0,
        "manifold_latent_cos": float(torch.stack(manifold_latent_cos).mean().item()) if manifold_latent_cos else 0.0,
        "compat_acc": float(torch.stack(compat_acc).mean().item()) if compat_acc else 0.0,
        "success_acc": float(torch.stack(success_acc).mean().item()) if success_acc else 0.0,
        "success_precision": float(precision),
        "success_recall": float(recall),
        "success_f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train state-conditioned dynamics D(s, action_chunk)->obs_transition from strong_gt obs start/end.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--action-trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--chunks-per-trace", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--compat-weight", type=float, default=1.0)
    parser.add_argument("--success-weight", type=float, default=0.25)
    parser.add_argument("--transition-manifold-weight", type=float, default=0.0)
    parser.add_argument("--transition-recon-weight", type=float, default=0.0)
    parser.add_argument("--transition-latent-dim", type=int, default=0)
    parser.add_argument("--compat-margin", type=float, default=0.2)
    parser.add_argument("--same-task-negative", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows, source_rows, reasons = load_samples(
        dataset_jsonl=args.dataset_jsonl,
        action_trace_dirs=list(args.action_trace_dir),
        chunk_len=int(args.chunk_len),
        chunks_per_trace=int(args.chunks_per_trace),
        seed=int(args.seed),
        max_source_rows=int(args.max_source_rows),
    )
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    train_by_task, train_all = build_task_pools(rows, train_idx)
    val_by_task, val_all = build_task_pools(rows, val_idx)
    transition_mean_np, transition_std_np = compute_stats(rows, train_idx)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    transition_mean = torch.from_numpy(transition_mean_np).to(device)
    transition_std = torch.from_numpy(transition_std_np).to(device)

    model = StateActionDynamicsProbe(
        state_dim=int(rows[0]["state"].shape[-1]),
        action_dim=int(rows[0]["action_chunk"].shape[-1]),
        hidden_dim=int(args.hidden_dim),
        z_dim=int(args.z_dim),
        dropout=float(args.dropout),
        transition_latent_dim=int(args.transition_latent_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        ids = rng.choices(train_idx, k=min(len(train_idx), args.batch_size))
        batch = make_batch(rows, ids, train_by_task, train_all, bool(args.same_task_negative), rng, transition_mean, transition_std, device)
        out = model(batch["state"], batch["action_chunk"], transition_target=batch["transition"])
        pred_transition = out.get("pred_transition_from_latent", out["pred_transition"])
        pred_norm = (pred_transition - transition_mean) / transition_std
        pos_sim = F.cosine_similarity(pred_norm, batch["transition_norm"], dim=-1)
        neg_sim = F.cosine_similarity(pred_norm, batch["neg_transition_norm"], dim=-1)
        transition_loss = F.smooth_l1_loss(pred_norm, batch["transition_norm"])
        compat_loss = F.relu(float(args.compat_margin) - (pos_sim - neg_sim)).mean()
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        manifold_loss = torch.zeros((), device=device)
        recon_loss = torch.zeros((), device=device)
        if "pred_transition_latent" in out and "target_transition_latent" in out:
            manifold_loss = 1.0 - F.cosine_similarity(
                out["pred_transition_latent"], out["target_transition_latent"], dim=-1
            ).mean()
        if "recon_transition" in out:
            recon_loss = F.smooth_l1_loss(out["recon_transition"], batch["transition"])
        loss = (
            float(args.transition_weight) * transition_loss
            + float(args.compat_weight) * compat_loss
            + float(args.success_weight) * success_loss
            + float(args.transition_manifold_weight) * manifold_loss
            + float(args.transition_recon_weight) * recon_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            metrics = compute_metrics(
                model,
                rows,
                val_idx,
                val_by_task,
                val_all,
                bool(args.same_task_negative),
                transition_mean,
                transition_std,
                device,
                int(args.eval_batch_size),
                args.seed + step,
            )
            rec = {
                "step": int(step),
                "loss": float(loss.item()),
                "transition_loss_train": float(transition_loss.item()),
                "compat_loss_train": float(compat_loss.item()),
                "success_loss_train": float(success_loss.item()),
                "manifold_loss_train": float(manifold_loss.item()),
                "recon_loss_train": float(recon_loss.item()),
                **metrics,
            }
            history.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            score = (
                metrics["transition_cos"]
                + metrics["compat_acc"]
                + metrics["success_f1"]
                + metrics["manifold_latent_cos"]
            )
            if best is None or score > best["score"]:
                best = {"score": float(score), "metrics": rec}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "state_dim": int(rows[0]["state"].shape[-1]),
                        "action_dim": int(rows[0]["action_chunk"].shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "z_dim": int(args.z_dim),
                        "dropout": float(args.dropout),
                        "transition_latent_dim": int(args.transition_latent_dim),
                        "transition_mean": transition_mean_np,
                        "transition_std": transition_std_np,
                        "args": vars(args),
                        "best_metrics": rec,
                    },
                    args.output_dir / "state_action_dynamics_probe.pt",
                )

    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_source_rows": int(source_rows),
        "num_train_chunks": int(len(train_idx)),
        "num_val_chunks": int(len(val_idx)),
        "steps": int(args.steps),
        "reason_counts": dict(reasons),
        "best": best["metrics"] if best is not None else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if history:
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
