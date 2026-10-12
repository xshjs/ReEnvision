from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from train_state_action_dynamics_probe import (
    StateActionDynamicsProbe,
    action_trace_from_row,
    chunk_starts,
    future_trace_from_row,
    iter_jsonl,
    pad_chunk,
    row_vec,
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-8 else float(np.dot(a, b) / denom)


def load_probe(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dim" not in ckpt:
        if "future_dim" in ckpt:
            raise ValueError(
                "expected state_action_dynamics_probe checkpoint, got future_action_dynamics_probe checkpoint"
            )
        raise ValueError("invalid state_action_dynamics_probe checkpoint: missing state_dim")
    model = StateActionDynamicsProbe(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        z_dim=int(ckpt["z_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
        transition_latent_dim=int(ckpt.get("transition_latent_dim", 0) or 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    transition_mean = torch.from_numpy(np.asarray(ckpt["transition_mean"], dtype=np.float32)).to(device)
    transition_std = torch.from_numpy(np.asarray(ckpt["transition_std"], dtype=np.float32)).to(device)
    return model, transition_mean, transition_std


def row_state_transition(row: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    robot_start = row_vec(row, "strong_gt_robot_obs_start")
    robot_end = row_vec(row, "strong_gt_robot_obs_end")
    scene_start = row_vec(row, "strong_gt_scene_obs_start")
    scene_end = row_vec(row, "strong_gt_scene_obs_end")
    if robot_start is None or robot_end is None or scene_start is None or scene_end is None:
        return None, None
    state = np.concatenate([robot_start, scene_start], axis=0).astype(np.float32)
    transition = np.concatenate([robot_end - robot_start, scene_end - scene_start], axis=0).astype(np.float32)
    return state, transition


def build_candidates(
    dataset_jsonl: Path,
    action_trace_dirs: list[Path],
    chunk_len: int,
    chunks_per_trace: int,
    same_task: str | None,
) -> list[dict[str, Any]]:
    items = []
    for row in iter_jsonl(dataset_jsonl):
        task = str(row.get("task", "unknown"))
        if same_task is not None and task != same_task:
            continue
        state, transition = row_state_transition(row)
        if state is None or transition is None:
            continue
        future_trace = future_trace_from_row(row)
        action_trace = action_trace_from_row(row, future_trace, action_trace_dirs)
        if action_trace is None:
            continue
        try:
            with np.load(action_trace) as data:
                actions = np.asarray(data["actions"], dtype=np.float32)
        except Exception:
            continue
        if actions.ndim != 2 or actions.shape[0] <= 0:
            continue
        starts = chunk_starts(actions.shape[0], chunk_len, chunks_per_trace, bool(row.get("success", False)), np.random.RandomState(0))
        for start in starts:
            items.append(
                {
                    "task": task,
                    "success": bool(row.get("success", False)),
                    "trace_path": str(row.get("trace_path", "")),
                    "action_trace_path": str(action_trace),
                    "chunk_start": int(start),
                    "state": state,
                    "transition": transition,
                    "action_chunk": pad_chunk(actions[start : start + chunk_len], chunk_len),
                }
            )
    if not items:
        raise ValueError("no candidate items built")
    return items


def score_pair(
    model: StateActionDynamicsProbe,
    transition_mean: torch.Tensor,
    transition_std: torch.Tensor,
    state: np.ndarray,
    action_chunk: np.ndarray,
    transition: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
        action_t = torch.from_numpy(action_chunk).unsqueeze(0).to(device)
        transition_t = torch.from_numpy(transition).unsqueeze(0).to(device)
        out = model(state_t, action_t, transition_target=transition_t)
        pred = out.get("pred_transition_from_latent", out["pred_transition"])
        pred_norm = (pred - transition_mean) / transition_std
        tgt_norm = (transition_t - transition_mean) / transition_std
        cos = float(F.cosine_similarity(pred_norm, tgt_norm, dim=-1).item())
        l1 = float(F.smooth_l1_loss(pred_norm, tgt_norm).item())
        success_prob = float(torch.sigmoid(out["success_logit"]).item())
        dynamics_score = float(cos - l1)
        metrics = {
        "dynamics_cos": cos,
        "dynamics_l1": l1,
        "success_prob": success_prob,
        "dynamics_score": dynamics_score,
        }
        if "pred_transition_latent" in out and "target_transition_latent" in out:
            metrics["manifold_latent_cos"] = float(
                F.cosine_similarity(out["pred_transition_latent"], out["target_transition_latent"], dim=-1).item()
            )
        if "recon_transition" in out:
            metrics["manifold_recon_l1"] = float(F.smooth_l1_loss(out["recon_transition"], transition_t).item())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Score state-action-transition compatibility with a trained dynamics probe.")
    parser.add_argument("--query-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--action-trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--probe-ckpt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--chunks-per-trace", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, transition_mean, transition_std = load_probe(args.probe_ckpt, device)

    all_candidates = build_candidates(
        dataset_jsonl=args.candidate_jsonl,
        action_trace_dirs=list(args.action_trace_dir),
        chunk_len=int(args.chunk_len),
        chunks_per_trace=int(args.chunks_per_trace),
        same_task=None,
    )

    results = []
    rerank_changed = 0
    total = 0
    for row in iter_jsonl(args.query_jsonl):
        query_state, _ = row_state_transition(row)
        if query_state is None:
            continue
        task = str(row.get("task", "unknown"))
        candidates = [c for c in all_candidates if (c["task"] == task if args.same_task_only else True)]
        if not candidates:
            continue
        sims = np.asarray([cosine(query_state, c["state"]) for c in candidates], dtype=np.float32)
        order = np.argsort(-sims)[: min(int(args.top_k), len(candidates))]
        chosen = []
        for rank, local_idx in enumerate(order.tolist(), start=1):
            cand = candidates[local_idx]
            score = score_pair(
                model,
                transition_mean,
                transition_std,
                query_state,
                np.asarray(cand["action_chunk"], dtype=np.float32),
                np.asarray(cand["transition"], dtype=np.float32),
                device,
            )
            chosen.append(
                {
                    "rank_retrieval": rank,
                    "task": cand["task"],
                    "success": bool(cand["success"]),
                    "trace_path": cand["trace_path"],
                    "action_trace_path": cand["action_trace_path"],
                    "chunk_start": int(cand["chunk_start"]),
                    "state_similarity": float(sims[local_idx]),
                    **score,
                }
            )
        best_state = chosen[0]
        best_dyn = max(chosen, key=lambda x: x["dynamics_score"])
        rerank_changed += int(best_state["chunk_start"] != best_dyn["chunk_start"] or best_state["trace_path"] != best_dyn["trace_path"])
        total += 1
        results.append(
            {
                "query_task": task,
                "query_trace_path": str(row.get("trace_path", "")),
                "best_state_similarity": best_state,
                "best_dynamics_score": best_dyn,
                "candidates": chosen,
            }
        )
        if args.max_queries > 0 and total >= args.max_queries:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    state_success = sum(int(r["best_state_similarity"]["success"]) for r in results)
    dyn_success = sum(int(r["best_dynamics_score"]["success"]) for r in results)
    summary = {
        "query_jsonl": str(args.query_jsonl),
        "candidate_jsonl": str(args.candidate_jsonl),
        "probe_ckpt": str(args.probe_ckpt),
        "output_jsonl": str(args.output_jsonl),
        "num_queries": int(total),
        "top_k": int(args.top_k),
        "same_task_only": bool(args.same_task_only),
        "rerank_changed_top1": int(rerank_changed),
        "rerank_change_rate": float(rerank_changed / max(total, 1)),
        "state_success_rate": float(state_success / max(total, 1)),
        "dynamics_success_rate": float(dyn_success / max(total, 1)),
        "success_delta": float((dyn_success - state_success) / max(total, 1)),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
