from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from score_state_action_dynamics import load_probe, score_pair
from train_joint_future_action_energy import JointFutureActionEnergy


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


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "base_future" and "original_base_future" in data:
                arr = np.asarray(data["base_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def future_summary(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[-1] - arr[0]], axis=0).astype(np.float32)


def future_summary_exec(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def hashed_task(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-8 else float(np.dot(a, b) / denom)


def load_memory(npz_path: Path, success_only: bool):
    with np.load(npz_path, allow_pickle=True) as data:
        mem = {k: data[k] for k in data.files}
    if success_only:
        mask = mem["success"] > 0.5
        for k, v in list(mem.items()):
            mem[k] = v[mask]
    return mem


def load_joint_energy(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = JointFutureActionEnergy(
        action_dim=int(ckpt["action_dim"]),
        future_dim=int(ckpt["future_dim"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "task_dim": int(ckpt["task_dim"]),
    }


def row_state(row: dict[str, Any]) -> np.ndarray | None:
    robot_start = row.get("strong_gt_robot_obs_start")
    scene_start = row.get("strong_gt_scene_obs_start")
    if not isinstance(robot_start, list) or not isinstance(scene_start, list) or not robot_start or not scene_start:
        return None
    try:
        return np.concatenate(
            [
                np.asarray(robot_start, dtype=np.float32).reshape(-1),
                np.asarray(scene_start, dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
    except Exception:
        return None


def memory_transition(memory: dict[str, np.ndarray], mem_idx: int) -> np.ndarray | None:
    if "state_transition" in memory:
        return np.asarray(memory["state_transition"][mem_idx], dtype=np.float32)
    return None


def is_success(outcome: str) -> bool:
    return str(outcome).startswith("success")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k future-action pairs and rerank by joint energy E(s, z_f, z_a).")
    parser.add_argument("--query-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--joint-energy-ckpt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--success-only-memory", action="store_true")
    parser.add_argument("--dynamics-probe-ckpt", type=Path, default=None)
    parser.add_argument("--lambda-dynamics", type=float, default=1.0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    bundle = load_joint_energy(args.joint_energy_ckpt, device)
    memory = load_memory(args.memory_npz, args.success_only_memory)
    dynamics_bundle = None
    if args.dynamics_probe_ckpt is not None:
        dynamics_bundle = load_probe(args.dynamics_probe_ckpt, device)

    results = []
    improved = 0
    improved_fused = 0
    total = 0

    for row in iter_jsonl(args.query_jsonl):
        future_trace = future_trace_from_row(row)
        if future_trace is None:
            continue
        base = load_future(future_trace, args.future_key)
        if base is None:
            continue
        base_summary = future_summary(base)
        base_summary_exec = future_summary_exec(base)
        task = str(row.get("task", "unknown"))
        sub = float(row.get("subtask_index", 0) or 0.0) / 5.0
        query_state = row_state(row)
        task_mask = np.ones(len(memory["task"]), dtype=bool)
        if args.same_task_only:
            task_mask = np.asarray(memory["task"] == task)
        candidate_ids = np.where(task_mask)[0]
        if candidate_ids.size == 0:
            continue
        sims = np.asarray([cosine(base_summary, memory["base_summary"][idx]) for idx in candidate_ids], dtype=np.float32)
        order = np.argsort(-sims)[: min(args.top_k, len(candidate_ids))]
        chosen_ids = candidate_ids[order]
        candidate_rows = []
        best_sim = None
        best_energy = None
        best_fused = None

        task_vec = hashed_task(task, bundle["task_dim"]).astype(np.float32)
        with torch.no_grad():
            for rank, mem_idx in enumerate(chosen_ids.tolist(), start=1):
                target_summary_exec = np.asarray(memory["target_summary_exec"][mem_idx], dtype=np.float32)
                target_delta_exec = np.asarray(memory["target_delta_exec"][mem_idx], dtype=np.float32)
                action_chunk = np.asarray(memory["action_chunk"][mem_idx], dtype=np.float32)
                energy = bundle["model"](
                    torch.from_numpy(base_summary_exec).unsqueeze(0).to(device),
                    torch.from_numpy(target_summary_exec).unsqueeze(0).to(device),
                    torch.from_numpy(target_delta_exec).unsqueeze(0).to(device),
                    torch.from_numpy(action_chunk).unsqueeze(0).to(device),
                    torch.from_numpy(task_vec).unsqueeze(0).to(device),
                    torch.tensor([[sub]], dtype=torch.float32, device=device),
                ).item()
                dynamics_score = 0.0
                dynamics_cos = 0.0
                dynamics_l1 = 0.0
                dynamics_success_prob = 0.0
                if dynamics_bundle is not None and query_state is not None:
                    mem_transition = memory_transition(memory, mem_idx)
                    if mem_transition is not None:
                        d_model, d_mean, d_std = dynamics_bundle
                        d_score = score_pair(
                            d_model,
                            d_mean,
                            d_std,
                            query_state,
                            action_chunk,
                            mem_transition,
                            device,
                        )
                        dynamics_score = float(d_score["dynamics_score"])
                        dynamics_cos = float(d_score["dynamics_cos"])
                        dynamics_l1 = float(d_score["dynamics_l1"])
                        dynamics_success_prob = float(d_score["success_prob"])
                fused_score = float(-energy + float(args.lambda_dynamics) * dynamics_score)
                rec = {
                    "rank_retrieval": rank,
                    "memory_index": int(mem_idx),
                    "memory_task": str(memory["task"][mem_idx]),
                    "memory_factor": str(memory["factor"][mem_idx]),
                    "memory_outcome": str(memory["outcome"][mem_idx]),
                    "memory_trace_path": str(memory["trace_path"][mem_idx]),
                    "memory_action_trace_path": str(memory["action_trace_path"][mem_idx]),
                    "chunk_start": int(memory["chunk_start"][mem_idx]),
                    "base_similarity": float(sims[order[rank - 1]]),
                    "joint_energy": float(energy),
                    "dynamics_score": float(dynamics_score),
                    "dynamics_cos": float(dynamics_cos),
                    "dynamics_l1": float(dynamics_l1),
                    "dynamics_success_prob": float(dynamics_success_prob),
                    "fused_score": float(fused_score),
                }
                candidate_rows.append(rec)
                if best_sim is None:
                    best_sim = rec
                if best_energy is None or energy < best_energy["joint_energy"]:
                    best_energy = rec
                if best_fused is None or fused_score > best_fused["fused_score"]:
                    best_fused = rec

        assert best_sim is not None and best_energy is not None and best_fused is not None
        improved += int(best_energy["memory_index"] != best_sim["memory_index"])
        improved_fused += int(best_fused["memory_index"] != best_sim["memory_index"])
        total += 1
        results.append(
            {
                "query_task": task,
                "query_trace_path": str(future_trace),
                "best_similarity": best_sim,
                "best_joint_energy": best_energy,
                "best_fused": best_fused,
                "candidates": candidate_rows,
            }
        )
        if args.max_queries > 0 and total >= args.max_queries:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sim_success = sum(int(is_success(r["best_similarity"]["memory_outcome"])) for r in results)
    energy_success = sum(int(is_success(r["best_joint_energy"]["memory_outcome"])) for r in results)
    fused_success = sum(int(is_success(r["best_fused"]["memory_outcome"])) for r in results)
    summary = {
        "query_jsonl": str(args.query_jsonl),
        "memory_npz": str(args.memory_npz),
        "joint_energy_ckpt": str(args.joint_energy_ckpt),
        "dynamics_probe_ckpt": str(args.dynamics_probe_ckpt) if args.dynamics_probe_ckpt is not None else None,
        "output_jsonl": str(args.output_jsonl),
        "num_queries": total,
        "top_k": int(args.top_k),
        "same_task_only": bool(args.same_task_only),
        "success_only_memory": bool(args.success_only_memory),
        "rerank_changed_top1": improved,
        "rerank_change_rate": float(improved / max(total, 1)),
        "fused_changed_top1": improved_fused,
        "fused_change_rate": float(improved_fused / max(total, 1)),
        "sim_success_rate": float(sim_success / max(total, 1)),
        "joint_energy_success_rate": float(energy_success / max(total, 1)),
        "success_delta": float((energy_success - sim_success) / max(total, 1)),
        "fused_success_rate": float(fused_success / max(total, 1)),
        "fused_success_delta": float((fused_success - sim_success) / max(total, 1)),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
