from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "base_future" and "original_base_future" in data:
                arr = np.asarray(data["base_future"], dtype=np.float32)
            elif key == "final_future" and "target_proxy_future" in data:
                arr = np.asarray(data["final_future"], dtype=np.float32)
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


class QCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def load_q_critic(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = QCritic(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1),
        "std": np.asarray(ckpt["std"], dtype=np.float32).reshape(-1),
        "task_dim": int(ckpt.get("task_dim", 128)),
    }


def load_exec_probe(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state"]
    action_dim = int(state["action_encoder.0.weight"].shape[1])
    future_tripled_dim = int(state["future_encoder.1.weight"].shape[1])
    if future_tripled_dim % 3 != 0:
        raise ValueError(f"unexpected future encoder input dim: {future_tripled_dim}")
    future_dim = future_tripled_dim // 3
    model = FutureActionExecutabilityProbe(
        action_dim=action_dim,
        future_dim=future_dim,
        z_dim=int(ckpt.get("z_dim", 128)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        dropout=0.1,
        num_tasks=len(ckpt["tasks"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "tasks": ckpt["tasks"]}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k future-action pairs and rerank by R_task + lambda * R_exec.")
    parser.add_argument("--query-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--q-critic-ckpt", type=Path, required=True)
    parser.add_argument("--exec-probe-ckpt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lambda-exec", type=float, default=1.0)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--success-only-memory", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    q_bundle = load_q_critic(args.q_critic_ckpt, device)
    exec_bundle = load_exec_probe(args.exec_probe_ckpt, device)
    memory = load_memory(args.memory_npz, args.success_only_memory)

    results = []
    improved = 0
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
        best_total = None
        for rank, mem_idx in enumerate(chosen_ids.tolist(), start=1):
            target_summary = np.asarray(memory["target_summary"][mem_idx], dtype=np.float32)
            target_delta = np.asarray(memory["target_delta"][mem_idx], dtype=np.float32)
            target_summary_exec = np.asarray(
                memory["target_summary_exec"][mem_idx] if "target_summary_exec" in memory else memory["target_summary"][mem_idx],
                dtype=np.float32,
            )
            target_delta_exec = np.asarray(
                memory["target_delta_exec"][mem_idx] if "target_delta_exec" in memory else memory["target_delta"][mem_idx],
                dtype=np.float32,
            )
            action_chunk = np.asarray(memory["action_chunk"][mem_idx], dtype=np.float32)
            q_feat = np.concatenate([target_summary, hashed_task(task, q_bundle["task_dim"]), np.asarray([sub], dtype=np.float32)], axis=0).astype(np.float32)
            q_feat = (q_feat - q_bundle["mean"]) / np.maximum(q_bundle["std"], 1e-6)
            q_tensor = torch.from_numpy(q_feat).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_success = torch.sigmoid(q_bundle["model"](q_tensor)).item()
                exec_out = exec_bundle["model"](
                    torch.from_numpy(action_chunk).unsqueeze(0).to(device),
                    torch.from_numpy(base_summary_exec).unsqueeze(0).to(device),
                    torch.from_numpy(target_summary_exec).unsqueeze(0).to(device),
                    torch.from_numpy(target_delta_exec).unsqueeze(0).to(device),
                )
                exec_prob = torch.sigmoid(exec_out["exec_logit"]).item()
            total_score = float(prob_success + args.lambda_exec * exec_prob)
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
                "r_task_prob_success": float(prob_success),
                "r_exec_prob_match": float(exec_prob),
                "r_total": total_score,
            }
            candidate_rows.append(rec)
            if best_sim is None:
                best_sim = rec
            if best_total is None or total_score > best_total["r_total"]:
                best_total = rec
        assert best_sim is not None and best_total is not None
        improved += int(best_total["memory_index"] != best_sim["memory_index"])
        total += 1
        results.append(
            {
                "query_task": task,
                "query_trace_path": str(future_trace),
                "best_similarity": best_sim,
                "best_reranked": best_total,
                "candidates": candidate_rows,
            }
        )
        if args.max_queries > 0 and total >= args.max_queries:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "query_jsonl": str(args.query_jsonl),
        "memory_npz": str(args.memory_npz),
        "q_critic_ckpt": str(args.q_critic_ckpt),
        "exec_probe_ckpt": str(args.exec_probe_ckpt),
        "output_jsonl": str(args.output_jsonl),
        "num_queries": total,
        "top_k": int(args.top_k),
        "lambda_exec": float(args.lambda_exec),
        "same_task_only": bool(args.same_task_only),
        "success_only_memory": bool(args.success_only_memory),
        "rerank_changed_top1": improved,
        "rerank_change_rate": float(improved / max(total, 1)),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
