from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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

    def forward(self, action_chunk: torch.Tensor, base_summary: torch.Tensor, target_summary: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
        h = self.action_encoder(action_chunk.transpose(1, 2))
        a_mean = h.mean(dim=-1)
        a_max = h.amax(dim=-1)
        f = self.future_encoder(torch.cat([base_summary, target_summary, target_delta], dim=-1))
        z = self.joint(torch.cat([a_mean, a_max, f], dim=-1))
        return self.exec_head(z).squeeze(-1)


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
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_memory(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def is_success(outcome: str) -> bool:
    return str(outcome).startswith("success")


def top_unique_future_indices(
    memory: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
    q_scores: np.ndarray,
    num_futures: int,
) -> list[int]:
    order = np.argsort(-q_scores)
    chosen = []
    seen = set()
    for pos in order.tolist():
        mem_idx = int(candidate_ids[pos])
        future_key = str(memory["trace_path"][mem_idx])
        if future_key in seen:
            continue
        chosen.append(mem_idx)
        seen.add(future_key)
        if len(chosen) >= num_futures:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict fixed-future, swap-action analysis.")
    parser.add_argument("--query-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--q-critic-ckpt", type=Path, required=True)
    parser.add_argument("--exec-probe-ckpt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--num-futures", type=int, default=4)
    parser.add_argument("--action-top-k", type=int, default=16)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    q_bundle = load_q_critic(args.q_critic_ckpt, device)
    exec_model = load_exec_probe(args.exec_probe_ckpt, device)
    memory = load_memory(args.memory_npz)

    results = []
    total = 0
    exec_gain = 0
    joint_gain = 0

    for row in iter_jsonl(args.query_jsonl):
        future_trace = future_trace_from_row(row)
        if future_trace is None:
            continue
        base = load_future(future_trace, args.future_key)
        if base is None:
            continue

        task = str(row.get("task", "unknown"))
        sub = float(row.get("subtask_index", 0) or 0.0) / 5.0
        task_mask = np.ones(len(memory["task"]), dtype=bool)
        if args.same_task_only:
            task_mask = np.asarray(memory["task"] == task)
        candidate_ids = np.where(task_mask)[0]
        if candidate_ids.size == 0:
            continue

        q_tail = np.concatenate([hashed_task(task, q_bundle["task_dim"]), np.asarray([sub], dtype=np.float32)], axis=0).astype(np.float32)
        candidate_target_summary = np.asarray(memory["target_summary"][candidate_ids], dtype=np.float32)
        q_tail_batch = np.repeat(q_tail[None, :], len(candidate_ids), axis=0)
        q_feat = np.concatenate([candidate_target_summary, q_tail_batch], axis=1).astype(np.float32)
        q_feat = (q_feat - q_bundle["mean"][None, :]) / np.maximum(q_bundle["std"][None, :], 1e-6)
        with torch.no_grad():
            q_scores = torch.sigmoid(q_bundle["model"](torch.from_numpy(q_feat).to(device))).cpu().numpy()

        future_ids = top_unique_future_indices(memory, candidate_ids, q_scores, args.num_futures)
        if not future_ids:
            continue

        base_summary_exec = future_summary_exec(base)
        action_ids = candidate_ids
        action_chunks = np.asarray(memory["action_chunk"][action_ids], dtype=np.float32)
        base_exec_batch = np.repeat(base_summary_exec[None, :], min(args.batch_size, len(action_ids)), axis=0).astype(np.float32)

        best_q_idx = future_ids[0]
        best_exec = None
        best_joint = None

        for future_mem_idx in future_ids:
            future_q = float(q_scores[np.where(candidate_ids == future_mem_idx)[0][0]])
            target_summary_exec = np.asarray(memory["target_summary_exec"][future_mem_idx], dtype=np.float32)
            target_delta_exec = np.asarray(memory["target_delta_exec"][future_mem_idx], dtype=np.float32)

            local_best_exec = None
            local_best_joint = None
            with torch.no_grad():
                for start in range(0, len(action_ids), args.batch_size):
                    batch_ids = action_ids[start : start + args.batch_size]
                    batch_actions = action_chunks[start : start + args.batch_size]
                    base_batch = np.repeat(base_summary_exec[None, :], len(batch_ids), axis=0).astype(np.float32)
                    target_batch = np.repeat(target_summary_exec[None, :], len(batch_ids), axis=0).astype(np.float32)
                    delta_batch = np.repeat(target_delta_exec[None, :], len(batch_ids), axis=0).astype(np.float32)
                    exec_scores = torch.sigmoid(
                        exec_model(
                            torch.from_numpy(batch_actions).to(device),
                            torch.from_numpy(base_batch).to(device),
                            torch.from_numpy(target_batch).to(device),
                            torch.from_numpy(delta_batch).to(device),
                        )
                    ).cpu().numpy()
                    for offset, action_mem_idx in enumerate(batch_ids.tolist()):
                        exec_score = float(exec_scores[offset])
                        joint_score = future_q + exec_score
                        rec = {
                            "future_memory_index": int(future_mem_idx),
                            "action_memory_index": int(action_mem_idx),
                            "future_score_q": future_q,
                            "exec_score": exec_score,
                            "joint_score": joint_score,
                            "future_outcome": str(memory["outcome"][future_mem_idx]),
                            "action_outcome": str(memory["outcome"][action_mem_idx]),
                            "future_factor": str(memory["factor"][future_mem_idx]),
                            "action_factor": str(memory["factor"][action_mem_idx]),
                            "future_trace_path": str(memory["trace_path"][future_mem_idx]),
                            "action_trace_path": str(memory["trace_path"][action_mem_idx]),
                            "action_chunk_start": int(memory["chunk_start"][action_mem_idx]),
                        }
                        if local_best_exec is None or exec_score > local_best_exec["exec_score"]:
                            local_best_exec = rec
                        if local_best_joint is None or joint_score > local_best_joint["joint_score"]:
                            local_best_joint = rec

            if local_best_exec is not None and (best_exec is None or local_best_exec["exec_score"] > best_exec["exec_score"]):
                best_exec = local_best_exec
            if local_best_joint is not None and (best_joint is None or local_best_joint["joint_score"] > best_joint["joint_score"]):
                best_joint = local_best_joint

        if best_exec is None or best_joint is None:
            continue

        q_success = is_success(memory["outcome"][best_q_idx])
        exec_success = is_success(best_exec["future_outcome"])
        joint_success = is_success(best_joint["future_outcome"])
        exec_gain += int((not q_success) and exec_success)
        joint_gain += int((not q_success) and joint_success)

        results.append(
            {
                "query_task": task,
                "query_trace_path": str(future_trace),
                "best_q_future": {
                    "memory_index": int(best_q_idx),
                    "memory_outcome": str(memory["outcome"][best_q_idx]),
                    "memory_factor": str(memory["factor"][best_q_idx]),
                    "future_score_q": float(q_scores[np.where(candidate_ids == best_q_idx)[0][0]]),
                },
                "best_exec_only": best_exec,
                "best_joint": best_joint,
            }
        )
        total += 1
        if args.max_queries > 0 and total >= args.max_queries:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def success_rate(key: str) -> float:
        if not results:
            return 0.0
        return float(sum(is_success(r[key]["memory_outcome"] if key == "best_q_future" else r[key]["future_outcome"]) for r in results) / len(results))

    summary = {
        "query_jsonl": str(args.query_jsonl),
        "memory_npz": str(args.memory_npz),
        "num_queries": total,
        "num_futures": int(args.num_futures),
        "action_top_k": int(args.action_top_k),
        "same_task_only": bool(args.same_task_only),
        "q_future_success_rate": success_rate("best_q_future"),
        "exec_only_success_rate": success_rate("best_exec_only"),
        "joint_success_rate": success_rate("best_joint"),
        "exec_gain_over_q": int(exec_gain),
        "joint_gain_over_q": int(joint_gain),
        "output_jsonl": str(args.output_jsonl),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
