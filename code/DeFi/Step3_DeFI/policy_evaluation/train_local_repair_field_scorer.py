from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRepairFieldScorer(nn.Module):
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features).squeeze(-1)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def build_pair_features(
    query: np.ndarray,
    failures: np.ndarray,
    successes: np.ndarray,
    deltas: np.ndarray,
    same_task: np.ndarray,
) -> np.ndarray:
    query_u = query / max(float(np.linalg.norm(query)), 1e-8)
    failure_u = normalize_rows(failures)
    success_u = normalize_rows(successes)
    delta_u = normalize_rows(deltas)
    delta_norm = np.linalg.norm(deltas, axis=1).astype(np.float32)
    delta_norm = delta_norm / max(float(np.median(delta_norm) if len(delta_norm) else 1.0), 1e-8)
    sim_failure = (failure_u @ query_u).astype(np.float32)
    sim_success = (success_u @ query_u).astype(np.float32)
    failure_distance = (1.0 - sim_failure).astype(np.float32)
    seed = np.maximum(sim_failure, 0.0) * np.maximum(same_task, 1e-6)
    if float(seed.sum()) <= 1e-8:
        seed = np.ones_like(seed, dtype=np.float32) / float(len(seed))
    else:
        seed = seed / seed.sum()
    mean_delta = np.tensordot(seed, delta_u, axes=(0, 0)).astype(np.float32)
    mean_delta = mean_delta / max(float(np.linalg.norm(mean_delta)), 1e-8)
    consistency = np.maximum(delta_u @ mean_delta, 0.0).astype(np.float32)
    return np.stack(
        [
            sim_failure,
            sim_success,
            same_task.astype(np.float32),
            consistency,
            delta_norm.astype(np.float32),
            failure_distance,
            (sim_success - sim_failure).astype(np.float32),
            (sim_failure * consistency).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def select_candidates(
    idx: int,
    failure_norm: np.ndarray,
    tasks: list[str],
    topk: int,
    same_task_only: bool,
) -> list[int]:
    task = tasks[idx]
    candidates = [j for j, t in enumerate(tasks) if j != idx and (not same_task_only or t == task)]
    if not candidates:
        candidates = [j for j in range(len(tasks)) if j != idx]
    if not candidates:
        return [idx]
    sims = failure_norm[candidates] @ failure_norm[idx]
    order = np.argsort(-sims)[: max(1, min(topk, len(candidates)))]
    return [candidates[int(i)] for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned local repair-field pair scorer.")
    parser.add_argument("--immune-memory-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.immune_memory_npz, allow_pickle=True)
    failures = np.asarray(data["failure_features"], dtype=np.float32)
    successes = np.asarray(data["success_features"], dtype=np.float32)
    deltas = np.asarray(data["attract_deltas"], dtype=np.float32)
    tasks = [str(x) for x in data["tasks"].tolist()]
    failure_norm = normalize_rows(failures)
    delta_norm = normalize_rows(deltas)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = LocalRepairFieldScorer(feature_dim=8, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    indices = np.arange(len(failures))
    feature_rows = []
    cand_delta_rows = []
    target_rows = []
    oracle_rows = []
    for idx in indices:
        cand = select_candidates(int(idx), failure_norm, tasks, args.topk, args.same_task_only)
        if len(cand) < args.topk:
            cand = cand + [cand[-1]] * (args.topk - len(cand))
        cand = cand[: args.topk]
        same = np.asarray([1.0 if tasks[j] == tasks[int(idx)] else 0.25 for j in cand], dtype=np.float32)
        feats = build_pair_features(failures[int(idx)], failures[cand], successes[cand], deltas[cand], same)
        target_u = delta_norm[int(idx)]
        oracle = np.maximum(delta_norm[cand] @ target_u, 0.0).astype(np.float32)
        oracle = oracle * np.maximum(feats[:, 0], 0.0) * np.maximum(same, 1e-6)
        if float(oracle.sum()) <= 1e-8:
            oracle = np.ones_like(oracle, dtype=np.float32) / float(len(oracle))
        else:
            oracle = oracle / oracle.sum()
        feature_rows.append(feats)
        cand_delta_rows.append(deltas[cand])
        target_rows.append(deltas[int(idx)])
        oracle_rows.append(oracle)
    feature_tensor = torch.from_numpy(np.stack(feature_rows, axis=0).astype(np.float32)).to(device)
    cand_delta_tensor = torch.from_numpy(np.stack(cand_delta_rows, axis=0).astype(np.float32)).to(device)
    target_tensor = torch.from_numpy(np.stack(target_rows, axis=0).astype(np.float32)).to(device)
    oracle_tensor = torch.from_numpy(np.stack(oracle_rows, axis=0).astype(np.float32)).to(device)
    logs = []
    for step in range(1, args.steps + 1):
        batch = torch.from_numpy(
            rng.choice(indices, size=min(args.batch_size, len(indices)), replace=len(indices) < args.batch_size)
        ).to(device=device, dtype=torch.long)
        feats_t = feature_tensor[batch]
        cand_delta_t = cand_delta_tensor[batch]
        target_t = target_tensor[batch]
        oracle_t = oracle_tensor[batch]
        logits = model(feats_t.reshape(-1, feats_t.shape[-1])).reshape(feats_t.shape[0], feats_t.shape[1])
        logits = logits / max(float(args.temperature), 1e-6)
        weights = torch.softmax(logits, dim=1)
        pred = torch.sum(weights[:, :, None] * cand_delta_t, dim=1)
        mse = F.mse_loss(pred, target_t)
        direction = 1.0 - F.cosine_similarity(pred, target_t, dim=-1).mean()
        ce = -(oracle_t * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        entropy = -(weights * torch.log(torch.clamp(weights, min=1e-8))).sum(dim=1).mean()
        batch_loss = mse + 0.5 * direction + 0.1 * ce - 0.005 * entropy
        opt.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                eval_n = min(64, len(indices))
                eval_feats = feature_tensor[:eval_n]
                eval_logits = model(eval_feats.reshape(-1, eval_feats.shape[-1])).reshape(eval_n, eval_feats.shape[1])
                eval_weights = torch.softmax(eval_logits, dim=1)
                eval_pred = torch.sum(eval_weights[:, :, None] * cand_delta_tensor[:eval_n], dim=1)
                eval_target = target_tensor[:eval_n]
                cos_vals = F.cosine_similarity(eval_pred, eval_target, dim=-1).detach().cpu().numpy().tolist()
                top1_hits = (
                    torch.argmax(eval_logits, dim=1) == torch.argmax(oracle_tensor[:eval_n], dim=1)
                ).detach().float().cpu().numpy().tolist()
                row = {
                    "step": step,
                    "loss": float(batch_loss.detach().cpu()),
                    "mse_loss": float(mse.detach().cpu()),
                    "direction_loss": float(direction.detach().cpu()),
                    "oracle_ce": float(ce.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "eval_delta_cosine": float(np.mean(cos_vals)),
                    "eval_top1_oracle_match": float(np.mean(top1_hits)),
                    "num_pairs": int(len(indices)),
                    "topk": int(args.topk),
                }
                logs.append(row)
                print(json.dumps(row, ensure_ascii=False))

    ckpt_path = output_dir / "local_repair_field_scorer.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": 8,
            "hidden_dim": int(args.hidden_dim),
            "topk": int(args.topk),
            "same_task_only": bool(args.same_task_only),
            "temperature": float(args.temperature),
            "memory_npz": str(args.immune_memory_npz),
        },
        ckpt_path,
    )
    summary = {
        "checkpoint_path": str(ckpt_path),
        "logs": logs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
