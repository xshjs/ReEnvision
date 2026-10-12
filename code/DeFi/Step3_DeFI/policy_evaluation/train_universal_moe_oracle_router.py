import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class OracleRouter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_embed = nn.Embedding(num_experts, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim + 16),
            nn.Linear(feature_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_experts, _ = x.shape
        expert_ids = torch.arange(num_experts, device=x.device).unsqueeze(0).expand(batch, -1)
        expert_embed = self.expert_embed(expert_ids)
        scores = self.score(torch.cat([x, expert_embed], dim=-1)).squeeze(-1)
        return scores


def load_rows(root_dirs: List[Path]) -> List[Tuple[Path, Dict[str, object]]]:
    rows: List[Tuple[Path, Dict[str, object]]] = []
    for root in root_dirs:
        base = root / "collect_moe_universal_self_dataset"
        index_path = base / "index.jsonl"
        if not index_path.exists():
            continue
        metadata_path = root / "metadata.json"
        sequence_offset = 0
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            sequence_offset = int(metadata.get("sequence_offset", 0))
        with index_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_sequence_offset"] = sequence_offset
                row["_global_sequence_index"] = sequence_offset + int(row.get("sequence_index", 0))
                npz_path = base / str(row["path"])
                if npz_path.exists():
                    rows.append((npz_path, row))
    return rows


def filter_rows_by_sequence(
    rows: List[Tuple[Path, Dict[str, object]]],
    exclude_start: int,
    exclude_end: int,
) -> List[Tuple[Path, Dict[str, object]]]:
    if exclude_start < 0 or exclude_end < 0:
        return rows
    if exclude_end <= exclude_start:
        raise ValueError("--exclude-sequence-end must be greater than --exclude-sequence-start")
    return [
        (npz_path, row)
        for npz_path, row in rows
        if not (exclude_start <= int(row.get("_global_sequence_index", -1)) < exclude_end)
    ]


BASE_FEATURE_LAYOUT = [
    "obs_embedding",
    "expert_future",
    "expert_future_minus_obs",
    "router_logit",
    "router_score",
    "expert_contradiction",
]


def make_features(rows: List[Tuple[Path, Dict[str, object]]], include_expert_steps_feature: bool = False) -> Dict[str, np.ndarray]:
    x_rows = []
    value_rows = []
    success_rows = []
    oracle_rows = []
    chosen_rows = []
    group_rows = []
    task_rows = []
    for npz_path, row in rows:
        with np.load(npz_path) as data:
            obs = data["obs_embedding"].astype(np.float32)
            futures = data["expert_futures"].astype(np.float32)
            future_delta = futures - obs[None, :]
            router_logits = data["router_logits"].astype(np.float32).reshape(-1, 1)
            router_scores = data["router_scores"].astype(np.float32).reshape(-1, 1)
            contradiction = data["expert_contradiction"].astype(np.float32).reshape(-1, 1)
            parts = [
                np.repeat(obs[None, :], futures.shape[0], axis=0),
                futures,
                future_delta,
                router_logits,
                router_scores,
                contradiction,
            ]
            if include_expert_steps_feature:
                steps = data["expert_steps"].astype(np.float32).reshape(-1, 1) / 160.0
                parts.append(steps)
            feature = np.concatenate(parts, axis=1)
            x_rows.append(feature)
            value_rows.append(data["expert_values"].astype(np.float32))
            success_rows.append(data["expert_success"].astype(np.float32))
            oracle_rows.append(int(data["oracle_expert"][0]))
            chosen_rows.append(int(data["chosen_expert"][0]))
            group_rows.append(f'{row.get("_global_sequence_index")}:{row.get("subtask_index")}')
            task_rows.append(str(row.get("task")))
    return {
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "values": np.stack(value_rows, axis=0).astype(np.float32),
        "success": np.stack(success_rows, axis=0).astype(np.float32),
        "steps": np.stack([np.load(npz_path)["expert_steps"].astype(np.float32) for npz_path, _ in rows], axis=0).astype(np.float32),
        "oracle": np.asarray(oracle_rows, dtype=np.int64),
        "chosen": np.asarray(chosen_rows, dtype=np.int64),
        "groups": np.asarray(group_rows),
        "tasks": np.asarray(task_rows),
    }


def target_labels(data: Dict[str, np.ndarray], target: str, step_lambda: float, ep_len: float) -> Tuple[np.ndarray, np.ndarray]:
    if target == "success":
        score = data["success"].astype(np.float32)
    elif target == "utility":
        score = data["success"].astype(np.float32) - float(step_lambda) * (data["steps"].astype(np.float32) / float(ep_len))
    elif target == "value":
        score = data["values"].astype(np.float32)
    else:
        raise ValueError(f"unknown target: {target}")
    return np.argmax(score, axis=1).astype(np.int64), score.astype(np.float32)


def split_indices(groups: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_groups = set(unique[:n_val].tolist())
    train_idx = np.asarray([i for i, group in enumerate(groups) if group not in val_groups], dtype=np.int64)
    val_idx = np.asarray([i for i, group in enumerate(groups) if group in val_groups], dtype=np.int64)
    return train_idx, val_idx


def probability_stats(scores: np.ndarray, idx: np.ndarray) -> Dict[str, object]:
    logits = scores[idx]
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)
    mean_probs = probs.mean(axis=0)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
    return {
        "mean_selection_prob": [float(value) for value in mean_probs],
        "mean_entropy": float(entropy.mean()),
        "normalized_mean_entropy": float(entropy.mean() / np.log(probs.shape[1])),
    }


def evaluate(scores: np.ndarray, data: Dict[str, np.ndarray], idx: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    chosen = data["chosen"][idx]
    oracle = data["oracle"][idx]
    target = labels[idx]
    pred = scores[idx].argmax(axis=1)
    success = data["success"][idx]
    values = data["values"][idx]
    router_success = success[np.arange(len(idx)), chosen]
    pred_success = success[np.arange(len(idx)), pred]
    oracle_success = success[np.arange(len(idx)), oracle]
    router_value = values[np.arange(len(idx)), chosen]
    pred_value = values[np.arange(len(idx)), pred]
    oracle_value = values[np.arange(len(idx)), oracle]
    top1_fail = router_success < 0.5
    recoverable = top1_fail & (success.max(axis=1) > 0.5)
    recovered = recoverable & (pred_success > 0.5)
    return {
        "num": int(len(idx)),
        "router_success": float(router_success.mean()),
        "learned_success": float(pred_success.mean()),
        "oracle_success": float(oracle_success.mean()),
        "router_value": float(router_value.mean()),
        "learned_value": float(pred_value.mean()),
        "oracle_value": float(oracle_value.mean()),
        "target_expert_acc": float((pred == target).mean()),
        "value_oracle_expert_acc": float((pred == oracle).mean()),
        "router_expert_acc": float((chosen == oracle).mean()),
        "recoverable_failures": int(recoverable.sum()),
        "recovered_by_learned": int(recovered.sum()),
        "recover_rate": float(recovered.sum() / max(recoverable.sum(), 1)),
        **probability_stats(scores, idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an oracle-supervised selector over Universal MoE expert futures.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=["success", "utility", "value"], default="value")
    parser.add_argument("--step-lambda", type=float, default=0.01)
    parser.add_argument("--ep-len", type=float, default=160.0)
    parser.add_argument("--exclude-sequence-start", type=int, default=-1)
    parser.add_argument("--exclude-sequence-end", type=int, default=-1)
    parser.add_argument("--balance-weight", type=float, default=0.0)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument(
        "--include-expert-steps-feature",
        action="store_true",
        help="Include post-rollout expert_steps as an input feature. This leaks execution outcome and is only for ablations.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.roots)
    rows_before_filter = len(rows)
    rows = filter_rows_by_sequence(rows, args.exclude_sequence_start, args.exclude_sequence_end)
    if not rows:
        raise ValueError("No Universal MoE rows found.")
    feature_layout = list(BASE_FEATURE_LAYOUT)
    if args.include_expert_steps_feature:
        feature_layout.append("expert_steps_norm")
    data = make_features(rows, include_expert_steps_feature=args.include_expert_steps_feature)
    labels_np, target_score_np = target_labels(data, args.target, args.step_lambda, args.ep_len)
    train_idx, val_idx = split_indices(data["groups"], args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OracleRouter(data["x"].shape[-1], args.hidden_dim, data["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(data["x"]).to(device)
    oracle = torch.from_numpy(labels_np).to(device)
    values = torch.from_numpy(data["values"]).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)

    best_state = None
    best_val = -float("inf")
    num_experts = data["x"].shape[1]
    for _ in range(args.epochs):
        model.train()
        scores = model(x[train_t])
        ce = F.cross_entropy(scores, oracle[train_t])
        probs = F.softmax(scores, dim=1)
        pred_value = torch.sum(probs * values[train_t], dim=1)
        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / float(num_experts))
        balance_loss = torch.sum((mean_probs - uniform) ** 2) * float(num_experts)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        diversity_loss = -entropy.mean() / np.log(float(num_experts))
        loss = (
            ce
            - float(args.value_weight) * pred_value.mean()
            + float(args.balance_weight) * balance_loss
            + float(args.diversity_weight) * diversity_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(x[val_t])
            val_pred = val_scores.argmax(dim=1)
            val_success_proxy = values[val_t][torch.arange(len(val_t), device=device), val_pred].mean()
        if float(val_success_proxy) > best_val:
            best_val = float(val_success_proxy)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores_np = model(x).detach().cpu().numpy()

    metrics = {
        "target": args.target,
        "step_lambda": float(args.step_lambda),
        "ep_len": float(args.ep_len),
        "num_samples": int(len(data["x"])),
        "num_samples_before_sequence_filter": int(rows_before_filter),
        "exclude_sequence_start": int(args.exclude_sequence_start),
        "exclude_sequence_end": int(args.exclude_sequence_end),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(data["x"].shape[-1]),
        "feature_layout": feature_layout,
        "include_expert_steps_feature": bool(args.include_expert_steps_feature),
        "balance_weight": float(args.balance_weight),
        "diversity_weight": float(args.diversity_weight),
        "value_weight": float(args.value_weight),
        "target_label_counts": {
            str(expert): int(np.sum(labels_np == expert))
            for expert in range(data["x"].shape[1])
        },
        "target_score_mean": float(target_score_np.mean()),
        "train": evaluate(scores_np, data, train_idx, labels_np),
        "val": evaluate(scores_np, data, val_idx, labels_np),
        "all": evaluate(scores_np, data, np.arange(len(data["x"]), dtype=np.int64), labels_np),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": int(data["x"].shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "num_experts": int(data["x"].shape[1]),
            "target": args.target,
            "step_lambda": float(args.step_lambda),
            "ep_len": float(args.ep_len),
            "feature_layout": feature_layout,
            "include_expert_steps_feature": bool(args.include_expert_steps_feature),
            "balance_weight": float(args.balance_weight),
            "diversity_weight": float(args.diversity_weight),
            "value_weight": float(args.value_weight),
        },
        args.output_dir / "universal_moe_oracle_router.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
