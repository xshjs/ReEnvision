import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train_universal_moe_oracle_router import BASE_FEATURE_LAYOUT, OracleRouter, evaluate, probability_stats, split_indices


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rows(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(path)):
        rows.append(row)
        if max_samples > 0 and idx + 1 >= max_samples:
            break
    return rows


def make_features(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    x_rows = []
    value_rows = []
    success_rows = []
    steps_rows = []
    chosen_rows = []
    oracle_rows = []
    group_rows = []
    meta_rows = []
    for row in rows:
        npz_path = Path(row["universal_npz_path"])
        with np.load(npz_path) as data:
            obs = data["obs_embedding"].astype(np.float32)
            futures = data["expert_futures"].astype(np.float32)
            future_delta = futures - obs[None, :]
            router_logits = data["router_logits"].astype(np.float32).reshape(-1, 1)
            router_scores = data["router_scores"].astype(np.float32).reshape(-1, 1)
            contradiction = data["expert_contradiction"].astype(np.float32).reshape(-1, 1)
            feature = np.concatenate(
                [
                    np.repeat(obs[None, :], futures.shape[0], axis=0),
                    futures,
                    future_delta,
                    router_logits,
                    router_scores,
                    contradiction,
                ],
                axis=1,
            )
            x_rows.append(feature)
            value_rows.append(data["expert_values"].astype(np.float32))
            success_rows.append(data["expert_success"].astype(np.float32))
            steps_rows.append(data["expert_steps"].astype(np.float32))
            chosen_rows.append(int(data["chosen_expert"][0]))
            oracle_rows.append(int(data["oracle_expert"][0]))
            group_rows.append(f'{int(row["global_sequence_index"])}:{int(row["subtask_index"])}')
            meta_rows.append(row)
    return {
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "values": np.stack(value_rows, axis=0).astype(np.float32),
        "success": np.stack(success_rows, axis=0).astype(np.float32),
        "steps": np.stack(steps_rows, axis=0).astype(np.float32),
        "chosen": np.asarray(chosen_rows, dtype=np.int64),
        "oracle": np.asarray(oracle_rows, dtype=np.int64),
        "groups": np.asarray(group_rows),
        "meta": meta_rows,
    }


def adjusted_target_scores(
    data: Dict[str, Any],
    oracle_bonus: float,
    chosen_penalty: float,
    recovery_bonus: float,
    long_step_lambda: float,
    operator_sample_boost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = data["values"].astype(np.float32).copy()
    sample_weights = np.ones((base.shape[0],), dtype=np.float32)
    expert_weights = np.ones_like(base, dtype=np.float32)

    for idx, row in enumerate(data["meta"]):
        adjusted = base[idx]
        chosen = int(data["chosen"][idx])
        oracle = int(data["oracle"][idx])
        row_steps = data["steps"][idx] / 160.0
        branches = row.get("branches", [])
        success = data["success"][idx]

        weights = row.get("data_reweight", {})
        tags = set(row.get("operator_tags", []))
        recovery_info = row.get("recovery_info", {})
        reflection = row.get("reflection", {})
        failure_type = str(reflection.get("failure_type", "uncertain"))

        adjusted[oracle] += float(oracle_bonus) * float(weights.get("oracle_expert", 1.0))

        if "avoid_collision_like_future" in tags or failure_type in {"wrong_expert", "unstable"}:
            adjusted[chosen] -= float(chosen_penalty) * max(0.0, 1.0 - float(weights.get("chosen_expert", 1.0)))
            expert_weights[idx, chosen] *= 0.75

        if "downweight_long_detour" in tags:
            adjusted -= float(long_step_lambda) * row_steps * (1.0 + float(weights.get("long_horizon_penalty", 0.0)))

        if (
            recovery_info.get("has_successful_alternative", False)
            or "prefer_recovery_capable" in tags
            or float(weights.get("recovery_bonus", 0.0)) > 0.0
        ):
            recover_mask = (success > 0.5).astype(np.float32)
            recover_mask[chosen] = 0.0
            adjusted += float(recovery_bonus) * (1.0 + float(weights.get("recovery_bonus", 0.0))) * recover_mask

        if row.get("policy_npz_path"):
            sample_weights[idx] += 0.1
        if tags:
            sample_weights[idx] += float(operator_sample_boost)
        if failure_type not in {"none", "uncertain"}:
            sample_weights[idx] += 0.1

        base[idx] = adjusted
    return base.astype(np.float32), sample_weights.astype(np.float32), expert_weights.astype(np.float32)


def soft_targets(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def weighted_evaluate(
    scores: np.ndarray,
    data: Dict[str, Any],
    labels: np.ndarray,
    idx: np.ndarray,
    sample_weights: np.ndarray,
) -> Dict[str, Any]:
    out = evaluate(scores, data, idx, labels)
    out["mean_sample_weight"] = float(sample_weights[idx].mean())
    out.update(probability_stats(scores, idx))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router with reflection-derived data reweight and preference shift.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle-bonus", type=float, default=0.2)
    parser.add_argument("--chosen-penalty", type=float, default=0.2)
    parser.add_argument("--recovery-bonus", type=float, default=0.15)
    parser.add_argument("--long-step-lambda", type=float, default=0.05)
    parser.add_argument("--operator-sample-boost", type=float, default=0.1)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset, max_samples=args.max_samples)
    if not rows:
        raise ValueError("No operator dataset rows found.")
    data = make_features(rows)
    adjusted_scores_np, sample_weights_np, expert_weights_np = adjusted_target_scores(
        data,
        oracle_bonus=args.oracle_bonus,
        chosen_penalty=args.chosen_penalty,
        recovery_bonus=args.recovery_bonus,
        long_step_lambda=args.long_step_lambda,
        operator_sample_boost=args.operator_sample_boost,
    )
    hard_labels_np = np.argmax(adjusted_scores_np, axis=1).astype(np.int64)
    soft_labels_np = soft_targets(adjusted_scores_np, args.target_temperature)
    train_idx, val_idx = split_indices(data["groups"], args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OracleRouter(data["x"].shape[-1], args.hidden_dim, data["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = torch.from_numpy(data["x"]).to(device)
    values = torch.from_numpy(data["values"]).to(device)
    soft_labels = torch.from_numpy(soft_labels_np).to(device)
    sample_weights = torch.from_numpy(sample_weights_np).to(device)
    expert_weights = torch.from_numpy(expert_weights_np).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)

    best_state = None
    best_val = -float("inf")
    for _ in range(args.epochs):
        model.train()
        scores = model(x[train_t])
        log_probs = F.log_softmax(scores, dim=1)
        weighted_targets = soft_labels[train_t] * expert_weights[train_t]
        weighted_targets = weighted_targets / torch.clamp(weighted_targets.sum(dim=1, keepdim=True), min=1e-12)
        kl = torch.sum(-weighted_targets * log_probs, dim=1)
        pred_value = torch.sum(F.softmax(scores, dim=1) * values[train_t], dim=1)
        loss = (kl * sample_weights[train_t]).mean() - float(args.value_weight) * pred_value.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(x[val_t])
            val_probs = F.softmax(val_scores, dim=1)
            val_pred_value = torch.sum(val_probs * values[val_t], dim=1).mean()
        if float(val_pred_value) > best_val:
            best_val = float(val_pred_value)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores_np = model(x).detach().cpu().numpy()

    metrics = {
        "method": "reflection_operator_router",
        "num_samples": int(len(rows)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(data["x"].shape[-1]),
        "feature_layout": list(BASE_FEATURE_LAYOUT),
        "oracle_bonus": float(args.oracle_bonus),
        "chosen_penalty": float(args.chosen_penalty),
        "recovery_bonus": float(args.recovery_bonus),
        "long_step_lambda": float(args.long_step_lambda),
        "operator_sample_boost": float(args.operator_sample_boost),
        "target_temperature": float(args.target_temperature),
        "value_weight": float(args.value_weight),
        "target_label_counts": {str(expert): int(np.sum(hard_labels_np == expert)) for expert in range(data["x"].shape[1])},
        "mean_sample_weight": float(sample_weights_np.mean()),
        "mean_expert_weight": float(expert_weights_np.mean()),
        "train": weighted_evaluate(scores_np, data, hard_labels_np, train_idx, sample_weights_np),
        "val": weighted_evaluate(scores_np, data, hard_labels_np, val_idx, sample_weights_np),
        "all": weighted_evaluate(scores_np, data, hard_labels_np, np.arange(len(rows), dtype=np.int64), sample_weights_np),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": int(data["x"].shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "num_experts": int(data["x"].shape[1]),
            "feature_layout": list(BASE_FEATURE_LAYOUT),
            "method": "reflection_operator_router",
            "oracle_bonus": float(args.oracle_bonus),
            "chosen_penalty": float(args.chosen_penalty),
            "recovery_bonus": float(args.recovery_bonus),
            "long_step_lambda": float(args.long_step_lambda),
            "operator_sample_boost": float(args.operator_sample_boost),
            "target_temperature": float(args.target_temperature),
            "value_weight": float(args.value_weight),
        },
        args.output_dir / "reflection_operator_router.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
