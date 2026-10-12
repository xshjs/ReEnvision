import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_future_portfolio_viability import execution_context_features, load_arrays
from train_universal_moe_oracle_router import filter_rows_by_sequence, load_rows, probability_stats, split_indices


BASE_FEATURE_LAYOUT = [
    "obs_embedding",
    "expert_future",
    "expert_future_minus_obs",
    "router_logit",
    "router_score",
    "expert_contradiction",
]


class ExecutionEnergyRouter(nn.Module):
    def __init__(self, feature_dim: int, execution_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.expert_embed = nn.Embedding(num_experts, 16)
        self.execution_encoder = nn.Sequential(
            nn.LayerNorm(execution_dim),
            nn.Linear(execution_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.feature_backbone = nn.Sequential(
            nn.LayerNorm(feature_dim + 16),
            nn.Linear(feature_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.step_head = nn.Linear(hidden_dim, 1)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, execution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_experts, _ = x.shape
        expert_ids = torch.arange(num_experts, device=x.device).unsqueeze(0).expand(batch, -1)
        expert_embed = self.expert_embed(expert_ids)
        z_exec = self.execution_encoder(execution)
        hidden = self.feature_backbone(torch.cat([x, expert_embed], dim=-1))
        hidden = hidden + z_exec.unsqueeze(1)
        energy = self.energy_head(hidden).squeeze(-1)
        pred_steps = torch.sigmoid(self.step_head(hidden).squeeze(-1))
        success_logits = self.success_head(hidden).squeeze(-1)
        return energy, pred_steps, success_logits, z_exec


def pairwise_energy_ranking_loss(
    energy: torch.Tensor,
    success: torch.Tensor,
    values: torch.Tensor,
    steps: torch.Tensor,
    chosen: torch.Tensor,
    margin: float,
    step_lambda: float,
    recoverable_upweight: float,
) -> torch.Tensor:
    target_score = values + success - float(step_lambda) * steps
    diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
    valid = diff > 1e-6
    if not torch.any(valid):
        return torch.zeros((), device=energy.device)
    energy_gap = energy.unsqueeze(2) - energy.unsqueeze(1)
    pair_loss = F.relu(float(margin) + energy_gap)
    weights = torch.ones_like(pair_loss)
    oracle_success = success.max(dim=1, keepdim=True).values
    chosen_success = success.gather(1, chosen.unsqueeze(1))
    recoverable_mask = (chosen_success < 0.5) & (oracle_success > 0.5)
    if float(recoverable_upweight) > 1.0:
        weights = torch.where(
            recoverable_mask.unsqueeze(2),
            weights * float(recoverable_upweight),
            weights,
        )
    pair_loss = pair_loss * weights
    return (pair_loss * valid.float()).sum() / torch.clamp(valid.float().sum(), min=1.0)


def evaluate_energy(
    energy: np.ndarray,
    pred_steps: np.ndarray,
    success_logits: np.ndarray,
    data: Dict[str, Any],
    idx: np.ndarray,
    ep_len: float,
) -> Dict[str, Any]:
    pred = np.argmin(energy[idx], axis=1)
    chosen = data["chosen"][idx]
    oracle = data["oracle"][idx]
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
    success_prob = 1.0 / (1.0 + np.exp(-success_logits[idx]))
    success_bce = -(
        success * np.log(np.clip(success_prob, 1e-7, 1.0))
        + (1.0 - success) * np.log(np.clip(1.0 - success_prob, 1e-7, 1.0))
    )
    success_mask = success > 0.5
    step_target = data["steps"][idx] / float(ep_len)
    step_mse = (pred_steps[idx] - step_target) ** 2
    return {
        "num": int(len(idx)),
        "router_success": float(router_success.mean()),
        "energy_success": float(pred_success.mean()),
        "oracle_success": float(oracle_success.mean()),
        "router_value": float(router_value.mean()),
        "energy_value": float(pred_value.mean()),
        "oracle_value": float(oracle_value.mean()),
        "router_expert_acc": float((chosen == oracle).mean()),
        "energy_expert_acc": float((pred == oracle).mean()),
        "recoverable_failures": int(recoverable.sum()),
        "recovered_by_energy": int(recovered.sum()),
        "recover_rate": float(recovered.sum() / max(recoverable.sum(), 1)),
        "success_bce": float(success_bce.mean()),
        "step_mse_success": float(step_mse[success_mask].mean()) if success_mask.any() else 0.0,
        "mean_energy": float(energy[idx].mean()),
        "mean_success_logit": float(success_logits[idx].mean()),
        **probability_stats(-energy, idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train execution-memory-conditioned energy router.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ep-len", type=float, default=160.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--success-weight", type=float, default=0.5)
    parser.add_argument("--step-weight", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.05)
    parser.add_argument("--pairwise-step-lambda", type=float, default=0.05)
    parser.add_argument("--recoverable-upweight", type=float, default=2.0)
    parser.add_argument("--exclude-sequence-start", type=int, default=-1)
    parser.add_argument("--exclude-sequence-end", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.roots)
    rows_before_filter = len(rows)
    rows = filter_rows_by_sequence(rows, args.exclude_sequence_start, args.exclude_sequence_end)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No Universal MoE rows found.")

    data = load_arrays(rows)
    execution_np = execution_context_features(data, args.ep_len)
    train_idx, val_idx = split_indices(data["groups"], args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExecutionEnergyRouter(
        feature_dim=int(data["x"].shape[-1]),
        execution_dim=int(execution_np.shape[-1]),
        hidden_dim=int(args.hidden_dim),
        num_experts=int(data["x"].shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = torch.from_numpy(data["x"]).to(device)
    execution = torch.from_numpy(execution_np).to(device)
    success_targets = torch.from_numpy(data["success"]).to(device)
    step_targets = torch.from_numpy(data["steps"] / float(args.ep_len)).to(device)
    raw_steps = torch.from_numpy(data["steps"]).to(device)
    values = torch.from_numpy(data["values"]).to(device)
    oracle = torch.from_numpy(data["oracle"]).to(device)
    chosen = torch.from_numpy(data["chosen"]).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)

    best_state = None
    best_val = -float("inf")
    for _ in range(args.epochs):
        model.train()
        energy, pred_steps, success_logits, _ = model(x[train_t], execution[train_t])
        rank_loss = F.cross_entropy(-energy, oracle[train_t])
        pairwise_loss = pairwise_energy_ranking_loss(
            energy,
            success_targets[train_t],
            values[train_t],
            raw_steps[train_t] / float(args.ep_len),
            chosen[train_t],
            args.pairwise_margin,
            args.pairwise_step_lambda,
            args.recoverable_upweight,
        )
        success_loss = F.binary_cross_entropy_with_logits(success_logits, success_targets[train_t])
        success_mask = success_targets[train_t] > 0.5
        step_loss = F.mse_loss(pred_steps[success_mask], step_targets[train_t][success_mask]) if success_mask.any() else torch.zeros((), device=device)
        pred_value = torch.sum(F.softmax(-energy, dim=1) * values[train_t], dim=1)
        loss = (
            float(args.rank_weight) * rank_loss
            + float(args.pairwise_weight) * pairwise_loss
            + float(args.success_weight) * success_loss
            + float(args.step_weight) * step_loss
            - float(args.value_weight) * pred_value.mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_energy, _, _, _ = model(x[val_t], execution[val_t])
            val_pred = torch.argmin(val_energy, dim=1)
            val_success = success_targets[val_t]
            score = val_success[torch.arange(len(val_t), device=device), val_pred].mean()
        if float(score) > best_val:
            best_val = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        energy_t, pred_steps_t, success_logits_t, _ = model(x, execution)
        energy_np = energy_t.detach().cpu().numpy()
        pred_steps_np = pred_steps_t.detach().cpu().numpy()
        success_logits_np = success_logits_t.detach().cpu().numpy()

    metrics = {
        "method": "execution_energy_router",
        "num_samples": int(len(data["x"])),
        "num_samples_before_sequence_filter": int(rows_before_filter),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(data["x"].shape[-1]),
        "execution_dim": int(execution_np.shape[-1]),
        "feature_layout": list(BASE_FEATURE_LAYOUT),
        "rank_weight": float(args.rank_weight),
        "pairwise_weight": float(args.pairwise_weight),
        "pairwise_margin": float(args.pairwise_margin),
        "pairwise_step_lambda": float(args.pairwise_step_lambda),
        "recoverable_upweight": float(args.recoverable_upweight),
        "success_weight": float(args.success_weight),
        "step_weight": float(args.step_weight),
        "value_weight": float(args.value_weight),
        "train": evaluate_energy(energy_np, pred_steps_np, success_logits_np, data, train_idx, args.ep_len),
        "val": evaluate_energy(energy_np, pred_steps_np, success_logits_np, data, val_idx, args.ep_len),
        "all": evaluate_energy(energy_np, pred_steps_np, success_logits_np, data, np.arange(len(data["x"]), dtype=np.int64), args.ep_len),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": int(data["x"].shape[-1]),
            "execution_dim": int(execution_np.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "num_experts": int(data["x"].shape[1]),
            "feature_layout": list(BASE_FEATURE_LAYOUT),
            "method": "execution_energy_router",
        },
        args.output_dir / "execution_energy_router.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
