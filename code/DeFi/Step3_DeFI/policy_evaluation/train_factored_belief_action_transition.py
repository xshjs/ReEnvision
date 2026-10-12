from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from policy_evaluation.train_chunk_effect_phi_mlp import split_by_sequence
    from policy_evaluation.train_joint_execution_update_mlp import build_meta
except ModuleNotFoundError:
    from train_chunk_effect_phi_mlp import split_by_sequence
    from train_joint_execution_update_mlp import build_meta


class BeliefTransitionMLP(nn.Module):
    """T_theta: h_t,o_t,a_t,o_{t+1} -> h_{t+1}."""

    def __init__(self, state_dim: int, summary_dim: int, action_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        in_dim = state_dim * 2 + summary_dim + action_dim + task_dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )

    def forward(self, h_t: torch.Tensor, state_t: torch.Tensor, action_t: torch.Tensor, state_tp1: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, state_t, action_t, state_tp1, task_vec, meta], dim=-1)
        return self.net(x)


class ExpectedTransitionMLP(nn.Module):
    """F_phi: h_t,o_t,a_t,task -> expected h_{t+1}; intentionally cannot see o_{t+1}."""

    def __init__(self, state_dim: int, summary_dim: int, action_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        in_dim = state_dim + summary_dim + action_dim + task_dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )

    def forward(self, h_t: torch.Tensor, state_t: torch.Tensor, action_t: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, state_t, action_t, task_vec, meta], dim=-1)
        return self.net(x)


class RSSMTransitionMLP(nn.Module):
    """Unified RSSM-style belief dynamics: prior h^- and posterior h^+."""

    def __init__(self, state_dim: int, summary_dim: int, action_dim: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        prior_in = summary_dim + action_dim + task_dim + 3
        post_in = summary_dim + state_dim + task_dim + 3
        self.prior_net = nn.Sequential(
            nn.LayerNorm(prior_in),
            nn.Linear(prior_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.LayerNorm(post_in),
            nn.Linear(post_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )

    def prior(self, h_t: torch.Tensor, action_t: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, action_t, task_vec, meta], dim=-1)
        return self.prior_net(x)

    def posterior(self, h_prior: torch.Tensor, state_tp1: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_prior, state_tp1, task_vec, meta], dim=-1)
        return self.posterior_net(x)

    def forward(self, h_t: torch.Tensor, action_t: torch.Tensor, state_tp1: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        prior_delta = self.prior(h_t, action_t, task_vec, meta)
        h_prior = h_t + prior_delta
        posterior_delta_from_prior = self.posterior(h_prior, state_tp1, task_vec, meta)
        h_posterior = h_prior + posterior_delta_from_prior
        return {
            "prior_delta": prior_delta,
            "h_prior": h_prior,
            "posterior_delta_from_prior": posterior_delta_from_prior,
            "h_posterior": h_posterior,
            "posterior_delta": h_posterior - h_t,
            "innovation": h_posterior - h_prior,
        }

    def rollout_prior_trajectory(
        self,
        h_t: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out h^-_{t+1:t+K} once at the beginning of an action chunk."""
        h = h_t
        traj = []
        for k in range(int(action_chunk.shape[1])):
            prior_delta = self.prior(h, action_chunk[:, k], task_vec, meta)
            h = h + prior_delta
            traj.append(h)
        return torch.stack(traj, dim=1)


class FullRSSMTransitionMLP(nn.Module):
    """Recurrent stochastic RSSM: deterministic GRU state + Gaussian prior/posterior."""

    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        action_dim: int,
        task_dim: int,
        hidden_dim: int,
        dropout: float,
        deter_dim: int = 512,
        stoch_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.task_dim = int(task_dim)
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)
        self.init_deter = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, deter_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(summary_dim + action_dim + task_dim + 3, deter_dim)
        self.prior_head = nn.Sequential(
            nn.LayerNorm(deter_dim),
            nn.Linear(deter_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(state_dim + task_dim + 3),
            nn.Linear(state_dim + task_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(deter_dim + hidden_dim),
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )
        self.h_decoder = nn.Sequential(
            nn.LayerNorm(deter_dim + stoch_dim),
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, summary_dim),
        )

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = stats.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 1e-4
        return mean, std

    @staticmethod
    def _sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean.requires_grad:
            return mean + std * torch.randn_like(std)
        return mean

    @staticmethod
    def _kl_normal(
        post_mean: torch.Tensor,
        post_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
    ) -> torch.Tensor:
        var_ratio = (post_std.pow(2) + 1e-8) / (prior_std.pow(2) + 1e-8)
        t1 = ((post_mean - prior_mean).pow(2)) / (prior_std.pow(2) + 1e-8)
        kl = 0.5 * (var_ratio + t1 - 1.0 + 2.0 * (prior_std.log() - post_std.log()))
        return kl.sum(dim=-1).mean()

    def prior_step(
        self,
        h_t: torch.Tensor,
        action_t: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
        deter_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if deter_t is None:
            deter_t = self.init_deter(h_t)
        gru_in = torch.cat([h_t, action_t, task_vec, meta], dim=-1)
        deter_tp1 = self.gru(gru_in, deter_t)
        prior_mean, prior_std = self._split_stats(self.prior_head(deter_tp1))
        prior_stoch = self._sample(prior_mean, prior_std)
        h_prior = h_t + self.h_decoder(torch.cat([deter_tp1, prior_stoch], dim=-1))
        return {
            "deter": deter_tp1,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "prior_stoch": prior_stoch,
            "h_prior": h_prior,
            "prior_delta": h_prior - h_t,
        }

    def posterior_step(
        self,
        h_prior: torch.Tensor,
        deter_tp1: torch.Tensor,
        state_tp1: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        obs_feat = self.obs_encoder(torch.cat([state_tp1, task_vec, meta], dim=-1))
        post_mean, post_std = self._split_stats(self.posterior_head(torch.cat([deter_tp1, obs_feat], dim=-1)))
        post_stoch = self._sample(post_mean, post_std)
        h_posterior = h_prior + self.h_decoder(torch.cat([deter_tp1, post_stoch], dim=-1))
        return {
            "post_mean": post_mean,
            "post_std": post_std,
            "post_stoch": post_stoch,
            "h_posterior": h_posterior,
        }

    def forward(
        self,
        h_t: torch.Tensor,
        action_t: torch.Tensor,
        state_tp1: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
        deter_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prior = self.prior_step(h_t, action_t, task_vec, meta, deter_t=deter_t)
        posterior = self.posterior_step(prior["h_prior"], prior["deter"], state_tp1, task_vec, meta)
        h_posterior = posterior["h_posterior"]
        kl = self._kl_normal(
            posterior["post_mean"],
            posterior["post_std"],
            prior["prior_mean"],
            prior["prior_std"],
        )
        return {
            **prior,
            **posterior,
            "posterior_delta_from_prior": h_posterior - prior["h_prior"],
            "posterior_delta": h_posterior - h_t,
            "innovation": h_posterior - prior["h_prior"],
            "kl": kl,
        }

    def rollout_prior_trajectory(
        self,
        h_t: torch.Tensor,
        action_chunk: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        h = h_t
        deter = self.init_deter(h_t)
        traj = []
        for k in range(int(action_chunk.shape[1])):
            prior = self.prior_step(h, action_chunk[:, k], task_vec, meta, deter_t=deter)
            h = prior["h_prior"]
            deter = prior["deter"]
            traj.append(h)
        return torch.stack(traj, dim=1)


class ActionAdaptationMLP(nn.Module):
    """D_theta: A_remain,h_t,h_{t+1} -> delta_A and learned mode."""

    def __init__(
        self,
        summary_dim: int,
        action_dim: int,
        chunk_len: int,
        task_dim: int,
        hidden_dim: int,
        dropout: float,
        adaptation_input: str = "ht_hnext",
    ):
        super().__init__()
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.task_dim = int(task_dim)
        self.adaptation_input = str(adaptation_input)
        in_dim = summary_dim * 2 + chunk_len * action_dim + task_dim + 3
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, chunk_len * action_dim)
        self.mode_head = nn.Linear(hidden_dim, 3)  # keep, patch, replan
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        action_remain: torch.Tensor,
        h_t: torch.Tensor,
        h_tp1: torch.Tensor,
        task_vec: torch.Tensor,
        meta: torch.Tensor,
        h_expected: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.adaptation_input == "post_innovation":
            if h_expected is None:
                h_expected = h_t
            belief_context = torch.cat([h_tp1, h_tp1 - h_expected], dim=-1)
        else:
            belief_context = torch.cat([h_t, h_tp1], dim=-1)
        x = torch.cat([action_remain.reshape(action_remain.shape[0], -1), belief_context, task_vec, meta], dim=-1)
        y = self.trunk(x)
        return {
            "action_delta": self.delta_head(y).reshape(action_remain.shape[0], self.chunk_len, self.action_dim),
            "mode_logits": self.mode_head(y),
            "confidence_logit": self.confidence_head(y).squeeze(-1),
        }


class PlanCompatibilityHead(nn.Module):
    """D_theta: h^-,h^+,A_remain -> keep/repair/replan + confidence only."""

    def __init__(self, summary_dim: int, action_dim: int, chunk_len: int, task_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.summary_dim = int(summary_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.task_dim = int(task_dim)
        in_dim = summary_dim * 2 + chunk_len * action_dim + task_dim + 3
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mode_head = nn.Linear(hidden_dim, 3)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, action_remain: torch.Tensor, h_prior: torch.Tensor, h_posterior: torch.Tensor, task_vec: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                action_remain.reshape(action_remain.shape[0], -1),
                h_prior,
                h_posterior,
                task_vec,
                meta,
            ],
            dim=-1,
        )
        y = self.trunk(x)
        return {
            "mode_logits": self.mode_head(y),
            "confidence_logit": self.confidence_head(y).squeeze(-1),
        }


def load_memory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    required = [
        "state_start",
        "state_end",
        "base_summary_exec",
        "target_summary_exec",
        "action_chunk",
        "action_delta_target",
        "success",
        "task_vec",
        "sequence_index",
        "subtask_index",
        "cut_idx",
        "remaining_steps",
        "full_chunk_len",
    ]
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"memory missing keys: {missing}")
    return out


def make_bank(memory: dict[str, np.ndarray], keep_delta_norm: float, replan_delta_norm: float, target_delta_clip: float) -> dict[str, np.ndarray]:
    action_delta = np.asarray(memory["action_delta_target"], dtype=np.float32)
    if target_delta_clip > 0:
        action_delta = np.clip(action_delta, -float(target_delta_clip), float(target_delta_clip))
    delta_norm = np.linalg.norm(action_delta.reshape(action_delta.shape[0], -1), axis=1)
    if "mode_label" in memory:
        # Balanced memories can provide explicit keep/patch/replan labels.
        mode = np.asarray(memory["mode_label"], dtype=np.int64).reshape(-1)
    else:
        mode = np.ones((action_delta.shape[0],), dtype=np.int64)
        mode[delta_norm <= keep_delta_norm] = 0
        if replan_delta_norm > 0:
            mode[delta_norm >= replan_delta_norm] = 2
    return {
        "state_start": np.asarray(memory["state_start"], dtype=np.float32),
        "state_end": np.asarray(memory["state_end"], dtype=np.float32),
        "h_t": np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "h_tp1": np.asarray(memory["target_summary_exec"], dtype=np.float32),
        "belief_delta": np.asarray(memory["target_summary_exec"], dtype=np.float32) - np.asarray(memory["base_summary_exec"], dtype=np.float32),
        "action_chunk": np.asarray(memory["action_chunk"], dtype=np.float32),
        "action_delta": action_delta,
        "executed_action": np.asarray(memory.get("executed_action", np.zeros((action_delta.shape[0], action_delta.shape[-1]), dtype=np.float32)), dtype=np.float32),
        "success": np.asarray(memory["success"], dtype=np.float32).reshape(-1),
        "task_vec": np.asarray(memory["task_vec"], dtype=np.float32),
        "meta": build_meta(memory),
        "mode": mode,
        "sequence_index": np.asarray(memory["sequence_index"], dtype=np.int32).reshape(-1),
    }


def gather(bank: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in bank.items():
        if value.dtype.kind in "f":
            out[key] = torch.from_numpy(value[ids]).to(device)
        elif key == "mode":
            out[key] = torch.from_numpy(value[ids]).long().to(device)
    return out


def run_models(
    t_model: BeliefTransitionMLP,
    d_model: ActionAdaptationMLP,
    batch: dict[str, torch.Tensor],
    f_model: ExpectedTransitionMLP | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
    pred_delta_h = t_model(batch["h_t"], batch["state_start"], batch["executed_action"], batch["state_end"], batch["task_vec"], batch["meta"])
    pred_h_tp1 = batch["h_t"] + pred_delta_h
    expected_delta_h = None
    pred_h_expected = None
    if f_model is not None:
        expected_delta_h = f_model(batch["h_t"], batch["state_start"], batch["executed_action"], batch["task_vec"], batch["meta"])
        pred_h_expected = batch["h_t"] + expected_delta_h
    d_out = d_model(batch["action_chunk"], batch["h_t"], pred_h_tp1, batch["task_vec"], batch["meta"], h_expected=pred_h_expected)
    return pred_delta_h, expected_delta_h, pred_h_tp1, d_out


def run_rssm_models(
    rssm_model: RSSMTransitionMLP | FullRSSMTransitionMLP,
    compat_head: PlanCompatibilityHead,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rssm_out = rssm_model(
        batch["h_t"],
        batch["executed_action"],
        batch["state_end"],
        batch["task_vec"],
        batch["meta"],
    )
    d_out = compat_head(
        batch["action_chunk"],
        rssm_out["h_prior"],
        rssm_out["h_posterior"],
        batch["task_vec"],
        batch["meta"],
    )
    return rssm_out, d_out


def eval_models(
    t_model: BeliefTransitionMLP,
    d_model: ActionAdaptationMLP,
    bank: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    f_model: ExpectedTransitionMLP | None = None,
    expected_loss_weight: float = 0.5,
    action_loss_weight: float = 0.5,
    mode_loss_weight: float = 0.1,
    confidence_loss_weight: float = 0.05,
) -> dict[str, Any]:
    t_model.eval()
    d_model.eval()
    if f_model is not None:
        f_model.eval()
    vals: dict[str, list[torch.Tensor]] = {"loss": [], "belief_l1": [], "action_l1": [], "mode_acc": [], "confidence_l1": []}
    if f_model is not None:
        vals["expected_l1"] = []
        vals["innovation_l1"] = []
    with torch.no_grad():
        for start in range(0, ids.size, batch_size):
            batch = gather(bank, ids[start : start + batch_size], device)
            pred_delta_h, expected_delta_h, pred_h_tp1, d_out = run_models(t_model, d_model, batch, f_model=f_model)
            belief_loss = F.smooth_l1_loss(pred_delta_h, batch["belief_delta"])
            expected_loss = torch.zeros_like(belief_loss)
            if expected_delta_h is not None:
                expected_loss = F.smooth_l1_loss(expected_delta_h, batch["belief_delta"])
            action_mask = (batch["mode"] != 0).float().reshape(-1, 1, 1)
            if action_mask.sum() > 0:
                action_loss = F.smooth_l1_loss(d_out["action_delta"] * action_mask, batch["action_delta"] * action_mask)
            else:
                action_loss = d_out["action_delta"].sum() * 0.0
            mode_loss = F.cross_entropy(d_out["mode_logits"], batch["mode"])
            conf_target = (batch["mode"] != 2).float()
            confidence_loss = F.binary_cross_entropy_with_logits(d_out["confidence_logit"], conf_target)
            loss = (
                belief_loss
                + expected_loss_weight * expected_loss
                + action_loss_weight * action_loss
                + mode_loss_weight * mode_loss
                + confidence_loss_weight * confidence_loss
            )
            vals["loss"].append(loss.cpu())
            vals["belief_l1"].append(F.l1_loss(pred_delta_h, batch["belief_delta"]).cpu())
            if expected_delta_h is not None:
                vals["expected_l1"].append(F.l1_loss(expected_delta_h, batch["belief_delta"]).cpu())
                vals["innovation_l1"].append(F.l1_loss(pred_delta_h - expected_delta_h, torch.zeros_like(pred_delta_h)).cpu())
            vals["action_l1"].append(F.l1_loss(d_out["action_delta"], batch["action_delta"]).cpu())
            vals["mode_acc"].append((d_out["mode_logits"].argmax(dim=-1) == batch["mode"]).float().mean().cpu())
            vals["confidence_l1"].append(F.l1_loss(d_out["confidence_logit"].sigmoid(), conf_target).cpu())
    return {key: float(torch.stack(value).mean().item()) for key, value in vals.items()}


def eval_rssm_models(
    rssm_model: RSSMTransitionMLP | FullRSSMTransitionMLP,
    compat_head: PlanCompatibilityHead,
    bank: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    prior_loss_weight: float = 0.5,
    mode_loss_weight: float = 0.1,
    confidence_loss_weight: float = 0.05,
    kl_loss_weight: float = 0.0,
) -> dict[str, Any]:
    rssm_model.eval()
    compat_head.eval()
    vals: dict[str, list[torch.Tensor]] = {
        "loss": [],
        "prior_l1": [],
        "posterior_l1": [],
        "innovation_l1": [],
        "mode_acc": [],
        "confidence_l1": [],
    }
    with torch.no_grad():
        for start in range(0, ids.size, batch_size):
            batch = gather(bank, ids[start : start + batch_size], device)
            rssm_out, d_out = run_rssm_models(rssm_model, compat_head, batch)
            prior_loss = F.smooth_l1_loss(rssm_out["prior_delta"], batch["belief_delta"])
            posterior_loss = F.smooth_l1_loss(rssm_out["posterior_delta"], batch["belief_delta"])
            mode_loss = F.cross_entropy(d_out["mode_logits"], batch["mode"])
            conf_target = (batch["mode"] != 2).float()
            confidence_loss = F.binary_cross_entropy_with_logits(d_out["confidence_logit"], conf_target)
            loss = (
                posterior_loss
                + float(prior_loss_weight) * prior_loss
                + float(mode_loss_weight) * mode_loss
                + float(confidence_loss_weight) * confidence_loss
                + float(kl_loss_weight) * rssm_out.get("kl", torch.zeros_like(posterior_loss))
            )
            vals["loss"].append(loss.cpu())
            vals["prior_l1"].append(F.l1_loss(rssm_out["prior_delta"], batch["belief_delta"]).cpu())
            vals["posterior_l1"].append(F.l1_loss(rssm_out["posterior_delta"], batch["belief_delta"]).cpu())
            vals["innovation_l1"].append(torch.mean(torch.abs(rssm_out["innovation"])).cpu())
            vals["mode_acc"].append((d_out["mode_logits"].argmax(dim=-1) == batch["mode"]).float().mean().cpu())
            vals["confidence_l1"].append(F.l1_loss(d_out["confidence_logit"].sigmoid(), conf_target).cpu())
    return {key: float(torch.stack(value).mean().item()) for key, value in vals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train factored T_theta belief dynamics and D_theta action adaptation.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--keep-delta-norm", type=float, default=0.02)
    parser.add_argument("--replan-delta-norm", type=float, default=0.0)
    parser.add_argument("--expected-transition", action="store_true", help="Train F_phi without state_end and use posterior-expected innovation for D.")
    parser.add_argument("--rssm-plan-head", action="store_true", help="Train one RSSM prior/posterior model and a mode/confidence-only plan compatibility head.")
    parser.add_argument("--full-rssm-plan-head", action="store_true", help="Train a GRU + stochastic latent RSSM and a mode/confidence-only plan compatibility head.")
    parser.add_argument("--rssm-deter-dim", type=int, default=512)
    parser.add_argument("--rssm-stoch-dim", type=int, default=256)
    parser.add_argument("--kl-loss-weight", type=float, default=1e-3)
    parser.add_argument("--expected-loss-weight", type=float, default=0.5)
    parser.add_argument("--action-loss-weight", type=float, default=0.5)
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.05)
    parser.add_argument("--target-delta-clip", type=float, default=0.0, help="Clip supervised action_delta_target before training; 0 disables clipping.")
    parser.add_argument("--adaptation-input", choices=["ht_hnext", "post_innovation"], default="ht_hnext")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    memory = load_memory(args.memory_npz)
    bank = make_bank(memory, args.keep_delta_norm, args.replan_delta_norm, args.target_delta_clip)
    for key, value in bank.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite values in {key}")

    train_ids, val_ids = split_by_sequence(bank["sequence_index"], args.val_ratio, args.seed)
    if val_ids.size == 0:
        rng = np.random.default_rng(args.seed)
        ids = np.arange(bank["sequence_index"].shape[0])
        rng.shuffle(ids)
        n_val = max(1, int(ids.size * args.val_ratio))
        val_ids, train_ids = ids[:n_val], ids[n_val:]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if args.rssm_plan_head or args.full_rssm_plan_head:
        rssm_cls = FullRSSMTransitionMLP if args.full_rssm_plan_head else RSSMTransitionMLP
        rssm_kwargs = {
            "state_dim": bank["state_start"].shape[-1],
            "summary_dim": bank["h_t"].shape[-1],
            "action_dim": bank["executed_action"].shape[-1],
            "task_dim": bank["task_vec"].shape[-1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        if args.full_rssm_plan_head:
            rssm_kwargs["deter_dim"] = int(args.rssm_deter_dim)
            rssm_kwargs["stoch_dim"] = int(args.rssm_stoch_dim)
        rssm_model = rssm_cls(**rssm_kwargs).to(device)
        compat_head = PlanCompatibilityHead(
            summary_dim=bank["h_t"].shape[-1],
            action_dim=bank["action_chunk"].shape[-1],
            chunk_len=bank["action_chunk"].shape[-2],
            task_dim=bank["task_vec"].shape[-1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        opt = torch.optim.AdamW(
            list(rssm_model.parameters()) + list(compat_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        rng = np.random.default_rng(args.seed)
        best: dict[str, Any] | None = None
        best_state = None
        log_path = args.output_dir / "train_log.jsonl"
        with log_path.open("w") as log_f:
            for step in range(1, args.steps + 1):
                ids = rng.choice(train_ids, size=args.batch_size, replace=train_ids.size < args.batch_size)
                batch = gather(bank, ids, device)
                rssm_model.train()
                compat_head.train()
                rssm_out, d_out = run_rssm_models(rssm_model, compat_head, batch)
                prior_loss = F.smooth_l1_loss(rssm_out["prior_delta"], batch["belief_delta"])
                posterior_loss = F.smooth_l1_loss(rssm_out["posterior_delta"], batch["belief_delta"])
                mode_loss = F.cross_entropy(d_out["mode_logits"], batch["mode"])
                conf_target = (batch["mode"] != 2).float()
                confidence_loss = F.binary_cross_entropy_with_logits(d_out["confidence_logit"], conf_target)
                loss = (
                    posterior_loss
                    + float(args.expected_loss_weight) * prior_loss
                    + float(args.mode_loss_weight) * mode_loss
                    + float(args.confidence_loss_weight) * confidence_loss
                    + (
                        float(args.kl_loss_weight) * rssm_out.get("kl", torch.zeros_like(posterior_loss))
                        if args.full_rssm_plan_head
                        else 0.0
                    )
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step == 1 or step % 200 == 0 or step == args.steps:
                    metrics = eval_rssm_models(
                        rssm_model,
                        compat_head,
                        bank,
                        val_ids,
                        device,
                        args.eval_batch_size,
                        prior_loss_weight=float(args.expected_loss_weight),
                        mode_loss_weight=float(args.mode_loss_weight),
                        confidence_loss_weight=float(args.confidence_loss_weight),
                        kl_loss_weight=float(args.kl_loss_weight) if args.full_rssm_plan_head else 0.0,
                    )
                    metrics.update({"step": int(step), "train_loss": float(loss.item())})
                    log_f.write(json.dumps(metrics) + "\n")
                    log_f.flush()
                    if best is None or metrics["loss"] < best["loss"]:
                        best = dict(metrics)
                        best_state = {
                            "rssm_model": {k: v.detach().cpu() for k, v in rssm_model.state_dict().items()},
                            "compat_head": {k: v.detach().cpu() for k, v in compat_head.state_dict().items()},
                        }
                    print(json.dumps(metrics), flush=True)
        if best_state is not None:
            rssm_model.load_state_dict(best_state["rssm_model"])
            compat_head.load_state_dict(best_state["compat_head"])
        runtime_type = "full_rssm_plan_compatibility" if args.full_rssm_plan_head else "rssm_plan_compatibility"
        ckpt_path = args.output_dir / "rssm_plan_compatibility.pt"
        torch.save(
            {
                "runtime_type": runtime_type,
                "rssm_model_state": rssm_model.state_dict(),
                "compat_head_state": compat_head.state_dict(),
                "state_dim": int(bank["state_start"].shape[-1]),
                "summary_dim": int(bank["h_t"].shape[-1]),
                "action_dim": int(bank["action_chunk"].shape[-1]),
                "chunk_len": int(bank["action_chunk"].shape[-2]),
                "task_dim": int(bank["task_vec"].shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "rssm_deter_dim": int(args.rssm_deter_dim) if args.full_rssm_plan_head else None,
                "rssm_stoch_dim": int(args.rssm_stoch_dim) if args.full_rssm_plan_head else None,
                "kl_loss_weight": float(args.kl_loss_weight) if args.full_rssm_plan_head else 0.0,
                "keep_delta_norm": float(args.keep_delta_norm),
                "replan_delta_norm": float(args.replan_delta_norm),
                "mode_loss_weight": float(args.mode_loss_weight),
                "confidence_loss_weight": float(args.confidence_loss_weight),
                "target_delta_clip": float(args.target_delta_clip),
                "best": best,
            },
            ckpt_path,
        )
        summary = {
            "memory_npz": str(args.memory_npz),
            "runtime_type": runtime_type,
            "rssm_deter_dim": int(args.rssm_deter_dim) if args.full_rssm_plan_head else None,
            "rssm_stoch_dim": int(args.rssm_stoch_dim) if args.full_rssm_plan_head else None,
            "kl_loss_weight": float(args.kl_loss_weight) if args.full_rssm_plan_head else 0.0,
            "num_train": int(train_ids.size),
            "num_val": int(val_ids.size),
            "mode_counts": {str(k): int(v) for k, v in zip(*np.unique(bank["mode"], return_counts=True))},
            "steps": int(args.steps),
            "best": best,
            "output_ckpt": str(ckpt_path),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    t_model = BeliefTransitionMLP(
        state_dim=bank["state_start"].shape[-1],
        summary_dim=bank["h_t"].shape[-1],
        action_dim=bank["executed_action"].shape[-1],
        task_dim=bank["task_vec"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    f_model = None
    if args.expected_transition:
        f_model = ExpectedTransitionMLP(
            state_dim=bank["state_start"].shape[-1],
            summary_dim=bank["h_t"].shape[-1],
            action_dim=bank["executed_action"].shape[-1],
            task_dim=bank["task_vec"].shape[-1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
    d_model = ActionAdaptationMLP(
        summary_dim=bank["h_t"].shape[-1],
        action_dim=bank["action_chunk"].shape[-1],
        chunk_len=bank["action_chunk"].shape[-2],
        task_dim=bank["task_vec"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        adaptation_input=("post_innovation" if args.expected_transition else args.adaptation_input),
    ).to(device)
    params = list(t_model.parameters()) + list(d_model.parameters())
    if f_model is not None:
        params += list(f_model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best: dict[str, Any] | None = None
    best_state = None
    log_path = args.output_dir / "train_log.jsonl"

    with log_path.open("w") as log_f:
        for step in range(1, args.steps + 1):
            ids = rng.choice(train_ids, size=args.batch_size, replace=train_ids.size < args.batch_size)
            batch = gather(bank, ids, device)
            t_model.train()
            d_model.train()
            if f_model is not None:
                f_model.train()
            pred_delta_h, expected_delta_h, pred_h_tp1, d_out = run_models(t_model, d_model, batch, f_model=f_model)
            belief_loss = F.smooth_l1_loss(pred_delta_h, batch["belief_delta"])
            expected_loss = torch.zeros_like(belief_loss)
            if expected_delta_h is not None:
                expected_loss = F.smooth_l1_loss(expected_delta_h, batch["belief_delta"])
            action_mask = (batch["mode"] != 0).float().reshape(-1, 1, 1)
            if action_mask.sum() > 0:
                action_loss = F.smooth_l1_loss(d_out["action_delta"] * action_mask, batch["action_delta"] * action_mask)
            else:
                action_loss = d_out["action_delta"].sum() * 0.0
            mode_loss = F.cross_entropy(d_out["mode_logits"], batch["mode"])
            conf_target = (batch["mode"] != 2).float()
            confidence_loss = F.binary_cross_entropy_with_logits(d_out["confidence_logit"], conf_target)
            loss = (
                belief_loss
                + float(args.expected_loss_weight) * expected_loss
                + float(args.action_loss_weight) * action_loss
                + float(args.mode_loss_weight) * mode_loss
                + float(args.confidence_loss_weight) * confidence_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % 200 == 0 or step == args.steps:
                metrics = eval_models(
                    t_model,
                    d_model,
                    bank,
                    val_ids,
                    device,
                    args.eval_batch_size,
                    f_model=f_model,
                    expected_loss_weight=float(args.expected_loss_weight),
                    action_loss_weight=float(args.action_loss_weight),
                    mode_loss_weight=float(args.mode_loss_weight),
                    confidence_loss_weight=float(args.confidence_loss_weight),
                )
                metrics.update({"step": int(step), "train_loss": float(loss.item())})
                log_f.write(json.dumps(metrics) + "\n")
                log_f.flush()
                if best is None or metrics["loss"] < best["loss"]:
                    best = dict(metrics)
                    best_state = {
                        "t_model": {k: v.detach().cpu() for k, v in t_model.state_dict().items()},
                        "d_model": {k: v.detach().cpu() for k, v in d_model.state_dict().items()},
                    }
                    if f_model is not None:
                        best_state["f_model"] = {k: v.detach().cpu() for k, v in f_model.state_dict().items()}
                print(json.dumps(metrics), flush=True)

    if best_state is not None:
        t_model.load_state_dict(best_state["t_model"])
        d_model.load_state_dict(best_state["d_model"])
        if f_model is not None and "f_model" in best_state:
            f_model.load_state_dict(best_state["f_model"])
    ckpt_path = args.output_dir / "factored_belief_action_transition.pt"
    torch.save(
        {
            "runtime_type": "factored_belief_action_transition",
            "t_model_state": t_model.state_dict(),
            "d_model_state": d_model.state_dict(),
            "f_model_state": f_model.state_dict() if f_model is not None else None,
            "has_expected_transition": bool(f_model is not None),
            "adaptation_input": str(d_model.adaptation_input),
            "state_dim": int(bank["state_start"].shape[-1]),
            "summary_dim": int(bank["h_t"].shape[-1]),
            "action_dim": int(bank["action_chunk"].shape[-1]),
            "chunk_len": int(bank["action_chunk"].shape[-2]),
            "task_dim": int(bank["task_vec"].shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "keep_delta_norm": float(args.keep_delta_norm),
            "replan_delta_norm": float(args.replan_delta_norm),
            "action_loss_weight": float(args.action_loss_weight),
            "mode_loss_weight": float(args.mode_loss_weight),
            "confidence_loss_weight": float(args.confidence_loss_weight),
            "target_delta_clip": float(args.target_delta_clip),
            "best": best,
        },
        ckpt_path,
    )
    summary = {
        "memory_npz": str(args.memory_npz),
        "num_train": int(train_ids.size),
        "num_val": int(val_ids.size),
        "mode_counts": {str(k): int(v) for k, v in zip(*np.unique(bank["mode"], return_counts=True))},
        "has_expected_transition": bool(f_model is not None),
        "adaptation_input": str(d_model.adaptation_input),
        "loss_weights": {
            "expected": float(args.expected_loss_weight),
            "action": float(args.action_loss_weight),
            "mode": float(args.mode_loss_weight),
            "confidence": float(args.confidence_loss_weight),
        },
        "target_delta_clip": float(args.target_delta_clip),
        "steps": int(args.steps),
        "best": best,
        "output_ckpt": str(ckpt_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
