from __future__ import annotations

import torch
from torch import nn


class ResidualActionHead(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        future_dim: int,
        motion_dim: int,
        goal_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.obs_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden_dim), nn.GELU())
        self.future_proj = nn.Sequential(nn.LayerNorm(future_dim), nn.Linear(future_dim, hidden_dim), nn.GELU())
        self.motion_proj = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden_dim), nn.GELU())
        self.goal_proj = nn.Sequential(nn.LayerNorm(goal_dim), nn.Linear(goal_dim, hidden_dim), nn.GELU())
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.gripper_head = nn.Linear(hidden_dim, 2)
        self.collision_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        motion_latent: torch.Tensor,
        latent_goal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        proj_weight = self.obs_proj[1].weight
        proj_dtype = proj_weight.dtype
        proj_device = proj_weight.device
        current_tokens = current_tokens.to(device=proj_device, dtype=proj_dtype)
        future_tokens = future_tokens.to(device=proj_device, dtype=proj_dtype)
        motion_latent = motion_latent.to(device=proj_device, dtype=proj_dtype)
        if motion_latent.ndim == 3:
            motion_latent = motion_latent.mean(dim=1)
        goal_feat = latent_goal.squeeze(1) if latent_goal.ndim == 3 else latent_goal
        goal_feat = goal_feat.to(device=proj_device, dtype=proj_dtype)
        obs_feat = self.obs_proj(current_tokens.mean(dim=1))
        future_feat = self.future_proj(future_tokens.mean(dim=1))
        motion_feat = self.motion_proj(motion_latent)
        goal_feat = self.goal_proj(goal_feat)
        fused = self.fuse(torch.cat([obs_feat, future_feat, motion_feat, goal_feat], dim=-1))
        return {
            "delta_xyz": self.delta_head(fused),
            "gripper_logits": self.gripper_head(fused),
            "collision_logits": self.collision_head(fused),
        }


class MotionAuxHead(nn.Module):
    def __init__(self, motion_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(motion_dim),
            nn.Linear(motion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.gripper_head = nn.Linear(hidden_dim, 2)

    def forward(self, motion_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        proj_weight = self.trunk[1].weight
        proj_dtype = proj_weight.dtype
        proj_device = proj_weight.device
        motion_latent = motion_latent.to(device=proj_device, dtype=proj_dtype)
        if motion_latent.ndim == 3:
            motion_latent = motion_latent.mean(dim=1)
        feat = self.trunk(motion_latent)
        return {
            "delta_xyz": self.delta_head(feat),
            "gripper_logits": self.gripper_head(feat),
        }
