from __future__ import annotations

import torch
from torch import nn


class GFDMFuturePredictor(nn.Module):
    def __init__(self, token_dim: int, goal_dim: int, hidden_dim: int = 512, depth: int = 2):
        super().__init__()
        self.goal_proj = nn.Sequential(nn.LayerNorm(goal_dim), nn.Linear(goal_dim, token_dim))
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, token_dim),
                ]
            )
        self.token_mlp = nn.Sequential(*layers)

    def forward(self, current_tokens: torch.Tensor, latent_goal: torch.Tensor) -> torch.Tensor:
        proj_weight = self.goal_proj[1].weight
        proj_dtype = proj_weight.dtype
        proj_device = proj_weight.device
        current_tokens = current_tokens.to(device=proj_device, dtype=proj_dtype)
        goal_feat = latent_goal.squeeze(1) if latent_goal.ndim == 3 else latent_goal
        goal_feat = goal_feat.to(device=proj_device, dtype=proj_dtype)
        goal_bias = self.goal_proj(goal_feat).unsqueeze(1)
        x = current_tokens + goal_bias
        return current_tokens + self.token_mlp(x)
