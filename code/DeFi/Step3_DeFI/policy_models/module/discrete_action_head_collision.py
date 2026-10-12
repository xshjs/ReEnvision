from typing import Dict, Optional

import torch
from torch import nn


class DiscreteActionHeadWithCollision(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dim: int,
        action_seq_len: int,
        n_trans_bins: int,
        n_rot_bins: int,
        use_latent_motion_tokens: bool = True,
        latent_motion_dim: int = 768,
    ):
        super().__init__()
        self.action_seq_len = action_seq_len
        self.n_trans_bins = n_trans_bins
        self.n_rot_bins = n_rot_bins
        self.use_latent_motion_tokens = use_latent_motion_tokens

        self.obs_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden_dim), nn.GELU())
        self.goal_proj = nn.Sequential(nn.LayerNorm(goal_dim), nn.Linear(goal_dim, hidden_dim), nn.GELU())
        if use_latent_motion_tokens:
            self.token_proj = nn.Sequential(nn.LayerNorm(latent_motion_dim), nn.Linear(latent_motion_dim, hidden_dim), nn.GELU())
            fused_in = hidden_dim * 3
        else:
            self.token_proj = None
            fused_in = hidden_dim * 2
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.step_emb = nn.Parameter(torch.randn(action_seq_len, hidden_dim) * 0.02)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.trans_head = nn.Linear(hidden_dim, 3 * n_trans_bins)
        self.rot_head = nn.Linear(hidden_dim, 3 * n_rot_bins)
        self.gripper_head = nn.Linear(hidden_dim, 2)
        self.collision_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        perceptual_emb: Dict[str, torch.Tensor],
        latent_goal: torch.Tensor,
        latent_motion_tokens_up: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_tokens = perceptual_emb["state_images_for_videoformer"]
        obs_tokens = obs_tokens.to(self.obs_proj[1].weight.dtype)
        obs_feat = self.obs_proj(obs_tokens.mean(dim=1))
        goal_feat = latent_goal.squeeze(1) if latent_goal.ndim == 3 else latent_goal
        goal_feat = goal_feat.to(self.goal_proj[1].weight.dtype)
        goal_feat = self.goal_proj(goal_feat)
        fused_parts = [obs_feat, goal_feat]
        if self.use_latent_motion_tokens:
            if latent_motion_tokens_up is None:
                raise ValueError("latent_motion_tokens_up is required when use_latent_motion_tokens=True")
            token_feat = latent_motion_tokens_up.mean(dim=1) if latent_motion_tokens_up.ndim == 3 else latent_motion_tokens_up
            token_feat = token_feat.to(self.token_proj[1].weight.dtype)
            token_feat = self.token_proj(token_feat)
            fused_parts.append(token_feat)
        fused = torch.cat(fused_parts, dim=-1).to(self.fuse[1].weight.dtype)
        base = self.fuse(fused)
        seq = self.trunk(base.unsqueeze(1) + self.step_emb.unsqueeze(0))
        return {
            "trans_logits": self.trans_head(seq).view(-1, self.action_seq_len, 3, self.n_trans_bins),
            "rot_logits": self.rot_head(seq).view(-1, self.action_seq_len, 3, self.n_rot_bins),
            "gripper_logits": self.gripper_head(seq),
            "collision_logits": self.collision_head(seq),
        }
