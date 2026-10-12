from typing import Dict, Optional

import torch
from torch import nn


class DiscreteActionHead(nn.Module):
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

        token_dim = latent_motion_dim if use_latent_motion_tokens else 0
        fused_dim = obs_dim + goal_dim + token_dim

        self.obs_proj = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
        )
        self.goal_proj = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
            nn.GELU(),
        )
        if use_latent_motion_tokens:
            self.token_proj = nn.Sequential(
                nn.LayerNorm(latent_motion_dim),
                nn.Linear(latent_motion_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            self.token_proj = None

        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * (3 if use_latent_motion_tokens else 2)),
            nn.Linear(hidden_dim * (3 if use_latent_motion_tokens else 2), hidden_dim),
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
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        perceptual_emb: Dict[str, torch.Tensor],
        latent_goal: torch.Tensor,
        latent_motion_tokens_up: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_tokens = perceptual_emb["state_images_for_videoformer"]
        obs_feat = obs_tokens.mean(dim=1)
        obs_feat = self.obs_proj(obs_feat)

        goal_feat = latent_goal.squeeze(1) if latent_goal.ndim == 3 else latent_goal
        goal_feat = self.goal_proj(goal_feat)

        fused_parts = [obs_feat, goal_feat]
        if self.use_latent_motion_tokens:
            if latent_motion_tokens_up is None:
                raise ValueError("latent_motion_tokens_up is required when use_latent_motion_tokens=True")
            token_feat = latent_motion_tokens_up
            if token_feat.ndim == 2:
                pass
            elif token_feat.ndim == 3:
                token_feat = token_feat.mean(dim=1)
            else:
                raise ValueError(f"Unsupported latent_motion_tokens_up shape: {token_feat.shape}")
            token_feat = self.token_proj(token_feat)
            fused_parts.append(token_feat)

        fused = torch.cat(fused_parts, dim=-1)
        base = self.fuse(fused)
        seq = base.unsqueeze(1) + self.step_emb.unsqueeze(0)
        seq = self.trunk(seq)

        trans_logits = self.trans_head(seq).view(-1, self.action_seq_len, 3, self.n_trans_bins)
        rot_logits = self.rot_head(seq).view(-1, self.action_seq_len, 3, self.n_rot_bins)
        gripper_logits = self.gripper_head(seq).squeeze(-1)
        return {
            "trans_logits": trans_logits,
            "rot_logits": rot_logits,
            "gripper_logits": gripper_logits,
        }
