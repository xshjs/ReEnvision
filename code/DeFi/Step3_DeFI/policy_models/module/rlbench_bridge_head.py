from typing import Dict

import torch
from torch import nn


class RLBenchBridgeHead(nn.Module):
    """Bridge-style discrete RLBench head over DeFi perceptual features."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dim: int,
        action_seq_len: int,
        n_trans_bins: int,
        n_rot_bins: int,
        motion_dim: int | None = None,
    ):
        super().__init__()
        self.action_seq_len = action_seq_len
        self.n_trans_bins = n_trans_bins
        self.n_rot_bins = n_rot_bins
        self.motion_dim = motion_dim

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
        self.voxel_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.motion_proj = None
        fuse_in_dim = hidden_dim * 3
        if motion_dim is not None:
            self.motion_proj = nn.Sequential(
                nn.LayerNorm(motion_dim),
                nn.Linear(motion_dim, hidden_dim),
                nn.GELU(),
            )
            fuse_in_dim = hidden_dim * 4
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in_dim),
            nn.Linear(fuse_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.step_emb = nn.Parameter(torch.randn(action_seq_len, hidden_dim) * 0.02)
        self.voxel_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
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

    def forward(self, perceptual_emb: Dict[str, torch.Tensor], latent_goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_with_motion(perceptual_emb, latent_goal, motion_latent=None)

    def forward_with_motion(
        self,
        perceptual_emb: Dict[str, torch.Tensor],
        latent_goal: torch.Tensor,
        motion_latent: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        proj_weight = self.obs_proj[1].weight
        proj_dtype = proj_weight.dtype
        proj_device = proj_weight.device
        obs_tokens = perceptual_emb["state_images_for_videoformer"]
        obs_tokens = obs_tokens.to(device=proj_device, dtype=proj_dtype)
        obs_feat = self.obs_proj(obs_tokens.mean(dim=1))
        goal_feat = latent_goal.squeeze(1) if latent_goal.ndim == 3 else latent_goal
        goal_feat = goal_feat.to(device=proj_device, dtype=proj_dtype)
        goal_feat = self.goal_proj(goal_feat)
        voxel_feat = perceptual_emb.get("voxel_features", torch.zeros_like(obs_feat))
        voxel_feat = voxel_feat.to(device=proj_device, dtype=proj_dtype)
        voxel_feat = self.voxel_proj(voxel_feat)

        fuse_inputs = [obs_feat, goal_feat, voxel_feat]
        if motion_latent is not None:
            if self.motion_proj is None:
                raise ValueError("motion_latent was provided but motion_dim was not configured for RLBenchBridgeHead")
            motion_latent = motion_latent.to(device=proj_device, dtype=proj_dtype)
            if motion_latent.ndim == 3:
                motion_latent = motion_latent.mean(dim=1)
            fuse_inputs.append(self.motion_proj(motion_latent))

        fused = self.fuse(torch.cat(fuse_inputs, dim=-1))
        seq = fused.unsqueeze(1) + self.step_emb.unsqueeze(0)
        voxel_tokens = perceptual_emb.get("voxel_tokens", None)
        voxel_mask = perceptual_emb.get("voxel_token_padding_mask", None)
        if voxel_tokens is not None:
            voxel_tokens = voxel_tokens.to(device=proj_device, dtype=proj_dtype)
            voxel_tokens = self.voxel_token_proj(voxel_tokens)
            attended, _ = self.cross_attn(
                query=seq,
                key=voxel_tokens,
                value=voxel_tokens,
                key_padding_mask=voxel_mask,
                need_weights=False,
            )
            seq = seq + attended
        seq = self.trunk(seq)
        return {
            "trans_logits": self.trans_head(seq).view(-1, self.action_seq_len, 3, self.n_trans_bins),
            "rot_logits": self.rot_head(seq).view(-1, self.action_seq_len, 3, self.n_rot_bins),
            "gripper_logits": self.gripper_head(seq),
            "collision_logits": self.collision_head(seq),
        }
