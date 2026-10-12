from typing import Dict

import torch
from torch import nn


class RLBenchWaypointHead(nn.Module):
    """Predict absolute RLBench waypoint actions: xyz + quaternion + gripper."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hidden_dim: int,
        action_seq_len: int,
        num_stage_classes: int = 4,
    ):
        super().__init__()
        self.action_seq_len = action_seq_len
        self.num_stage_classes = num_stage_classes
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
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.step_emb = nn.Parameter(torch.randn(action_seq_len, hidden_dim) * 0.02)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pos_head = nn.Linear(hidden_dim, 3)
        self.quat_head = nn.Linear(hidden_dim, 4)
        self.gripper_head = nn.Linear(hidden_dim, 1)
        self.stage_head = nn.Linear(hidden_dim, num_stage_classes)

    def forward(self, perceptual_emb: Dict[str, torch.Tensor], latent_goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        obs_tokens = perceptual_emb["state_images_for_videoformer"]
        obs_feat = self.obs_proj(obs_tokens.mean(dim=1))

        goal_feat = latent_goal.squeeze(1) if latent_goal.ndim == 3 else latent_goal
        goal_feat = self.goal_proj(goal_feat)

        base = self.fuse(torch.cat([obs_feat, goal_feat], dim=-1))
        seq = base.unsqueeze(1) + self.step_emb.unsqueeze(0)
        seq = self.trunk(seq)

        quat = self.quat_head(seq)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "pos": self.pos_head(seq),
            "quat": quat,
            "gripper_logits": self.gripper_head(seq).squeeze(-1),
            "stage_logits": self.stage_head(seq),
        }


def quaternion_geodesic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dot = torch.sum(pred * target, dim=-1).abs().clamp(max=1.0)
    return (1.0 - dot).mean()
