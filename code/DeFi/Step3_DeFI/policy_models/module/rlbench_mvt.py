from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class RLBenchMVT(nn.Module):
    """Lightweight MVT-style 3D encoder for RLBench RGBD observations.

    It uses sparse voxel tokens plus a transformer aggregator. This is a
    compact DeFi-native approximation of the Bridge/PerAct MVT path.
    """

    def __init__(
        self,
        scene_bounds: list[float],
        voxel_size: int = 100,
        out_dim: int = 512,
        token_dim: int = 256,
        max_tokens: int = 256,
        depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.register_buffer("scene_mins", torch.tensor(scene_bounds[:3], dtype=torch.float32))
        self.register_buffer("scene_maxs", torch.tensor(scene_bounds[3:], dtype=torch.float32))
        self.voxel_size = voxel_size
        self.out_dim = out_dim
        self.token_dim = token_dim
        self.max_tokens = max_tokens

        self.feature_proj = nn.Sequential(
            nn.Linear(7, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.x_embed = nn.Embedding(voxel_size, token_dim)
        self.y_embed = nn.Embedding(voxel_size, token_dim)
        self.z_embed = nn.Embedding(voxel_size, token_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=heads,
                dim_feedforward=token_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.token_out_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, out_dim),
        )

    def _build_sparse_tokens(self, points: torch.Tensor, colors: torch.Tensor):
        device = points.device
        mins = self.scene_mins.to(device)
        maxs = self.scene_maxs.to(device)

        valid = torch.isfinite(points).all(dim=-1)
        points = points[valid]
        colors = colors[valid]
        if points.numel() == 0:
            return None

        scaled = (points - mins) / (maxs - mins).clamp_min(1e-6)
        valid = ((scaled >= 0.0) & (scaled <= 1.0)).all(dim=-1)
        points = points[valid]
        colors = colors[valid]
        scaled = scaled[valid]
        if points.numel() == 0:
            return None

        idx = torch.clamp((scaled * (self.voxel_size - 1)).round().long(), 0, self.voxel_size - 1)
        lin = idx[:, 0] * self.voxel_size * self.voxel_size + idx[:, 1] * self.voxel_size + idx[:, 2]
        unique_lin, inverse, counts = torch.unique(lin, sorted=False, return_inverse=True, return_counts=True)

        num_voxels = unique_lin.shape[0]
        centers = torch.zeros((num_voxels, 3), dtype=torch.float32, device=device)
        rgb_sum = torch.zeros((num_voxels, 3), dtype=torch.float32, device=device)
        centers.scatter_add_(0, inverse[:, None].expand(-1, 3), points)
        rgb_sum.scatter_add_(0, inverse[:, None].expand(-1, 3), colors)
        counts_f = counts.float().unsqueeze(-1).clamp_min(1.0)
        centers = centers / counts_f
        rgb_avg = rgb_sum / counts_f

        vox_idx = torch.stack(
            [
                unique_lin // (self.voxel_size * self.voxel_size),
                (unique_lin // self.voxel_size) % self.voxel_size,
                unique_lin % self.voxel_size,
            ],
            dim=-1,
        ).long()

        if num_voxels > self.max_tokens:
            keep = torch.topk(counts, k=self.max_tokens, largest=True).indices
            vox_idx = vox_idx[keep]
            centers = centers[keep]
            rgb_avg = rgb_avg[keep]
            counts = counts[keep]

        occupancy = (counts.float() / counts.float().max().clamp_min(1.0)).unsqueeze(-1)
        features = torch.cat([centers, rgb_avg, occupancy], dim=-1)
        return vox_idx, features

    def forward(
        self,
        point_cloud_obs: Dict[str, torch.Tensor],
        rgb_obs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not point_cloud_obs:
            raise ValueError("point_cloud_obs is required for RLBenchMVT")

        key_to_rgb = {
            "pcd_front": "rgb_static",
            "pcd_wrist": "rgb_gripper",
            "pcd_left_shoulder": "rgb_left_shoulder",
            "pcd_right_shoulder": "rgb_right_shoulder",
        }
        sample_key = next(iter(point_cloud_obs.keys()))
        batch_size = point_cloud_obs[sample_key].shape[0]
        device = point_cloud_obs[sample_key].device

        all_tokens = []
        token_counts = []
        for b in range(batch_size):
            points_all = []
            colors_all = []
            for pcd_key, pcd_tensor in point_cloud_obs.items():
                rgb_key = key_to_rgb.get(pcd_key)
                if rgb_key is None or rgb_key not in rgb_obs:
                    continue
                pts = pcd_tensor[b]
                if pts.ndim == 4:
                    pts = pts[0]
                rgb = rgb_obs[rgb_key][b]
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if pts.shape[-2:] != rgb.shape[-2:]:
                    pts = torch.nn.functional.interpolate(pts.unsqueeze(0), size=rgb.shape[-2:], mode="nearest").squeeze(0)
                pts = pts.permute(1, 2, 0).reshape(-1, 3)
                rgb = ((rgb.permute(1, 2, 0) + 1.0) * 0.5).reshape(-1, 3).clamp(0.0, 1.0)
                n_pts = min(pts.shape[0], rgb.shape[0])
                pts = pts[:n_pts]
                rgb = rgb[:n_pts]
                points_all.append(pts)
                colors_all.append(rgb)
            if not points_all:
                raise ValueError("No matching RGB/point-cloud camera pairs found.")

            sparse = self._build_sparse_tokens(torch.cat(points_all, dim=0), torch.cat(colors_all, dim=0))
            if sparse is None:
                vox_idx = torch.zeros((1, 3), dtype=torch.long, device=device)
                feat = torch.zeros((1, 7), dtype=torch.float32, device=device)
            else:
                vox_idx, feat = sparse
            token = self.feature_proj(feat)
            token = token + self.x_embed(vox_idx[:, 0]) + self.y_embed(vox_idx[:, 1]) + self.z_embed(vox_idx[:, 2])
            all_tokens.append(token)
            token_counts.append(token.shape[0])

        max_len = max(token_counts)
        padded = torch.zeros((batch_size, max_len, self.token_dim), dtype=torch.float32, device=device)
        padding_mask = torch.ones((batch_size, max_len + 1), dtype=torch.bool, device=device)
        for b, token in enumerate(all_tokens):
            padded[b, : token.shape[0]] = token
            padding_mask[b, 1 : token.shape[0] + 1] = False

        cls = self.cls_token.expand(batch_size, -1, -1).to(device)
        seq = torch.cat([cls, padded], dim=1)
        encoded = self.transformer(seq, src_key_padding_mask=padding_mask)
        return {
            "global_feature": self.out_proj(encoded[:, 0]),
            "token_features": self.token_out_proj(encoded[:, 1:]),
            "token_padding_mask": padding_mask[:, 1:],
        }
