from __future__ import annotations

import torch
from torch import nn


class RenderedViewTokenAdapter(nn.Module):
    """Convert RLBench renderer views into token grids for GIDM and action heads.

    Input: rendered views [B, V, C, H, W]
    Output:
      - gidm_tokens: [B, 2, 256, 768]
      - head_tokens: [B, 256, 384]
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        gidm_dim: int = 768,
        head_dim: int = 384,
        num_views_for_gidm: int = 2,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_views_for_gidm = num_views_for_gidm
        patch_vec_dim = 3 * patch_size * patch_size
        self.gidm_proj = nn.Linear(patch_vec_dim, gidm_dim)
        self.head_proj = nn.Linear(patch_vec_dim, head_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, P]
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, self.num_patches, c * p * p)
        return x

    def forward(self, rendered_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        # rendered_rgb: [B, V, 3, H, W]
        proj_weight = self.gidm_proj.weight
        proj_dtype = proj_weight.dtype
        proj_device = proj_weight.device
        rendered_rgb = rendered_rgb.to(device=proj_device, dtype=proj_dtype)
        b, v, c, h, w = rendered_rgb.shape
        assert c == 3
        use_v = min(v, self.num_views_for_gidm)
        gidm_views = rendered_rgb[:, :use_v]
        gidm_tokens = []
        for i in range(use_v):
            gidm_tokens.append(self.gidm_proj(self._patchify(gidm_views[:, i])).unsqueeze(1))
        if use_v < self.num_views_for_gidm:
            pad = torch.zeros(
                b,
                self.num_views_for_gidm - use_v,
                self.num_patches,
                self.gidm_proj.out_features,
                device=rendered_rgb.device,
                dtype=rendered_rgb.dtype,
            )
            gidm_tokens = torch.cat(gidm_tokens + [pad], dim=1) if gidm_tokens else pad
        else:
            gidm_tokens = torch.cat(gidm_tokens, dim=1)

        head_tokens = self.head_proj(self._patchify(rendered_rgb[:, 0]))
        return {
            "gidm_tokens": gidm_tokens,
            "head_tokens": head_tokens,
        }
