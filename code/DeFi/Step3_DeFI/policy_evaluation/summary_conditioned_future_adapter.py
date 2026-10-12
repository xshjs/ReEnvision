from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class SummaryConditionedFutureAdapter(nn.Module):
    def __init__(
        self,
        future_dim: int,
        summary_dim: int,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.future_dim = int(future_dim)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)

        cond_dim = summary_dim * 3
        self.input_proj = nn.Linear(future_dim + cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + future_dim + cond_dim),
            nn.Linear(hidden_dim + future_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + future_dim + cond_dim),
            nn.Linear(hidden_dim + future_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(
        self,
        base_future: torch.Tensor,
        source_summary: torch.Tensor,
        target_summary: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        if base_future.ndim != 3:
            raise ValueError("base_future must be [B, T, D]")
        if source_summary.ndim != 2 or target_summary.ndim != 2:
            raise ValueError("source_summary and target_summary must be [B, S]")
        if base_future.shape[-1] != self.future_dim:
            raise ValueError(f"future dim mismatch: got {base_future.shape[-1]}, expected {self.future_dim}")
        if source_summary.shape[-1] != self.summary_dim or target_summary.shape[-1] != self.summary_dim:
            raise ValueError(
                f"summary dim mismatch: got {source_summary.shape[-1]} and {target_summary.shape[-1]}, expected {self.summary_dim}"
            )

        summary_delta = target_summary - source_summary
        cond = torch.cat([source_summary, target_summary, summary_delta], dim=-1)
        cond_tokens = cond.unsqueeze(1).expand(-1, base_future.shape[1], -1)
        fused = torch.cat([base_future, cond_tokens], dim=-1)
        h0 = self.input_proj(fused)
        h = self.net(h0)
        adapter_in = torch.cat([h, base_future, cond_tokens], dim=-1)
        gate = self.gate(adapter_in)
        delta = self.delta_head(adapter_in)
        corrected = base_future + gate * delta
        return corrected, {
            "gate": gate,
            "delta": delta,
            "summary_delta": summary_delta,
        }
