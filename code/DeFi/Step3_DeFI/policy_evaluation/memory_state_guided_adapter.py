from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MemoryStateGuidedAdapter(nn.Module):
    def __init__(
        self,
        future_dim: int,
        memory_state_dim: int,
        hidden_dim: int = 1024,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.future_dim = int(future_dim)
        self.memory_state_dim = int(memory_state_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.Linear(future_dim + memory_state_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=int(gru_layers), batch_first=True)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + future_dim + memory_state_dim),
            nn.Linear(hidden_dim + future_dim + memory_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + future_dim + memory_state_dim),
            nn.Linear(hidden_dim + future_dim + memory_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(
        self,
        future_feature: torch.Tensor,
        memory_state: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        if memory_state.ndim != 2:
            raise ValueError("memory_state must be [B, D]")
        squeeze_time = False
        if future_feature.ndim == 2:
            future_tokens = future_feature.unsqueeze(1)
            squeeze_time = True
        elif future_feature.ndim == 3:
            future_tokens = future_feature
        else:
            raise ValueError("future_feature must be [B, D] or [B, T, D]")
        if future_tokens.shape[-1] != self.future_dim:
            raise ValueError(f"future dim mismatch: got {future_tokens.shape[-1]}, expected {self.future_dim}")
        if memory_state.shape[-1] != self.memory_state_dim:
            raise ValueError(f"memory_state dim mismatch: got {memory_state.shape[-1]}, expected {self.memory_state_dim}")

        memory_tokens = memory_state.unsqueeze(1).expand(-1, future_tokens.shape[1], -1)
        fused = torch.cat([future_tokens, memory_tokens], dim=-1)
        seq_in = self.input_proj(fused)
        gru_out, next_hidden = self.gru(seq_in, hidden_state)
        adapter_in = torch.cat([gru_out, future_tokens, memory_tokens], dim=-1)
        gate = self.gate(adapter_in)
        delta = self.delta_head(adapter_in)
        corrected_future = future_tokens + gate * delta
        if squeeze_time:
            corrected_future = corrected_future.squeeze(1)
            gate = gate.squeeze(1)
            delta = delta.squeeze(1)
        return corrected_future, next_hidden, {
            "gate": gate,
            "delta": delta,
            "gru_out": gru_out,
        }
