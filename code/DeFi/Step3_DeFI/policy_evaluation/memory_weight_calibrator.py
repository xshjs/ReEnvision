from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MemoryWeightCalibrator(nn.Module):
    def __init__(
        self,
        future_dim: int,
        memory_dim: int,
        max_memories: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.future_dim = int(future_dim)
        self.memory_dim = int(memory_dim)
        self.max_memories = int(max_memories)
        self.hidden_dim = int(hidden_dim)

        pair_dim = future_dim + memory_dim + 4
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(future_dim + memory_dim),
            nn.Linear(future_dim + memory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(
        self,
        future_feature: torch.Tensor,
        memory_features: torch.Tensor,
        qwen_weights: torch.Tensor,
        similarity_scores: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if future_feature.ndim != 2:
            raise ValueError("future_feature must be [B, Df]")
        if memory_features.ndim != 3:
            raise ValueError("memory_features must be [B, K, Dm]")
        if qwen_weights.ndim != 2 or similarity_scores.ndim != 2:
            raise ValueError("qwen_weights and similarity_scores must be [B, K]")
        if memory_features.shape[1] != self.max_memories:
            raise ValueError(f"expected K={self.max_memories}, got {memory_features.shape[1]}")

        batch, topk, _ = memory_features.shape
        future_rep = future_feature.unsqueeze(1).expand(batch, topk, future_feature.shape[-1])
        qwen_weights = qwen_weights.to(memory_features.dtype)
        similarity_scores = similarity_scores.to(memory_features.dtype)
        pair = torch.cat(
            [
                future_rep,
                memory_features,
                qwen_weights.unsqueeze(-1),
                similarity_scores.unsqueeze(-1),
                (qwen_weights - similarity_scores).unsqueeze(-1),
                (qwen_weights * similarity_scores).unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.score_mlp(pair).squeeze(-1)
        if memory_mask is not None:
            logits = logits.masked_fill(~memory_mask.bool(), -1e9)
        calibrated_weights = torch.softmax(logits, dim=-1)
        aggregated_memory = torch.sum(calibrated_weights.unsqueeze(-1) * memory_features, dim=1)
        context = self.context_mlp(torch.cat([future_feature, aggregated_memory], dim=-1))
        return calibrated_weights, context, {
            "raw_logits": logits,
            "aggregated_memory": aggregated_memory,
        }


def pad_weight_inputs(
    qwen_weight_map: Dict[str, float],
    similarity_map: Dict[str, float],
    memory_ids: list[str],
    max_memories: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qwen = torch.zeros((max_memories,), dtype=torch.float32)
    sims = torch.zeros((max_memories,), dtype=torch.float32)
    mask = torch.zeros((max_memories,), dtype=torch.bool)
    for idx, memory_id in enumerate(memory_ids[:max_memories]):
        qwen[idx] = float(qwen_weight_map.get(memory_id, 0.0))
        sims[idx] = float(similarity_map.get(memory_id, 0.0))
        mask[idx] = True
    if float(qwen.sum()) > 1e-12:
        qwen = qwen / qwen.sum()
    return qwen, sims, mask

