from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn


def stable_text_features(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / max(float(norm), 1e-6)


def summarize_future_np(future: np.ndarray) -> np.ndarray:
    future = np.asarray(future, dtype=np.float32)
    n = future.shape[0]
    one = max(1, n // 3)
    early = future[:one].mean(axis=0)
    middle = future[one : max(one + 1, 2 * one)].mean(axis=0)
    late = future[max(0, 2 * one) :].mean(axis=0)
    mean = future.mean(axis=0)
    std = future.std(axis=0)
    amin = future.min(axis=0)
    amax = future.max(axis=0)
    delta = future[-1] - future[0]
    early_to_mid = middle - early
    mid_to_late = late - middle
    return np.concatenate(
        [mean, std, amin, amax, early, middle, late, delta, early_to_mid, mid_to_late],
        axis=0,
    ).astype(np.float32)


def summarize_future_torch(future: torch.Tensor) -> torch.Tensor:
    future = future.float()
    n = future.shape[0]
    one = max(1, n // 3)
    early = future[:one].mean(dim=0)
    middle = future[one : max(one + 1, 2 * one)].mean(dim=0)
    late = future[max(0, 2 * one) :].mean(dim=0)
    mean = future.mean(dim=0)
    std = future.std(dim=0, unbiased=False)
    amin = future.min(dim=0).values
    amax = future.max(dim=0).values
    delta = future[-1] - future[0]
    early_to_mid = middle - early
    mid_to_late = late - middle
    return torch.cat([mean, std, amin, amax, early, middle, late, delta, early_to_mid, mid_to_late], dim=0)


class TaskConditionedFutureEnergy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def build_energy_features_np(future: np.ndarray, task: str, subtask_index: int | float, task_dim: int) -> np.ndarray:
    future_sig = summarize_future_np(future)
    task_sig = stable_text_features(task, task_dim)
    sub_idx = np.asarray([float(subtask_index) / 5.0], dtype=np.float32)
    return np.concatenate([future_sig, task_sig, sub_idx], axis=0).astype(np.float32)


def build_energy_features_torch(
    future: torch.Tensor,
    task: str,
    subtask_index: int | float,
    task_dim: int,
    device: torch.device,
) -> torch.Tensor:
    future_sig = summarize_future_torch(future)
    task_sig_np = stable_text_features(task, task_dim)
    task_sig = torch.from_numpy(task_sig_np).to(device=device, dtype=future_sig.dtype)
    sub_idx = torch.tensor([float(subtask_index) / 5.0], device=device, dtype=future_sig.dtype)
    return torch.cat([future_sig, task_sig, sub_idx], dim=0)
