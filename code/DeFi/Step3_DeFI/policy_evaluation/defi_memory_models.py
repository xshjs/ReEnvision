import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def hashed_text_features(texts: Sequence[str], dim: int = 768) -> np.ndarray:
    feats = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        tokens = str(text).lower().split()
        if not tokens:
            continue
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            feats[row_idx, index] += sign
        norm = float(np.linalg.norm(feats[row_idx]))
        if norm > 1e-12:
            feats[row_idx] /= norm
    return feats


def load_feature(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_rollout_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if rows and "row_id" not in rows[0]:
        return {idx: dict(row, row_id=idx) for idx, row in enumerate(rows)}
    return {int(row["row_id"]): row for row in rows}


def load_labeled_rows(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def load_memory_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def summarize_feature(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32)
    if arr.ndim <= 1:
        return arr.reshape(-1).astype(np.float32)
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)


def sketch_vector(vec: np.ndarray, dim: int = 768) -> np.ndarray:
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)
    if dim <= 0:
        return flat.astype(np.float32)
    if flat.size == 0:
        return np.zeros((dim,), dtype=np.float32)
    if flat.size == dim:
        out = flat.astype(np.float32)
    else:
        out = np.zeros((dim,), dtype=np.float32)
        for idx, value in enumerate(flat):
            out[idx % dim] += float(value)
    norm = float(np.linalg.norm(out))
    if norm > 1e-12:
        out = out / norm
    return out.astype(np.float32)


def feature_signature(x: np.ndarray, dim: int = 768) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((dim,), dtype=np.float32)
    if arr.ndim > 1:
        try:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        except Exception:
            arr = arr.reshape(-1)
    else:
        arr = arr.reshape(-1)
    return sketch_vector(arr, dim=dim)


def encode_texts_t5(
    texts: Sequence[str],
    model_path: Path,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except Exception:
        return hashed_text_features(texts)

    tokenizer = T5Tokenizer.from_pretrained(str(model_path))
    encoder = T5EncoderModel.from_pretrained(str(model_path)).to(device).eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(F.normalize(pooled.float(), dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 768), dtype=np.float32)


class T5TextEmbedder:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        self.device = device
        self.backend = "t5"
        try:
            from transformers import T5EncoderModel, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(str(model_path))
            self.encoder = T5EncoderModel.from_pretrained(str(model_path)).to(device).eval()
        except Exception:
            self.tokenizer = None
            self.encoder = None
            self.backend = "hash"

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if self.backend == "hash":
            return torch.from_numpy(hashed_text_features(texts)).to(self.device)
        tokens = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled.float(), dim=-1)


def split_sequence_indices(sequence_indices: Sequence[int], val_every: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    seq = np.asarray(sequence_indices, dtype=np.int64)
    val_mask = (seq % val_every) == 0
    train_idx = np.nonzero(~val_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    return train_idx, val_idx


def repair_label(row: Dict[str, Any]) -> int:
    return 0 if str(row.get("mismatch_type", "none")) == "none" else 1


def repair_target_mask(repair_delta: np.ndarray, row: Dict[str, Any]) -> bool:
    if repair_label(row) == 0:
        return False
    return float(np.linalg.norm(repair_delta.reshape(-1))) > 1e-6


def retrieve_memory_topk(
    task: str,
    future_feature: np.ndarray,
    memory: Dict[str, np.ndarray],
    topk: int,
    exclude_row_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_ids = memory["row_ids"].astype(np.int64)
    tasks = memory["tasks"]
    future_bank = memory["future_features"].astype(np.float32)
    delta_bank = memory["repair_deltas"].astype(np.float32)
    task_mask = np.asarray([str(item) == str(task) for item in tasks], dtype=bool)
    if exclude_row_id is not None:
        task_mask &= row_ids != int(exclude_row_id)
    candidate_idx = np.nonzero(task_mask)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.nonzero(row_ids != int(exclude_row_id) if exclude_row_id is not None else np.ones_like(row_ids, dtype=bool))[0]
    if candidate_idx.size == 0:
        dim = 768
        return (
            np.zeros((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
        )

    query_signature = feature_signature(future_feature, dim=768)
    bank_signatures = np.stack(
        [feature_signature(future_bank[idx], dim=query_signature.size) for idx in candidate_idx],
        axis=0,
    ).astype(np.float32)
    scores = np.asarray([cosine(query_signature, bank_signatures[idx]) for idx in range(bank_signatures.shape[0])], dtype=np.float32)
    order = np.argsort(-scores)[: max(1, topk)]
    chosen = candidate_idx[order]
    selected_futures = np.stack(
        [feature_signature(future_bank[idx], dim=query_signature.size) for idx in chosen],
        axis=0,
    ).astype(np.float32)
    selected_deltas = np.stack(
        [feature_signature(delta_bank[idx], dim=query_signature.size) for idx in chosen],
        axis=0,
    ).astype(np.float32)
    return (
        selected_futures.mean(axis=0).astype(np.float32),
        selected_deltas.mean(axis=0).astype(np.float32),
        row_ids[chosen].astype(np.int64),
        scores[order].astype(np.float32),
    )


def build_gate_input(
    current_feature: np.ndarray,
    future_feature: np.ndarray,
    memory_future: np.ndarray,
    memory_delta: np.ndarray,
    reflection_embedding: np.ndarray,
    memory_scores: np.ndarray,
) -> np.ndarray:
    score_stats = np.zeros((2,), dtype=np.float32)
    if memory_scores.size > 0:
        score_stats[0] = float(memory_scores[0])
        score_stats[1] = float(memory_scores.mean())
    return np.concatenate(
        [
            current_feature.astype(np.float32).reshape(-1),
            future_feature.astype(np.float32).reshape(-1),
            memory_future.astype(np.float32).reshape(-1),
            memory_delta.astype(np.float32).reshape(-1),
            reflection_embedding.astype(np.float32).reshape(-1),
            score_stats,
        ],
        axis=0,
    ).astype(np.float32)


def build_repair_input(
    future_feature: np.ndarray,
    memory_future: np.ndarray,
    memory_delta: np.ndarray,
    reflection_embedding: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            future_feature.astype(np.float32).reshape(-1),
            memory_future.astype(np.float32).reshape(-1),
            memory_delta.astype(np.float32).reshape(-1),
            reflection_embedding.astype(np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


class GateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RepairAdapter(nn.Module):
    def __init__(self, input_dim: int, future_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
