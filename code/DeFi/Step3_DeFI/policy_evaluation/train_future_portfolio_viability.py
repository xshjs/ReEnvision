import argparse
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if "distutils.spawn" not in sys.modules:
    distutils_mod = sys.modules.setdefault("distutils", types.ModuleType("distutils"))
    spawn_mod = types.ModuleType("distutils.spawn")
    spawn_mod.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_mod
    setattr(distutils_mod, "spawn", spawn_mod)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from future_reflection_correction_mlp import encode_reflections_qwen, encode_reflections_t5
from train_universal_moe_oracle_router import (
    BASE_FEATURE_LAYOUT,
    filter_rows_by_sequence,
    load_rows,
    probability_stats,
    split_indices,
)


class PortfolioViabilityModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_experts: int,
        context_dim: int = 0,
        execution_dim: int = 0,
        num_failure_types: int = 0,
        num_correction_directions: int = 0,
    ) -> None:
        super().__init__()
        self.expert_embed = nn.Embedding(num_experts, 16)
        self.backbone = nn.Sequential(
            nn.LayerNorm(feature_dim + 16),
            nn.Linear(feature_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.viability = nn.Linear(hidden_dim, 1)
        self.steps = nn.Linear(hidden_dim, 1)
        self.utility = nn.Linear(hidden_dim, 1)
        self.utility_correction = nn.Linear(hidden_dim * 2, 1) if execution_dim > 0 else None
        self.recovery = nn.Linear(hidden_dim, 1)
        self.context_dim = int(context_dim)
        self.execution_dim = int(execution_dim)
        if self.context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            self.context_proj = None
        if self.execution_dim > 0:
            self.execution_proj = nn.Sequential(
                nn.LayerNorm(execution_dim),
                nn.Linear(execution_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.execution_readout = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.execution_proj = None
            self.execution_readout = None
        aux_input_dim = hidden_dim + (hidden_dim if self.context_proj is not None else 0)
        self.failure_head = nn.Linear(aux_input_dim, num_failure_types) if num_failure_types > 0 else None
        self.correction_head = (
            nn.Linear(aux_input_dim, num_correction_directions) if num_correction_directions > 0 else None
        )
        self.recoverability_head = nn.Linear(aux_input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        execution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, num_experts, _ = x.shape
        expert_ids = torch.arange(num_experts, device=x.device).unsqueeze(0).expand(batch, -1)
        expert_embed = self.expert_embed(expert_ids)
        base_hidden = self.backbone(torch.cat([x, expert_embed], dim=-1))
        hidden = base_hidden
        execution_repr = None
        utility_base = self.utility(base_hidden).squeeze(-1)
        utility_delta = torch.zeros_like(utility_base)
        if self.execution_proj is not None and execution is not None:
            execution_repr = self.execution_proj(execution)
            hidden = hidden + self.execution_readout(execution_repr).unsqueeze(1)
            if self.utility_correction is not None:
                exec_expand = execution_repr.unsqueeze(1).expand(-1, num_experts, -1)
                utility_delta = self.utility_correction(torch.cat([base_hidden, exec_expand], dim=-1)).squeeze(-1)
        viability_logits = self.viability(hidden).squeeze(-1)
        steps_norm = torch.sigmoid(self.steps(hidden).squeeze(-1))
        utility_logits = utility_base
        sample_hidden = hidden.mean(dim=1)
        if self.context_proj is not None and context is not None:
            context_hidden = self.context_proj(context)
            aux_hidden = torch.cat([sample_hidden, context_hidden], dim=-1)
        else:
            aux_hidden = sample_hidden
        failure_logits = self.failure_head(aux_hidden) if self.failure_head is not None else None
        correction_logits = self.correction_head(aux_hidden) if self.correction_head is not None else None
        recoverability_logit = self.recoverability_head(aux_hidden).squeeze(-1)
        recovery_logits = self.recovery(hidden).squeeze(-1)
        return (
            viability_logits,
            steps_norm,
            utility_logits,
            utility_delta,
            failure_logits,
            correction_logits,
            recoverability_logit,
            recovery_logits,
            execution_repr,
        )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_reflection_texts(path: Path) -> Dict[Tuple[int, int], str]:
    rows = {}
    for row in iter_jsonl(path):
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        text = row.get("reflection_text")
        if not text:
            fields = [
                "progress_status",
                "failure_type",
                "prefer_next",
                "avoid_next",
                "correction_direction",
                "memory_query",
            ]
            text = " | ".join(f"{field}: {row.get(field, 'uncertain')}" for field in fields)
        rows[key] = str(text)
    return rows


def load_reflection_rows(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    rows = {}
    for row in iter_jsonl(path):
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        rows[key] = row
    return rows


def parse_expert_token(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text.startswith("expert "):
        return None
    try:
        return int(text.split()[1])
    except Exception:
        return None


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros((size,), dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def normalize_rows(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cos01":
        return np.clip((values + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    if mode == "minmax":
        lo = values.min(axis=1, keepdims=True)
        hi = values.max(axis=1, keepdims=True)
        return ((values - lo) / np.clip(hi - lo, 1e-6, None)).astype(np.float32)
    if mode == "softmax":
        z = values - values.max(axis=1, keepdims=True)
        exp = np.exp(z / 0.001)
        return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)
    raise ValueError(f"unknown gt cosine normalization: {mode}")


def load_arrays(rows: List[Tuple[Path, Dict[str, object]]]) -> Dict[str, Any]:
    x_rows = []
    success_rows = []
    steps_rows = []
    value_rows = []
    chosen_rows = []
    oracle_rows = []
    gt_cos_rows = []
    exec_rows = []
    group_rows = []
    meta_rows = []
    for npz_path, row in rows:
        with np.load(npz_path) as data:
            obs = data["obs_embedding"].astype(np.float32)
            futures = data["expert_futures"].astype(np.float32)
            gt_future = data["next_task_obs_embedding"].astype(np.float32)
            future_delta = futures - obs[None, :]
            router_logits = data["router_logits"].astype(np.float32).reshape(-1, 1)
            router_scores = data["router_scores"].astype(np.float32).reshape(-1, 1)
            contradiction = data["expert_contradiction"].astype(np.float32).reshape(-1, 1)
            feature = np.concatenate(
                [
                    np.repeat(obs[None, :], futures.shape[0], axis=0),
                    futures,
                    future_delta,
                    router_logits,
                    router_scores,
                    contradiction,
                ],
                axis=1,
            )
            future_norm = futures / np.clip(np.linalg.norm(futures, axis=1, keepdims=True), 1e-12, None)
            gt_norm = gt_future / np.clip(np.linalg.norm(gt_future), 1e-12, None)
            gt_cos = np.sum(future_norm * gt_norm[None, :], axis=1)

            global_seq = int(row.get("_global_sequence_index", row.get("sequence_index", -1)))
            subtask_idx = int(row.get("subtask_index", -1))
            x_rows.append(feature)
            success_rows.append(data["expert_success"].astype(np.float32))
            steps_rows.append(data["expert_steps"].astype(np.float32))
            value_rows.append(data["expert_values"].astype(np.float32))
            exec_rows.append(data["expert_execution_features"].astype(np.float32))
            chosen_rows.append(int(data["chosen_expert"][0]))
            oracle_rows.append(int(data["oracle_expert"][0]))
            gt_cos_rows.append(gt_cos.astype(np.float32))
            group_rows.append(f"{global_seq}:{subtask_idx}")
            meta_rows.append(
                {
                    "global_sequence_index": global_seq,
                    "subtask_index": subtask_idx,
                    "task": str(row.get("task")),
                }
            )
    return {
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "success": np.stack(success_rows, axis=0).astype(np.float32),
        "steps": np.stack(steps_rows, axis=0).astype(np.float32),
        "values": np.stack(value_rows, axis=0).astype(np.float32),
        "chosen": np.asarray(chosen_rows, dtype=np.int64),
        "oracle": np.asarray(oracle_rows, dtype=np.int64),
        "gt_cosine": np.stack(gt_cos_rows, axis=0).astype(np.float32),
        "expert_execution_features": np.stack(exec_rows, axis=0).astype(np.float32),
        "groups": np.asarray(group_rows),
        "meta": meta_rows,
    }


def previous_outcome_features(data: Dict[str, Any], ep_len: float) -> np.ndarray:
    key_to_idx = {
        (int(row["global_sequence_index"]), int(row["subtask_index"])): idx
        for idx, row in enumerate(data["meta"])
    }
    num_rows, num_experts = data["success"].shape
    rows = []
    for row in data["meta"]:
        prev_idx = key_to_idx.get((int(row["global_sequence_index"]), int(row["subtask_index"]) - 1))
        if prev_idx is None:
            feat = np.zeros((22,), dtype=np.float32)
        else:
            chosen = int(data["chosen"][prev_idx])
            oracle = int(data["oracle"][prev_idx])
            prev_success = data["success"][prev_idx]
            prev_steps = data["steps"][prev_idx] / float(ep_len)
            chosen_success = float(prev_success[chosen]) if 0 <= chosen < num_experts else 0.0
            chosen_steps = float(prev_steps[chosen]) if 0 <= chosen < num_experts else 1.0
            feat = np.concatenate(
                [
                    np.asarray([1.0, chosen_success, chosen_steps], dtype=np.float32),
                    one_hot(chosen, num_experts),
                    one_hot(oracle, num_experts),
                    prev_success.astype(np.float32),
                    prev_steps.astype(np.float32),
                    np.asarray(
                        [
                            float(chosen == oracle),
                            float(prev_success.sum() / max(num_experts, 1)),
                            float(data["values"][prev_idx].max()),
                        ],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        rows.append(feat)
    return np.stack(rows, axis=0).astype(np.float32)


def execution_context_features(data: Dict[str, Any], ep_len: float) -> np.ndarray:
    prev = previous_outcome_features(data, ep_len)
    exec_feat = data["expert_execution_features"].astype(np.float32)
    exec_mean = exec_feat.mean(axis=1)
    exec_max = exec_feat.max(axis=1)
    success = data["success"].astype(np.float32)
    steps = (data["steps"].astype(np.float32) / float(ep_len))
    values = data["values"].astype(np.float32)
    summary = np.concatenate(
        [
            success,
            steps,
            values,
            success.mean(axis=1, keepdims=True),
            steps.mean(axis=1, keepdims=True),
            values.mean(axis=1, keepdims=True),
            success.max(axis=1, keepdims=True),
            values.max(axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([prev, exec_mean, exec_max, summary], axis=1).astype(np.float32)


def build_vocab(values: Sequence[str], min_count: int = 1, default_token: str = "other") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    vocab = {default_token: 0}
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= int(min_count) and value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def auxiliary_targets(
    data: Dict[str, Any],
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
    reflection_map = load_reflection_rows(path)
    failure_values: List[str] = []
    correction_values: List[str] = []
    for row in data["meta"]:
        ref = reflection_map.get((int(row["global_sequence_index"]), int(row["subtask_index"])), {})
        failure_values.append(str(ref.get("failure_type", "none")))
        correction_values.append(str(ref.get("correction_direction", "maintain")))
    failure_vocab = build_vocab(failure_values, min_count=1, default_token="other")
    correction_vocab = build_vocab(correction_values, min_count=2, default_token="other")

    num_rows = len(data["meta"])
    failure_idx = np.zeros((num_rows,), dtype=np.int64)
    correction_idx = np.zeros((num_rows,), dtype=np.int64)
    recoverability = np.zeros((num_rows,), dtype=np.float32)
    for idx, row in enumerate(data["meta"]):
        ref = reflection_map.get((int(row["global_sequence_index"]), int(row["subtask_index"])), {})
        failure_key = str(ref.get("failure_type", "none"))
        correction_key = str(ref.get("correction_direction", "maintain"))
        failure_idx[idx] = failure_vocab.get(failure_key, 0)
        correction_idx[idx] = correction_vocab.get(correction_key, 0)
        chosen = int(data["chosen"][idx])
        router_success = float(data["success"][idx, chosen]) if 0 <= chosen < data["success"].shape[1] else 0.0
        oracle_success = float(data["success"][idx].max())
        recoverability[idx] = 1.0 if router_success < 0.5 and oracle_success > 0.5 else 0.0
    return failure_idx, correction_idx, recoverability, {
        "failure_type_vocab": failure_vocab,
        "correction_direction_vocab": correction_vocab,
    }


def reflection_features(
    data: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    if args.conditioning == "none":
        return np.zeros((len(data["x"]), 0), dtype=np.float32), []
    if args.conditioning == "outcome":
        return previous_outcome_features(data, args.ep_len), [
            "prev_has_outcome",
            "prev_chosen_success",
            "prev_chosen_steps_norm",
            "prev_chosen_expert_onehot",
            "prev_oracle_expert_onehot",
            "prev_expert_success",
            "prev_expert_steps_norm",
            "prev_chosen_is_oracle",
            "prev_success_count_norm",
            "prev_best_value",
        ]
    if args.conditioning != "qwen":
        raise ValueError(f"unknown conditioning: {args.conditioning}")
    if args.qwen_reflections is None:
        raise ValueError("--qwen-reflections is required for --conditioning qwen")
    reflection_map = load_reflection_texts(args.qwen_reflections)
    texts = []
    missing = 0
    for row in data["meta"]:
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        text = reflection_map.get(key)
        if text is None:
            missing += 1
            text = (
                f"progress_status: uncertain | failure_type: uncertain | "
                f"prefer_next: choose viable future for {row['task']} | "
                "avoid_next: avoid failed futures | correction_direction: use success and GT consistency | "
                f"memory_query: {row['task']}"
            )
        texts.append(text)
    if missing:
        print(json.dumps({"qwen_reflection_missing": int(missing), "total": len(texts)}), flush=True)
    if args.reflection_encoder_type == "t5":
        encoded = encode_reflections_t5(texts, args.t5_path, device, args.text_batch_size)
    elif args.reflection_encoder_type == "qwen":
        encoded = encode_reflections_qwen(texts, args.qwen_path, device, args.text_batch_size)
    else:
        raise ValueError(f"unknown reflection encoder type: {args.reflection_encoder_type}")
    return encoded.astype(np.float32), [f"qwen_reflection_{args.reflection_encoder_type}_embedding"]


def apply_conditioning(
    data: Dict[str, Any],
    cond: np.ndarray,
) -> Dict[str, Any]:
    if cond.shape[1] == 0:
        return data
    repeated = np.repeat(cond[:, None, :], data["x"].shape[1], axis=1)
    out = dict(data)
    out["x"] = np.concatenate([data["x"], repeated], axis=-1).astype(np.float32)
    return out


def make_targets(data: Dict[str, Any], alpha_success: float, gt_cos_norm: str) -> np.ndarray:
    cos_target = normalize_rows(data["gt_cosine"], gt_cos_norm)
    target = float(alpha_success) * data["success"] + (1.0 - float(alpha_success)) * cos_target
    return np.clip(target, 0.0, 1.0).astype(np.float32)


def utility_scores(data: Dict[str, Any], target: str, step_lambda: float, ep_len: float) -> np.ndarray:
    if target == "success_steps":
        utility = data["success"].astype(np.float32) - float(step_lambda) * (data["steps"].astype(np.float32) / float(ep_len))
    elif target == "value":
        utility = data["values"].astype(np.float32)
    else:
        raise ValueError(f"unknown utility target: {target}")
    return utility.astype(np.float32)


def utility_targets(data: Dict[str, Any], target: str, step_lambda: float, ep_len: float, temperature: float) -> np.ndarray:
    utility = utility_scores(data, target, step_lambda, ep_len)
    utility = utility / float(temperature)
    utility = utility - utility.max(axis=1, keepdims=True)
    probs = np.exp(utility)
    return (probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def reflection_teacher_targets(
    data: Dict[str, Any],
    path: Path,
    alpha: float,
    require_explicit_expert: bool,
) -> np.ndarray:
    reflection_map = load_reflection_rows(path)
    num_rows, num_experts = data["success"].shape
    targets = np.full((num_rows, num_experts), 1.0 / max(num_experts, 1), dtype=np.float32)
    missing = 0
    usable = 0
    filtered = 0
    for idx, row in enumerate(data["meta"]):
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        ref = reflection_map.get(key)
        if ref is None:
            missing += 1
            continue
        prefer = parse_expert_token(ref.get("prefer_next"))
        avoid = parse_expert_token(ref.get("avoid_next"))
        if require_explicit_expert and prefer is None and avoid is None:
            filtered += 1
            continue
        teacher = np.zeros((num_experts,), dtype=np.float32)
        if prefer is not None and 0 <= prefer < num_experts:
            teacher[prefer] = 1.0
            usable += 1
        else:
            if require_explicit_expert:
                filtered += 1
                continue
            teacher[:] = 1.0 / max(num_experts, 1)
        if avoid is not None and 0 <= avoid < num_experts:
            teacher[avoid] = 0.0
            if teacher.sum() <= 0:
                teacher[:] = 1.0 / max(num_experts, 1)
            else:
                teacher /= teacher.sum()
            usable += 1
        targets[idx] = teacher
    print(
        json.dumps(
            {
                "teacher_reflection_missing": int(missing),
                "teacher_reflection_usable": int(usable),
                "teacher_reflection_filtered": int(filtered),
                "teacher_alpha": float(alpha),
                "teacher_require_explicit_expert": bool(require_explicit_expert),
            }
        ),
        flush=True,
    )
    return targets.astype(np.float32)


def portfolio_score(
    viability_logits: np.ndarray,
    pred_steps_norm: np.ndarray,
    utility_logits: np.ndarray,
    recovery_logits: np.ndarray | None,
    score_step_lambda: float,
    utility_score_weight: float,
    viability_score_weight: float,
    recovery_score_weight: float,
) -> np.ndarray:
    viability = 1.0 / (1.0 + np.exp(-viability_logits))
    score = (
        float(utility_score_weight) * utility_logits
        + float(viability_score_weight) * viability
        - float(score_step_lambda) * pred_steps_norm
    )
    if recovery_logits is not None:
        recovery = 1.0 / (1.0 + np.exp(-recovery_logits))
        score = score + float(recovery_score_weight) * recovery
    return score


def evaluate(
    viability_logits: np.ndarray,
    pred_steps_norm: np.ndarray,
    utility_logits: np.ndarray,
    recovery_logits: np.ndarray | None,
    data: Dict[str, Any],
    idx: np.ndarray,
    labels: np.ndarray,
    score_step_lambda: float,
    utility_score_weight: float,
    viability_score_weight: float,
    recovery_score_weight: float,
    ep_len: float,
) -> Dict[str, Any]:
    scores = portfolio_score(
        viability_logits[idx],
        pred_steps_norm[idx],
        utility_logits[idx],
        recovery_logits[idx] if recovery_logits is not None else None,
        score_step_lambda,
        utility_score_weight,
        viability_score_weight,
        recovery_score_weight,
    )
    pred = scores.argmax(axis=1)
    chosen = data["chosen"][idx]
    success = data["success"][idx]
    values = data["values"][idx]
    gt_cos = data["gt_cosine"][idx]
    pred_success = success[np.arange(len(idx)), pred]
    router_success = success[np.arange(len(idx)), chosen]
    pred_value = values[np.arange(len(idx)), pred]
    router_value = values[np.arange(len(idx)), chosen]
    pred_gt_cos = gt_cos[np.arange(len(idx)), pred]
    router_gt_cos = gt_cos[np.arange(len(idx)), chosen]
    oracle_success = success.max(axis=1)
    recoverable = (router_success < 0.5) & (oracle_success > 0.5)
    recovered = recoverable & (pred_success > 0.5)
    probs = 1.0 / (1.0 + np.exp(-viability_logits[idx]))
    bce = -(
        labels[idx] * np.log(np.clip(probs, 1e-7, 1.0))
        + (1.0 - labels[idx]) * np.log(np.clip(1.0 - probs, 1e-7, 1.0))
    )
    success_mask = success > 0.5
    step_target = data["steps"][idx] / float(ep_len)
    step_mse = (pred_steps_norm[idx] - step_target) ** 2
    if success_mask.any():
        step_mse_value = float(step_mse[success_mask].mean())
    else:
        step_mse_value = 0.0
    return {
        "num": int(len(idx)),
        "router_success": float(router_success.mean()),
        "portfolio_success": float(pred_success.mean()),
        "oracle_success": float(oracle_success.mean()),
        "router_value": float(router_value.mean()),
        "portfolio_value": float(pred_value.mean()),
        "router_gt_cosine": float(router_gt_cos.mean()),
        "portfolio_gt_cosine": float(pred_gt_cos.mean()),
        "bce": float(bce.mean()),
        "step_mse_success": step_mse_value,
        "target_mean": float(labels[idx].mean()),
        "positive_rate": float((labels[idx] > 0.5).mean()),
        "recoverable_failures": int(recoverable.sum()),
        "recovered_by_portfolio": int(recovered.sum()),
        "recover_rate": float(recovered.sum() / max(recoverable.sum(), 1)),
        "utility_logit_mean": float(utility_logits[idx].mean()),
        "utility_logit_std": float(utility_logits[idx].std()),
        **probability_stats(scores, np.arange(len(idx), dtype=np.int64)),
    }


def per_expert_recovery_targets(data: Dict[str, Any]) -> np.ndarray:
    chosen = data["chosen"]
    success = data["success"].astype(np.float32)
    num_rows, num_experts = success.shape
    out = np.zeros((num_rows, num_experts), dtype=np.float32)
    for idx in range(num_rows):
        chosen_idx = int(chosen[idx])
        chosen_success = float(success[idx, chosen_idx]) if 0 <= chosen_idx < num_experts else 0.0
        if chosen_success < 0.5:
            out[idx] = success[idx]
    return out


def aux_accuracy(logits: np.ndarray, target: np.ndarray) -> float:
    if logits.size == 0:
        return 0.0
    pred = logits.argmax(axis=1)
    return float((pred == target).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-positive future portfolio viability selector.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditioning", choices=["none", "outcome", "qwen"], default="none")
    parser.add_argument("--qwen-reflections", type=Path, default=None)
    parser.add_argument("--teacher-reflections", type=Path, default=None)
    parser.add_argument("--teacher-preference-alpha", type=float, default=0.0)
    parser.add_argument("--teacher-require-explicit-expert", action="store_true")
    parser.add_argument("--use-trajectory-context", action="store_true")
    parser.add_argument("--aux-reflections", type=Path, default=None)
    parser.add_argument("--failure-weight", type=float, default=0.0)
    parser.add_argument("--correction-weight", type=float, default=0.0)
    parser.add_argument("--recoverability-weight", type=float, default=0.0)
    parser.add_argument("--coupled-recovery-weight", type=float, default=0.0)
    parser.add_argument("--recovery-score-weight", type=float, default=0.0)
    parser.add_argument("--use-execution-augmentation", action="store_true")
    parser.add_argument("--use-execution-residual-correction", action="store_true")
    parser.add_argument("--execution-correction-scale", type=float, default=0.0)
    parser.add_argument("--execution-repr-weight", type=float, default=0.0)
    parser.add_argument("--reflection-encoder-type", choices=["t5", "qwen"], default="t5")
    parser.add_argument("--t5-path", type=Path, default=Path("ckpts/t5_base"))
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/data/manipulation/llm_models/Qwen3-8B"))
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-success", type=float, default=0.7)
    parser.add_argument("--gt-cos-norm", choices=["minmax", "cos01", "softmax"], default="cos01")
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--step-weight", type=float, default=0.1)
    parser.add_argument("--viability-weight", type=float, default=0.25)
    parser.add_argument("--utility-target", choices=["success_steps", "value"], default="value")
    parser.add_argument("--utility-score-weight", type=float, default=1.0)
    parser.add_argument("--viability-score-weight", type=float, default=0.1)
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    parser.add_argument("--score-step-lambda", type=float, default=0.05)
    parser.add_argument("--ep-len", type=float, default=160.0)
    parser.add_argument("--exclude-sequence-start", type=int, default=-1)
    parser.add_argument("--exclude-sequence-end", type=int, default=-1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.roots)
    rows_before_filter = len(rows)
    rows = filter_rows_by_sequence(rows, args.exclude_sequence_start, args.exclude_sequence_end)
    if not rows:
        raise ValueError("No Universal MoE rows found.")
    data = load_arrays(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_np = previous_outcome_features(data, args.ep_len) if args.use_trajectory_context else np.zeros((len(data["x"]), 0), dtype=np.float32)
    execution_np = execution_context_features(data, args.ep_len) if args.use_execution_augmentation else np.zeros((len(data["x"]), 0), dtype=np.float32)
    cond, cond_layout = reflection_features(data, args, device)
    data = apply_conditioning(data, cond)
    labels_np = make_targets(data, args.alpha_success, args.gt_cos_norm)
    rank_targets_np = utility_targets(data, args.utility_target, args.score_step_lambda, args.ep_len, args.rank_temperature)
    recovery_targets_np = per_expert_recovery_targets(data)
    teacher_target_stats = None
    if args.teacher_reflections is not None and args.teacher_preference_alpha > 0.0:
        teacher_rank_targets = reflection_teacher_targets(
            data,
            args.teacher_reflections,
            args.teacher_preference_alpha,
            args.teacher_require_explicit_expert,
        )
        rank_targets_np = (
            (1.0 - float(args.teacher_preference_alpha)) * rank_targets_np
            + float(args.teacher_preference_alpha) * teacher_rank_targets
        ).astype(np.float32)
        teacher_target_stats = {
            "teacher_reflections": str(args.teacher_reflections),
            "teacher_preference_alpha": float(args.teacher_preference_alpha),
            "teacher_require_explicit_expert": bool(args.teacher_require_explicit_expert),
        }
    aux_stats = None
    failure_targets_np = np.zeros((len(data["x"]),), dtype=np.int64)
    correction_targets_np = np.zeros((len(data["x"]),), dtype=np.int64)
    recoverability_targets_np = np.zeros((len(data["x"]),), dtype=np.float32)
    aux_reflections_path = args.aux_reflections or args.teacher_reflections or args.qwen_reflections
    if (
        aux_reflections_path is not None
        and (args.failure_weight > 0.0 or args.correction_weight > 0.0 or args.recoverability_weight > 0.0)
    ):
        failure_targets_np, correction_targets_np, recoverability_targets_np, aux_stats = auxiliary_targets(data, aux_reflections_path)
    train_idx, val_idx = split_indices(data["groups"], args.val_fraction, args.seed)

    model = PortfolioViabilityModel(
        data["x"].shape[-1],
        args.hidden_dim,
        data["x"].shape[1],
        context_dim=context_np.shape[1],
        execution_dim=execution_np.shape[1],
        num_failure_types=(max(failure_targets_np) + 1) if aux_stats is not None else 0,
        num_correction_directions=(max(correction_targets_np) + 1) if aux_stats is not None else 0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(data["x"]).to(device)
    context = torch.from_numpy(context_np).to(device)
    execution = torch.from_numpy(execution_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    rank_targets = torch.from_numpy(rank_targets_np).to(device)
    recovery_targets = torch.from_numpy(recovery_targets_np).to(device)
    step_targets = torch.from_numpy(data["steps"] / float(args.ep_len)).to(device)
    success_targets = torch.from_numpy(data["success"]).to(device)
    failure_targets = torch.from_numpy(failure_targets_np).to(device)
    correction_targets = torch.from_numpy(correction_targets_np).to(device)
    recoverability_targets = torch.from_numpy(recoverability_targets_np).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)

    best_state = None
    best_val = -float("inf")
    for _ in range(args.epochs):
        model.train()
        viability_logits, pred_steps, utility_logits, utility_delta, failure_logits, correction_logits, recoverability_logit, recovery_logits, execution_repr = model(
            x[train_t], context[train_t], execution[train_t]
        )
        utility_logits = utility_logits + float(args.execution_correction_scale) * utility_delta if args.use_execution_residual_correction else utility_logits
        viable_loss = F.binary_cross_entropy_with_logits(viability_logits, labels[train_t])
        rank_loss = F.kl_div(
            F.log_softmax(utility_logits, dim=1),
            rank_targets[train_t],
            reduction="batchmean",
        )
        success_mask = success_targets[train_t] > 0.5
        if success_mask.any():
            step_loss = F.mse_loss(pred_steps[success_mask], step_targets[train_t][success_mask])
        else:
            step_loss = torch.zeros((), device=device)
        failure_loss = (
            F.cross_entropy(failure_logits, failure_targets[train_t])
            if failure_logits is not None and args.failure_weight > 0.0
            else torch.zeros((), device=device)
        )
        correction_loss = (
            F.cross_entropy(correction_logits, correction_targets[train_t])
            if correction_logits is not None and args.correction_weight > 0.0
            else torch.zeros((), device=device)
        )
        recoverability_loss = (
            F.binary_cross_entropy_with_logits(recoverability_logit, recoverability_targets[train_t])
            if args.recoverability_weight > 0.0
            else torch.zeros((), device=device)
        )
        coupled_recovery_loss = (
            F.binary_cross_entropy_with_logits(recovery_logits, recovery_targets[train_t])
            if args.coupled_recovery_weight > 0.0
            else torch.zeros((), device=device)
        )
        execution_loss = (
            F.mse_loss(execution_repr, x[train_t].mean(dim=1)[:, : execution_repr.shape[1]])
            if execution_repr is not None and args.execution_repr_weight > 0.0
            else torch.zeros((), device=device)
        )
        loss = (
            float(args.viability_weight) * viable_loss
            + float(args.rank_weight) * rank_loss
            + float(args.step_weight) * step_loss
            + float(args.failure_weight) * failure_loss
            + float(args.correction_weight) * correction_loss
            + float(args.recoverability_weight) * recoverability_loss
            + float(args.coupled_recovery_weight) * coupled_recovery_loss
            + float(args.execution_repr_weight) * execution_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_viability_logits, val_steps, val_utility_logits, val_utility_delta, _, _, _, val_recovery_logits, _ = model(
                x[val_t], context[val_t], execution[val_t]
            )
            val_utility_logits = (
                val_utility_logits + float(args.execution_correction_scale) * val_utility_delta
                if args.use_execution_residual_correction else val_utility_logits
            )
            val_scores = (
                float(args.utility_score_weight) * val_utility_logits
                + float(args.viability_score_weight) * torch.sigmoid(val_viability_logits)
                + float(args.recovery_score_weight) * torch.sigmoid(val_recovery_logits)
                - float(args.score_step_lambda) * val_steps
            )
            val_pred = val_scores.argmax(dim=1)
            val_success = torch.from_numpy(data["success"][val_idx]).to(device)
            score = val_success[torch.arange(len(val_t), device=device), val_pred].mean()
        if float(score) > best_val:
            best_val = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        viability_logits_t, pred_steps_t, utility_logits_t, utility_delta_t, failure_logits_t, correction_logits_t, recoverability_logit_t, recovery_logits_t, execution_repr_t = model(
            x, context, execution
        )
        if args.use_execution_residual_correction:
            utility_logits_t = utility_logits_t + float(args.execution_correction_scale) * utility_delta_t
        logits_np = viability_logits_t.detach().cpu().numpy()
        pred_steps_np = pred_steps_t.detach().cpu().numpy()
        utility_logits_np = utility_logits_t.detach().cpu().numpy()
        utility_delta_np = utility_delta_t.detach().cpu().numpy()
        recovery_logits_np = recovery_logits_t.detach().cpu().numpy()
        failure_logits_np = failure_logits_t.detach().cpu().numpy() if failure_logits_t is not None else np.zeros((len(data["x"]), 0), dtype=np.float32)
        correction_logits_np = correction_logits_t.detach().cpu().numpy() if correction_logits_t is not None else np.zeros((len(data["x"]), 0), dtype=np.float32)
        recoverability_prob_np = torch.sigmoid(recoverability_logit_t).detach().cpu().numpy()

    metrics = {
        "method": "future_portfolio_viability",
        "conditioning": args.conditioning,
        "alpha_success": float(args.alpha_success),
        "gt_cos_norm": args.gt_cos_norm,
        "rank_weight": float(args.rank_weight),
        "step_weight": float(args.step_weight),
        "viability_weight": float(args.viability_weight),
        "utility_target": args.utility_target,
        "utility_score_weight": float(args.utility_score_weight),
        "viability_score_weight": float(args.viability_score_weight),
        "rank_temperature": float(args.rank_temperature),
        "score_step_lambda": float(args.score_step_lambda),
        "num_samples": int(len(data["x"])),
        "num_samples_before_sequence_filter": int(rows_before_filter),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(data["x"].shape[-1]),
        "feature_layout": list(BASE_FEATURE_LAYOUT) + cond_layout,
        "qwen_reflections": str(args.qwen_reflections) if args.qwen_reflections is not None else None,
        "reflection_encoder_type": args.reflection_encoder_type,
        "teacher_target_stats": teacher_target_stats,
        "use_trajectory_context": bool(args.use_trajectory_context),
        "context_dim": int(context_np.shape[1]),
        "use_execution_augmentation": bool(args.use_execution_augmentation),
        "use_execution_residual_correction": bool(args.use_execution_residual_correction),
        "execution_correction_scale": float(args.execution_correction_scale),
        "execution_dim": int(execution_np.shape[1]),
        "execution_repr_weight": float(args.execution_repr_weight),
        "execution_correction_mean_abs": float(np.mean(np.abs(utility_delta_np))) if args.use_execution_residual_correction else 0.0,
        "context_layout": [
            "prev_has_outcome",
            "prev_chosen_success",
            "prev_chosen_steps_norm",
            "prev_chosen_expert_onehot",
            "prev_oracle_expert_onehot",
            "prev_expert_success",
            "prev_expert_steps_norm",
            "prev_chosen_is_oracle",
            "prev_success_count_norm",
            "prev_best_value",
        ] if args.use_trajectory_context else [],
        "aux_stats": aux_stats,
        "failure_weight": float(args.failure_weight),
        "correction_weight": float(args.correction_weight),
        "recoverability_weight": float(args.recoverability_weight),
        "coupled_recovery_weight": float(args.coupled_recovery_weight),
        "recovery_score_weight": float(args.recovery_score_weight),
        "success_positive_rate": float(data["success"].mean()),
        "gt_cosine_mean": float(data["gt_cosine"].mean()),
        "gt_cosine_min": float(data["gt_cosine"].min()),
        "gt_cosine_max": float(data["gt_cosine"].max()),
        "target_mean": float(labels_np.mean()),
        "train": evaluate(
            logits_np,
            pred_steps_np,
            utility_logits_np,
            recovery_logits_np,
            data,
            train_idx,
            labels_np,
            args.score_step_lambda,
            args.utility_score_weight,
            args.viability_score_weight,
            args.recovery_score_weight,
            args.ep_len,
        ),
        "val": evaluate(
            logits_np,
            pred_steps_np,
            utility_logits_np,
            recovery_logits_np,
            data,
            val_idx,
            labels_np,
            args.score_step_lambda,
            args.utility_score_weight,
            args.viability_score_weight,
            args.recovery_score_weight,
            args.ep_len,
        ),
        "all": evaluate(
            logits_np,
            pred_steps_np,
            utility_logits_np,
            recovery_logits_np,
            data,
            np.arange(len(data["x"]), dtype=np.int64),
            labels_np,
            args.score_step_lambda,
            args.utility_score_weight,
            args.viability_score_weight,
            args.recovery_score_weight,
            args.ep_len,
        ),
    }
    if aux_stats is not None:
        metrics["aux_eval"] = {
            "train_failure_acc": aux_accuracy(failure_logits_np[train_idx], failure_targets_np[train_idx]),
            "val_failure_acc": aux_accuracy(failure_logits_np[val_idx], failure_targets_np[val_idx]),
            "train_correction_acc": aux_accuracy(correction_logits_np[train_idx], correction_targets_np[train_idx]),
            "val_correction_acc": aux_accuracy(correction_logits_np[val_idx], correction_targets_np[val_idx]),
            "train_recoverability_brier": float(np.mean((recoverability_prob_np[train_idx] - recoverability_targets_np[train_idx]) ** 2)),
            "val_recoverability_brier": float(np.mean((recoverability_prob_np[val_idx] - recoverability_targets_np[val_idx]) ** 2)),
            "train_coupled_recovery_bce": float(
                np.mean(
                    -(
                        recovery_targets_np[train_idx] * np.log(np.clip(1.0 / (1.0 + np.exp(-recovery_logits_np[train_idx])), 1e-7, 1.0))
                        + (1.0 - recovery_targets_np[train_idx]) * np.log(np.clip(1.0 - 1.0 / (1.0 + np.exp(-recovery_logits_np[train_idx])), 1e-7, 1.0))
                    )
                )
            ),
            "val_coupled_recovery_bce": float(
                np.mean(
                    -(
                        recovery_targets_np[val_idx] * np.log(np.clip(1.0 / (1.0 + np.exp(-recovery_logits_np[val_idx])), 1e-7, 1.0))
                        + (1.0 - recovery_targets_np[val_idx]) * np.log(np.clip(1.0 - 1.0 / (1.0 + np.exp(-recovery_logits_np[val_idx])), 1e-7, 1.0))
                    )
                )
            ),
        }
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": int(data["x"].shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "num_experts": int(data["x"].shape[1]),
            "feature_layout": metrics["feature_layout"],
            "conditioning": args.conditioning,
            "alpha_success": float(args.alpha_success),
            "gt_cos_norm": args.gt_cos_norm,
            "rank_weight": float(args.rank_weight),
            "step_weight": float(args.step_weight),
            "viability_weight": float(args.viability_weight),
            "utility_target": args.utility_target,
            "utility_score_weight": float(args.utility_score_weight),
            "viability_score_weight": float(args.viability_score_weight),
            "rank_temperature": float(args.rank_temperature),
            "score_step_lambda": float(args.score_step_lambda),
            "qwen_reflections": metrics["qwen_reflections"],
            "reflection_encoder_type": args.reflection_encoder_type,
            "use_execution_residual_correction": bool(args.use_execution_residual_correction),
            "execution_correction_scale": float(args.execution_correction_scale),
        },
        args.output_dir / "future_portfolio_viability.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
