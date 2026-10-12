from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from policy_evaluation.train_chunk_effect_phi_mlp import split_by_sequence
    from policy_evaluation.train_directional_sensitivity_tcn import DirectionalSensitivityTCN, _finite_rows
    from policy_evaluation.train_joint_execution_update_mlp import build_meta
except ModuleNotFoundError:
    from train_chunk_effect_phi_mlp import split_by_sequence
    from train_directional_sensitivity_tcn import DirectionalSensitivityTCN, _finite_rows
    from train_joint_execution_update_mlp import build_meta


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _as_2d_stat(value: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] != dim:
        raise ValueError(f"normalization stat dim mismatch: expected {dim}, got {arr.shape[1]}")
    return arr


def _standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values.astype(np.float64), q))


def _mask_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float | int]:
    pred = pred_mask.astype(bool).reshape(-1)
    target = target_mask.astype(bool).reshape(-1)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_positive": int(pred.sum()),
        "target_positive": int(target.sum()),
        "total_actions": int(pred.size),
    }


def _topk_mask(scores: np.ndarray, k: int, floor: float = 0.0, min_margin: float = 0.0) -> np.ndarray:
    if k <= 0:
        return scores > floor
    out = np.zeros_like(scores, dtype=bool)
    for i, row in enumerate(scores):
        if row.size == 0:
            continue
        max_score = float(row.max())
        median_score = float(np.median(row))
        if max_score < floor:
            continue
        if min_margin > 0.0 and (max_score - median_score) < min_margin:
            continue
        kk = min(k, row.size)
        ids = np.argpartition(row, -kk)[-kk:]
        out[i, ids] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DirectionalSensitivityTCN against finite-difference/proxy affected-action labels."
    )
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["sequence", "all"], default="sequence")
    parser.add_argument("--target-delta-clip", type=float, default=-1.0)
    parser.add_argument("--target-threshold", type=float, default=-1.0)
    parser.add_argument("--target-quantile", type=float, default=0.80)
    parser.add_argument("--pred-threshold", type=float, default=-1.0)
    parser.add_argument("--pred-quantile", type=float, default=0.80)
    parser.add_argument("--pred-top-k", type=int, default=0)
    parser.add_argument("--pred-top-k-floor", type=float, default=0.0)
    parser.add_argument("--pred-top-k-min-margin", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    memory = _load_npz(args.memory_npz)
    required = ["state_start", "base_summary_exec", "target_summary_exec", "action_chunk", "action_delta_target", "task_vec"]
    missing = [key for key in required if key not in memory]
    if missing:
        raise ValueError(f"memory missing keys: {missing}")

    state = np.asarray(memory["state_start"], dtype=np.float32)
    h = np.asarray(memory["base_summary_exec"], dtype=np.float32)
    delta_h = (np.asarray(memory["target_summary_exec"], dtype=np.float32) - h).astype(np.float32)
    action = np.asarray(memory["action_chunk"], dtype=np.float32)
    target = np.asarray(memory["action_delta_target"], dtype=np.float32)
    if args.target_delta_clip > 0:
        target = np.clip(target, -float(args.target_delta_clip), float(args.target_delta_clip)).astype(np.float32)
    task = np.asarray(memory["task_vec"], dtype=np.float32)
    meta = build_meta(memory).astype(np.float32)
    finite = _finite_rows(state, h, delta_h, action, target, task, meta)
    if not finite.any():
        raise ValueError("no finite rows")
    state, h, delta_h, action, target, task, meta = (
        arr[finite] for arr in [state, h, delta_h, action, target, task, meta]
    )
    seq = np.asarray(memory.get("sequence_index", np.arange(len(state))), dtype=np.int64)[finite]

    if args.split_mode == "sequence":
        _, eval_idx = split_by_sequence(seq, args.val_ratio, args.seed)
        if len(eval_idx) == 0:
            eval_idx = np.arange(len(state), dtype=np.int64)
    else:
        eval_idx = np.arange(len(state), dtype=np.int64)

    state, h, delta_h, action, target, task, meta = (
        arr[eval_idx] for arr in [state, h, delta_h, action, target, task, meta]
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = DirectionalSensitivityTCN(
        state_dim=int(ckpt["state_dim"]),
        summary_dim=int(ckpt["summary_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_dim=int(ckpt["task_dim"]),
        chunk_len=int(ckpt["chunk_len"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        cond_dim=int(ckpt.get("cond_dim", 256)),
        num_blocks=int(ckpt.get("num_blocks", 3)),
        kernel_size=int(ckpt.get("kernel_size", 3)),
        dropout=float(ckpt.get("dropout", 0.0)),
        output_scale=float(ckpt.get("output_scale", 1.0)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    state = _standardize(state, _as_2d_stat(ckpt["state_mean"], state.shape[-1]), _as_2d_stat(ckpt["state_std"], state.shape[-1]))
    h = _standardize(h, _as_2d_stat(ckpt["h_mean"], h.shape[-1]), _as_2d_stat(ckpt["h_std"], h.shape[-1]))
    delta_h = _standardize(
        delta_h,
        _as_2d_stat(ckpt["delta_h_mean"], delta_h.shape[-1]),
        _as_2d_stat(ckpt["delta_h_std"], delta_h.shape[-1]),
    )

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(state), args.batch_size):
            sl = slice(start, start + args.batch_size)
            pred = model(
                torch.from_numpy(state[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(h[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(action[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(delta_h[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(task[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(meta[sl]).to(device=device, dtype=torch.float32),
            )
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0)

    pred_step_norm = np.linalg.norm(pred, axis=-1)
    target_step_norm = np.linalg.norm(target, axis=-1)
    target_threshold = args.target_threshold if args.target_threshold >= 0 else _quantile(target_step_norm.reshape(-1), args.target_quantile)
    pred_threshold = args.pred_threshold if args.pred_threshold >= 0 else _quantile(pred_step_norm.reshape(-1), args.pred_quantile)
    target_mask = target_step_norm > float(target_threshold)
    if args.pred_top_k > 0:
        pred_mask = _topk_mask(
            pred_step_norm,
            int(args.pred_top_k),
            floor=float(args.pred_top_k_floor if args.pred_top_k_floor > 0 else pred_threshold),
            min_margin=float(args.pred_top_k_min_margin),
        )
    else:
        pred_mask = pred_step_norm > float(pred_threshold)

    pred_t = torch.from_numpy(pred.reshape(pred.shape[0], -1))
    target_t = torch.from_numpy(target.reshape(target.shape[0], -1))
    result = {
        "memory_npz": str(args.memory_npz),
        "ckpt": str(args.ckpt),
        "num_rows_eval": int(len(pred)),
        "chunk_len": int(pred.shape[1]),
        "action_dim": int(pred.shape[2]),
        "mse": float(F.mse_loss(pred_t, target_t).item()),
        "l1": float(torch.mean(torch.abs(pred_t - target_t)).item()),
        "cosine": float(F.cosine_similarity(pred_t, target_t, dim=-1).mean().item()),
        "pred_norm_mean": float(pred_step_norm.mean()),
        "target_norm_mean": float(target_step_norm.mean()),
        "pred_step_norm_p50": _quantile(pred_step_norm.reshape(-1), 0.50),
        "pred_step_norm_p90": _quantile(pred_step_norm.reshape(-1), 0.90),
        "target_step_norm_p50": _quantile(target_step_norm.reshape(-1), 0.50),
        "target_step_norm_p90": _quantile(target_step_norm.reshape(-1), 0.90),
        "target_threshold": float(target_threshold),
        "pred_threshold": float(pred_threshold),
        "pred_top_k": int(args.pred_top_k),
        "detection": _mask_metrics(pred_mask, target_mask),
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
