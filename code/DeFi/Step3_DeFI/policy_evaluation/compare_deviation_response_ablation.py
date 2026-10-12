from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from policy_evaluation.train_directional_sensitivity_tcn import DirectionalSensitivityTCN
    from policy_evaluation.train_joint_future_action_energy import hashed_task
except ModuleNotFoundError:
    from train_directional_sensitivity_tcn import DirectionalSensitivityTCN
    from train_joint_future_action_energy import hashed_task


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _resolve_feature_path(rows_path: Path, feature_path: str) -> Path:
    path = Path(feature_path)
    return path if path.is_absolute() else rows_path.parent / path


def _pool_exec_summary_np(summary: np.ndarray, out_dim: int) -> np.ndarray:
    summary = np.asarray(summary, dtype=np.float32).reshape(-1)
    if summary.size == int(out_dim):
        return summary.astype(np.float32)
    if summary.size < int(out_dim):
        out = np.zeros((int(out_dim),), dtype=np.float32)
        out[: summary.size] = summary
        return out
    trim = (summary.size // int(out_dim)) * int(out_dim)
    return summary[:trim].reshape(int(out_dim), trim // int(out_dim)).mean(axis=1).astype(np.float32)


def _coerce_chunk(arr: np.ndarray, chunk_len: int, action_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((chunk_len, action_dim), dtype=np.float32)
    arr = arr.reshape(-1, action_dim)
    out = np.zeros((chunk_len, action_dim), dtype=np.float32)
    n = min(chunk_len, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-6)).astype(np.float32)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values.astype(np.float64), q))


def _segments(mask: np.ndarray) -> int:
    count = 0
    active = False
    for value in mask.astype(bool).reshape(-1):
        if value and not active:
            count += 1
            active = True
        elif not value:
            active = False
    return count


def _mask_metrics(pred_mask: np.ndarray, oracle_mask: np.ndarray) -> dict[str, float | int]:
    pred = pred_mask.astype(bool).reshape(-1)
    oracle = oracle_mask.astype(bool).reshape(-1)
    tp = int(np.logical_and(pred, oracle).sum())
    fp = int(np.logical_and(pred, ~oracle).sum())
    fn = int(np.logical_and(~pred, oracle).sum())
    tn = int(np.logical_and(~pred, ~oracle).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(1, tp + fp + fn + tn)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_positive": int(pred.sum()),
        "oracle_positive": int(oracle.sum()),
    }


def _delta_metrics(pred: np.ndarray, oracle: np.ndarray, action: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred_flat = torch.from_numpy(pred.reshape(pred.shape[0], -1).astype(np.float32))
    oracle_flat = torch.from_numpy(oracle.reshape(oracle.shape[0], -1).astype(np.float32))
    oracle_norm = torch.norm(oracle_flat, dim=-1)
    pred_norm = torch.norm(pred_flat, dim=-1)
    action_norm = np.linalg.norm(action.reshape(action.shape[0], -1), axis=-1)
    delta_norm = np.linalg.norm(pred.reshape(pred.shape[0], -1), axis=-1)
    cos = F.cosine_similarity(pred_flat, oracle_flat, dim=-1)
    affected_fraction = mask.astype(bool).reshape(mask.shape[0], -1).mean(axis=1)
    return {
        "l1_to_oracle": float(torch.mean(torch.abs(pred_flat - oracle_flat)).item()),
        "mse_to_oracle": float(F.mse_loss(pred_flat, oracle_flat).item()),
        "cos_to_oracle": float(cos.mean().item()),
        "pred_delta_norm_mean": float(pred_norm.mean().item()),
        "oracle_delta_norm_mean": float(oracle_norm.mean().item()),
        "delta_norm_ratio_to_oracle": float((pred_norm / torch.clamp(oracle_norm, min=1e-8)).mean().item()),
        "patched_action_fraction": float(affected_fraction.mean()),
        "patched_action_fraction_p90": _safe_quantile(affected_fraction, 0.90),
        "suffix_preservation_ratio": float(np.mean(1.0 - np.minimum(1.0, delta_norm / np.maximum(action_norm, 1e-8)))),
        "segments_mean": float(np.mean([_segments(m) for m in mask])),
    }


def _make_topk_mask(scores: np.ndarray, k: int, threshold: float) -> np.ndarray:
    mask = np.zeros_like(scores, dtype=bool)
    if k <= 0:
        return scores > threshold
    for i, row in enumerate(scores):
        if row.size == 0 or float(row.max()) < threshold:
            continue
        ids = np.argpartition(row, -min(k, row.size))[-min(k, row.size) :]
        mask[i, ids] = True
    return mask


def _structured_mask_np(scores: np.ndarray, patch_cost: float, switch_cost: float, threshold: float) -> np.ndarray:
    out = np.zeros_like(scores, dtype=bool)
    patch_c = float(max(0.0, patch_cost))
    switch_c = float(max(0.0, switch_cost))
    for row_id, row in enumerate(scores.astype(np.float64)):
        n = int(row.size)
        if n == 0:
            continue
        dp = np.zeros((n, 2), dtype=np.float64)
        back = np.zeros((n, 2), dtype=np.int64)
        dp[0, 0] = row[0]
        dp[0, 1] = patch_c
        for i in range(1, n):
            for state in (0, 1):
                unary = row[i] if state == 0 else patch_c
                stay = dp[i - 1, state]
                switch = dp[i - 1, 1 - state] + switch_c
                if stay <= switch:
                    dp[i, state] = stay + unary
                    back[i, state] = state
                else:
                    dp[i, state] = switch + unary
                    back[i, state] = 1 - state
        state = int(np.argmin(dp[-1]))
        mask = np.zeros((n,), dtype=bool)
        for i in range(n - 1, -1, -1):
            mask[i] = state == 1
            state = int(back[i, state])
        if threshold > 0.0:
            mask &= row >= threshold
        out[row_id] = mask
    return out


def _load_cases(rows_path: Path, max_rows: int, min_oracle_norm: float) -> dict[str, np.ndarray]:
    rows = _read_jsonl(rows_path)
    states: list[np.ndarray] = []
    hs: list[np.ndarray] = []
    delta_hs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    oracle_deltas: list[np.ndarray] = []
    oracle_scores: list[np.ndarray] = []
    task_vecs: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for row in rows:
        feature = _resolve_feature_path(rows_path, str(row.get("feature_path", "")))
        if not feature.exists():
            continue
        with np.load(feature, allow_pickle=True) as data:
            if "teacher_suffix_delta_action" not in data or "teacher_suffix_action" not in data:
                continue
            action = _coerce_chunk(data["teacher_suffix_action"], 10, 7)
            oracle = _coerce_chunk(data["teacher_suffix_delta_action"], 10, 7)
            scores = np.linalg.norm(oracle, axis=-1)
            if "teacher_step_scores" in data and np.asarray(data["teacher_step_scores"]).size:
                scores_raw = np.asarray(data["teacher_step_scores"], dtype=np.float32).reshape(-1)
                scores[: min(len(scores), len(scores_raw))] = scores_raw[: min(len(scores), len(scores_raw))]
            if not np.isfinite(action).all() or not np.isfinite(oracle).all():
                continue
            if float(np.linalg.norm(oracle.reshape(-1), ord=2)) < float(min_oracle_norm):
                continue
            h = _pool_exec_summary_np(data.get("h_future_before", np.zeros((0,), dtype=np.float32)), 512)
            h_action = _pool_exec_summary_np(data.get("h_action_before", np.zeros((0,), dtype=np.float32)), 512)
            h = np.concatenate([h, h_action], axis=0).astype(np.float32)
            delta_h = _pool_exec_summary_np(data.get("delta_h", np.zeros((0,), dtype=np.float32)), 1024)
            state = np.asarray(data.get("state_before", np.zeros((39,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if state.size < 39:
                fixed = np.zeros((39,), dtype=np.float32)
                fixed[: state.size] = state
                state = fixed
            state = state[:39]
            if not np.isfinite(state).all() or not np.isfinite(h).all() or not np.isfinite(delta_h).all():
                continue
            task = str(row.get("task", ""))
            subtask_index = int(row.get("subtask_index", 0) or 0)
            states.append(state)
            hs.append(h)
            delta_hs.append(delta_h)
            actions.append(action)
            oracle_deltas.append(oracle)
            oracle_scores.append(scores.astype(np.float32))
            task_vecs.append(hashed_task(task, 128).astype(np.float32))
            metas.append(np.asarray([subtask_index / 5.0, 0.1, 0.9], dtype=np.float32))
            meta_rows.append(
                {
                    "row_id": int(row.get("row_id", -1)),
                    "sequence_index": int(row.get("sequence_index", -1)),
                    "subtask_index": subtask_index,
                    "task": task,
                    "step": int(row.get("step", -1)),
                    "feature_path": str(feature),
                    "teacher_suffix_delta_norm": float(row.get("teacher_suffix_delta_norm", 0.0) or 0.0),
                }
            )
            if max_rows > 0 and len(states) >= max_rows:
                break
    if not states:
        raise ValueError("no valid deviation cases found")
    return {
        "state": np.stack(states).astype(np.float32),
        "h": np.stack(hs).astype(np.float32),
        "delta_h": np.stack(delta_hs).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "oracle_delta": np.stack(oracle_deltas).astype(np.float32),
        "oracle_scores": np.stack(oracle_scores).astype(np.float32),
        "task_vec": np.stack(task_vecs).astype(np.float32),
        "meta": np.stack(metas).astype(np.float32),
        "rows": np.asarray(meta_rows, dtype=object),
    }


def _predict_tcn(
    bank: dict[str, np.ndarray], ckpt_path: Path, device_name: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    model.load_state_dict(ckpt["model_state"], strict=False)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    state = _standardize(bank["state"], np.asarray(ckpt["state_mean"], dtype=np.float32), np.asarray(ckpt["state_std"], dtype=np.float32))
    h = _standardize(bank["h"], np.asarray(ckpt["h_mean"], dtype=np.float32), np.asarray(ckpt["h_std"], dtype=np.float32))
    delta_h = _standardize(
        bank["delta_h"],
        np.asarray(ckpt["delta_h_mean"], dtype=np.float32),
        np.asarray(ckpt["delta_h_std"], dtype=np.float32),
    )
    out: list[np.ndarray] = []
    invalidities: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(state), batch_size):
            sl = slice(start, start + batch_size)
            if bool(ckpt.get("has_validity_head", False)) and hasattr(model, "forward_with_heads"):
                heads = model.forward_with_heads(
                    torch.from_numpy(state[sl]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(h[sl]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(bank["action"][sl]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(delta_h[sl]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(bank["task_vec"][sl]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(bank["meta"][sl]).to(device=device, dtype=torch.float32),
                )
                pred = heads["delta"]
                invalidities.append(torch.sigmoid(heads["invalidity_logit"]).detach().cpu().numpy().astype(np.float32))
                uncertainties.append(torch.sigmoid(heads["uncertainty_logit"]).detach().cpu().numpy().astype(np.float32))
            else:
                pred = model(
                torch.from_numpy(state[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(h[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(bank["action"][sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(delta_h[sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(bank["task_vec"][sl]).to(device=device, dtype=torch.float32),
                torch.from_numpy(bank["meta"][sl]).to(device=device, dtype=torch.float32),
                )
            out.append(pred.detach().cpu().numpy().astype(np.float32))
    invalidity = np.concatenate(invalidities, axis=0) if invalidities else None
    uncertainty = np.concatenate(uncertainties, axis=0) if uncertainties else None
    return np.concatenate(out, axis=0), invalidity, uncertainty


def _apply_mask(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask.astype(bool)[..., None], delta, np.zeros_like(delta))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deviation response strategies on identical sensitivity-teacher cases."
    )
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--tcn-ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-oracle-norm", type=float, default=0.0)
    parser.add_argument("--oracle-threshold", type=float, default=-1.0)
    parser.add_argument("--oracle-quantile", type=float, default=0.80)
    parser.add_argument("--ours-threshold", type=float, default=-1.0)
    parser.add_argument("--ours-quantile", type=float, default=0.80)
    parser.add_argument("--ours-top-k", type=int, default=1)
    parser.add_argument("--ours-mask-source", choices=["delta_norm", "validity"], default="delta_norm")
    parser.add_argument("--ours-structured-mask", action="store_true")
    parser.add_argument("--ours-selection-lambda", type=float, default=0.5)
    parser.add_argument("--ours-selection-beta", type=float, default=0.15)
    parser.add_argument("--ours-max-abs-delta", type=float, default=0.003)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bank = _load_cases(args.rows_jsonl, int(args.max_rows), float(args.min_oracle_norm))
    oracle = bank["oracle_delta"]
    action = bank["action"]
    oracle_scores = bank["oracle_scores"]
    oracle_threshold = (
        float(args.oracle_threshold)
        if args.oracle_threshold >= 0
        else _safe_quantile(oracle_scores.reshape(-1), float(args.oracle_quantile))
    )
    oracle_mask = oracle_scores > oracle_threshold

    ours_raw, ours_invalidity, ours_uncertainty = _predict_tcn(bank, args.tcn_ckpt, args.device, int(args.batch_size))
    if float(args.ours_max_abs_delta) > 0.0:
        ours_raw = np.clip(ours_raw, -float(args.ours_max_abs_delta), float(args.ours_max_abs_delta)).astype(np.float32)
    ours_scores = (
        ours_invalidity
        if args.ours_mask_source == "validity" and ours_invalidity is not None
        else np.linalg.norm(ours_raw, axis=-1)
    )
    ours_threshold = (
        float(args.ours_threshold)
        if args.ours_threshold >= 0
        else _safe_quantile(ours_scores.reshape(-1), float(args.ours_quantile))
    )
    if args.ours_structured_mask:
        ours_mask = _structured_mask_np(
            ours_scores,
            patch_cost=float(args.ours_selection_lambda),
            switch_cost=float(args.ours_selection_beta),
            threshold=float(ours_threshold),
        )
    else:
        ours_mask = _make_topk_mask(ours_scores, int(args.ours_top_k), ours_threshold)

    keep_delta = np.zeros_like(oracle)
    whole_delta = oracle.copy()
    ours_delta = _apply_mask(ours_raw, ours_mask)
    oracle_mask_delta = _apply_mask(oracle, oracle_mask)

    random_metrics = []
    for _ in range(max(1, int(args.random_trials))):
        random_mask = np.zeros_like(oracle_mask, dtype=bool)
        for i, target in enumerate(oracle_mask):
            count = int(target.sum())
            if count <= 0:
                continue
            random_mask[i, rng.choice(target.size, size=count, replace=False)] = True
        random_delta = _apply_mask(oracle, random_mask)
        random_metrics.append(_delta_metrics(random_delta, oracle, action, random_mask))
    random_summary = {
        key: float(np.mean([m[key] for m in random_metrics]))
        for key in random_metrics[0].keys()
    }

    strategies = {
        "keep": {
            **_delta_metrics(keep_delta, oracle, action, np.zeros_like(oracle_mask, dtype=bool)),
            "mask": _mask_metrics(np.zeros_like(oracle_mask, dtype=bool), oracle_mask),
            "note": "A_remain unchanged.",
        },
        "whole_suffix_patch": {
            **_delta_metrics(whole_delta, oracle, action, np.ones_like(oracle_mask, dtype=bool)),
            "mask": _mask_metrics(np.ones_like(oracle_mask, dtype=bool), oracle_mask),
            "note": "Oracle delta applied to every suffix timestep; tests whole-suffix editing cost.",
        },
        "random_mask_patch": {
            **random_summary,
            "mask": {"note": "random mask uses the same positive count as oracle per case"},
            "note": f"Mean over {int(args.random_trials)} random masks.",
        },
        "ours_selective_patch": {
            **_delta_metrics(ours_delta, oracle, action, ours_mask),
            "mask": _mask_metrics(ours_mask, oracle_mask),
            "note": "DirectionalSensitivityTCN predicts J_h delta_h, then masks affected actions.",
        },
        "oracle_mask_upper_bound": {
            **_delta_metrics(oracle_mask_delta, oracle, action, oracle_mask),
            "mask": _mask_metrics(oracle_mask, oracle_mask),
            "note": "Uses oracle affected mask and oracle delta; upper bound for selective patch.",
        },
        "replan": {
            "note": "True replan calls GIDM/env and cannot be faithfully scored from fixed offline suffix cases.",
            "requires_rollout": True,
        },
        "direct_correction": {
            "note": "Requires a direct observation/state-to-action correction model evaluated on the same case schema; not inferred here.",
            "requires_direct_model_adapter": True,
        },
    }

    result = {
        "rows_jsonl": str(args.rows_jsonl),
        "tcn_ckpt": str(args.tcn_ckpt),
        "num_cases": int(len(action)),
        "oracle_threshold": float(oracle_threshold),
        "oracle_threshold_source": "fixed" if args.oracle_threshold >= 0 else f"q{args.oracle_quantile}",
        "ours_threshold": float(ours_threshold),
        "ours_threshold_source": "fixed" if args.ours_threshold >= 0 else f"q{args.ours_quantile}",
        "ours_top_k": int(args.ours_top_k),
        "ours_mask_source": str(args.ours_mask_source),
        "ours_structured_mask": bool(args.ours_structured_mask),
        "ours_selection_lambda": float(args.ours_selection_lambda),
        "ours_selection_beta": float(args.ours_selection_beta),
        "ours_max_abs_delta": float(args.ours_max_abs_delta),
        "oracle_affected_fraction": float(oracle_mask.mean()),
        "ours_affected_fraction": float(ours_mask.mean()),
        "strategies": strategies,
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)

    if args.examples_jsonl is not None:
        args.examples_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.examples_jsonl.open("w", encoding="utf-8") as handle:
            rows = bank["rows"].tolist()
            for i in range(min(20, len(rows))):
                handle.write(
                    json.dumps(
                        {
                            **rows[i],
                            "oracle_scores": oracle_scores[i].astype(float).tolist(),
                            "oracle_mask": oracle_mask[i].astype(int).tolist(),
                            "ours_scores": ours_scores[i].astype(float).tolist(),
                            "ours_mask": ours_mask[i].astype(int).tolist(),
                            "oracle_delta_norm": float(np.linalg.norm(oracle[i].reshape(-1), ord=2)),
                            "ours_delta_norm": float(np.linalg.norm(ours_delta[i].reshape(-1), ord=2)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
