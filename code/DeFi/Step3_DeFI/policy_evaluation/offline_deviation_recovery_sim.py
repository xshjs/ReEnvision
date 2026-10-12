from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from policy_evaluation.train_directional_sensitivity_tcn import DirectionalSensitivityTCN
    from policy_evaluation.train_joint_execution_update_mlp import build_meta
except ModuleNotFoundError:
    from train_directional_sensitivity_tcn import DirectionalSensitivityTCN
    from train_joint_execution_update_mlp import build_meta


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones((arrays[0].shape[0],), dtype=bool)
    for arr in arrays:
        flat = np.asarray(arr).reshape(arr.shape[0], -1)
        mask &= np.isfinite(flat).all(axis=1)
    return mask


def _standardize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((value - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-6)).astype(np.float32)


def _structured_mask(scores: np.ndarray, patch_cost: float, switch_cost: float, threshold: float) -> np.ndarray:
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
        mask &= row >= float(threshold)
        out[row_id] = mask
    return out


def _segments(mask: np.ndarray) -> np.ndarray:
    values = []
    for row in mask.astype(bool):
        count = 0
        active = False
        for value in row:
            if value and not active:
                count += 1
                active = True
            elif not value:
                active = False
        values.append(count)
    return np.asarray(values, dtype=np.float32)


def _smooth_cost(action: np.ndarray) -> float:
    if action.shape[1] < 3:
        return 0.0
    second = action[:, 2:] - 2.0 * action[:, 1:-1] + action[:, :-2]
    return float(np.linalg.norm(second, axis=-1).mean())


def _strategy_metrics(
    base_action: np.ndarray,
    strategy_delta: np.ndarray,
    target_delta: np.ndarray,
    mask: np.ndarray,
    recovery_rel_threshold: float,
    replan_mask: np.ndarray | None = None,
) -> dict[str, float]:
    pred_action = base_action + strategy_delta
    err = np.linalg.norm((strategy_delta - target_delta).reshape(len(base_action), -1), axis=1)
    tgt = np.linalg.norm(target_delta.reshape(len(base_action), -1), axis=1)
    rel = err / np.maximum(tgt, 1e-8)
    step_delta_norm = np.linalg.norm(strategy_delta, axis=-1)
    step_target_norm = np.linalg.norm(target_delta, axis=-1)
    modified = step_delta_norm > 1e-8
    target_affected = step_target_norm > 1e-8
    if replan_mask is None:
        replan_mask = np.zeros((len(base_action),), dtype=bool)
    return {
        "proxy_recovery_success": float(np.mean(rel <= float(recovery_rel_threshold))),
        "relative_error_mean": float(rel.mean()),
        "relative_error_p50": float(np.quantile(rel, 0.50)),
        "relative_error_p90": float(np.quantile(rel, 0.90)),
        "delta_l1_to_teacher": float(np.mean(np.abs(strategy_delta - target_delta))),
        "delta_norm_mean": float(np.linalg.norm(strategy_delta.reshape(len(base_action), -1), axis=1).mean()),
        "teacher_delta_norm_mean": float(tgt.mean()),
        "actions_modified_pct": float(modified.mean() * 100.0),
        "affected_pct": float(mask.astype(bool).mean() * 100.0),
        "segments_mean": float(_segments(mask).mean()),
        "replan_rate_pct": float(replan_mask.astype(bool).mean() * 100.0),
        "suffix_preservation_ratio": float(
            np.mean(
                1.0
                - np.minimum(
                    1.0,
                    np.linalg.norm(strategy_delta.reshape(len(base_action), -1), axis=1)
                    / np.maximum(np.linalg.norm(base_action.reshape(len(base_action), -1), axis=1), 1e-8),
                )
            )
        ),
        "smooth_cost": _smooth_cost(pred_action),
        "target_actions_affected_pct": float(target_affected.mean() * 100.0),
    }


def _load_bank(memory: dict[str, np.ndarray], max_rows: int) -> dict[str, np.ndarray]:
    required = ["state_start", "base_summary_exec", "target_summary_exec", "action_chunk", "action_delta_target", "task_vec"]
    missing = [key for key in required if key not in memory]
    if missing:
        raise ValueError(f"memory missing keys: {missing}")
    state = np.asarray(memory["state_start"], dtype=np.float32)
    h = np.asarray(memory["base_summary_exec"], dtype=np.float32)
    delta_h = (np.asarray(memory["target_summary_exec"], dtype=np.float32) - h).astype(np.float32)
    action = np.asarray(memory["action_chunk"], dtype=np.float32)
    delta = np.asarray(memory["action_delta_target"], dtype=np.float32)
    task = np.asarray(memory["task_vec"], dtype=np.float32)
    meta = build_meta(memory).astype(np.float32)
    finite = _finite_rows(state, h, delta_h, action, delta, task, meta)
    ids = np.flatnonzero(finite)
    if max_rows > 0:
        ids = ids[: int(max_rows)]
    if ids.size == 0:
        raise ValueError("no finite rows found")
    return {
        "state": state[ids],
        "h": h[ids],
        "delta_h": delta_h[ids],
        "action": action[ids],
        "teacher_delta": delta[ids],
        "task": task[ids],
        "meta": meta[ids],
        "row_id": np.asarray(memory.get("row_id", ids), dtype=np.int64)[ids],
        "sequence_index": np.asarray(memory.get("sequence_index", ids), dtype=np.int64)[ids],
        "subtask_index": np.asarray(memory.get("subtask_index", np.zeros_like(ids)), dtype=np.int64)[ids],
        "task_name": np.asarray(memory.get("task", np.asarray([""] * len(state), dtype=object)), dtype=object)[ids],
    }


def _predict_tcn(
    bank: dict[str, np.ndarray],
    ckpt_path: Path,
    severity_scale: float,
    batch_size: int,
    device_name: str,
    max_abs_delta: float,
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
    scaled_delta_h = bank["delta_h"] * float(severity_scale)
    delta_h = _standardize(
        scaled_delta_h,
        np.asarray(ckpt["delta_h_mean"], dtype=np.float32),
        np.asarray(ckpt["delta_h_std"], dtype=np.float32),
    )
    deltas: list[np.ndarray] = []
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
                    torch.from_numpy(bank["task"][sl]).to(device=device, dtype=torch.float32),
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
                    torch.from_numpy(bank["task"][sl]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(bank["meta"][sl]).to(device=device, dtype=torch.float32),
                )
            deltas.append(pred.detach().cpu().numpy().astype(np.float32))
    delta = np.concatenate(deltas, axis=0)
    if max_abs_delta > 0:
        delta = np.clip(delta, -float(max_abs_delta), float(max_abs_delta)).astype(np.float32)
    invalidity = np.concatenate(invalidities, axis=0) if invalidities else None
    uncertainty = np.concatenate(uncertainties, axis=0) if uncertainties else None
    return delta, invalidity, uncertainty


def _first_affected_delta(target_delta: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(target_delta)
    mask = np.zeros_like(target_mask, dtype=bool)
    for i, row in enumerate(target_mask):
        ids = np.flatnonzero(row)
        if ids.size:
            k = int(ids[0])
            out[i, k] = target_delta[i, k]
            mask[i, k] = True
    return out, mask


def _random_mask_like(target_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(target_mask, dtype=bool)
    for i, row in enumerate(target_mask):
        count = int(row.sum())
        if count > 0:
            out[i, rng.choice(row.size, size=count, replace=False)] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline deviation/recovery proxy simulator without CALVIN rollout.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--tcn-ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--severity", action="append", default=["small:0.5", "medium:1.0", "severe:1.5"])
    parser.add_argument("--affected-threshold", type=float, default=0.0001)
    parser.add_argument("--recovery-rel-threshold", type=float, default=0.5)
    parser.add_argument("--tcn-mask-source", choices=["validity", "delta_norm"], default="validity")
    parser.add_argument("--tcn-validity-threshold", type=float, default=0.5)
    parser.add_argument("--tcn-delta-threshold", type=float, default=0.0001)
    parser.add_argument("--structured-mask", action="store_true")
    parser.add_argument("--selection-lambda", type=float, default=0.5)
    parser.add_argument("--selection-beta", type=float, default=0.15)
    parser.add_argument("--tcn-max-abs-delta", type=float, default=0.003)
    parser.add_argument("--replan-affected-frac", type=float, default=0.6)
    parser.add_argument("--replan-uncertainty-threshold", type=float, default=0.9)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    memory = _load_npz(args.memory_npz)
    bank = _load_bank(memory, int(args.max_rows))
    rng = np.random.default_rng(int(args.seed))

    results: dict[str, object] = {
        "memory_npz": str(args.memory_npz),
        "tcn_ckpt": str(args.tcn_ckpt),
        "num_cases": int(len(bank["action"])),
        "note": (
            "Offline proxy only: recovery_success is closeness to paired-policy teacher correction, "
            "not environment task success. full_replan_proxy uses the full paired-policy response A+DeltaA*."
        ),
        "config": {
            "affected_threshold": float(args.affected_threshold),
            "recovery_rel_threshold": float(args.recovery_rel_threshold),
            "tcn_mask_source": str(args.tcn_mask_source),
            "tcn_validity_threshold": float(args.tcn_validity_threshold),
            "tcn_delta_threshold": float(args.tcn_delta_threshold),
            "structured_mask": bool(args.structured_mask),
            "selection_lambda": float(args.selection_lambda),
            "selection_beta": float(args.selection_beta),
            "tcn_max_abs_delta": float(args.tcn_max_abs_delta),
            "replan_affected_frac": float(args.replan_affected_frac),
            "replan_uncertainty_threshold": float(args.replan_uncertainty_threshold),
        },
        "severities": {},
    }

    examples: list[dict[str, object]] = []
    for spec in args.severity:
        name, value = str(spec).split(":", 1)
        scale = float(value)
        target_delta = (bank["teacher_delta"] * scale).astype(np.float32)
        target_scores = np.linalg.norm(target_delta, axis=-1)
        target_mask = target_scores > float(args.affected_threshold)

        tcn_delta, invalidity, uncertainty = _predict_tcn(
            bank,
            args.tcn_ckpt,
            severity_scale=scale,
            batch_size=int(args.batch_size),
            device_name=str(args.device),
            max_abs_delta=float(args.tcn_max_abs_delta),
        )
        if args.tcn_mask_source == "validity" and invalidity is not None:
            tcn_scores = invalidity
            tcn_threshold = float(args.tcn_validity_threshold)
        else:
            tcn_scores = np.linalg.norm(tcn_delta, axis=-1)
            tcn_threshold = float(args.tcn_delta_threshold)
        if args.structured_mask:
            tcn_mask = _structured_mask(
                tcn_scores,
                patch_cost=float(args.selection_lambda),
                switch_cost=float(args.selection_beta),
                threshold=tcn_threshold,
            )
        else:
            tcn_mask = tcn_scores >= tcn_threshold

        affected_fraction = tcn_mask.mean(axis=1)
        if uncertainty is not None:
            uncertainty_max = uncertainty.max(axis=1)
        else:
            uncertainty_max = np.zeros((len(tcn_mask),), dtype=np.float32)
        replan_mask = (affected_fraction > float(args.replan_affected_frac)) | (
            uncertainty_max > float(args.replan_uncertainty_threshold)
        )

        no_delta = np.zeros_like(target_delta)
        oracle_delta = np.where(target_mask[..., None], target_delta, np.zeros_like(target_delta))
        random_metrics = []
        for _ in range(max(1, int(args.random_trials))):
            random_mask = _random_mask_like(target_mask, rng)
            random_delta = np.where(random_mask[..., None], target_delta, np.zeros_like(target_delta))
            random_metrics.append(
                _strategy_metrics(
                    bank["action"], random_delta, target_delta, random_mask, float(args.recovery_rel_threshold)
                )
            )
        random_summary = {key: float(np.mean([m[key] for m in random_metrics])) for key in random_metrics[0].keys()}
        step_delta, step_mask = _first_affected_delta(target_delta, target_mask)
        tcn_patch_delta = np.where(tcn_mask[..., None], tcn_delta, np.zeros_like(tcn_delta))
        reenv_delta = np.where(replan_mask[:, None, None], target_delta, tcn_patch_delta)
        reenv_mask = np.where(replan_mask[:, None], np.ones_like(tcn_mask, dtype=bool), tcn_mask)

        severity_result = {
            "scale": scale,
            "teacher_affected_pct": float(target_mask.mean() * 100.0),
            "teacher_segments_mean": float(_segments(target_mask).mean()),
            "strategies": {
                "no_repair": _strategy_metrics(
                    bank["action"], no_delta, target_delta, np.zeros_like(target_mask), float(args.recovery_rel_threshold)
                ),
                "step_wise_correction_proxy": _strategy_metrics(
                    bank["action"], step_delta, target_delta, step_mask, float(args.recovery_rel_threshold)
                ),
                "full_replan_proxy": _strategy_metrics(
                    bank["action"],
                    target_delta,
                    target_delta,
                    np.ones_like(target_mask, dtype=bool),
                    float(args.recovery_rel_threshold),
                    replan_mask=np.ones((len(target_mask),), dtype=bool),
                ),
                "oracle_selective": _strategy_metrics(
                    bank["action"], oracle_delta, target_delta, target_mask, float(args.recovery_rel_threshold)
                ),
                "random_mask": random_summary,
                "reenvision_tcn": _strategy_metrics(
                    bank["action"],
                    reenv_delta,
                    target_delta,
                    reenv_mask,
                    float(args.recovery_rel_threshold),
                    replan_mask=replan_mask,
                ),
            },
        }
        results["severities"][name] = severity_result  # type: ignore[index]

        for i in range(min(10, len(bank["action"]))):
            examples.append(
                {
                    "severity": name,
                    "row_id": int(bank["row_id"][i]),
                    "sequence_index": int(bank["sequence_index"][i]),
                    "subtask_index": int(bank["subtask_index"][i]),
                    "task": str(bank["task_name"][i]),
                    "teacher_scores": target_scores[i].astype(float).tolist(),
                    "teacher_mask": target_mask[i].astype(int).tolist(),
                    "tcn_scores": tcn_scores[i].astype(float).tolist(),
                    "tcn_mask": tcn_mask[i].astype(int).tolist(),
                    "replan": bool(replan_mask[i]),
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    if args.examples_jsonl is not None:
        args.examples_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.examples_jsonl.open("w", encoding="utf-8") as handle:
            for row in examples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
