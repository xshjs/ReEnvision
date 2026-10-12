from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from policy_evaluation.train_directional_sensitivity_tcn import (
        DirectionalSensitivityTCN,
        _batch,
        _load_npz,
        _make_bank,
        _split_stratified_by_mode,
        _standardize,
    )
    from policy_evaluation.train_chunk_effect_phi_mlp import split_by_sequence
except ModuleNotFoundError:
    from train_directional_sensitivity_tcn import (
        DirectionalSensitivityTCN,
        _batch,
        _load_npz,
        _make_bank,
        _split_stratified_by_mode,
        _standardize,
    )
    from train_chunk_effect_phi_mlp import split_by_sequence


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask.astype(bool).tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / union) if union > 0 else 0.0


def _segment_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred_segs = _segments(pred)
    target_segs = _segments(target)
    if not pred_segs and not target_segs:
        return 1.0, float("nan")
    if not pred_segs or not target_segs:
        return 0.0, 0.0
    gt_to_pred = [max(_interval_iou(t, p) for p in pred_segs) for t in target_segs]
    pred_to_gt = [max(_interval_iou(p, t) for t in target_segs) for p in pred_segs]
    symmetric = 0.5 * (float(np.mean(gt_to_pred)) + float(np.mean(pred_to_gt)))
    gt_only = float(np.mean(gt_to_pred))
    return symmetric, gt_only


def _mask_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0, float("nan")
    inter = np.logical_and(pred, target).sum()
    return float(inter / union), float(inter / union)


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[DirectionalSensitivityTCN, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
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
    ).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TCN affected-action localization with segment IoU.")
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split-mode", choices=["stratified_mode", "sequence"], default="stratified_mode")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--target-delta-clip", type=float, default=0.05)
    parser.add_argument("--affected-threshold", type=float, default=1e-4)
    parser.add_argument("--validity-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    memory = _load_npz(args.memory_npz)
    bank = _make_bank(memory, args.target_delta_clip)
    if args.split_mode == "sequence":
        train_idx, val_idx = split_by_sequence(bank["sequence_index"], args.val_ratio, args.seed)
    else:
        train_idx, val_idx = _split_stratified_by_mode(bank["mode"], args.val_ratio, args.seed)

    bank["state"], _, _ = _standardize(bank["state"][train_idx], bank["state"])
    bank["h"], _, _ = _standardize(bank["h"][train_idx], bank["h"])
    bank["delta_h"], _, _ = _standardize(bank["delta_h"][train_idx], bank["delta_h"])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, _ = _load_model(args.ckpt, device)

    tp = fp = fn = tn = 0
    delta_abs = []
    mask_ious_all = []
    mask_ious_nonempty = []
    seg_ious_all = []
    seg_ious_nonempty = []
    gt_seg_counts = []
    pred_seg_counts = []
    gt_frac = []
    pred_frac = []
    n_rows = 0

    with torch.no_grad():
        for start in range(0, len(val_idx), args.batch_size):
            ids = val_idx[start : start + args.batch_size]
            b = _batch(bank, ids, device)
            heads = model.forward_with_heads(b["state"], b["h"], b["action"], b["delta_h"], b["task"], b["meta"])
            pred_delta = heads["delta"]
            target_delta = b["target_delta_action"]
            pred_mask = (torch.sigmoid(heads["invalidity_logit"]) > args.validity_threshold).cpu().numpy()
            target_mask = (torch.norm(target_delta, dim=-1) > args.affected_threshold).cpu().numpy()
            delta_abs.append(torch.abs(pred_delta - target_delta).mean().item() * len(ids))

            p_flat = pred_mask.reshape(-1)
            t_flat = target_mask.reshape(-1)
            tp += int(np.logical_and(p_flat, t_flat).sum())
            fp += int(np.logical_and(p_flat, ~t_flat).sum())
            fn += int(np.logical_and(~p_flat, t_flat).sum())
            tn += int(np.logical_and(~p_flat, ~t_flat).sum())

            for p, t in zip(pred_mask, target_mask):
                miou_all, miou_nonempty = _mask_iou(p, t)
                siou_all, siou_nonempty = _segment_iou(p, t)
                mask_ious_all.append(miou_all)
                seg_ious_all.append(siou_all)
                if np.isfinite(miou_nonempty):
                    mask_ious_nonempty.append(miou_nonempty)
                if np.isfinite(siou_nonempty):
                    seg_ious_nonempty.append(siou_nonempty)
                gt_seg_counts.append(len(_segments(t)))
                pred_seg_counts.append(len(_segments(p)))
                gt_frac.append(float(np.mean(t)))
                pred_frac.append(float(np.mean(p)))
            n_rows += len(ids)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    summary = {
        "memory_npz": str(args.memory_npz),
        "ckpt": str(args.ckpt),
        "num_eval_rows": int(n_rows),
        "affected_threshold": float(args.affected_threshold),
        "validity_threshold": float(args.validity_threshold),
        "delta_mae": float(sum(delta_abs) / max(1, n_rows)),
        "mask_precision": float(precision),
        "mask_recall": float(recall),
        "mask_f1": float(f1),
        "mask_tp": int(tp),
        "mask_fp": int(fp),
        "mask_fn": int(fn),
        "mask_tn": int(tn),
        "temporal_mask_iou_all": float(np.mean(mask_ious_all)) if mask_ious_all else 0.0,
        "temporal_mask_iou_nonempty": float(np.mean(mask_ious_nonempty)) if mask_ious_nonempty else 0.0,
        "segment_iou_all": float(np.mean(seg_ious_all)) if seg_ious_all else 0.0,
        "segment_iou_nonempty": float(np.mean(seg_ious_nonempty)) if seg_ious_nonempty else 0.0,
        "gt_segments_mean": float(np.mean(gt_seg_counts)) if gt_seg_counts else 0.0,
        "pred_segments_mean": float(np.mean(pred_seg_counts)) if pred_seg_counts else 0.0,
        "gt_affected_fraction_mean": float(np.mean(gt_frac)) if gt_frac else 0.0,
        "pred_affected_fraction_mean": float(np.mean(pred_frac)) if pred_frac else 0.0,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
