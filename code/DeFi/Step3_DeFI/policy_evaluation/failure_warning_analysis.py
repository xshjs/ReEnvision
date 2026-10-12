import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CONTACT_RICH_TASKS = {
    "push_red_block_right",
    "push_blue_block_right",
    "push_pink_block_right",
    "push_into_drawer",
    "stack_block",
}

NON_CONTACT_RICH_TASKS = {
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "open_drawer",
    "close_drawer",
}


@dataclass
class SignalCurves:
    trajectory_id: str
    task: str
    failed: bool
    visual: np.ndarray
    contact: np.ndarray


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0 if np.linalg.norm(a - b) <= 1e-12 else 1.0
    return float(1.0 - np.dot(a, b) / denom)


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def visual_mismatch(step: Dict[str, Any], metric: str) -> Optional[float]:
    if "visual_mismatch" in step:
        return float(step["visual_mismatch"])
    visual = step.get("visual", step)
    if "mismatch" in visual:
        return float(visual["mismatch"])
    pred = _as_array(_first_present(visual, ["pred", "predicted", "visual_pred"]))
    obs = _as_array(_first_present(visual, ["obs", "observed", "visual_obs"]))
    if pred is None or obs is None:
        return None
    if pred.shape != obs.shape:
        min_len = min(pred.size, obs.size)
        pred = pred[:min_len]
        obs = obs[:min_len]
    if metric == "cosine":
        return _cosine_distance(pred, obs)
    if metric == "l2":
        return _l2_distance(pred, obs)
    raise ValueError(f"Unsupported visual metric: {metric}")


def _contact_items(state: Any) -> List[Dict[str, Any]]:
    if state is None:
        return []
    if isinstance(state, dict):
        if "contacts" in state:
            state = state["contacts"]
        else:
            state = [state]
    if not isinstance(state, list):
        return []
    return [item for item in state if isinstance(item, dict)]


def _contact_key(contact: Dict[str, Any]) -> Tuple[str, str]:
    body_a = str(contact.get("body_a", contact.get("a", "")))
    link_a = str(contact.get("link_a", contact.get("linkA", contact.get("link_a_id", ""))))
    body_b = str(contact.get("body_b", contact.get("b", "")))
    link_b = str(contact.get("link_b", contact.get("linkB", contact.get("link_b_id", ""))))
    first = f"{body_a}:{link_a}"
    second = f"{body_b}:{link_b}"
    return tuple(sorted((first, second)))


def _contact_position(contact: Dict[str, Any]) -> Optional[np.ndarray]:
    return _as_array(_first_present(contact, ["position", "point", "contact_point"]))


def _contact_force(contact: Dict[str, Any]) -> Optional[np.ndarray]:
    force = _first_present(contact, ["force", "force_vector", "normal_force_vector"])
    if force is not None:
        return _as_array(force)
    magnitude = _first_present(contact, ["force_magnitude", "normal_force"])
    direction = _as_array(_first_present(contact, ["force_direction", "normal"]))
    if magnitude is None or direction is None:
        return None
    return float(magnitude) * direction


def _mean_matched_contact_error(
    pred_contacts: List[Dict[str, Any]],
    obs_contacts: List[Dict[str, Any]],
    field_fn,
    distance_fn,
) -> float:
    obs_by_key = {_contact_key(item): item for item in obs_contacts}
    values = []
    for pred_item in pred_contacts:
        obs_item = obs_by_key.get(_contact_key(pred_item))
        if obs_item is None:
            continue
        pred_value = field_fn(pred_item)
        obs_value = field_fn(obs_item)
        if pred_value is None or obs_value is None:
            continue
        min_len = min(pred_value.size, obs_value.size)
        values.append(distance_fn(pred_value[:min_len], obs_value[:min_len]))
    if not values:
        return 0.0
    return float(np.mean(values))


def contact_mismatch(
    step: Dict[str, Any],
    prev_obs_contact: Any,
    weights: Dict[str, float],
    point_scale: float,
    force_scale: float,
) -> Optional[float]:
    if "contact_mismatch" in step:
        return float(step["contact_mismatch"])
    contact = step.get("contact", step)
    if "mismatch" in contact:
        return float(contact["mismatch"])
    pred_state = _first_present(contact, ["pred", "predicted", "contact_pred"])
    obs_state = _first_present(contact, ["obs", "observed", "contact_obs"])
    if pred_state is None or obs_state is None:
        return None

    pred_contacts = _contact_items(pred_state)
    obs_contacts = _contact_items(obs_state)
    prev_obs_contacts = _contact_items(prev_obs_contact)

    pred_keys = {_contact_key(item) for item in pred_contacts}
    obs_keys = {_contact_key(item) for item in obs_contacts}
    prev_obs_keys = {_contact_key(item) for item in prev_obs_contacts}

    existence_error = float(bool(pred_contacts) != bool(obs_contacts))
    if pred_keys or obs_keys:
        link_identity_error = 1.0 - len(pred_keys & obs_keys) / max(len(pred_keys | obs_keys), 1)
    else:
        link_identity_error = 0.0

    point_error = _mean_matched_contact_error(
        pred_contacts,
        obs_contacts,
        _contact_position,
        lambda a, b: min(_l2_distance(a, b) / max(point_scale, 1e-12), 1.0),
    )
    force_direction_error = _mean_matched_contact_error(
        pred_contacts,
        obs_contacts,
        _contact_force,
        lambda a, b: max(0.0, min(_cosine_distance(a, b) / 2.0, 1.0)),
    )
    force_magnitude_error = _mean_matched_contact_error(
        pred_contacts,
        obs_contacts,
        _contact_force,
        lambda a, b: min(abs(np.linalg.norm(a) - np.linalg.norm(b)) / max(force_scale, 1e-12), 1.0),
    )
    persistence_break_error = float(bool(prev_obs_keys) and bool(pred_keys & prev_obs_keys) and not bool(obs_keys & prev_obs_keys))

    return float(
        weights["existence"] * existence_error
        + weights["link"] * link_identity_error
        + weights["point"] * point_error
        + weights["force_direction"] * force_direction_error
        + weights["force_magnitude"] * force_magnitude_error
        + weights["persistence"] * persistence_break_error
    )


def load_curves(
    path: Path,
    visual_metric: str,
    contact_weights: Dict[str, float],
    point_scale: float,
    force_scale: float,
    tasks: Optional[set],
) -> List[SignalCurves]:
    curves = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            traj = json.loads(line)
            task = traj.get("task") or traj.get("subtask")
            if tasks is not None and task not in tasks:
                continue
            steps = traj.get("steps", [])
            visual_values = []
            contact_values = []
            prev_obs_contact = None
            for step in steps:
                visual_value = visual_mismatch(step, visual_metric)
                contact_value = contact_mismatch(step, prev_obs_contact, contact_weights, point_scale, force_scale)
                obs_contact = _first_present(step.get("contact", step), ["obs", "observed", "contact_obs"])
                prev_obs_contact = obs_contact if obs_contact is not None else prev_obs_contact
                if visual_value is not None:
                    visual_values.append(visual_value)
                if contact_value is not None:
                    contact_values.append(contact_value)
            if not visual_values and not contact_values:
                raise ValueError(f"No usable mismatch fields found at {path}:{line_number}")
            success = bool(traj.get("success", traj.get("succeeded", False)))
            curves.append(
                SignalCurves(
                    trajectory_id=str(traj.get("trajectory_id", traj.get("id", line_number))),
                    task=str(task),
                    failed=not success,
                    visual=np.asarray(visual_values, dtype=np.float64),
                    contact=np.asarray(contact_values, dtype=np.float64),
                )
            )
    return curves


def _curve_features(values: np.ndarray, early_fraction: float) -> Dict[str, float]:
    if values.size == 0:
        return {"max": 0.0, "mean": 0.0, "early_mean": 0.0, "slope": 0.0}
    early_len = max(1, int(math.ceil(values.size * early_fraction)))
    if values.size >= 2:
        x = np.arange(values.size, dtype=np.float64)
        slope = float(np.polyfit(x, values, deg=1)[0])
    else:
        slope = 0.0
    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "early_mean": float(np.mean(values[:early_len])),
        "slope": slope,
    }


def aggregate_scores(curves: Sequence[SignalCurves], signal: str, early_fraction: float) -> np.ndarray:
    rows = []
    for curve in curves:
        values = getattr(curve, signal)
        feats = _curve_features(values, early_fraction)
        rows.append([feats["max"], feats["mean"], feats["early_mean"], feats["slope"]])
    return np.asarray(rows, dtype=np.float64)


def _standardize(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (values - values.mean(axis=0, keepdims=True)) / std


def score_from_features(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros((features.shape[0],), dtype=np.float64)
    return _standardize(features).mean(axis=1)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(labels) == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.sum(labels == 1)
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * recall_delta))


def best_threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    best = {
        "F1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": float("nan"),
    }
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["F1"]:
            best = {
                "F1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold),
            }
    return best


def threshold_from_success(curves: Sequence[SignalCurves], signal: str, quantile: float) -> float:
    success_values = []
    for curve in curves:
        if not curve.failed:
            values = getattr(curve, signal)
            success_values.extend(values.tolist())
    if not success_values:
        all_values = np.concatenate([getattr(curve, signal) for curve in curves if getattr(curve, signal).size > 0])
        return float(np.quantile(all_values, quantile))
    return float(np.quantile(np.asarray(success_values, dtype=np.float64), quantile))


def first_alarm(values: np.ndarray, threshold: float) -> Optional[int]:
    indices = np.flatnonzero(values > threshold)
    if indices.size == 0:
        return None
    return int(indices[0])


def warning_stats(curves: Sequence[SignalCurves], signal: str, threshold: float) -> Dict[str, float]:
    leads = []
    detected = 0
    false_alarms = 0
    successes = 0
    failures = 0
    for curve in curves:
        values = getattr(curve, signal)
        alarm = first_alarm(values, threshold)
        if curve.failed:
            failures += 1
            if alarm is not None:
                detected += 1
                leads.append(max(values.size - 1 - alarm, 0))
        else:
            successes += 1
            if alarm is not None:
                false_alarms += 1
    return {
        "threshold": threshold,
        "failure_detection_rate": detected / max(failures, 1),
        "success_false_alarm_rate": false_alarms / max(successes, 1),
        "avg_warning_lead_steps": float(np.mean(leads)) if leads else float("nan"),
        "median_warning_lead_steps": float(np.median(leads)) if leads else float("nan"),
    }


def classification_table(curves: Sequence[SignalCurves], early_fraction: float) -> List[Dict[str, float]]:
    labels = np.asarray([int(curve.failed) for curve in curves], dtype=np.int64)
    visual_features = aggregate_scores(curves, "visual", early_fraction)
    contact_features = aggregate_scores(curves, "contact", early_fraction)
    scores = {
        "visual_mismatch": score_from_features(visual_features),
        "contact_mismatch": score_from_features(contact_features),
        "visual_plus_contact": score_from_features(np.concatenate([visual_features, contact_features], axis=1)),
    }
    rows = []
    for name, score in scores.items():
        threshold_metrics = best_threshold_metrics(labels, score)
        rows.append(
            {
                "signal": name,
                "AUROC": auroc(labels, score),
                "AUPRC": auprc(labels, score),
                "F1": threshold_metrics["F1"],
                "precision": threshold_metrics["precision"],
                "recall": threshold_metrics["recall"],
                "best_threshold": threshold_metrics["threshold"],
            }
        )
    return rows


def write_markdown_report(path: Path, curves: Sequence[SignalCurves], warning: Dict[str, Dict[str, float]], table: List[Dict[str, float]]) -> None:
    failures = sum(curve.failed for curve in curves)
    tasks = sorted({curve.task for curve in curves})
    lines = [
        "# Failure Warning Analysis",
        "",
        f"Trajectories: {len(curves)}",
        f"Failures: {failures}",
        f"Tasks: {', '.join(tasks)}",
        "",
        "## Early Warning",
        "",
        "| Signal | Threshold | Detection | False Alarm | Avg Lead | Median Lead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for signal, stats in warning.items():
        lines.append(
            f"| {signal} | {stats['threshold']:.6g} | {stats['failure_detection_rate']:.3f} | "
            f"{stats['success_false_alarm_rate']:.3f} | {stats['avg_warning_lead_steps']:.3f} | "
            f"{stats['median_warning_lead_steps']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Success/Failure Classification",
            "",
            "| Signal | AUROC | AUPRC | F1 | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table:
        lines.append(
            f"| {row['signal']} | {row['AUROC']:.3f} | {row['AUPRC']:.3f} | {row['F1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_tasks(value: str) -> Optional[set]:
    if value == "all":
        return None
    if value == "contact_rich":
        return CONTACT_RICH_TASKS
    if value == "non_contact_rich":
        return NON_CONTACT_RICH_TASKS
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare visual/contact mismatch as failure early-warning signals.")
    parser.add_argument("--trajectories", type=Path, required=True, help="JSONL trajectories with per-step predictions/observations.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="all", help="all, contact_rich, non_contact_rich, or comma-separated task names.")
    parser.add_argument("--visual-metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--success-quantile", type=float, default=0.95)
    parser.add_argument("--early-fraction", type=float, default=0.33)
    parser.add_argument("--point-scale", type=float, default=0.05)
    parser.add_argument("--force-scale", type=float, default=10.0)
    parser.add_argument("--lambda-existence", type=float, default=1.0)
    parser.add_argument("--lambda-link", type=float, default=1.0)
    parser.add_argument("--lambda-point", type=float, default=1.0)
    parser.add_argument("--lambda-force-direction", type=float, default=1.0)
    parser.add_argument("--lambda-force-magnitude", type=float, default=0.5)
    parser.add_argument("--lambda-persistence", type=float, default=1.0)
    args = parser.parse_args()

    weights = {
        "existence": args.lambda_existence,
        "link": args.lambda_link,
        "point": args.lambda_point,
        "force_direction": args.lambda_force_direction,
        "force_magnitude": args.lambda_force_magnitude,
        "persistence": args.lambda_persistence,
    }
    curves = load_curves(
        args.trajectories,
        args.visual_metric,
        weights,
        args.point_scale,
        args.force_scale,
        parse_tasks(args.tasks),
    )
    if not curves:
        raise ValueError("No trajectories matched the requested task filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    warning = {
        "visual_mismatch": warning_stats(curves, "visual", threshold_from_success(curves, "visual", args.success_quantile)),
        "contact_mismatch": warning_stats(curves, "contact", threshold_from_success(curves, "contact", args.success_quantile)),
    }
    table = classification_table(curves, args.early_fraction)
    report = {"warning": warning, "classification": table}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown_report(args.output_dir / "report.md", curves, warning, table)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
