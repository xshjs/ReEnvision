import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.failure_warning_analysis import (  # noqa: E402
    contact_mismatch,
    visual_mismatch,
)


REFLECTION_CLASSES = ["stable", "drifting", "high-error", "recovering"]


@dataclass
class ReflectionExample:
    trajectory_id: str
    task: str
    failed: int
    features: np.ndarray
    error_values: np.ndarray
    error_max: float
    drift_slope: float
    future_history_consistency: float
    reflection_label: int
    reflection_score_target: float
    first_contradiction: Optional[int]


class ReflectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, num_classes)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.state_head(h), self.score_head(h).squeeze(-1)


def _curve_stats(values: np.ndarray, prefix: str) -> Tuple[List[float], List[str]]:
    names = [
        f"{prefix}_mean",
        f"{prefix}_max",
        f"{prefix}_std",
        f"{prefix}_first",
        f"{prefix}_last",
        f"{prefix}_slope",
        f"{prefix}_range",
        f"{prefix}_mean_abs_delta",
    ]
    if values.size == 0:
        return [0.0] * len(names), names
    slope = float(np.polyfit(np.arange(values.size, dtype=np.float64), values, deg=1)[0]) if values.size >= 2 else 0.0
    mean_abs_delta = float(np.mean(np.abs(np.diff(values)))) if values.size >= 2 else 0.0
    vals = [
        float(np.mean(values)),
        float(np.max(values)),
        float(np.std(values)),
        float(values[0]),
        float(values[-1]),
        slope,
        float(np.max(values) - np.min(values)),
        mean_abs_delta,
    ]
    return vals, names


def _normalize01(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(labels == 1) == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(np.sum(labels == 1), 1)
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * recall_delta))


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    best = {"F1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": float("nan")}
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["F1"]:
            best = {"F1": float(f1), "precision": float(precision), "recall": float(recall), "threshold": float(threshold)}
    return best


def _load_contradictions(path: Optional[Path]) -> Tuple[Dict[str, int], str]:
    if path is None:
        return {}, "derived_from_error_threshold"
    first_by_id: Dict[str, int] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            trajectory_id = str(row.get("trajectory_id", row.get("id", line_number)))
            first = None
            for step in row.get("steps", []):
                count = int(step.get("count", len(step.get("contradictions", []))))
                if count > 0:
                    first = int(step.get("t", 0))
                    break
            if first is not None:
                first_by_id[trajectory_id] = first
    return first_by_id, str(path)


def _extract_raw(path: Path, visual_metric: str) -> List[Dict[str, Any]]:
    rows = []
    contact_weights = {
        "existence": 1.0,
        "link": 1.0,
        "point": 1.0,
        "force_direction": 1.0,
        "force_magnitude": 0.5,
        "persistence": 1.0,
    }
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            visual_values = []
            contact_values = []
            prev_obs_contact = None
            for step in row.get("steps", []):
                visual_value = visual_mismatch(step, visual_metric)
                contact_value = contact_mismatch(step, prev_obs_contact, contact_weights, 0.05, 10.0)
                if visual_value is not None:
                    visual_values.append(float(visual_value))
                if contact_value is not None:
                    contact_values.append(float(contact_value))
                contact = step.get("contact", step)
                prev_obs_contact = contact.get("obs", contact.get("observed", prev_obs_contact))
            if not visual_values:
                continue
            rows.append(
                {
                    "trajectory_id": str(row.get("trajectory_id", row.get("id", line_number))),
                    "task": str(row.get("task", row.get("subtask", ""))),
                    "failed": int(not bool(row.get("success", row.get("succeeded", False)))),
                    "visual": np.asarray(visual_values, dtype=np.float64),
                    "contact": np.asarray(contact_values, dtype=np.float64),
                }
            )
    return rows


def build_examples(path: Path, contradiction_path: Optional[Path], visual_metric: str, high_error_quantile: float) -> Tuple[List[ReflectionExample], List[str], Dict[str, Any]]:
    raw = _extract_raw(path, visual_metric)
    if not raw:
        raise ValueError(f"No usable trajectories found in {path}")
    contradiction_first, contradiction_source = _load_contradictions(contradiction_path)
    error_maxes = np.asarray([float(np.max(item["visual"])) for item in raw], dtype=np.float64)
    slopes = np.asarray([
        float(np.polyfit(np.arange(item["visual"].size, dtype=np.float64), item["visual"], deg=1)[0]) if item["visual"].size >= 2 else 0.0
        for item in raw
    ])
    consistencies = np.asarray([
        1.0 / (1.0 + (float(np.mean(np.abs(np.diff(item["visual"])))) if item["visual"].size >= 2 else 0.0))
        for item in raw
    ])
    high_error_threshold = float(np.quantile(error_maxes, high_error_quantile))
    recovery_slope_threshold = max(float(np.quantile(np.abs(slopes), 0.75)), 1e-8)
    tasks = sorted({item["task"] for item in raw})
    task_index = {task: idx for idx, task in enumerate(tasks)}
    risk = (
        0.35 * _normalize01(error_maxes)
        + 0.25 * _normalize01(np.maximum(slopes, 0.0))
        + 0.20 * _normalize01(1.0 - consistencies)
        + 0.20 * np.asarray([item["failed"] for item in raw], dtype=np.float64)
    )

    examples: List[ReflectionExample] = []
    feature_names: List[str] = []
    for idx, item in enumerate(raw):
        visual_stats, visual_names = _curve_stats(item["visual"], "error")
        contact_stats, contact_names = _curve_stats(item["contact"], "contact")
        task_one_hot = [0.0] * len(tasks)
        task_one_hot[task_index[item["task"]]] = 1.0
        names = visual_names + contact_names + [f"task={task}" for task in tasks]
        if not feature_names:
            feature_names = names
        error_max = float(error_maxes[idx])
        slope = float(slopes[idx])
        first = contradiction_first.get(item["trajectory_id"])
        if first is None:
            high_indices = np.flatnonzero(item["visual"] >= high_error_threshold)
            first = int(high_indices[0]) if high_indices.size else None
        if error_max >= high_error_threshold:
            label = REFLECTION_CLASSES.index("high-error")
        elif slope > 0.0:
            label = REFLECTION_CLASSES.index("drifting")
        elif slope < -recovery_slope_threshold and float(item["visual"][-1]) < float(item["visual"][0]):
            label = REFLECTION_CLASSES.index("recovering")
        else:
            label = REFLECTION_CLASSES.index("stable")
        examples.append(
            ReflectionExample(
                trajectory_id=item["trajectory_id"],
                task=item["task"],
                failed=item["failed"],
                features=np.asarray(visual_stats + contact_stats + task_one_hot, dtype=np.float64),
                error_values=item["visual"],
                error_max=error_max,
                drift_slope=slope,
                future_history_consistency=float(consistencies[idx]),
                reflection_label=label,
                reflection_score_target=float(risk[idx]),
                first_contradiction=first,
            )
        )
    metadata = {
        "label_rule": {
            "classes": REFLECTION_CLASSES,
            "high_error_threshold": high_error_threshold,
            "high_error_quantile": high_error_quantile,
            "drift_positive_threshold": 0.0,
            "recovery_abs_slope_threshold": recovery_slope_threshold,
            "reflection_score_target": "0.35*norm(error_max)+0.25*norm(positive_drift_slope)+0.20*norm(1-future_history_consistency)+0.20*failure",
            "future_history_consistency": "1 / (1 + mean_abs_delta(error_curve))",
            "first_contradiction_source": contradiction_source,
        },
        "tasks": tasks,
    }
    return examples, feature_names, metadata


def _make_folds(labels: np.ndarray, folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        buckets[int(label)].append(idx)
    fold_indices = [[] for _ in range(folds)]
    for indices in buckets.values():
        rng.shuffle(indices)
        for pos, idx in enumerate(indices):
            fold_indices[pos % folds].append(idx)
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in fold_indices if indices]


def train_predict_cv(examples: Sequence[ReflectionExample], folds: int, epochs: int, seed: int, hidden_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.stack([example.features for example in examples], axis=0).astype(np.float32)
    y_class = np.asarray([example.reflection_label for example in examples], dtype=np.int64)
    y_score = np.asarray([example.reflection_score_target for example in examples], dtype=np.float32)
    pred_class_prob = np.zeros((x.shape[0], len(REFLECTION_CLASSES)), dtype=np.float64)
    pred_score = np.zeros((x.shape[0],), dtype=np.float64)
    fold_indices = _make_folds(y_class, min(folds, max(2, x.shape[0])), seed)
    for fold_id, val_idx in enumerate(fold_indices):
        train_idx = np.setdiff1d(np.arange(x.shape[0]), val_idx)
        mean = x[train_idx].mean(axis=0, keepdims=True)
        std = x[train_idx].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        tx = torch.from_numpy((x[train_idx] - mean) / std).float()
        vx = torch.from_numpy((x[val_idx] - mean) / std).float()
        ty_class = torch.from_numpy(y_class[train_idx]).long()
        ty_score = torch.from_numpy(y_score[train_idx]).float()
        torch.manual_seed(seed + fold_id)
        model = ReflectionHead(x.shape[1], hidden_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        for _ in range(epochs):
            model.train()
            logits, score_logits = model(tx)
            loss = nn.CrossEntropyLoss()(logits, ty_class) + nn.MSELoss()(torch.sigmoid(score_logits), ty_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, score_logits = model(vx)
            pred_class_prob[val_idx] = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_score[val_idx] = torch.sigmoid(score_logits).cpu().numpy()
    return pred_class_prob, pred_score


def evaluate(examples: Sequence[ReflectionExample], class_prob: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    labels = {
        "failure": np.asarray([example.failed for example in examples], dtype=np.int64),
        "first_contradiction": np.asarray([int(example.first_contradiction is not None) for example in examples], dtype=np.int64),
        "high_error_max": np.asarray([
            int(example.reflection_label == REFLECTION_CLASSES.index("high-error")) for example in examples
        ], dtype=np.int64),
        "positive_drift_slope": np.asarray([int(example.drift_slope > 0.0) for example in examples], dtype=np.int64),
    }
    metrics = {}
    high_error_prob = class_prob[:, REFLECTION_CLASSES.index("high-error")]
    drifting_prob = class_prob[:, REFLECTION_CLASSES.index("drifting")]
    combined = np.maximum(score, np.maximum(high_error_prob, drifting_prob))
    scores = {
        "reflection_score": score,
        "high_error_prob": high_error_prob,
        "drifting_prob": drifting_prob,
        "combined_reflection_risk": combined,
    }
    for target, y in labels.items():
        target_rows = {}
        for score_name, values in scores.items():
            f1 = _best_f1(y, values)
            target_rows[score_name] = {
                "AUROC": _auroc(y, values),
                "AUPRC": _auprc(y, values),
                **f1,
            }
        metrics[target] = target_rows
    y_class = np.asarray([example.reflection_label for example in examples], dtype=np.int64)
    pred_class = np.argmax(class_prob, axis=1)
    metrics["reflection_state_accuracy"] = float(np.mean(pred_class == y_class))
    metrics["reflection_state_distribution"] = {
        REFLECTION_CLASSES[idx]: int(count) for idx, count in Counter(y_class.tolist()).items()
    }
    return metrics


def write_report(path: Path, examples: Sequence[ReflectionExample], metrics: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    failures = sum(example.failed for example in examples)
    lines = [
        "# Reflection Head Analysis",
        "",
        f"Trajectories: {len(examples)}",
        f"Failures: {failures}",
        f"Tasks: {', '.join(metadata['tasks'])}",
        "",
        "## Label Rule",
        "",
        f"- states: {', '.join(REFLECTION_CLASSES)}",
        f"- high_error_threshold: {metadata['label_rule']['high_error_threshold']:.6g}",
        f"- drift_positive_threshold: {metadata['label_rule']['drift_positive_threshold']:.6g}",
        f"- recovery_abs_slope_threshold: {metadata['label_rule']['recovery_abs_slope_threshold']:.6g}",
        f"- future_history_consistency: {metadata['label_rule']['future_history_consistency']}",
        "",
        "## Target Prediction",
        "",
        "| Target | Best score | AUROC | AUPRC | F1 | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for target, score_rows in metrics.items():
        if target.startswith("reflection_state"):
            continue
        best_name, best = max(score_rows.items(), key=lambda item: (-1 if math.isnan(item[1]["AUROC"]) else item[1]["AUROC"]))
        lines.append(
            f"| {target} | {best_name} | {best['AUROC']:.3f} | {best['AUPRC']:.3f} | "
            f"{best['F1']:.3f} | {best['precision']:.3f} | {best['recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reflection State",
            "",
            f"- CV accuracy: {metrics['reflection_state_accuracy']:.3f}",
            f"- distribution: {metrics['reflection_state_distribution']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate an explicit Reflection Head from rollout statistics.")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contradictions", type=Path, default=None)
    parser.add_argument("--visual-metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--high-error-quantile", type=float, default=0.75)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples, feature_names, metadata = build_examples(args.trajectories, args.contradictions, args.visual_metric, args.high_error_quantile)
    class_prob, score = train_predict_cv(examples, args.folds, args.epochs, args.seed, args.hidden_dim)
    metrics = evaluate(examples, class_prob, score)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, example in enumerate(examples):
        rows.append(
            {
                "trajectory_id": example.trajectory_id,
                "task": example.task,
                "failed": bool(example.failed),
                "error_max": example.error_max,
                "drift_slope": example.drift_slope,
                "future_history_consistency": example.future_history_consistency,
                "reflection_label": REFLECTION_CLASSES[example.reflection_label],
                "reflection_score": float(score[idx]),
                "reflection_probs": {
                    REFLECTION_CLASSES[class_idx]: float(class_prob[idx, class_idx]) for class_idx in range(len(REFLECTION_CLASSES))
                },
                "first_contradiction": example.first_contradiction,
            }
        )
    (args.output_dir / "metrics.json").write_text(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2) + "\n")
    (args.output_dir / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (args.output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2) + "\n")
    write_report(args.output_dir / "report.md", examples, metrics, metadata)
    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
