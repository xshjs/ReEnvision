import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
class Example:
    trajectory_id: str
    task: str
    success: int
    visual: np.ndarray
    contact: np.ndarray


def as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0 if np.linalg.norm(a - b) <= 1e-12 else 1.0
    return float(1.0 - np.dot(a, b) / denom)


def summarize_curve(values: Sequence[float], prefix: str) -> Tuple[List[float], List[str]]:
    arr = np.asarray(values, dtype=np.float64)
    names = [f"{prefix}_mean", f"{prefix}_max", f"{prefix}_std", f"{prefix}_first", f"{prefix}_last", f"{prefix}_slope"]
    if arr.size == 0:
        return [0.0] * len(names), names
    slope = 0.0
    if arr.size >= 2:
        slope = float(np.polyfit(np.arange(arr.size, dtype=np.float64), arr, deg=1)[0])
    return [float(arr.mean()), float(arr.max()), float(arr.std()), float(arr[0]), float(arr[-1]), slope], names


def pair_key(contact: Dict[str, Any]) -> str:
    first = f"{contact.get('body_a', '')}:{contact.get('link_a', '')}"
    second = f"{contact.get('body_b', '')}:{contact.get('link_b', '')}"
    return "|".join(sorted((first, second)))


def contact_flags(contact: Dict[str, Any]) -> Dict[str, float]:
    body_a = str(contact.get("body_a", ""))
    body_b = str(contact.get("body_b", ""))
    link_a = str(contact.get("link_a", ""))
    link_b = str(contact.get("link_b", ""))
    bodies = {body_a, body_b}
    links = {link_a, link_b}
    block_count = sum("block" in body for body in bodies)
    return {
        "has_robot": float("robot" in bodies),
        "has_drawer_link": float("drawer_link" in links),
        "has_plank_link": float("plank_link" in links),
        "has_table": float("table" in bodies or "base_link" in links),
        "block_block": float(block_count >= 2),
        "block_fixed": float(block_count == 1),
    }


def force_magnitude(contact: Dict[str, Any]) -> float:
    if "force_magnitude" in contact:
        return float(contact["force_magnitude"])
    force = as_array(contact.get("force"))
    if force is None:
        return 0.0
    return float(np.linalg.norm(force))


def visual_features(steps: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    cosine_values = []
    l2_values = []
    pred_norms = []
    obs_norms = []
    for step in steps:
        visual = step.get("visual", {})
        pred = as_array(visual.get("pred"))
        obs = as_array(visual.get("obs"))
        if pred is None or obs is None:
            continue
        min_len = min(pred.size, obs.size)
        pred = pred[:min_len]
        obs = obs[:min_len]
        cosine_values.append(cosine_distance(pred, obs))
        l2_values.append(float(np.linalg.norm(pred - obs) / math.sqrt(max(min_len, 1))))
        pred_norms.append(float(np.linalg.norm(pred) / math.sqrt(max(min_len, 1))))
        obs_norms.append(float(np.linalg.norm(obs) / math.sqrt(max(min_len, 1))))
    features = []
    names = []
    for values, prefix in [
        (cosine_values, "vis_cos"),
        (l2_values, "vis_l2"),
        (pred_norms, "vis_pred_norm"),
        (obs_norms, "vis_obs_norm"),
    ]:
        vals, val_names = summarize_curve(values, prefix)
        features.extend(vals)
        names.extend(val_names)
    return np.asarray(features, dtype=np.float64), names


def contact_features(steps: List[Dict[str, Any]], contact_feature_set: str) -> Tuple[np.ndarray, List[str]]:
    counts = []
    unique_counts = []
    force_means = []
    force_maxes = []
    persistence_jaccard = []
    flag_curves = defaultdict(list)
    previous_pairs: Optional[set] = None
    all_pairs = set()
    for step in steps:
        contacts = step.get("contact", {}).get("obs", [])
        if not isinstance(contacts, list):
            contacts = []
        pairs = {pair_key(contact) for contact in contacts if isinstance(contact, dict)}
        all_pairs |= pairs
        counts.append(float(len(contacts)))
        unique_counts.append(float(len(pairs)))
        forces = [force_magnitude(contact) for contact in contacts if isinstance(contact, dict)]
        force_means.append(float(np.mean(forces)) if forces else 0.0)
        force_maxes.append(float(np.max(forces)) if forces else 0.0)
        if previous_pairs is None:
            persistence_jaccard.append(1.0)
        else:
            union = len(previous_pairs | pairs)
            persistence_jaccard.append(float(len(previous_pairs & pairs) / union) if union else 1.0)
        previous_pairs = pairs
        step_flags = defaultdict(float)
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            for name, value in contact_flags(contact).items():
                step_flags[name] += value
        if contact_feature_set == "full":
            flag_names = ["has_robot", "has_drawer_link", "has_plank_link", "has_table", "block_block", "block_fixed"]
        elif contact_feature_set == "safe":
            flag_names = ["has_robot", "has_table", "block_fixed"]
        elif contact_feature_set == "minimal":
            flag_names = []
        else:
            raise ValueError(f"Unknown contact feature set: {contact_feature_set}")
        for name in flag_names:
            flag_curves[name].append(step_flags[name])

    features = [float(len(all_pairs))]
    names = ["contact_total_unique_pairs"]
    for values, prefix in [
        (counts, "contact_count"),
        (unique_counts, "contact_unique"),
        (force_means, "contact_force_mean"),
        (force_maxes, "contact_force_max"),
        (persistence_jaccard, "contact_persistence"),
    ]:
        vals, val_names = summarize_curve(values, prefix)
        features.extend(vals)
        names.extend(val_names)
    for flag_name in sorted(flag_curves):
        vals, val_names = summarize_curve(flag_curves[flag_name], f"contact_{flag_name}")
        features.extend(vals)
        names.extend(val_names)
    return np.asarray(features, dtype=np.float64), names


def parse_tasks(value: str) -> Optional[set]:
    if value == "all":
        return None
    if value == "contact_rich":
        return CONTACT_RICH_TASKS
    if value == "non_contact_rich":
        return NON_CONTACT_RICH_TASKS
    return {item.strip() for item in value.split(",") if item.strip()}


def load_examples(
    path: Path,
    tasks: Optional[set],
    prefix_steps: Optional[int],
    prefix_fraction: Optional[float],
    require_prefix_steps: bool,
    contact_feature_set: str,
) -> Tuple[List[Example], List[str], List[str]]:
    examples = []
    visual_names = []
    contact_names = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task"))
            if tasks is not None and task not in tasks:
                continue
            steps = row.get("steps", [])
            if prefix_fraction is not None:
                if not 0 < prefix_fraction <= 1:
                    raise ValueError("--prefix-fraction must be in (0, 1].")
                prefix_len = max(1, int(math.ceil(len(steps) * prefix_fraction)))
                steps = steps[:prefix_len]
            if prefix_steps is not None:
                if require_prefix_steps and len(steps) < prefix_steps:
                    continue
                steps = steps[:prefix_steps]
            visual, visual_names = visual_features(steps)
            contact, contact_names = contact_features(steps, contact_feature_set)
            examples.append(
                Example(
                    trajectory_id=str(row.get("trajectory_id", line_number)),
                    task=task,
                    success=int(bool(row.get("success", False))),
                    visual=visual,
                    contact=contact,
                )
            )
    return examples, visual_names, contact_names


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def fit_predict_logistic(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float, steps: int, lr: float) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full((x_test.shape[0],), float(np.mean(y_train)), dtype=np.float64)
    x_train, x_test = standardize_train_test(x_train, x_test)
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    weights = np.zeros((x_train.shape[1],), dtype=np.float64)
    for _ in range(steps):
        probs = sigmoid(x_train @ weights)
        grad = (x_train.T @ (probs - y_train)) / len(y_train)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return sigmoid(x_test @ weights)


def leave_one_out_scores(features: np.ndarray, labels: np.ndarray, l2: float, steps: int, lr: float) -> np.ndarray:
    scores = np.zeros((features.shape[0],), dtype=np.float64)
    for idx in range(features.shape[0]):
        train_mask = np.ones((features.shape[0],), dtype=bool)
        train_mask[idx] = False
        scores[idx] = fit_predict_logistic(
            features[train_mask],
            labels[train_mask],
            features[idx : idx + 1],
            l2=l2,
            steps=steps,
            lr=lr,
        )[0]
    return scores


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(labels == 1) == 0:
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
    best = {"F1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": float("nan")}
    for threshold in np.unique(scores):
        pred = scores >= threshold
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["F1"]:
            best = {"F1": float(f1), "precision": float(precision), "recall": float(recall), "threshold": float(threshold)}
    return best


def metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    threshold = best_threshold_metrics(labels, scores)
    return {
        "AUROC": auroc(labels, scores),
        "AUPRC": auprc(labels, scores),
        "F1": threshold["F1"],
        "precision": threshold["precision"],
        "recall": threshold["recall"],
        "best_threshold": threshold["threshold"],
    }


def finite_greater(left: float, right: float, eps: float = 1e-9) -> bool:
    return np.isfinite(left) and np.isfinite(right) and left > right + eps


def write_report(path: Path, examples: List[Example], aggregate: Dict[str, Dict[str, float]], per_task: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    task_counts = Counter(example.task for example in examples)
    success_counts = Counter((example.task, example.success) for example in examples)
    methods = ["Visual", "Contact", "Visual+Contact", "Visual+ContactFeatures"]
    valid_tasks = [task for task, rows in per_task.items() if np.isfinite(rows["Visual"]["AUROC"]) and np.isfinite(rows["Visual+Contact"]["AUROC"])]
    improved_tasks = [task for task in valid_tasks if finite_greater(per_task[task]["Visual+Contact"]["AUROC"], per_task[task]["Visual"]["AUROC"])]
    mean_task_auroc = {
        method: float(np.mean([per_task[task][method]["AUROC"] for task in valid_tasks])) if valid_tasks else float("nan")
        for method in methods
    }
    lines = [
        "# Learned Skill Verifier Analysis",
        "",
        f"Trajectories: {len(examples)}",
        f"Successes: {sum(example.success for example in examples)}",
        f"Failures: {sum(1 - example.success for example in examples)}",
        "Positive label: skill success",
        "",
        "## Aggregate",
        "",
        "| Method | AUROC | AUPRC | F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = aggregate[method]
        lines.append(
            f"| {method} | {row['AUROC']:.3f} | {row['AUPRC']:.3f} | {row['F1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Task Improvement",
            "",
            f"Visual+Contact improved over Visual on AUROC for {len(improved_tasks)}/{len(valid_tasks)} valid tasks.",
            f"Improved tasks: {', '.join(improved_tasks) if improved_tasks else 'none'}",
            "",
            "| Method | Mean Per-Task AUROC |",
            "|---|---:|",
        ]
    )
    for method in methods:
        lines.append(f"| {method} | {mean_task_auroc[method]:.3f} |")
    lines.extend(
        [
            "",
            "## Per Task",
            "",
            "| Task | N | Success | Method | AUROC | AUPRC | F1 | Precision | Recall |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in sorted(per_task):
        n = task_counts[task]
        n_success = success_counts[(task, 1)]
        for method in methods:
            row = per_task[task][method]
            lines.append(
                f"| {task} | {n} | {n_success} | {method} | {row['AUROC']:.3f} | {row['AUPRC']:.3f} | "
                f"{row['F1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate learned visual/contact skill verifiers with leave-one-out CV.")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--prefix-steps", type=int, default=None, help="Use only the first K recorded steps to avoid post-outcome leakage.")
    parser.add_argument("--prefix-fraction", type=float, default=None, help="Use only the first fraction of recorded trajectory steps, e.g. 0.1, 0.2, 0.3, 0.5.")
    parser.add_argument("--require-prefix-steps", action="store_true", help="Drop trajectories with fewer than --prefix-steps records.")
    parser.add_argument(
        "--contact-feature-set",
        choices=["full", "safe", "minimal"],
        default="full",
        help="full includes semantic contact flags; safe removes oracle-like drawer_link/block_block flags; minimal keeps only counts/force/persistence.",
    )
    args = parser.parse_args()

    examples, visual_names, contact_names = load_examples(
        args.trajectories,
        parse_tasks(args.tasks),
        args.prefix_steps,
        args.prefix_fraction,
        args.require_prefix_steps,
        args.contact_feature_set,
    )
    if len(examples) < 3:
        raise ValueError("Need at least 3 trajectories for leave-one-out verifier evaluation.")
    labels = np.asarray([example.success for example in examples], dtype=np.float64)
    visual = np.stack([example.visual for example in examples], axis=0)
    contact = np.stack([example.contact for example in examples], axis=0)
    combined = np.concatenate([visual, contact], axis=1)
    feature_sets = {"Visual": visual, "Contact": contact, "Visual+ContactFeatures": combined}
    scores = {
        name: leave_one_out_scores(values, labels, l2=args.l2, steps=args.steps, lr=args.lr)
        for name, values in feature_sets.items()
    }
    scores["Visual+Contact"] = 0.5 * (scores["Visual"] + scores["Contact"])
    aggregate = {name: metrics(labels.astype(np.int64), score) for name, score in scores.items()}

    per_task = {}
    tasks = sorted({example.task for example in examples})
    for task in tasks:
        mask = np.asarray([example.task == task for example in examples], dtype=bool)
        per_task[task] = {name: metrics(labels[mask].astype(np.int64), score[mask]) for name, score in scores.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "positive_label": "skill_success",
        "n": len(examples),
        "prefix_steps": args.prefix_steps,
        "prefix_fraction": args.prefix_fraction,
        "require_prefix_steps": args.require_prefix_steps,
        "contact_feature_set": args.contact_feature_set,
        "visual_feature_names": visual_names,
        "contact_feature_names": contact_names,
        "aggregate": aggregate,
        "per_task": per_task,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(args.output_dir / "report.md", examples, aggregate, per_task)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
