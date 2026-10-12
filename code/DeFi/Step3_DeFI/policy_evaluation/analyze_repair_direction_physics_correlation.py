import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _norm_list(value):
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return 0.0
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))


def _read_jsonl(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _summary_norm(summary):
    if not isinstance(summary, dict):
        return 0.0
    if "l2" in summary:
        return _safe_float(summary.get("l2"))
    if "norm" in summary:
        return _safe_float(summary.get("norm"))
    return 0.0


def _object_motion_norm(row):
    motion = row.get("object_motion") or row.get("block_displacement")
    if motion is None:
        return 0.0
    if isinstance(motion, dict):
        vals = []
        for value in motion.values():
            if isinstance(value, dict):
                for key in ("delta", "pos_delta", "translation_delta", "position_delta", "l2", "norm"):
                    if key in value:
                        vals.append(_norm_list(value[key]) if not isinstance(value[key], (int, float)) else abs(float(value[key])))
            elif isinstance(value, (list, tuple)):
                vals.append(_norm_list(value))
            elif isinstance(value, (int, float)):
                vals.append(abs(float(value)))
        return float(max(vals)) if vals else 0.0
    if isinstance(motion, (list, tuple)):
        return _norm_list(motion)
    return 0.0


def _progress_norm(row):
    delta = row.get("drawer_slider_progress_delta")
    if not isinstance(delta, dict):
        return 0.0
    vals = []
    for value in delta.values():
        if isinstance(value, (int, float)):
            vals.append(abs(float(value)))
        elif isinstance(value, (list, tuple)):
            vals.append(_norm_list(value))
        elif isinstance(value, dict):
            vals.extend(abs(_safe_float(v)) for v in value.values() if isinstance(v, (int, float)))
    return float(np.linalg.norm(vals)) if vals else 0.0


def _contact_delta(row):
    contact = row.get("contact_state")
    if not isinstance(contact, dict):
        return 0.0
    start = contact.get("start_contacts") or contact.get("start_contact_sets") or []
    end = contact.get("end_contacts") or contact.get("end_contact_sets") or []
    try:
        return float(abs(len(end) - len(start)))
    except Exception:
        return 0.0


def _gripper_delta(row):
    start = row.get("gripper_pose_start") or {}
    end = row.get("gripper_pose_end") or {}
    vals = []
    for key in ("tcp_pos", "tcp_orn", "gripper_width_or_action"):
        if key in start and key in end:
            vals.append(_norm_list(np.asarray(end[key], dtype=np.float32) - np.asarray(start[key], dtype=np.float32)))
    return float(np.linalg.norm(vals)) if vals else _summary_norm(row.get("strong_gt_robot_obs_delta"))


def _scene_delta(row):
    return _summary_norm(row.get("strong_gt_scene_obs_delta"))


def _goal_value(row):
    goal = row.get("goal_predicate")
    if isinstance(goal, dict):
        return 1.0 if goal.get("success") or goal.get("strong_gt_success") else 0.0
    return 1.0 if row.get("success") else 0.0


def _load_trace_future(row):
    path = row.get("collection_trace_path") or row.get("trace_path") or row.get("trace")
    if not path or not Path(path).exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    for key in ("base_future", "original_base_future"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32)
            return arr.mean(axis=0) if arr.ndim == 2 else arr.reshape(-1)
    return None


def _cosine_matrix(query, keys):
    query = query.astype(np.float32)
    keys = keys.astype(np.float32)
    q = query / max(float(np.linalg.norm(query)), 1e-8)
    kn = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8)
    return kn @ q


def _repair_direction_from_memory(base, task, memory, topk, same_task_only):
    failure = memory["failure_features"].astype(np.float32)
    deltas = memory["attract_deltas"].astype(np.float32)
    tasks = memory["tasks"].astype(str) if "tasks" in memory.files else None
    mask = np.ones((failure.shape[0],), dtype=bool)
    if same_task_only and tasks is not None:
        mask = tasks == str(task)
        if not mask.any():
            mask = np.ones((failure.shape[0],), dtype=bool)
    idx = np.where(mask)[0]
    sims = _cosine_matrix(base, failure[idx])
    order = idx[np.argsort(-sims)[: max(1, topk)]]
    weights = np.maximum(sims[np.argsort(-sims)[: max(1, topk)]], 0.0)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones_like(weights) / max(1, len(weights))
    else:
        weights = weights / weights.sum()
    direction = (deltas[order] * weights[:, None]).sum(axis=0)
    return direction, order.tolist(), weights.tolist()


def _group_norms(direction, group_size):
    flat = direction.reshape(-1).astype(np.float32)
    groups = []
    for start in range(0, flat.size, group_size):
        groups.append(float(np.linalg.norm(flat[start : start + group_size])))
    return np.asarray(groups, dtype=np.float32)


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-gt-jsonl", required=True)
    parser.add_argument("--immune-memory-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = _read_jsonl(args.strong_gt_jsonl)
    tasks = {x.strip() for x in args.task_filter.split(",") if x.strip()}
    if tasks:
        rows = [r for r in rows if str(r.get("task")) in tasks]
    if args.limit > 0:
        rows = rows[: args.limit]

    memory = np.load(args.immune_memory_npz, allow_pickle=True)
    examples = []
    group_values = []
    metrics = defaultdict(list)
    by_task_top = defaultdict(Counter)
    by_factor_top = defaultdict(Counter)

    for row in rows:
        base = _load_trace_future(row)
        if base is None:
            continue
        direction, nn_ids, weights = _repair_direction_from_memory(base, row.get("task"), memory, args.topk, args.same_task_only)
        groups = _group_norms(direction, args.group_size)
        top_groups = np.argsort(-groups)[:5].tolist()
        factor = str(row.get("factor_label") or row.get("wrong_component") or "unknown")
        item_metrics = {
            "object_motion": _object_motion_norm(row),
            "progress": _progress_norm(row),
            "contact_delta": _contact_delta(row),
            "gripper_delta": _gripper_delta(row),
            "scene_delta": _scene_delta(row),
            "goal_success": _goal_value(row),
            "failure": 0.0 if row.get("success") else 1.0,
            "steps": _safe_float(row.get("steps")),
        }
        for k, v in item_metrics.items():
            metrics[k].append(v)
        group_values.append(groups)
        for g in top_groups:
            by_task_top[str(row.get("task"))][int(g)] += 1
            by_factor_top[factor][int(g)] += 1
        examples.append(
            {
                "task": row.get("task"),
                "success": bool(row.get("success")),
                "factor": factor,
                "metrics": item_metrics,
                "top_groups": top_groups,
                "top_group_norms": [float(groups[g]) for g in top_groups],
                "nearest_memory_ids": nn_ids[:5],
                "nearest_weights": [float(x) for x in weights[:5]],
            }
        )

    group_arr = np.stack(group_values, axis=0) if group_values else np.zeros((0, 0), dtype=np.float32)
    correlations = {}
    for metric_name, values in metrics.items():
        vals = np.asarray(values, dtype=np.float32)
        per_group = []
        for g in range(group_arr.shape[1]):
            c = _corr(group_arr[:, g], vals)
            if c is not None:
                per_group.append((g, c))
        correlations[metric_name] = {
            "top_positive": [{"group": int(g), "corr": float(c)} for g, c in sorted(per_group, key=lambda x: -x[1])[:10]],
            "top_negative": [{"group": int(g), "corr": float(c)} for g, c in sorted(per_group, key=lambda x: x[1])[:10]],
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "repair_direction_physics_examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    summary = {
        "strong_gt_jsonl": args.strong_gt_jsonl,
        "immune_memory_npz": args.immune_memory_npz,
        "num_rows_used": len(examples),
        "group_size": args.group_size,
        "num_groups": int(group_arr.shape[1]) if group_arr.size else 0,
        "task_counts": Counter(ex["task"] for ex in examples),
        "factor_counts": Counter(ex["factor"] for ex in examples),
        "metric_mean": {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()},
        "correlations": correlations,
        "top_groups_by_task": {
            task: [{"group": int(g), "count": int(c)} for g, c in counter.most_common(10)]
            for task, counter in by_task_top.items()
        },
        "top_groups_by_factor": {
            factor: [{"group": int(g), "count": int(c)} for g, c in counter.most_common(10)]
            for factor, counter in by_factor_top.items()
        },
        "output_examples": str(out_dir / "repair_direction_physics_examples.jsonl"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
