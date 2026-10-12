from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }
    return out


def pass_rate(delta_energy: np.ndarray, action_shift: np.ndarray, future_shift: np.ndarray, min_delta: float, max_action: float, max_future: float) -> float:
    passed = (delta_energy > min_delta) & (action_shift <= max_action) & (future_shift <= max_future)
    return float(passed.mean()) if len(passed) > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze joint latent refine outputs and suggest stricter gating thresholds.")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--topk-tasks", type=int, default=10)
    parser.add_argument("--novel-joint-threshold", type=float, default=0.05)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.results_jsonl))
    if not rows:
        raise ValueError("no rows found")

    delta_energy: list[float] = []
    action_shift: list[float] = []
    future_shift: list[float] = []
    final_energy: list[float] = []
    init_energy: list[float] = []
    applied: list[int] = []
    generated_new_pair: list[int] = []
    novelty_gain_joint_l2: list[float] = []
    refined_nearest_joint_l2: list[float] = []
    init_nearest_joint_l2: list[float] = []
    tasks = Counter()
    per_task_delta: dict[str, list[float]] = {}
    per_task_action: dict[str, list[float]] = {}
    per_task_new_pair: dict[str, list[int]] = {}

    for row in rows:
        task = str(row.get("query_task", "unknown"))
        info = row.get("refine", {}) or {}
        de = float(info.get("delta_energy", 0.0))
        ash = float(info.get("action_shift", 0.0))
        fsh = float(info.get("future_shift", 0.0))
        fe = float(info.get("final_energy", 0.0))
        ie = float(info.get("initial_energy", 0.0))
        ap = 1 if bool(info.get("applied", False)) else 0
        gn = 1 if bool(info.get("generated_new_pair", False)) else 0
        ng = float(info.get("novelty_gain_joint_l2", 0.0))
        rnj = float(info.get("refined_nearest_joint_l2", 0.0))
        inj = float(info.get("init_nearest_joint_l2", 0.0))

        delta_energy.append(de)
        action_shift.append(ash)
        future_shift.append(fsh)
        final_energy.append(fe)
        init_energy.append(ie)
        applied.append(ap)
        generated_new_pair.append(gn)
        novelty_gain_joint_l2.append(ng)
        refined_nearest_joint_l2.append(rnj)
        init_nearest_joint_l2.append(inj)
        tasks[task] += 1
        per_task_delta.setdefault(task, []).append(de)
        per_task_action.setdefault(task, []).append(ash)
        per_task_new_pair.setdefault(task, []).append(gn)

    delta_arr = np.asarray(delta_energy, dtype=np.float64)
    action_arr = np.asarray(action_shift, dtype=np.float64)
    future_arr = np.asarray(future_shift, dtype=np.float64)
    novelty_arr = np.asarray(novelty_gain_joint_l2, dtype=np.float64)
    refined_joint_arr = np.asarray(refined_nearest_joint_l2, dtype=np.float64)
    init_joint_arr = np.asarray(init_nearest_joint_l2, dtype=np.float64)

    delta_stats = summarize(delta_energy)
    action_stats = summarize(action_shift)
    future_stats = summarize(future_shift)
    init_stats = summarize(init_energy)
    final_stats = summarize(final_energy)
    novelty_stats = summarize(novelty_gain_joint_l2)
    refined_joint_stats = summarize(refined_nearest_joint_l2)
    init_joint_stats = summarize(init_nearest_joint_l2)

    suggested = {
        "strict": {
            "gate_min_delta_energy": float(np.quantile(delta_arr, 0.75)),
            "gate_max_action_shift": float(np.quantile(action_arr, 0.50)),
            "gate_max_future_shift": float(np.quantile(future_arr, 0.95)),
        },
        "medium": {
            "gate_min_delta_energy": float(np.quantile(delta_arr, 0.50)),
            "gate_max_action_shift": float(np.quantile(action_arr, 0.75)),
            "gate_max_future_shift": float(np.quantile(future_arr, 0.95)),
        },
        "loose": {
            "gate_min_delta_energy": float(np.quantile(delta_arr, 0.25)),
            "gate_max_action_shift": float(np.quantile(action_arr, 0.90)),
            "gate_max_future_shift": float(np.quantile(future_arr, 0.99)),
        },
    }

    for cfg in suggested.values():
        cfg["estimated_apply_rate"] = pass_rate(
            delta_arr,
            action_arr,
            future_arr,
            float(cfg["gate_min_delta_energy"]),
            float(cfg["gate_max_action_shift"]),
            float(cfg["gate_max_future_shift"]),
        )

    novelty_thresholds = [1e-4, 0.01, 0.05, 0.1, 0.25]
    novelty_pass = {}
    for thr in novelty_thresholds:
        novelty_pass[f"joint_l2_ge_{thr:g}"] = float((refined_joint_arr >= thr).mean())
    user_thr = float(args.novel_joint_threshold)
    novelty_pass["user_threshold"] = user_thr
    novelty_pass["user_threshold_rate"] = float((refined_joint_arr >= user_thr).mean())

    top_tasks = []
    for task, count in tasks.most_common(args.topk_tasks):
        top_tasks.append(
            {
                "task": task,
                "count": int(count),
                "delta_energy_mean": float(np.mean(per_task_delta[task])),
                "delta_energy_p50": float(np.median(per_task_delta[task])),
                "action_shift_mean": float(np.mean(per_task_action[task])),
                "action_shift_p50": float(np.median(per_task_action[task])),
                "generated_new_pair_rate": float(np.mean(per_task_new_pair[task])),
            }
        )

    summary = {
        "results_jsonl": str(args.results_jsonl),
        "num_rows": int(len(rows)),
        "original_apply_rate": float(np.mean(np.asarray(applied, dtype=np.float64))),
        "generated_new_pair_rate": float(np.mean(np.asarray(generated_new_pair, dtype=np.float64))),
        "delta_energy": delta_stats,
        "action_shift": action_stats,
        "future_shift": future_stats,
        "initial_energy": init_stats,
        "final_energy": final_stats,
        "novelty_gain_joint_l2": novelty_stats,
        "init_nearest_joint_l2": init_joint_stats,
        "refined_nearest_joint_l2": refined_joint_stats,
        "novelty_threshold_rates": novelty_pass,
        "refined_farther_than_init_rate": float((refined_joint_arr > init_joint_arr).mean()),
        "refined_farther_than_init_mean_delta": float((refined_joint_arr - init_joint_arr).mean()),
        "top_tasks": top_tasks,
        "suggested_gates": suggested,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
