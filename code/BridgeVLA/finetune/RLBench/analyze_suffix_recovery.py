from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "baseline": "checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_split_zfza_conservative",
    "replan": "checkpoints/RLBench/eval/bridgevla_replan_only_18tasks_50ep_prob0010_collect",
    "direct": "checkpoints/RLBench/eval/bridgevla_direct_suffix_patch_18tasks_50ep_prob0010_mix010",
    "ours_patch": "checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_patchonly_clip005_prob012_mix010",
    "ours_retry": "checkpoints/RLBench/eval/bridgevla_suffix_td_retry_only_triggered_18tasks_50ep",
}


def _read_eval_scores(run_dir: Path) -> dict[str, float]:
    path = run_dir / "model_80" / "eval_results.csv"
    if not path.exists():
        return {}
    scores: dict[str, float] = {}
    with path.open(newline="", errors="ignore") as fp:
        for row in csv.DictReader(fp):
            try:
                scores[str(row["task"])] = float(row["success rate"])
            except Exception:
                continue
    return scores


def _read_rows(run_dir: Path) -> list[dict]:
    rows_path = run_dir / "suffix_collect" / "step_rollout_rows.jsonl"
    if not rows_path.exists():
        return []
    rows = []
    with rows_path.open(errors="ignore") as fp:
        for line in fp:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize_rows(rows: list[dict], moderate_low: float, moderate_high: float, max_delta: float) -> dict:
    active = [r for r in rows if r.get("agent_suffix_td_active")]
    requested = [r for r in active if r.get("agent_suffix_td_requested")]
    applied = [r for r in active if r.get("agent_suffix_td_applied")]
    moderate = [
        r
        for r in active
        if moderate_low <= _safe_float(r, "agent_suffix_td_patch_prob", -1.0) <= moderate_high
        and _safe_float(r, "agent_suffix_td_delta_norm", 1e9) <= max_delta
    ]
    moderate_requested = [r for r in moderate if r.get("agent_suffix_td_requested")]
    moderate_applied = [r for r in moderate if r.get("agent_suffix_td_applied")]
    task_active = Counter(str(r.get("task", "")) for r in active)
    task_requested = Counter(str(r.get("task", "")) for r in requested)
    task_applied = Counter(str(r.get("task", "")) for r in applied)
    reason = Counter(str(r.get("agent_suffix_td_trigger_reason", "missing")) for r in active)
    mode = Counter(str(r.get("agent_suffix_td_mode", "missing")) for r in active)
    deltas = [_safe_float(r, "agent_suffix_td_delta_norm") for r in active if "agent_suffix_td_delta_norm" in r]
    patch_probs = [_safe_float(r, "agent_suffix_td_patch_prob") for r in active if "agent_suffix_td_patch_prob" in r]
    return {
        "rows": len(rows),
        "active": len(active),
        "requested": len(requested),
        "applied": len(applied),
        "request_rate": len(requested) / len(active) if active else 0.0,
        "apply_rate": len(applied) / len(active) if active else 0.0,
        "moderate": len(moderate),
        "moderate_requested": len(moderate_requested),
        "moderate_applied": len(moderate_applied),
        "moderate_success": mean([float(bool(r.get("final_episode_success"))) for r in moderate]) if moderate else 0.0,
        "moderate_requested_success": mean([float(bool(r.get("final_episode_success"))) for r in moderate_requested]) if moderate_requested else 0.0,
        "moderate_applied_success": mean([float(bool(r.get("final_episode_success"))) for r in moderate_applied]) if moderate_applied else 0.0,
        "delta_mean": mean(deltas) if deltas else 0.0,
        "patch_prob_mean": mean(patch_probs) if patch_probs else 0.0,
        "reason": dict(reason),
        "mode": dict(mode),
        "task_active": dict(task_active),
        "task_requested": dict(task_requested),
        "task_applied": dict(task_applied),
    }


def _print_score_table(all_scores: dict[str, dict[str, float]]):
    tasks = sorted({task for scores in all_scores.values() for task in scores})
    names = list(all_scores)
    print("\n=== success scores ===")
    print("task".ljust(38), *(name.rjust(12) for name in names))
    for task in tasks:
        print(task.ljust(38), *(f"{all_scores[name].get(task, float('nan')):12.3f}" for name in names))
    print("\n=== mean ===")
    for name in names:
        vals = list(all_scores[name].values())
        print(f"{name:18s} n={len(vals):2d} mean={mean(vals):.4f}" if vals else f"{name:18s} n=0 mean=nan")


def _print_row_summary(name: str, summary: dict):
    print(f"\n=== {name}: suffix / moderate-deviation stats ===")
    print(
        "rows={rows} active={active} requested={requested} applied={applied} "
        "req_rate={request_rate:.3f} app_rate={apply_rate:.3f}".format(**summary)
    )
    print(
        "moderate={moderate} moderate_requested={moderate_requested} moderate_applied={moderate_applied} "
        "moderate_success={moderate_success:.3f} requested_success={moderate_requested_success:.3f} "
        "applied_success={moderate_applied_success:.3f}".format(**summary)
    )
    print(f"delta_mean={summary['delta_mean']:.4f} patch_prob_mean={summary['patch_prob_mean']:.4f}")
    print("reason", summary["reason"])
    print("mode", summary["mode"])
    print("top task triggers")
    for task, active in sorted(summary["task_active"].items(), key=lambda x: x[0]):
        req = summary["task_requested"].get(task, 0)
        app = summary["task_applied"].get(task, 0)
        rate = req / active if active else 0.0
        print(f"  {task:38s} active={active:5d} requested={req:4d} applied={app:4d} req_rate={rate:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RLBench suffix patch/replan recovery logs.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", action="append", default=[], help="name=relative/or/absolute/run_dir. Can be repeated.")
    parser.add_argument("--moderate-low", type=float, default=0.01)
    parser.add_argument("--moderate-high", type=float, default=0.20)
    parser.add_argument("--max-delta", type=float, default=0.30)
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be name=path, got {item}")
        name, value = item.split("=", 1)
        runs[name] = value

    all_scores: dict[str, dict[str, float]] = {}
    row_summaries: dict[str, dict] = {}
    for name, value in runs.items():
        run_dir = Path(value)
        if not run_dir.is_absolute():
            run_dir = args.root / run_dir
        if not run_dir.exists():
            continue
        scores = _read_eval_scores(run_dir)
        rows = _read_rows(run_dir)
        all_scores[name] = scores
        if rows:
            row_summaries[name] = summarize_rows(rows, args.moderate_low, args.moderate_high, args.max_delta)
        print(f"[run] {name}: {run_dir} scores={len(scores)} rows={len(rows)}")

    _print_score_table(all_scores)
    for name, summary in row_summaries.items():
        _print_row_summary(name, summary)


if __name__ == "__main__":
    main()
