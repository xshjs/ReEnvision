from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "keep": "checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_split_zfza_conservative",
    "replan": "checkpoints/RLBench/eval/bridgevla_replan_only_18tasks_50ep_prob0010_collect",
    "direct": "checkpoints/RLBench/eval/bridgevla_direct_suffix_patch_18tasks_50ep_prob0010_mix010",
    "patch": "checkpoints/RLBench/eval/bridgevla_suffix_td_eval_18tasks_50ep_patchonly_clip005_prob012_mix010",
}


def _rows_path(run_dir: Path) -> Path:
    return run_dir / "suffix_collect" / "step_rollout_rows.jsonl"


def _score_path(run_dir: Path) -> Path:
    return run_dir / "model_80" / "eval_results.csv"


def _read_rows(run_dir: Path) -> list[dict]:
    path = _rows_path(run_dir)
    if not path.exists():
        return []
    rows = []
    with path.open(errors="ignore") as fp:
        for line in fp:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _read_scores(run_dir: Path) -> dict[str, float]:
    path = _score_path(run_dir)
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


def _key(row: dict) -> tuple[str, int, int]:
    return str(row.get("task")), int(row.get("episode", -1)), int(row.get("step", -1))


def _episode_key(row: dict) -> tuple[str, int]:
    return str(row.get("task")), int(row.get("episode", -1))


def _safe_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _index_best_rows(rows: list[dict]) -> dict[tuple[str, int, int], dict]:
    indexed: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        key = _key(row)
        old = indexed.get(key)
        # Prefer rows where the response was actually requested/applied.
        score = int(bool(row.get("agent_suffix_td_requested"))) + int(bool(row.get("agent_suffix_td_applied")))
        old_score = -1
        if old is not None:
            old_score = int(bool(old.get("agent_suffix_td_requested"))) + int(bool(old.get("agent_suffix_td_applied")))
        if old is None or score >= old_score:
            indexed[key] = row
    return indexed


def _select_moderate_cases(
    rows: list[dict],
    low_prob: float,
    high_prob: float,
    max_delta: float,
    requested_only: bool,
) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        if not row.get("agent_suffix_td_active"):
            continue
        prob = _safe_float(row, "agent_suffix_td_patch_prob", -1.0)
        delta = _safe_float(row, "agent_suffix_td_delta_norm", 1e9)
        if prob < low_prob or prob > high_prob or delta > max_delta:
            continue
        if requested_only and not row.get("agent_suffix_td_requested"):
            continue
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def _summarize_matches(cases: list[dict], run_rows: dict[tuple[str, int, int], dict]) -> dict:
    matched = []
    missing = 0
    for case in cases:
        row = run_rows.get(_key(case))
        if row is None:
            missing += 1
            continue
        matched.append(row)

    active = [r for r in matched if r.get("agent_suffix_td_active")]
    requested = [r for r in active if r.get("agent_suffix_td_requested")]
    applied = [r for r in active if r.get("agent_suffix_td_applied")]
    final_success = [float(bool(r.get("final_episode_success"))) for r in matched]
    step_success = [float(bool(r.get("success"))) for r in matched]
    deltas = [_safe_float(r, "agent_suffix_td_delta_norm") for r in active if "agent_suffix_td_delta_norm" in r]
    probs = [_safe_float(r, "agent_suffix_td_patch_prob") for r in active if "agent_suffix_td_patch_prob" in r]
    reasons = Counter(str(r.get("agent_suffix_td_trigger_reason", "missing")) for r in active)
    modes = Counter(str(r.get("agent_suffix_td_mode", "missing")) for r in active)
    task_success: dict[str, list[float]] = defaultdict(list)
    for row in matched:
        task_success[str(row.get("task"))].append(float(bool(row.get("final_episode_success"))))
    return {
        "matched_steps": len(matched),
        "missing_steps": missing,
        "active_steps": len(active),
        "requested_steps": len(requested),
        "applied_steps": len(applied),
        "final_success": mean(final_success) if final_success else 0.0,
        "step_success": mean(step_success) if step_success else 0.0,
        "request_rate": len(requested) / len(active) if active else 0.0,
        "apply_rate": len(applied) / len(active) if active else 0.0,
        "delta_mean": mean(deltas) if deltas else 0.0,
        "patch_prob_mean": mean(probs) if probs else 0.0,
        "reasons": dict(reasons),
        "modes": dict(modes),
        "task_final_success": {k: mean(v) for k, v in sorted(task_success.items())},
    }


def _summarize_episode_level(cases: list[dict], rows: list[dict]) -> dict:
    case_eps = {_episode_key(r) for r in cases}
    eps: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = _episode_key(row)
        if key in case_eps:
            eps[key] = row
    vals = [float(bool(r.get("final_episode_success"))) for r in eps.values()]
    lengths = [_safe_float(r, "final_episode_length", 0.0) for r in eps.values()]
    return {
        "matched_episodes": len(eps),
        "final_success": mean(vals) if vals else 0.0,
        "final_length_mean": mean(lengths) if lengths else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare keep/replan/direct/patch on the same moderate-deviation keys from existing RLBench logs."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", default="patch", help="Run name used to select moderate cases.")
    parser.add_argument("--low-prob", type=float, default=0.01)
    parser.add_argument("--high-prob", type=float, default=0.20)
    parser.add_argument("--max-delta", type=float, default=0.30)
    parser.add_argument("--requested-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--run", action="append", default=[], help="name=run_dir override/addition.")
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    for item in args.run:
        name, value = item.split("=", 1)
        runs[name] = value

    run_dirs = {}
    all_rows = {}
    all_scores = {}
    for name, value in runs.items():
        path = Path(value)
        if not path.is_absolute():
            path = args.root / path
        run_dirs[name] = path
        all_rows[name] = _read_rows(path)
        all_scores[name] = _read_scores(path)
        print(f"[run] {name:8s} rows={len(all_rows[name]):5d} scores={len(all_scores[name]):2d} path={path}")

    if args.source not in all_rows or not all_rows[args.source]:
        raise FileNotFoundError(f"No rows found for source run: {args.source}")

    cases = _select_moderate_cases(
        all_rows[args.source],
        low_prob=args.low_prob,
        high_prob=args.high_prob,
        max_delta=args.max_delta,
        requested_only=args.requested_only,
    )
    print(
        f"\n[selected] source={args.source} cases={len(cases)} "
        f"prob=[{args.low_prob},{args.high_prob}] max_delta={args.max_delta} requested_only={args.requested_only}"
    )

    output = {
        "source": args.source,
        "selection": {
            "low_prob": args.low_prob,
            "high_prob": args.high_prob,
            "max_delta": args.max_delta,
            "requested_only": args.requested_only,
            "num_cases": len(cases),
            "num_episodes": len({_episode_key(r) for r in cases}),
        },
        "runs": {},
    }

    print("\n=== matched moderate step response ===")
    for name, rows in all_rows.items():
        summary = _summarize_matches(cases, _index_best_rows(rows))
        ep_summary = _summarize_episode_level(cases, rows)
        score_vals = list(all_scores[name].values())
        summary["overall_task_mean"] = mean(score_vals) if score_vals else 0.0
        summary["episode_level"] = ep_summary
        output["runs"][name] = summary
        print(
            f"{name:8s} matched={summary['matched_steps']:4d} missing={summary['missing_steps']:4d} "
            f"final_success={summary['final_success']:.3f} step_success={summary['step_success']:.3f} "
            f"requested={summary['requested_steps']:4d} applied={summary['applied_steps']:4d} "
            f"ep_success={ep_summary['final_success']:.3f} overall_mean={summary['overall_task_mean']:.3f}"
        )
        print(f"          reason={summary['reasons']} mode={summary['modes']}")

    print("\n=== per-task final success on matched steps ===")
    tasks = sorted({str(r.get("task")) for r in cases})
    print("task".ljust(38), *(name.rjust(10) for name in runs))
    for task in tasks:
        values = []
        for name in runs:
            value = output["runs"][name]["task_final_success"].get(task)
            values.append("nan" if value is None else f"{value:.3f}")
        print(task.ljust(38), *(v.rjust(10) for v in values))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"\n[wrote] {args.output_json}")


if __name__ == "__main__":
    main()
