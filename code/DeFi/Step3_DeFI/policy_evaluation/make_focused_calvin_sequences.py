"""Create CALVIN eval sequence JSON focused on high-failure tasks."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from policy_evaluation.multistep_sequences import get_sequences


def parse_tasks(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Comma-separated target task names.")
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--num-sequences", type=int, default=300)
    parser.add_argument("--min-target-count", type=int, default=1)
    parser.add_argument("--require-first-target", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    target_tasks = parse_tasks(args.tasks)
    rng = random.Random(args.seed)
    pool = get_sequences(args.pool_size)
    selected = []
    for item in pool:
        _, sequence = item
        hits = sum(1 for task in sequence if task in target_tasks)
        if hits < args.min_target_count:
            continue
        if args.require_first_target and (not sequence or sequence[0] not in target_tasks):
            continue
        selected.append(item)
    rng.shuffle(selected)
    selected = selected[: args.num_sequences]
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "target_tasks": sorted(target_tasks),
        "pool_size": int(args.pool_size),
        "num_selected": len(selected),
        "min_target_count": int(args.min_target_count),
        "require_first_target": bool(args.require_first_target),
        "task_counts": dict(Counter(task for _, seq in selected for task in seq).most_common()),
        "target_task_counts": dict(Counter(task for _, seq in selected for task in seq if task in target_tasks).most_common()),
        "output_json": str(out),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
