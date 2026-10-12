from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_TASKS = [
    "stack_block",
    "push_into_drawer",
    "push_blue_block_right",
    "push_red_block_right",
    "push_pink_block_right",
]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_task_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def gather_success_expert_rows(
    root_dirs: List[Path],
    allowed_tasks: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    used_roots: List[str] = []
    traj_id = 0
    for root in sorted(root_dirs):
        base = root / "collect_success_expert_policy_dataset"
        index_path = base / "index.jsonl"
        if not index_path.exists():
            continue
        used_roots.append(str(root))
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in iter_jsonl(index_path):
            task = str(row.get("task", ""))
            if task not in allowed_tasks:
                continue
            grouped[str(row["trajectory_id"])].append(row)
        for _, traj_rows in sorted(grouped.items()):
            traj_rows.sort(key=lambda item: int(item.get("subtask_index", 0)))
            new_traj_id = f"traj_{traj_id:04d}"
            traj_id += 1
            for item in traj_rows:
                src = base / str(item["path"])
                if not src.exists():
                    continue
                step_name = Path(str(item["path"])).name
                dst_rel = Path("policy_dataset") / new_traj_id / step_name
                rows.append(
                    {
                        "trajectory_id": new_traj_id,
                        "sequence_index": int(item.get("sequence_index", 0)),
                        "subtask_index": int(item.get("subtask_index", 0)),
                        "task": str(item.get("task", "")),
                        "lang_text": str(item.get("lang_text", "")),
                        "path": str(dst_rel),
                        "expert": int(item.get("expert", -1)),
                        "router_rank": int(item.get("router_rank", -1)),
                        "steps": int(item.get("steps", 0)),
                        "num_actions": int(item.get("num_actions", 0)),
                        "source": str(item.get("source", "successful_moe_expert_rollout")),
                        "_src_path": str(src.resolve()),
                        "_src_root": str(root),
                    }
                )
                counts[str(item.get("task", ""))] += 1
    return rows, dict(sorted(counts.items())), used_roots


def gather_clean110_rows(
    root: Path,
    allowed_tasks: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows_path = root / "rollout_rows.jsonl"
    out_rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    row_id = 0
    for row in iter_jsonl(rows_path):
        task = str(row.get("task", ""))
        if task not in allowed_tasks or not bool(row.get("success", False)):
            continue
        src = root / str(row["feature_path"])
        if not src.exists():
            continue
        dst_rel = Path("future_features") / f"row_{row_id:06d}.npz"
        row_id += 1
        out_rows.append(
            {
                "row_id": int(row.get("row_id", -1)),
                "sequence_index": int(row.get("sequence_index", 0)),
                "subtask_index": int(row.get("subtask_index", 0)),
                "task": task,
                "lang_text": str(row.get("lang_text", "")),
                "next_task": row.get("next_task"),
                "success": True,
                "steps": int(row.get("steps", 0)),
                "ep_len": int(row.get("ep_len", 160)),
                "feature_path": str(dst_rel),
                "_src_path": str(src.resolve()),
                "_src_root": str(root),
            }
        )
        counts[task] += 1
    return out_rows, dict(sorted(counts.items()))


def gather_heuristic_counts(root: Path, allowed_tasks: set[str]) -> Dict[str, int]:
    sequences_path = root / "heuristic_selector" / "sequences.jsonl"
    counts: Counter[str] = Counter()
    for seq in iter_jsonl(sequences_path):
        for subtask in seq.get("subtasks", []):
            task = str(subtask.get("task", ""))
            if task in allowed_tasks and bool(subtask.get("success", False)):
                counts[task] += 1
    return dict(sorted(counts.items()))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_symlinks(output_dir: Path, rows: List[Dict[str, Any]], key: str) -> None:
    for row in rows:
        src = Path(str(row[key]))
        dst_key = "path" if key == "_src_path" and "path" in row else "feature_path"
        dst = output_dir / str(row[dst_key])
        ensure_symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect weak-task success assets into one merged output.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated weak-task list.",
    )
    parser.add_argument(
        "--success-expert-roots",
        nargs="*",
        type=Path,
        default=sorted(Path("/mnt/data/manipulation/DeFi/outputs").glob("policy_finetune_success_expert_d_shard_*")),
    )
    parser.add_argument(
        "--clean110-root",
        type=Path,
        default=Path("/mnt/data/manipulation/DeFi/outputs/defi_single_future_clean110_rerun"),
    )
    parser.add_argument(
        "--heuristic-root",
        type=Path,
        default=Path("/mnt/data/manipulation/DeFi/outputs/lt90_tasks_heuristic_n1000_ep120"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/manipulation/DeFi/outputs/weak_task_success_assets_v1"),
    )
    args = parser.parse_args()

    tasks = parse_task_list(args.tasks)
    allowed_tasks = set(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows, policy_counts, policy_roots = gather_success_expert_rows(args.success_expert_roots, allowed_tasks)
    clean_rows, clean_counts = gather_clean110_rows(args.clean110_root, allowed_tasks)
    heuristic_counts = gather_heuristic_counts(args.heuristic_root, allowed_tasks)

    materialize_symlinks(args.output_dir, policy_rows, "_src_path")
    materialize_symlinks(args.output_dir, clean_rows, "_src_path")

    policy_index_rows = []
    for row in policy_rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        policy_index_rows.append(clean)
    clean_feature_rows = []
    for row in clean_rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean_feature_rows.append(clean)

    write_jsonl(args.output_dir / "policy_index.jsonl", policy_index_rows)
    write_jsonl(args.output_dir / "clean110_success_rows.jsonl", clean_feature_rows)

    summary = {
        "tasks": tasks,
        "policy_dataset": {
            "num_rows": len(policy_index_rows),
            "task_counts": policy_counts,
            "roots": policy_roots,
            "index_jsonl": str((args.output_dir / "policy_index.jsonl").resolve()),
        },
        "clean110_success_features": {
            "num_rows": len(clean_feature_rows),
            "task_counts": clean_counts,
            "rows_jsonl": str((args.output_dir / "clean110_success_rows.jsonl").resolve()),
        },
        "heuristic_success_counts": {
            "task_counts": heuristic_counts,
            "note": "Counts only. No ready-made action trace npz files were found under heuristic_selector.",
        },
        "notes": [
            "policy_dataset contains symlinked action trajectories from collect_success_expert_policy_dataset.",
            "clean110_success_rows contains symlinked future-feature rows for success-only auxiliary training or analysis.",
            "heuristic_selector currently contributes statistics only; if you want trainable action traces, those rollouts need to be re-collected with saving enabled.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
