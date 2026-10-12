import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _resolve_trajectories(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "trajectories.jsonl"
        if candidate.exists():
            return candidate
    return path


def _load_rollout(paths: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        path = _resolve_trajectories(path)
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    for row in rows:
        task = str(row.get("task", row.get("subtask", "")))
        success = bool(row.get("success", row.get("succeeded", False)))
        by_task[task]["total"] += 1
        by_task[task]["success"] += int(success)
    total = sum(item["total"] for item in by_task.values())
    success = sum(item["success"] for item in by_task.values())
    return {
        "total": total,
        "success": success,
        "success_rate": success / max(total, 1),
        "by_task": {
            task: {
                **counts,
                "success_rate": counts["success"] / max(counts["total"], 1),
            }
            for task, counts in sorted(by_task.items())
        },
    }


def write_report(path: Path, variants: List[Tuple[str, Dict[str, Any]]]) -> None:
    lines = [
        "# Rollout Ablation Report",
        "",
        "| Variant | Trajectories | Success | Success Rate |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in variants:
        lines.append(f"| {name} | {summary['total']} | {summary['success']} | {summary['success_rate']:.3f} |")
    tasks = sorted({task for _, summary in variants for task in summary["by_task"]})
    lines.extend(["", "## By Task", ""])
    for task in tasks:
        lines.extend(
            [
                f"### {task}",
                "",
                "| Variant | Trajectories | Success | Success Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, summary in variants:
            row = summary["by_task"].get(task, {"total": 0, "success": 0, "success_rate": 0.0})
            lines.append(f"| {name} | {row['total']} | {row['success']} | {row['success_rate']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Current / Meta-Foresight / Reflection rollout ablations.")
    parser.add_argument("--current", type=Path, nargs="+", required=True, help="Current Refiner trajectories.jsonl or output directories.")
    parser.add_argument("--meta", type=Path, nargs="+", required=True, help="Meta-Foresight Refiner trajectories.jsonl or output directories.")
    parser.add_argument("--reflection", type=Path, nargs="+", required=True, help="Meta-Foresight + Reflection Head trajectories.jsonl or output directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants = [
        ("Current Refiner", _summarize(_load_rollout(args.current))),
        ("Meta-Foresight Refiner", _summarize(_load_rollout(args.meta))),
        ("Meta-Foresight + Reflection Head", _summarize(_load_rollout(args.reflection))),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: summary for name, summary in variants}
    (args.output_dir / "ablation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(args.output_dir / "ablation_report.md", variants)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
