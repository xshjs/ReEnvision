from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a reflection SFT JSONL down to failure samples only.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-tasks", nargs="*", default=None)
    args = parser.parse_args()

    keep_tasks = set(args.keep_tasks) if args.keep_tasks else None
    rows = []
    for row in read_jsonl(args.input):
        meta = dict(row.get("metadata", {}))
        if bool(meta.get("success", False)):
            continue
        if keep_tasks is not None and str(meta.get("task", "")) not in keep_tasks:
            continue
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "tasks": sorted({str(row.get("metadata", {}).get("task", "")) for row in rows}),
    }
    (args.output.parent / "failure_only_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
