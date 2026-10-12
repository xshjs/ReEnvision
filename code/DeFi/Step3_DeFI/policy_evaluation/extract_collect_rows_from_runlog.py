#!/usr/bin/env python3
"""Extract collect rows from an evaluation run.log into a training jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(args.run_log).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[collect-row]" not in line:
                continue
            try:
                row = json.loads(line.split("[collect-row]", 1)[1].strip())
            except Exception:
                continue
            trace = row.get("trace") or row.get("trace_path") or row.get("collection_trace_path")
            if trace:
                row["trace_path"] = trace
            if "sequence_index" not in row and "seq" in row:
                row["sequence_index"] = row["seq"]
            if "subtask_index" not in row and "sub" in row:
                row["subtask_index"] = row["sub"]
            rows.append(row)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"run_log": args.run_log, "output_jsonl": str(out), "num_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
