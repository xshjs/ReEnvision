import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_sequence_success = defaultdict(int)
    task_total = Counter()
    task_success = Counter()
    max_seq = -1
    for row in rows:
        seq_idx = int(row["sequence_index"])
        max_seq = max(max_seq, seq_idx)
        task = str(row["task"])
        task_total[task] += 1
        task_success[task] += int(bool(row["success"]))
        per_sequence_success[seq_idx] += int(bool(row["success"]))
    num_sequences = max_seq + 1 if max_seq >= 0 else 0
    results = [per_sequence_success[idx] for idx in range(num_sequences)]
    avg_seq_len = float(np.mean(results)) if results else 0.0
    chain_sr = {}
    for chain_idx in range(10):
        if not results:
            chain_sr[str(chain_idx + 1)] = 0.0
            continue
        chain_sr[str(chain_idx + 1)] = float(np.mean([value >= (chain_idx + 1) for value in results]))
    return {
        "num_sequences": num_sequences,
        "avg_seq_len": avg_seq_len,
        "chain_sr": chain_sr,
        "results": results,
        "task_info": {
            task: {"success": int(task_success[task]), "total": int(total)}
            for task, total in sorted(task_total.items())
        },
        "row_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple DeFi single-future rollout shards.")
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.output_dir / "features"
    if features_dir.exists():
        shutil.rmtree(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    merged_sequences: List[Dict[str, Any]] = []
    next_row_id = 0
    next_sequence_index = 0
    source_summaries = []

    for shard_dir in args.input_dirs:
        rows_path = shard_dir / "rollout_rows.jsonl"
        if not rows_path.exists():
            continue
        shard_rows = list(iter_jsonl(rows_path))
        shard_sequence_rows = list(iter_jsonl(shard_dir / "sequences.jsonl")) if (shard_dir / "sequences.jsonl").exists() else []
        seq_map: Dict[int, int] = {}
        for item in shard_sequence_rows:
            old_idx = int(item["sequence_index"])
            new_idx = next_sequence_index
            next_sequence_index += 1
            seq_map[old_idx] = new_idx
            merged_sequences.append(
                {
                    **item,
                    "sequence_index": new_idx,
                    "source_sequence_index": old_idx,
                    "source_dir": str(shard_dir),
                }
            )

        for row in shard_rows:
            old_row_id = int(row["row_id"])
            old_seq_idx = int(row["sequence_index"])
            if old_seq_idx not in seq_map:
                seq_map[old_seq_idx] = next_sequence_index
                next_sequence_index += 1
            feature_path = shard_dir / row["feature_path"]
            new_feature_rel = Path("features") / f"row_{next_row_id:07d}.npz"
            shutil.copy2(feature_path, args.output_dir / new_feature_rel)
            merged_rows.append(
                {
                    **row,
                    "row_id": next_row_id,
                    "sequence_index": seq_map[old_seq_idx],
                    "feature_path": str(new_feature_rel),
                    "source_row_id": old_row_id,
                    "source_sequence_index": old_seq_idx,
                    "source_dir": str(shard_dir),
                }
            )
            next_row_id += 1

        source_summary = {"input_dir": str(shard_dir), "num_rows": len(shard_rows), "num_sequences": len(seq_map)}
        if (shard_dir / "summary.json").exists():
            source_summary["summary"] = json.loads((shard_dir / "summary.json").read_text())
        if (shard_dir / "metadata.json").exists():
            source_summary["metadata"] = json.loads((shard_dir / "metadata.json").read_text())
        source_summaries.append(source_summary)

    write_jsonl(args.output_dir / "rollout_rows.jsonl", merged_rows)
    write_jsonl(args.output_dir / "sequences.jsonl", merged_sequences)
    summary = build_summary(merged_rows)
    metadata = {
        "num_input_dirs": len(args.input_dirs),
        "sources": source_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
