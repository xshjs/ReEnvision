from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repair-advantage dataset for critic gating.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--include-negated", action="store_true")
    parser.add_argument("--advantage-scale", type=float, default=1.0)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.pairs_jsonl))
    if not rows:
        raise ValueError(f"no rows in {args.pairs_jsonl}")

    out_rows: list[dict[str, Any]] = []
    pos_count = 0
    id_count = 0
    neg_count = 0
    for row in rows:
        state = np.asarray(row["state_neg"], dtype=np.float32)
        future = np.asarray(row["future_neg"], dtype=np.float32)
        action = np.asarray(row["action_neg"], dtype=np.float32)
        future_pos = np.asarray(row["future_pos"], dtype=np.float32)
        action_pos = np.asarray(row["action_pos"], dtype=np.float32)
        delta_future = (future_pos - future).astype(np.float32)
        delta_action = (action_pos - action).astype(np.float32)
        pending_effect = delta_future.astype(np.float32)
        base = {
            "task": str(row.get("task", "unknown")),
            "stage": float(row.get("stage", row.get("subtask_index_neg", 0))),
            "sequence_index": int(row.get("sequence_index_neg", -1)),
            "state": state.tolist(),
            "future": future.tolist(),
            "action": action.tolist(),
            "delta_future": delta_future.tolist(),
            "delta_action": delta_action.tolist(),
            "pending_effect": pending_effect.tolist(),
            "advantage": float(row.get("return_gap", 1.0)) * float(args.advantage_scale),
            "label": 1,
            "source": "positive_pair",
        }
        out_rows.append(base)
        pos_count += 1

        if args.include_identity:
            out_rows.append(
                {
                    **base,
                    "delta_future": np.zeros_like(delta_future).tolist(),
                    "delta_action": np.zeros_like(delta_action).tolist(),
                    "pending_effect": np.zeros_like(pending_effect).tolist(),
                    "advantage": 0.0,
                    "label": 0,
                    "source": "identity_control",
                }
            )
            id_count += 1

        if args.include_negated:
            out_rows.append(
                {
                    **base,
                    "delta_future": (-delta_future).tolist(),
                    "delta_action": (-delta_action).tolist(),
                    "pending_effect": pending_effect.tolist(),
                    "advantage": -float(row.get("return_gap", 1.0)) * float(args.advantage_scale),
                    "label": 0,
                    "source": "negated_control",
                }
            )
            neg_count += 1

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    advantages = np.asarray([float(r["advantage"]) for r in out_rows], dtype=np.float32)
    summary = {
        "pairs_jsonl": str(args.pairs_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "num_input_pairs": int(len(rows)),
        "num_rows": int(len(out_rows)),
        "positive_rows": int(pos_count),
        "identity_rows": int(id_count),
        "negated_rows": int(neg_count),
        "advantage_mean": float(advantages.mean()) if advantages.size else 0.0,
        "advantage_p50": float(np.median(advantages)) if advantages.size else 0.0,
        "advantage_min": float(advantages.min()) if advantages.size else 0.0,
        "advantage_max": float(advantages.max()) if advantages.size else 0.0,
    }
    args.output_jsonl.with_name("summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
