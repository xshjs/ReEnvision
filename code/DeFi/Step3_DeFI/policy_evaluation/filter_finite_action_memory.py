from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter memory npz rows with finite action_chunk values.")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    keys = list(data.files)
    if "action_chunk" not in keys:
        raise ValueError("input npz missing action_chunk")

    action_chunk = np.asarray(data["action_chunk"])
    keep = np.isfinite(action_chunk).reshape(action_chunk.shape[0], -1).all(axis=1)
    kept = int(keep.sum())
    total = int(len(keep))
    if kept <= 0:
        raise ValueError("no finite action_chunk rows found")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **{k: data[k][keep] for k in keys})

    summary = {
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "total_rows": total,
        "kept_rows": kept,
        "dropped_rows": total - kept,
        "keep_ratio": float(kept / max(total, 1)),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
