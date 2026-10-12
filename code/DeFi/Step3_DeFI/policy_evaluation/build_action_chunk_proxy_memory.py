from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_action_chunk(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        chunk = np.asarray(data["action_chunk"], dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"expected action_chunk to be rank-2, got shape={chunk.shape} from {path}")
    if not np.isfinite(chunk).all():
        raise ValueError(f"non-finite action_chunk in {path}")
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal action-only proxy memory npz from collected CALVIN baseline action chunks.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    rows = list(iter_jsonl(args.rows_jsonl))
    if args.max_rows > 0:
        rows = rows[: int(args.max_rows)]
    if args.success_only:
        rows = [row for row in rows if bool(row.get("success", False))]
    if not rows:
        raise ValueError("no rows selected")

    action_chunks = []
    success = []
    tasks = []
    lang_text = []
    sequence_index = []
    subtask_index = []
    row_id = []
    feature_path = []
    steps = []
    ep_len = []

    for idx, row in enumerate(rows):
        rel = Path(str(row["feature_path"]))
        chunk = load_action_chunk(args.feature_root / rel)
        action_chunks.append(chunk.astype(np.float32))
        success.append(1.0 if bool(row.get("success", False)) else 0.0)
        tasks.append(str(row.get("task", "unknown")))
        lang_text.append(str(row.get("lang_text", row.get("task", "unknown"))))
        sequence_index.append(int(row.get("sequence_index", 0)))
        subtask_index.append(int(row.get("subtask_index", 0)))
        row_id.append(int(row.get("row_id", idx)))
        feature_path.append(str(rel))
        steps.append(int(row.get("steps", 0)))
        ep_len.append(int(row.get("ep_len", 0)))

    action_arr = np.stack(action_chunks).astype(np.float32)
    n = int(action_arr.shape[0])

    # This proxy memory is intentionally action-only. We keep the standard keys so
    # existing training code can load it, but the future/state channels are
    # placeholders and should be treated as unavailable.
    zero_summary = np.zeros((n, 1), dtype=np.float32)
    zero_state = np.zeros((n, 1), dtype=np.float32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        row_id=np.asarray(row_id, dtype=np.int64),
        base_summary_exec=zero_summary,
        target_summary_exec=zero_summary.copy(),
        target_delta_exec=zero_summary.copy(),
        action_chunk=action_arr,
        state_start=zero_state,
        state_end=zero_state.copy(),
        state_transition=zero_state.copy(),
        has_state=np.zeros((n,), dtype=np.float32),
        success=np.asarray(success, dtype=np.float32),
        task=np.asarray(tasks, dtype=object),
        lang_text=np.asarray(lang_text, dtype=object),
        sequence_index=np.asarray(sequence_index, dtype=np.int32),
        subtask_index=np.asarray(subtask_index, dtype=np.int32),
        feature_path=np.asarray(feature_path, dtype=object),
        steps=np.asarray(steps, dtype=np.int32),
        ep_len=np.asarray(ep_len, dtype=np.int32),
        memory_mode=np.asarray(["action_only_proxy"] * n, dtype=object),
    )

    summary = {
        "rows_jsonl": str(args.rows_jsonl),
        "feature_root": str(args.feature_root),
        "output_npz": str(args.output_npz),
        "num_rows": n,
        "num_success": int(sum(1 for x in success if x > 0.5)),
        "num_failure": int(sum(1 for x in success if x <= 0.5)),
        "action_shape": list(action_arr.shape[1:]),
        "memory_mode": "action_only_proxy",
        "placeholder_summary_dim": 1,
        "placeholder_state_dim": 1,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
