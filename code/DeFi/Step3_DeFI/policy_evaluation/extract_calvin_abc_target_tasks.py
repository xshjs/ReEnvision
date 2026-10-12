from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TARGET_TASKS = {
    "push_into_drawer",
    "push_pink_block_right",
    "stack_block",
    "push_blue_block_right",
    "push_red_block_right",
    "push_pink_block_left",
    "lift_blue_block_slider",
}


def classify_ann(text: str) -> str:
    t = text.lower()
    if any(
        x in t
        for x in (
            "sweep the block into the drawer",
            "sweep the object into the drawer",
            "push the object into the drawer",
            "push the block into the drawer",
            "slide the block into the drawer",
            "falls into the drawer",
        )
    ):
        return "push_into_drawer"
    if "stack" in t and not any(x in t for x in ("unstack", "remove", "take off", "collapse")):
        return "stack_block"
    if "pink block" in t and "right" in t and any(x in t for x in ("push", "slide", "sweep")):
        return "push_pink_block_right"
    if "pink block" in t and "left" in t and any(x in t for x in ("push", "slide", "sweep")):
        return "push_pink_block_left"
    if "blue block" in t and "right" in t and any(x in t for x in ("push", "slide", "sweep")):
        return "push_blue_block_right"
    if "red block" in t and "right" in t and any(x in t for x in ("push", "slide", "sweep")):
        return "push_red_block_right"
    if "blue block" in t and any(x in t for x in ("slider", "sliding cabinet", "shelf")) and any(
        x in t for x in ("pick up", "lift", "grasp")
    ):
        return "lift_blue_block_slider"
    return "unknown"


def episode_name(idx: int) -> str:
    return f"task_ABC_D/training/episode_{idx:07d}.npz"


def load_ann(zf: zipfile.ZipFile, path: str) -> dict[str, Any]:
    return np.load(io.BytesIO(zf.read(path)), allow_pickle=True).item()


def dump_npy(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, obj, allow_pickle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected CALVIN ABC training demonstrations from task_ABC_D.zip.")
    parser.add_argument("--zip-path", type=Path, default=Path("/mnt/data/calvin/task_ABC_D.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default=",".join(sorted(TARGET_TASKS)))
    parser.add_argument("--max-per-task", type=int, default=100)
    parser.add_argument("--pad-before", type=int, default=0)
    parser.add_argument("--pad-after", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = {x.strip() for x in args.tasks.split(",") if x.strip()}
    src_ann = "task_ABC_D/training/lang_annotations/auto_lang_ann.npy"
    out_root = args.output_dir
    out_train = out_root / "training"
    out_ann = out_train / "lang_annotations" / "auto_lang_ann.npy"

    with zipfile.ZipFile(args.zip_path) as zf:
        ann = load_ann(zf, src_ann)
        texts = list(ann["language"]["ann"])
        indices = list(ann["info"]["indx"])
        selected = []
        counts = Counter()
        for text, (start, end) in zip(texts, indices):
            task = classify_ann(str(text))
            if task not in wanted:
                continue
            if counts[task] >= args.max_per_task:
                continue
            counts[task] += 1
            selected.append({"task": task, "ann": str(text), "start": int(start), "end": int(end)})

        episode_ids = set()
        for row in selected:
            for idx in range(max(0, row["start"] - args.pad_before), row["end"] + args.pad_after + 1):
                episode_ids.add(idx)

        summary = {
            "zip_path": str(args.zip_path),
            "output_dir": str(args.output_dir),
            "tasks": sorted(wanted),
            "max_per_task": int(args.max_per_task),
            "num_selected_annotations": len(selected),
            "task_counts": dict(counts.most_common()),
            "num_episode_files": len(episode_ids),
            "dry_run": bool(args.dry_run),
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        (out_root / "selected_annotations.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        if args.dry_run:
            return

        out_train.mkdir(parents=True, exist_ok=True)
        missing = []
        for i, idx in enumerate(sorted(episode_ids), 1):
            name = episode_name(idx)
            if name not in zf.NameToInfo:
                missing.append(idx)
                continue
            target = out_train / Path(name).name
            if not target.exists():
                target.write_bytes(zf.read(name))
            if i % 500 == 0:
                print(json.dumps({"progress": i, "total": len(episode_ids), "missing": len(missing)}), flush=True)

        slim_ann = {
            "language": {"ann": [r["ann"] for r in selected], "task": [r["task"] for r in selected]},
            "info": {"indx": [(r["start"], r["end"]) for r in selected], "source_zip": str(args.zip_path)},
        }
        dump_npy(out_ann, slim_ann)
        summary["missing_episode_files"] = len(missing)
        summary["missing_episode_ids_sample"] = missing[:20]
        (out_root / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
