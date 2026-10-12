import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_npy(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data, allow_pickle=True)


def load_annotations(split_dir: Path) -> dict:
    ann_path = split_dir / "lang_clip_resnet50" / "auto_lang_ann.npy"
    if not ann_path.exists():
        ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    return np.load(ann_path, allow_pickle=True).item()


def select_indices(
    tasks: list[str],
    include_tasks: set[str],
    include_prefixes: tuple[str, ...],
    per_task_limit: int | None,
) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    keep: list[int] = []
    for idx, task in enumerate(tasks):
        match = task in include_tasks or any(task.startswith(prefix) for prefix in include_prefixes)
        if not match:
            continue
        if per_task_limit is not None and counts[task] >= per_task_limit:
            continue
        keep.append(idx)
        counts[task] += 1
    return np.asarray(keep, dtype=np.int64)


def filter_annotation_payload(data: dict, keep: np.ndarray) -> dict:
    num_annotations = len(data["language"]["task"])
    filtered_info = {}
    for key, value in data["info"].items():
        array = np.asarray(value)
        filtered_info[key] = array[keep] if len(array) == num_annotations else array
    return {
        "language": {key: np.asarray(value)[keep] for key, value in data["language"].items()},
        "info": filtered_info,
    }


def copy_episodes(source_split: Path, target_split: Path, filtered: dict, symlink: bool = False) -> int:
    needed = set()
    for start, end in filtered["info"]["indx"]:
        start = int(start)
        end = int(end)
        if end < start:
            start, end = end, start
        # Training reads contiguous frame windows, so the full span must exist.
        needed.update(range(start, end + 1))
    copied = 0
    for episode_idx in sorted(needed):
        src = source_split / f"episode_{episode_idx:07d}.npz"
        dst = target_split / f"episode_{episode_idx:07d}.npz"
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if symlink:
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
        copied += 1
    return copied


def extract_split(
    source_root: Path,
    output_root: Path,
    split: str,
    include_tasks: set[str],
    include_prefixes: tuple[str, ...],
    per_task_limit: int | None,
    symlink: bool,
) -> dict:
    source_split = source_root / split
    output_split = output_root / split
    data = load_annotations(source_split)
    tasks = [str(task) for task in data["language"]["task"]]
    keep = select_indices(tasks, include_tasks, include_prefixes, per_task_limit)
    filtered = filter_annotation_payload(data, keep)

    write_npy(output_split / "lang_clip_resnet50" / "auto_lang_ann.npy", filtered)
    write_npy(output_split / "lang_annotations" / "auto_lang_ann.npy", filtered)
    copied = copy_episodes(source_split, output_split, filtered, symlink=symlink)

    kept_tasks = [str(task) for task in filtered["language"]["task"]]
    return {
        "split": split,
        "segments": int(len(keep)),
        "episodes": int(copied),
        "task_counts": dict(sorted(Counter(kept_tasks).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a task-filtered local CALVIN subset while preserving file layout.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-tasks", default="")
    parser.add_argument("--include-prefixes", default="")
    parser.add_argument("--train-per-task", type=int, default=None)
    parser.add_argument("--val-per-task", type=int, default=None)
    parser.add_argument("--symlink", action="store_true", help="Symlink episode files instead of copying them.")
    parser.add_argument("--splits", default="training,validation", help="Comma-separated splits to extract.")
    args = parser.parse_args()

    include_tasks = set(parse_csv(args.include_tasks))
    include_prefixes = tuple(parse_csv(args.include_prefixes))
    splits = parse_csv(args.splits)
    if not include_tasks and not include_prefixes:
        raise ValueError("At least one of --include-tasks or --include-prefixes is required")
    if not splits:
        raise ValueError("--splits must not be empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dir": str(args.source_dir),
        "include_tasks": sorted(include_tasks),
        "include_prefixes": list(include_prefixes),
        "splits": splits,
    }
    for split in splits:
        per_task_limit = args.train_per_task if split == "training" else args.val_per_task
        summary[split] = extract_split(
            args.source_dir,
            args.output_dir,
            split,
            include_tasks,
            include_prefixes,
            per_task_limit,
            args.symlink,
        )
    (args.output_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
