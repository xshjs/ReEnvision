import argparse
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_zip_npy(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as handle:
        return np.load(io.BytesIO(handle.read()), allow_pickle=True).item()


def filter_annotations(data, per_task: int, max_total: int | None):
    counts = defaultdict(int)
    keep = []
    for idx, task in enumerate(data["language"]["task"]):
        task = str(task)
        if counts[task] >= per_task:
            continue
        keep.append(idx)
        counts[task] += 1
        if max_total is not None and len(keep) >= max_total:
            break
    keep = np.asarray(keep, dtype=np.int64)
    num_annotations = len(data["language"]["task"])
    filtered_info = {}
    for key, value in data["info"].items():
        array = np.asarray(value)
        filtered_info[key] = array[keep] if len(array) == num_annotations else array
    return {
        "language": {
            key: np.asarray(value)[keep]
            for key, value in data["language"].items()
        },
        "info": filtered_info,
    }


def write_npy(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data, allow_pickle=True)


def extract_member(zf: zipfile.ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    with zf.open(member) as src, target.open("wb") as dst:
        dst.write(src.read())


def extract_split(zf: zipfile.ZipFile, root: str, split: str, out_dir: Path, per_task: int, max_total: int | None) -> dict:
    ann_name = f"{root}/{split}/lang_clip_resnet50/auto_lang_ann.npy"
    data = load_zip_npy(zf, ann_name)
    subset = filter_annotations(data, per_task, max_total)
    split_out = out_dir / split
    write_npy(split_out / "lang_clip_resnet50" / "auto_lang_ann.npy", subset)
    # Some utilities look for lang_annotations; keep it identical for compatibility.
    write_npy(split_out / "lang_annotations" / "auto_lang_ann.npy", subset)

    needed = set()
    for start, end in subset["info"]["indx"]:
        needed.add(int(start))
        needed.add(int(end))
    extracted = 0
    for episode_idx in sorted(needed):
        member = f"{root}/{split}/episode_{episode_idx:07d}.npz"
        target = split_out / f"episode_{episode_idx:07d}.npz"
        try:
            extract_member(zf, member, target)
            extracted += 1
        except KeyError:
            pass
    return {
        "split": split,
        "segments": int(len(subset["language"]["task"])),
        "tasks": int(len(set(map(str, subset["language"]["task"])))),
        "episodes": int(extracted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a small CALVIN task_ABC_D subset from the official zip.")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", default="task_ABC_D")
    parser.add_argument("--train-per-task", type=int, default=8)
    parser.add_argument("--val-per-task", type=int, default=2)
    parser.add_argument("--train-max-total", type=int, default=None)
    parser.add_argument("--val-max-total", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        summary = {
            "source_zip": str(args.zip),
            "root": args.root,
            "training": extract_split(zf, args.root, "training", args.output_dir, args.train_per_task, args.train_max_total),
            "validation": extract_split(zf, args.root, "validation", args.output_dir, args.val_per_task, args.val_max_total),
        }
    write_npy(args.output_dir / "subset_summary.npy", summary)
    print(summary)


if __name__ == "__main__":
    main()
