import argparse
import json
from pathlib import Path


PRESETS = {
    "pull_like": {
        "open_drawer",
        "close_drawer",
        "move_slider_left",
        "move_slider_right",
        "push_into_drawer",
        "place_in_drawer",
        "lift_red_block_drawer",
        "lift_blue_block_drawer",
        "lift_pink_block_drawer",
        "lift_red_block_slider",
        "lift_blue_block_slider",
        "lift_pink_block_slider",
    },
}


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter future-reflection prompts by CALVIN task.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    selected = set()
    if args.preset:
        selected.update(PRESETS[args.preset])
    if args.tasks:
        selected.update(task.strip() for task in args.tasks.split(",") if task.strip())
    if not selected:
        raise SystemExit("Provide --preset or --tasks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for row in iter_jsonl(args.input):
        if row.get("task") in selected:
            kept.append(row)
            if args.limit and len(kept) >= args.limit:
                break

    with args.output.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "selected_tasks": sorted(selected),
                "num_rows": len(kept),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
