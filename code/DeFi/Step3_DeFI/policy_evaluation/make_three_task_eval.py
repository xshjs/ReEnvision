import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def parse_tasks(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def base_state() -> Dict[str, object]:
    return {
        "led": 0,
        "lightbulb": 0,
        "slider": "left",
        "drawer": "closed",
        "red_block": "table",
        "blue_block": "slider_right",
        "pink_block": "slider_left",
        "grasped": 0,
    }


def state_for_task(task_name: str) -> Dict[str, object]:
    state = base_state()
    if task_name == "open_drawer":
        state["drawer"] = "closed"
    elif task_name == "close_drawer":
        state["drawer"] = "open"
    elif task_name == "move_slider_left":
        state["slider"] = "right"
    elif task_name == "move_slider_right":
        state["slider"] = "left"
    elif task_name == "turn_on_lightbulb":
        state["lightbulb"] = 0
    elif task_name == "turn_off_lightbulb":
        state["lightbulb"] = 1
    elif task_name == "turn_on_led":
        state["led"] = 0
    elif task_name == "turn_off_led":
        state["led"] = 1
    else:
        raise KeyError(
            "Unsupported task for the lightweight 3-task eval generator: "
            f"{task_name}. Supported tasks: open_drawer, close_drawer, "
            "move_slider_left, move_slider_right, turn_on_lightbulb, "
            "turn_off_lightbulb, turn_on_led, turn_off_led."
        )
    return deepcopy(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small CALVIN eval sequence file for three single-task evaluations.")
    parser.add_argument("--tasks", default="open_drawer,move_slider_left,turn_on_lightbulb")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_names = parse_tasks(args.tasks)
    if len(task_names) != 3:
        raise ValueError("--tasks must contain exactly 3 comma-separated CALVIN task names")

    rows = [[state_for_task(task_name), [task_name]] for task_name in task_names]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tasks": task_names, "num_sequences": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
