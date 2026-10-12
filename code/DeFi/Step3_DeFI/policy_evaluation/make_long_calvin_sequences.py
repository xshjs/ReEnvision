import argparse
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from policy_evaluation.multistep_sequences import check_sequence, task_categories, tasks, valid_task


def serializable_pair(item: Tuple[Dict[str, Any], Sequence[str]]) -> List[Any]:
    state, sequence = item
    return [
        {str(key): value.item() if hasattr(value, "item") else value for key, value in state.items()},
        [str(task) for task in sequence],
    ]


def pair_key(item: Tuple[Dict[str, Any], Sequence[str]]) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
    state, sequence = item
    return tuple(sorted((str(key), str(value)) for key, value in state.items())), tuple(str(task) for task in sequence)


def load_keys(path: Path) -> set:
    if not path.exists():
        return set()
    rows = json.loads(path.read_text())
    return {pair_key((row[0], row[1])) for row in rows}


def initial_states() -> List[Dict[str, Any]]:
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    valid = lambda vals: vals.count("table") in [1, 2] and vals.count("slider_right") < 2 and vals.count("slider_left") < 2
    value_combinations = filter(valid, product(*possible_conditions.values()))
    return [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]


def sample_sequence(
    rng: np.random.Generator,
    initial_state: Dict[str, Any],
    sequence_len: int,
    allow_repeat: bool,
) -> List[str] | None:
    state = deepcopy(initial_state)
    sequence: List[str] = []
    used_tasks = set()
    used_categories = set()

    for _ in range(sequence_len):
        candidates = []
        for task_name, task in tasks.items():
            if not allow_repeat and task_name in used_tasks:
                continue
            category = task_categories[task_name]
            if category in used_categories:
                continue
            next_states = valid_task(state, task)
            if len(next_states) == 1:
                candidates.append((task_name, next_states[0]))
        if not candidates:
            return None
        idx = int(rng.integers(0, len(candidates)))
        task_name, state = candidates[idx]
        sequence.append(str(task_name))
        used_tasks.add(task_name)
        used_categories.add(task_categories[task_name])

    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed longer CALVIN task sequences.")
    parser.add_argument("--num-sequences", type=int, default=110)
    parser.add_argument("--sequence-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-sequences", type=Path, default=None)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1000000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    states = initial_states()
    exclude = load_keys(args.exclude_sequences) if args.exclude_sequences is not None else set()
    seen = set()
    rows: List[Tuple[Dict[str, Any], Sequence[str]]] = []
    attempts = 0
    while len(rows) < args.num_sequences and attempts < args.max_attempts:
        attempts += 1
        state = deepcopy(states[int(rng.integers(0, len(states)))])
        sequence = sample_sequence(rng, state, args.sequence_len, args.allow_repeat)
        if sequence is None:
            continue
        item = (state, sequence)
        key = pair_key(item)
        if key in seen or key in exclude:
            continue
        if not check_sequence(deepcopy(state), sequence):
            continue
        seen.add(key)
        rows.append(item)

    if len(rows) < args.num_sequences:
        raise RuntimeError(f"Only generated {len(rows)} sequences after {attempts} attempts.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([serializable_pair(row) for row in rows], indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_sequences": len(rows),
                "sequence_len": args.sequence_len,
                "seed": args.seed,
                "excluded_keys": len(exclude),
                "attempts": attempts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
