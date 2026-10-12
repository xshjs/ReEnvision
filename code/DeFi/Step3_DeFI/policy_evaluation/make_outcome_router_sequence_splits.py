import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from policy_evaluation.multistep_sequences import get_sequences


def normalize_pair(item: Tuple[Dict[str, Any], Sequence[str]]) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
    state, sequence = item
    state_key = tuple(sorted((str(key), str(value)) for key, value in state.items()))
    sequence_key = tuple(str(task) for task in sequence)
    return state_key, sequence_key


def serializable_pair(item: Tuple[Dict[str, Any], Sequence[str]]) -> List[Any]:
    state, sequence = item
    return [
        {str(key): value.item() if hasattr(value, "item") else value for key, value in state.items()},
        [str(task) for task in sequence],
    ]


def unique_sequences(sequences: Iterable[Tuple[Dict[str, Any], Sequence[str]]]) -> List[Tuple[Dict[str, Any], Sequence[str]]]:
    seen = set()
    out = []
    for item in sequences:
        key = normalize_pair(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def write_json(path: Path, rows: List[Tuple[Dict[str, Any], Sequence[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([serializable_pair(row) for row in rows], indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create disjoint outcome-router train/heldout evaluation sequence files.")
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--num-heldout", type=int, default=110)
    parser.add_argument("--num-train", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--prefix", default="outcome_router_clean")
    args = parser.parse_args()

    pool = unique_sequences(get_sequences(args.pool_size))
    need = args.num_heldout + args.num_train
    if len(pool) < need:
        raise ValueError(f"Only found {len(pool)} unique sequences, need {need}. Increase --pool-size.")

    heldout = pool[: args.num_heldout]
    train = pool[args.num_heldout : args.num_heldout + args.num_train]
    train_keys = {normalize_pair(row) for row in train}
    heldout_keys = {normalize_pair(row) for row in heldout}
    overlap = train_keys & heldout_keys
    if overlap:
        raise RuntimeError(f"Split overlap detected: {len(overlap)}")

    train_path = args.output_dir / f"{args.prefix}_train{args.num_train}.json"
    heldout_path = args.output_dir / f"{args.prefix}_heldout{args.num_heldout}.json"
    write_json(train_path, train)
    write_json(heldout_path, heldout)
    print(
        json.dumps(
            {
                "pool_size": args.pool_size,
                "unique_pool_size": len(pool),
                "train": len(train),
                "heldout": len(heldout),
                "overlap": len(overlap),
                "train_path": str(train_path),
                "heldout_path": str(heldout_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
