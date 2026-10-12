import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from train_universal_moe_oracle_router import load_rows


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_reflections(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        rows[key] = row
    return rows


def load_policy_rows(roots: List[Path]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for root in roots:
        base = root / "collect_success_expert_policy_dataset"
        index_path = base / "index.jsonl"
        if not index_path.exists():
            continue
        metadata_path = root / "metadata.json"
        sequence_offset = 0
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            sequence_offset = int(metadata.get("sequence_offset", 0))
        for row in iter_jsonl(index_path):
            key = (sequence_offset + int(row.get("sequence_index", 0)), int(row.get("subtask_index", 0)))
            joined = dict(row)
            joined["_global_sequence_index"] = key[0]
            joined["_npz_path"] = str((base / str(row["path"])).resolve())
            rows[key] = joined
    return rows


def safe_text(row: Dict[str, Any], key: str, default: str = "uncertain") -> str:
    return str(row.get(key, default)).strip()


def derive_recovery_info(row: Dict[str, Any]) -> Dict[str, Any]:
    chosen = int(row.get("chosen_expert", -1))
    oracle = int(row.get("oracle_expert", -1))
    branches = row.get("branches", [])
    chosen_branch = next((branch for branch in branches if int(branch.get("expert", -1)) == chosen), None)
    successful = [branch for branch in branches if bool(branch.get("success", False))]
    recoverable = bool(successful) and not bool(row.get("success", False))
    return {
        "chosen_failed": bool(chosen_branch is not None and not bool(chosen_branch.get("success", False))),
        "recoverable": recoverable,
        "has_successful_alternative": any(int(branch.get("expert", -1)) != chosen for branch in successful),
        "oracle_beats_chosen": oracle != chosen,
        "oracle_success": any(int(branch.get("expert", -1)) == oracle and bool(branch.get("success", False)) for branch in branches),
    }


def derive_operator(reflection: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    prefer_next = safe_text(reflection, "prefer_next", "")
    avoid_next = safe_text(reflection, "avoid_next", "")
    failure_type = safe_text(reflection, "failure_type", "uncertain")
    correction_direction = safe_text(reflection, "correction_direction", "uncertain")
    progress_status = safe_text(reflection, "progress_status", "uncertain")
    steps = int(row.get("steps", 160))
    weights = {
        "chosen_expert": 1.0,
        "oracle_expert": 1.0,
        "long_horizon_penalty": 0.0,
        "recovery_bonus": 0.0,
    }
    tags: List[str] = []
    if failure_type in {"slow_success", "unstable"} or steps >= 120 or "long" in avoid_next or "detour" in avoid_next:
        weights["long_horizon_penalty"] = 0.2
        tags.append("downweight_long_detour")
    if "recover" in prefer_next or "recovery" in prefer_next or "recovery" in correction_direction:
        weights["recovery_bonus"] = 0.2
        tags.append("prefer_recovery_capable")
    if "collision" in avoid_next or "crash" in avoid_next or failure_type in {"wrong_expert", "unstable"}:
        weights["chosen_expert"] = 0.7
        tags.append("avoid_collision_like_future")
    if progress_status == "cold_start" and failure_type == "none":
        tags.append("keep_router_prior")
    shift = {
        "prefer_next": prefer_next,
        "avoid_next": avoid_next,
        "correction_direction": correction_direction,
        "failure_type": failure_type,
        "progress_status": progress_status,
    }
    return {
        "operator_tags": tags,
        "data_reweight": weights,
        "preference_shift": shift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare joined operator dataset from rollout rows and aligned reflections.")
    parser.add_argument("--universal-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--policy-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--reflections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    universal_rows = load_rows(args.universal_roots)
    reflections = load_reflections(args.reflections)
    policy_rows = load_policy_rows(args.policy_roots)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "reflection_operator_dataset.jsonl"

    total = 0
    matched_reflections = 0
    matched_policy = 0
    tag_counts: Dict[str, int] = {}

    with output_path.open("w") as handle:
        for npz_path, row in universal_rows:
            global_seq = int(row.get("_global_sequence_index", -1))
            subtask_idx = int(row.get("subtask_index", -1))
            key = (global_seq, subtask_idx)
            reflection = reflections.get(key, {})
            policy = policy_rows.get(key)
            recovery_info = derive_recovery_info(row)
            operator = derive_operator(reflection, row)
            for tag in operator["operator_tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            joined = {
                "global_sequence_index": global_seq,
                "subtask_index": subtask_idx,
                "task": str(row.get("task")),
                "universal_npz_path": str(npz_path.resolve()),
                "policy_npz_path": None if policy is None else policy["_npz_path"],
                "success": bool(row.get("success", False)),
                "steps": int(row.get("steps", 160)),
                "chosen_expert": int(row.get("chosen_expert", -1)),
                "oracle_expert": int(row.get("oracle_expert", -1)),
                "oracle_value": float(row.get("oracle_value", 0.0)),
                "router_order": [int(x) for x in row.get("router_order", [])],
                "branches": row.get("branches", []),
                "reflection": {
                    "progress_status": safe_text(reflection, "progress_status"),
                    "failure_type": safe_text(reflection, "failure_type"),
                    "prefer_next": safe_text(reflection, "prefer_next"),
                    "avoid_next": safe_text(reflection, "avoid_next"),
                    "correction_direction": safe_text(reflection, "correction_direction"),
                    "memory_query": safe_text(reflection, "memory_query"),
                    "reflection_text": safe_text(reflection, "reflection_text", ""),
                },
                "recovery_info": recovery_info,
                **operator,
            }
            handle.write(json.dumps(joined, ensure_ascii=False) + "\n")
            total += 1
            matched_reflections += int(bool(reflection))
            matched_policy += int(policy is not None)

    summary = {
        "total_rows": total,
        "matched_reflections": matched_reflections,
        "matched_policy_rows": matched_policy,
        "output": str(output_path),
        "operator_tag_counts": tag_counts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
