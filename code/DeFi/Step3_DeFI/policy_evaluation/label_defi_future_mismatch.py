import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.oracle_hypothesis_rollout import QwenReflectionEncoder


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_feature(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def infer_mismatch_type(task: str, success: bool, steps: int, ep_len: int) -> str:
    if success and steps < max(ep_len // 2, 1):
        return "none"
    if "drawer" in task and not success:
        return "drawer_progress_hallucinated"
    if task.startswith("push_") and not success:
        return "object_displacement_overestimated"
    if task.startswith("lift_") and not success:
        return "contact_not_realized"
    if task == "stack_block" and not success:
        return "wrong_object_identity"
    if not success:
        return "future_outcome_mismatch"
    return "slow_success"


def nearest_success_target(
    task: str,
    current_feature: np.ndarray,
    success_bank: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Optional[np.ndarray], Optional[int], float]:
    candidates = success_bank.get(task, [])
    if not candidates:
        return None, None, -1.0
    best_idx = None
    best_score = -float("inf")
    best_target = None
    for item in candidates:
        score = cosine(current_feature, item["obs_current_feature"])
        if score > best_score:
            best_score = score
            best_idx = int(item["row_id"])
            best_target = item["defi_future_feature"]
    return best_target, best_idx, float(best_score)


def build_reflection_prompt(row: Dict[str, Any], mismatch_type: str, future_alignment: float) -> str:
    outcome = "success" if row["success"] else "failure"
    next_task = row.get("next_task") or "none"
    return (
        "You are labeling a robot imagined future after one executed subtask.\n"
        "Return one compact JSON object only with keys: "
        '{"trust_score": number, "mismatch_type": string, "correction_direction": string, '
        '"recoverability": string, "explanation": string}.\n'
        "Use short phrases. trust_score must be in [0,1].\n"
        f"task: {row['task']}\n"
        f"outcome: {outcome}\n"
        f"steps: {int(row['steps'])}/{int(row['ep_len'])}\n"
        f"rule_mismatch: {mismatch_type}\n"
        f"future_alignment_to_actual: {future_alignment:.4f}\n"
        f"next_task: {next_task}\n"
    )


def parse_qwen_json(text: str) -> Dict[str, Any]:
    clean = text.strip()
    try:
        return json.loads(clean)
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(clean[start : end + 1])
            except Exception:
                pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label DeFi single-future rollout rows with mismatch/reflection/repair targets.")
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.rollout_jsonl))
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded: List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]] = []
    success_bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    mismatch_counts = Counter()

    for row in rows:
        feat = load_feature(args.feature_root / row["feature_path"])
        loaded.append((row, feat))
        if bool(row["success"]):
            success_bank[str(row["task"])].append(
                {
                    "row_id": int(row["row_id"]),
                    "obs_current_feature": feat["obs_current_feature"],
                    "defi_future_feature": feat["defi_future_feature"],
                }
            )

    qwen = None
    if args.qwen_model_path is not None:
        device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        qwen = QwenReflectionEncoder(args.qwen_model_path, device)

    labeled_rows: List[Dict[str, Any]] = []
    qwen_rows: List[Dict[str, Any]] = []
    row_ids: List[int] = []
    future_features: List[np.ndarray] = []
    repair_targets: List[np.ndarray] = []
    repair_deltas: List[np.ndarray] = []

    for row, feat in tqdm(loaded, desc="label_defi_future_mismatch"):
        current_feature = feat["obs_current_feature"]
        future_feature = feat["defi_future_feature"]
        end_feature = feat["end_obs_current_feature"]
        pooled_future_feature = future_feature.mean(axis=0) if future_feature.ndim > 1 else future_feature
        future_alignment = cosine(pooled_future_feature, end_feature)
        mismatch_type = infer_mismatch_type(str(row["task"]), bool(row["success"]), int(row["steps"]), int(row["ep_len"]))
        mismatch_counts[mismatch_type] += 1

        qwen_payload = {}
        raw_qwen_text = ""
        if qwen is not None:
            raw_qwen_text = qwen.generate_reflection(
                build_reflection_prompt(row, mismatch_type, future_alignment),
                max_new_tokens=args.max_new_tokens,
            )
            qwen_payload = parse_qwen_json(raw_qwen_text)

        target_feature, target_row_id, retrieval_score = nearest_success_target(str(row["task"]), current_feature, success_bank)
        if target_feature is None:
            target_feature = future_feature
            target_row_id = None
            retrieval_score = -1.0

        labeled = {
            "row_id": int(row["row_id"]),
            "sequence_index": int(row["sequence_index"]),
            "subtask_index": int(row["subtask_index"]),
            "task": str(row["task"]),
            "success": bool(row["success"]),
            "steps": int(row["steps"]),
            "ep_len": int(row["ep_len"]),
            "future_alignment_to_actual": float(future_alignment),
            "mismatch_type": mismatch_type,
            "trust_score": float(qwen_payload.get("trust_score", 0.5) or 0.5),
            "correction_direction": str(qwen_payload.get("correction_direction", "stabilize contact"))[:256],
            "recoverability": str(qwen_payload.get("recoverability", "uncertain"))[:64],
            "reflection_text": raw_qwen_text[:1000],
            "retrieved_success_row_id": target_row_id,
            "retrieved_success_score": float(retrieval_score),
        }
        labeled_rows.append(labeled)
        qwen_rows.append(
            {
                "row_id": int(row["row_id"]),
                "task": str(row["task"]),
                "mismatch_type": mismatch_type,
                "raw_response": raw_qwen_text[:1000],
                "parsed": qwen_payload,
            }
        )
        row_ids.append(int(row["row_id"]))
        future_features.append(future_feature.astype(np.float32))
        repair_targets.append(np.asarray(target_feature, dtype=np.float32))
        repair_deltas.append(np.asarray(target_feature, dtype=np.float32) - future_feature.astype(np.float32))

    with (args.output_dir / "labeled_rows.jsonl").open("w") as handle:
        for row in labeled_rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "qwen_reflections.jsonl").open("w") as handle:
        for row in qwen_rows:
            handle.write(json.dumps(row) + "\n")

    np.savez_compressed(
        args.output_dir / "repair_targets.npz",
        row_ids=np.asarray(row_ids, dtype=np.int64),
        future_features=np.stack(future_features, axis=0).astype(np.float32),
        repair_targets=np.stack(repair_targets, axis=0).astype(np.float32),
        repair_deltas=np.stack(repair_deltas, axis=0).astype(np.float32),
    )
    summary = {
        "num_rows": len(labeled_rows),
        "num_success_rows": int(sum(1 for row in labeled_rows if row["success"])),
        "num_failure_rows": int(sum(1 for row in labeled_rows if not row["success"])),
        "mismatch_counts": dict(mismatch_counts),
        "qwen_enabled": bool(qwen is not None),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
