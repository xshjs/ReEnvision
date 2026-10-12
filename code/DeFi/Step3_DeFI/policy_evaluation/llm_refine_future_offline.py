import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train_vjepa2_moe_future import MoeFuture, STAGE_NAMES, infer_stage, topk_same_task


def load_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    )


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def task_to_caption(task: str) -> str:
    return str(task).replace("_", " ")


def load_moe(path: Path, device: torch.device) -> Tuple[MoeFuture, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    model = MoeFuture(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["heads"]),
        int(checkpoint["layers"]),
        int(checkpoint["num_experts"]),
        bool(checkpoint.get("future_aware_router", False)),
        len(checkpoint.get("task_vocab", {})),
        len(checkpoint.get("stage_names", STAGE_NAMES)),
        bool(checkpoint.get("condition_router", False)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.task_vocab = checkpoint.get("task_vocab", {})
    return model.to(device).eval(), checkpoint


def task_ids(tasks: Sequence[str], task_vocab: Dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([int(task_vocab.get(str(task), 0)) for task in tasks], dtype=torch.long, device=device)


def stage_ids(tasks: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([infer_stage(str(task)) for task in tasks], dtype=torch.long, device=device)


@torch.no_grad()
def predict_val_futures(
    model: MoeFuture,
    checkpoint: Dict[str, Any],
    features: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_start = normalize_rows(features["train_start"].astype(np.float32))
    train_end = normalize_rows(features["train_end"].astype(np.float32))
    val_start = normalize_rows(features["val_start"].astype(np.float32))
    train_tasks = features["train_tasks"].astype(str)
    val_tasks = features["val_tasks"].astype(str)
    k = int(checkpoint.get("k", 8))
    val_knn = topk_same_task(val_start, val_tasks, train_start, train_tasks, k, False)

    x = torch.from_numpy(val_start).to(device)
    dc = torch.from_numpy(train_start[val_knn]).to(device)
    df = torch.from_numpy(train_end[val_knn]).to(device)
    tids = sids = None
    if bool(checkpoint.get("condition_router", False)):
        tids = task_ids(val_tasks, checkpoint.get("task_vocab", {}), device)
        sids = stage_ids(val_tasks, device)
    futures, logits = model(x, dc, df, tids, sids)
    return futures.cpu().numpy(), logits.cpu().numpy(), val_knn


def nearest_future_bank(
    queries: np.ndarray,
    train_end: np.ndarray,
    train_tasks: Sequence[str],
    allowed_task: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if allowed_task is None:
        idx = np.arange(len(train_end), dtype=np.int64)
    else:
        idx = np.asarray([i for i, task in enumerate(train_tasks) if str(task) == str(allowed_task)], dtype=np.int64)
        if len(idx) == 0:
            idx = np.arange(len(train_end), dtype=np.int64)
    sims = queries @ train_end[idx].T
    local = np.argmax(sims, axis=1)
    chosen = idx[local]
    return chosen, sims[np.arange(len(queries)), local]


def write_prompts(
    path: Path,
    futures: np.ndarray,
    logits: np.ndarray,
    features: Dict[str, np.ndarray],
    caption_bank_scope: str,
) -> List[Dict[str, Any]]:
    train_end = normalize_rows(features["train_end"].astype(np.float32))
    val_start = normalize_rows(features["val_start"].astype(np.float32))
    train_tasks = features["train_tasks"].astype(str)
    val_tasks = features["val_tasks"].astype(str)
    router_idx = np.argmax(logits, axis=1)
    rows = []
    with path.open("w") as f:
        for sample_idx, task in enumerate(val_tasks):
            expert = int(router_idx[sample_idx])
            query = futures[sample_idx, expert : expert + 1]
            allowed_task = str(task) if caption_bank_scope == "same_task" else None
            nearest_idx, nearest_sim = nearest_future_bank(query, train_end, train_tasks, allowed_task)
            hyp_task = str(train_tasks[int(nearest_idx[0])])
            row = {
                "sample_index": int(sample_idx),
                "expert": expert,
                "task": str(task),
                "current_observation_summary": f"Current validation sample for task: {task_to_caption(str(task))}.",
                "future_hypothesis": {
                    "nearest_train_future_index": int(nearest_idx[0]),
                    "nearest_train_future_task": hyp_task,
                    "nearest_train_future_caption": task_to_caption(hyp_task),
                    "similarity_to_candidate_future": float(nearest_sim[0]),
                    "current_to_candidate_future_cosine": float(val_start[sample_idx] @ query[0]),
                },
                "llm_instruction": (
                    "Return JSON with keys: consistent (boolean), confidence (0-1), "
                    "corrected_task (snake_case or null), corrected_future_description (string or null)."
                ),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_decisions(path: Optional[Path]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if path is None:
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(int(row["sample_index"]), int(row["expert"]))] = row
    return out


def first_index_by_task(tasks: Iterable[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        out[str(task)].append(idx)
    return out


def corrected_embeddings(
    prompts: Sequence[Dict[str, Any]],
    decisions: Dict[Tuple[int, int], Dict[str, Any]],
    original_router_futures: np.ndarray,
    train_end: np.ndarray,
    train_tasks: Sequence[str],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    by_task = first_index_by_task(train_tasks)
    corrected = original_router_futures.copy()
    audit = []
    for row_idx, prompt in enumerate(prompts):
        key = (int(prompt["sample_index"]), int(prompt["expert"]))
        decision = decisions.get(key)
        encoded = False
        source = "original_no_decision"
        if decision is not None and not bool(decision.get("consistent", True)):
            corrected_task = decision.get("corrected_task")
            if corrected_task in by_task:
                candidates = np.asarray(by_task[str(corrected_task)], dtype=np.int64)
                sims = train_end[candidates] @ original_router_futures[row_idx]
                best = int(candidates[int(np.argmax(sims))])
                corrected[row_idx] = train_end[best]
                encoded = True
                source = "corrected_task_nearest_train_future"
            else:
                source = "correction_unencoded_missing_task"
        elif decision is not None:
            source = "consistent_noop"
        audit.append({"sample_index": key[0], "expert": key[1], "source": source, "encoded": encoded})
    return corrected, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LLM-refine probe for MoE V-JEPA future embeddings.")
    parser.add_argument("--features", type=Path, default=Path("outputs/vjepa2_retrieval_probe_n200/features.npz"))
    parser.add_argument("--moe", type=Path, default=Path("outputs/vjepa2_moe_future_aware_router_k8_e4_balanced/vjepa2_moe_future.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-decisions", type=Path, default=None)
    parser.add_argument("--caption-bank-scope", choices=["all", "same_task"], default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    features = load_features(args.features)
    features["train_start"] = normalize_rows(features["train_start"].astype(np.float32))
    features["train_end"] = normalize_rows(features["train_end"].astype(np.float32))
    features["val_start"] = normalize_rows(features["val_start"].astype(np.float32))
    features["val_end"] = normalize_rows(features["val_end"].astype(np.float32))

    model, checkpoint = load_moe(args.moe, device)
    futures, logits, _ = predict_val_futures(model, checkpoint, features, device)

    prompts = write_prompts(args.output_dir / "prompts.jsonl", futures, logits, features, args.caption_bank_scope)

    val_end = features["val_end"].astype(np.float32)
    router_idx = np.argmax(logits, axis=1)
    router_futures = futures[np.arange(len(futures)), router_idx]
    expert_cos = cosine_rows(futures, val_end[:, None, :])
    router_cos = cosine_rows(router_futures, val_end)
    oracle4_cos = np.max(expert_cos, axis=1)

    metrics: Dict[str, Any] = {
        "num_val": int(len(val_end)),
        "num_experts": int(futures.shape[1]),
        "features": str(args.features),
        "moe": str(args.moe),
        "caption_bank_scope": args.caption_bank_scope,
        "router_original_cosine": summarize(router_cos),
        "oracle_best_4_experts_cosine": summarize(oracle4_cos),
        "router_matches_oracle_expert": float(np.mean(router_idx == np.argmax(expert_cos, axis=1))),
        "prompts_jsonl": str(args.output_dir / "prompts.jsonl"),
    }

    decisions = load_decisions(args.llm_decisions)
    if decisions:
        corrected, audit = corrected_embeddings(
            prompts,
            decisions,
            router_futures,
            features["train_end"].astype(np.float32),
            features["train_tasks"].astype(str),
        )
        corrected_cos = cosine_rows(corrected, val_end)
        oracle5_cos = np.maximum(oracle4_cos, corrected_cos)
        metrics.update(
            {
                "llm_decisions": str(args.llm_decisions),
                "corrected_router_future_cosine": summarize(corrected_cos),
                "corrected_minus_original_cosine": summarize(corrected_cos - router_cos),
                "corrected_improved_rate": float(np.mean(corrected_cos > router_cos)),
                "corrected_degraded_rate": float(np.mean(corrected_cos < router_cos)),
                "oracle_best_5_with_corrected_cosine": summarize(oracle5_cos),
                "oracle5_minus_oracle4_cosine": summarize(oracle5_cos - oracle4_cos),
                "encoded_corrections": int(sum(1 for row in audit if row["encoded"])),
            }
        )
        (args.output_dir / "correction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
