import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from train_vjepa2_moe_future import MoeFuture, STAGE_NAMES, infer_stage, topk_same_task


def load_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def parse_task_list(raw: str):
    return [task.strip() for task in str(raw).split(",") if task.strip()]


def filter_split(start: np.ndarray, tasks: np.ndarray, include_tasks):
    if not include_tasks:
        return start, tasks
    selected = set(include_tasks)
    keep = np.asarray([str(task) in selected for task in tasks], dtype=bool)
    return start[keep], tasks[keep]


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


def ids_for_tasks(tasks: Sequence[str], checkpoint: Dict[str, Any], device: torch.device):
    if not bool(checkpoint.get("condition_router", False)):
        return None, None
    vocab = checkpoint.get("task_vocab", {})
    task_ids = torch.tensor([int(vocab.get(str(task), 0)) for task in tasks], dtype=torch.long, device=device)
    stage_ids = torch.tensor([infer_stage(str(task)) for task in tasks], dtype=torch.long, device=device)
    return task_ids, stage_ids


@torch.no_grad()
def predict_split(
    model: MoeFuture,
    checkpoint: Dict[str, Any],
    train_start: np.ndarray,
    train_end: np.ndarray,
    train_tasks: np.ndarray,
    split_start: np.ndarray,
    split_tasks: np.ndarray,
    split_name: str,
    device: torch.device,
):
    k = int(checkpoint.get("k", 8))
    exclude_self = split_name == "train" and len(split_start) == len(train_start)
    knn = topk_same_task(split_start, split_tasks, train_start, train_tasks, k, exclude_self)
    task_ids, stage_ids = ids_for_tasks(split_tasks, checkpoint, device)
    futures, logits = model(
        torch.from_numpy(split_start).to(device),
        torch.from_numpy(train_start[knn]).to(device),
        torch.from_numpy(train_end[knn]).to(device),
        task_ids,
        stage_ids,
    )
    return futures.cpu().numpy(), logits.cpu().numpy()


def nearest_future_task(query: np.ndarray, train_end: np.ndarray, train_tasks: np.ndarray) -> Tuple[int, str, float]:
    sims = train_end @ query
    idx = int(np.argmax(sims))
    return idx, str(train_tasks[idx]), float(sims[idx])


def write_prompts(path: Path, futures: np.ndarray, logits: np.ndarray, split_tasks: np.ndarray, train_end: np.ndarray, train_tasks: np.ndarray, mode: str) -> None:
    if mode == "router":
        experts = np.argmax(logits, axis=1)[:, None]
    elif mode == "all":
        experts = np.tile(np.arange(futures.shape[1])[None, :], (len(futures), 1))
    else:
        raise ValueError(f"unknown mode: {mode}")
    with path.open("w") as f:
        for sample_idx, row_experts in enumerate(experts):
            for expert in row_experts:
                expert = int(expert)
                idx, hyp_task, sim = nearest_future_task(futures[sample_idx, expert], train_end, train_tasks)
                row = {
                    "sample_index": int(sample_idx),
                    "expert": expert,
                    "task": str(split_tasks[sample_idx]),
                    "current_observation_summary": f"Current sample for task: {task_to_caption(str(split_tasks[sample_idx]))}.",
                    "future_hypothesis": {
                        "nearest_train_future_index": idx,
                        "nearest_train_future_task": hyp_task,
                        "nearest_train_future_caption": task_to_caption(hyp_task),
                        "similarity_to_candidate_future": sim,
                    },
                    "llm_instruction": (
                        "Return JSON with keys: consistent (boolean), confidence (0-1), "
                        "corrected_task (snake_case or null), corrected_future_description (string or null)."
                    ),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make train/val prompts for reflection correction.")
    parser.add_argument("--features", type=Path, default=Path("outputs/vjepa2_retrieval_probe_n200/features.npz"))
    parser.add_argument("--moe", type=Path, default=Path("outputs/vjepa2_moe_future_aware_router_k8_e4_balanced/vjepa2_moe_future.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--mode", choices=["router", "all"], default="router")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-tasks", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    features = load_features(args.features)
    train_start = normalize_rows(features["train_start"].astype(np.float32))
    train_end = normalize_rows(features["train_end"].astype(np.float32))
    val_start = normalize_rows(features["val_start"].astype(np.float32))
    train_tasks = features["train_tasks"].astype(str)
    val_tasks = features["val_tasks"].astype(str)
    include_tasks = parse_task_list(args.include_tasks)
    train_start, train_tasks = filter_split(train_start, train_tasks, include_tasks)
    train_end, _ = filter_split(train_end, features["train_tasks"].astype(str), include_tasks)
    val_start, val_tasks = filter_split(val_start, val_tasks, include_tasks)
    model, checkpoint = load_moe(args.moe, device)

    if args.split == "train":
        futures, logits = predict_split(model, checkpoint, train_start, train_end, train_tasks, train_start, train_tasks, "train", device)
        split_tasks = train_tasks
    else:
        futures, logits = predict_split(model, checkpoint, train_start, train_end, train_tasks, val_start, val_tasks, "val", device)
        split_tasks = val_tasks
    write_prompts(args.output_dir / f"{args.split}_{args.mode}_prompts.jsonl", futures, logits, split_tasks, train_end, train_tasks, args.mode)
    print(
        json.dumps(
            {
                "split": args.split,
                "mode": args.mode,
                "include_tasks": include_tasks,
                "num_samples": int(len(split_tasks)),
                "path": str(args.output_dir / f"{args.split}_{args.mode}_prompts.jsonl"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
