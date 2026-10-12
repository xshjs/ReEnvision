import argparse
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if "distutils.spawn" not in sys.modules:
    distutils_mod = sys.modules.setdefault("distutils", types.ModuleType("distutils"))
    spawn_mod = types.ModuleType("distutils.spawn")
    spawn_mod.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_mod
    setattr(distutils_mod, "spawn", spawn_mod)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_universal_moe_oracle_router import (
    BASE_FEATURE_LAYOUT,
    OracleRouter,
    evaluate,
    filter_rows_by_sequence,
    load_rows,
    split_indices,
    target_labels,
)

from future_reflection_correction_mlp import encode_reflections_qwen, encode_reflections_t5


def parse_task_list(raw: str) -> List[str]:
    return [task.strip() for task in str(raw).split(",") if task.strip()]


def filter_rows_by_task(
    rows: List[Tuple[Path, Dict[str, object]]],
    include_tasks: Sequence[str],
) -> List[Tuple[Path, Dict[str, object]]]:
    if not include_tasks:
        return rows
    selected = set(include_tasks)
    return [(npz_path, row) for npz_path, row in rows if str(row.get("task")) in selected]


def load_base_arrays(rows: List[Tuple[Path, Dict[str, object]]]) -> Dict[str, np.ndarray]:
    x_rows = []
    value_rows = []
    success_rows = []
    steps_rows = []
    oracle_rows = []
    chosen_rows = []
    group_rows = []
    task_rows = []
    meta_rows = []
    for npz_path, row in rows:
        with np.load(npz_path) as data:
            obs = data["obs_embedding"].astype(np.float32)
            futures = data["expert_futures"].astype(np.float32)
            future_delta = futures - obs[None, :]
            router_logits = data["router_logits"].astype(np.float32).reshape(-1, 1)
            router_scores = data["router_scores"].astype(np.float32).reshape(-1, 1)
            contradiction = data["expert_contradiction"].astype(np.float32).reshape(-1, 1)
            feature = np.concatenate(
                [
                    np.repeat(obs[None, :], futures.shape[0], axis=0),
                    futures,
                    future_delta,
                    router_logits,
                    router_scores,
                    contradiction,
                ],
                axis=1,
            )
            x_rows.append(feature)
            value_rows.append(data["expert_values"].astype(np.float32))
            success_rows.append(data["expert_success"].astype(np.float32))
            steps_rows.append(data["expert_steps"].astype(np.float32))
            oracle_rows.append(int(data["oracle_expert"][0]))
            chosen_rows.append(int(data["chosen_expert"][0]))
            global_seq = int(row.get("_global_sequence_index", row.get("sequence_index", -1)))
            subtask_idx = int(row.get("subtask_index", -1))
            group_rows.append(f"{global_seq}:{subtask_idx}")
            task_rows.append(str(row.get("task")))
            meta_rows.append(
                {
                    "global_sequence_index": global_seq,
                    "subtask_index": subtask_idx,
                    "task": str(row.get("task")),
                }
            )
    return {
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "values": np.stack(value_rows, axis=0).astype(np.float32),
        "success": np.stack(success_rows, axis=0).astype(np.float32),
        "steps": np.stack(steps_rows, axis=0).astype(np.float32),
        "oracle": np.asarray(oracle_rows, dtype=np.int64),
        "chosen": np.asarray(chosen_rows, dtype=np.int64),
        "groups": np.asarray(group_rows),
        "tasks": np.asarray(task_rows),
        "meta": meta_rows,
    }


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros((size,), dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def previous_outcome_features(data: Dict[str, np.ndarray], ep_len: float) -> np.ndarray:
    meta = data["meta"]
    key_to_idx = {
        (int(row["global_sequence_index"]), int(row["subtask_index"])): idx
        for idx, row in enumerate(meta)
    }
    num_rows, num_experts = data["success"].shape
    reflection_rows = []
    for idx, row in enumerate(meta):
        prev_idx = key_to_idx.get((int(row["global_sequence_index"]), int(row["subtask_index"]) - 1))
        if prev_idx is None:
            reflection = np.zeros((22,), dtype=np.float32)
            reflection[0] = 0.0
        else:
            chosen = int(data["chosen"][prev_idx])
            oracle = int(data["oracle"][prev_idx])
            prev_success = data["success"][prev_idx]
            prev_steps = data["steps"][prev_idx] / float(ep_len)
            chosen_success = float(prev_success[chosen]) if 0 <= chosen < num_experts else 0.0
            chosen_steps = float(prev_steps[chosen]) if 0 <= chosen < num_experts else 1.0
            reflection = np.concatenate(
                [
                    np.asarray([1.0, chosen_success, chosen_steps], dtype=np.float32),
                    one_hot(chosen, num_experts),
                    one_hot(oracle, num_experts),
                    prev_success.astype(np.float32),
                    prev_steps.astype(np.float32),
                    np.asarray(
                        [
                            float(chosen == oracle),
                            float(prev_success.sum() / max(num_experts, 1)),
                            float(data["values"][prev_idx].max()),
                        ],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        reflection_rows.append(reflection)
    reflection_arr = np.stack(reflection_rows, axis=0).astype(np.float32)
    if reflection_arr.shape != (num_rows, 22):
        raise ValueError(f"unexpected reflection shape {reflection_arr.shape}")
    return reflection_arr


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_qwen_reflection_texts(path: Path) -> Dict[Tuple[int, int], str]:
    rows: Dict[Tuple[int, int], str] = {}
    for row in iter_jsonl(path):
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        text = row.get("reflection_text")
        if not text:
            fields = [
                "progress_status",
                "failure_type",
                "prefer_next",
                "avoid_next",
                "correction_direction",
                "memory_query",
            ]
            text = " | ".join(f"{field}: {row.get(field, 'uncertain')}" for field in fields)
        rows[key] = str(text)
    return rows


def qwen_texts_for_rows(data: Dict[str, np.ndarray], reflections_path: Path) -> List[str]:
    reflection_map = load_qwen_reflection_texts(reflections_path)
    texts = []
    missing = 0
    for row in data["meta"]:
        key = (int(row["global_sequence_index"]), int(row["subtask_index"]))
        text = reflection_map.get(key)
        if text is None:
            missing += 1
            text = (
                f"progress_status: uncertain | failure_type: uncertain | "
                f"prefer_next: choose the expert most likely to complete {row['task']} | "
                "avoid_next: avoid failed or slow futures | "
                "correction_direction: use rollout utility | "
                f"memory_query: {row['task']}"
            )
        texts.append(text)
    if missing:
        print(json.dumps({"qwen_reflection_missing": int(missing), "total": len(texts)}), flush=True)
    return texts


def encode_qwen_conditioning(
    texts: Sequence[str],
    encoder_type: str,
    t5_path: Path,
    qwen_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if encoder_type == "t5":
        return encode_reflections_t5(texts, t5_path, device, batch_size)
    if encoder_type == "qwen":
        return encode_reflections_qwen(texts, qwen_path, device, batch_size)
    raise ValueError(f"unknown reflection encoder type: {encoder_type}")


def apply_conditioning(
    data: Dict[str, np.ndarray],
    mode: str,
    ep_len: float,
    qwen_reflections: Path = None,
    reflection_encoder_type: str = "t5",
    t5_path: Path = Path("ckpts/t5_base"),
    qwen_path: Path = Path("/mnt/data/manipulation/llm_models/Qwen3-8B"),
    text_batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if mode == "none":
        return data, list(BASE_FEATURE_LAYOUT)
    if mode == "outcome":
        reflection = previous_outcome_features(data, ep_len)
        layout_suffix = [
            "prev_has_outcome",
            "prev_chosen_success",
            "prev_chosen_steps_norm",
            "prev_chosen_expert_onehot",
            "prev_oracle_expert_onehot",
            "prev_expert_success",
            "prev_expert_steps_norm",
            "prev_chosen_is_oracle",
            "prev_success_count_norm",
            "prev_best_value",
        ]
    elif mode == "qwen":
        if qwen_reflections is None:
            raise ValueError("--qwen-reflections is required when --mode=qwen")
        texts = qwen_texts_for_rows(data, qwen_reflections)
        reflection = encode_qwen_conditioning(
            texts,
            reflection_encoder_type,
            t5_path,
            qwen_path,
            device,
            text_batch_size,
        )
        layout_suffix = [f"qwen_reflection_{reflection_encoder_type}_embedding"]
    else:
        raise ValueError(f"unknown mode: {mode}")
    repeated = np.repeat(reflection[:, None, :], data["x"].shape[1], axis=1)
    out = dict(data)
    out["x"] = np.concatenate([data["x"], repeated], axis=-1).astype(np.float32)
    layout = list(BASE_FEATURE_LAYOUT) + layout_suffix
    return out, layout


def train_model(data: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[OracleRouter, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels_np, target_score_np = target_labels(data, args.target, args.step_lambda, args.ep_len)
    train_idx, val_idx = split_indices(data["groups"], args.val_fraction, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OracleRouter(data["x"].shape[-1], args.hidden_dim, data["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(data["x"]).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    values = torch.from_numpy(data["values"]).to(device)
    train_t = torch.from_numpy(train_idx).to(device)
    val_t = torch.from_numpy(val_idx).to(device)
    best_state = None
    best_val = -float("inf")
    for _ in range(args.epochs):
        model.train()
        scores = model(x[train_t])
        ce = F.cross_entropy(scores, labels[train_t])
        probs = F.softmax(scores, dim=1)
        pred_value = torch.sum(probs * values[train_t], dim=1)
        loss = ce - float(args.value_weight) * pred_value.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(x[val_t])
            val_pred = val_scores.argmax(dim=1)
            val_value = values[val_t][torch.arange(len(val_t), device=device), val_pred].mean()
        if float(val_value) > best_val:
            best_val = float(val_value)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores_np = model(x).detach().cpu().numpy()
    return model, scores_np, labels_np, target_score_np, (train_idx, val_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train outcome/reflection-conditioned expert selector ablations.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["none", "outcome", "qwen"], required=True)
    parser.add_argument("--qwen-reflections", type=Path, default=None)
    parser.add_argument("--reflection-encoder-type", choices=["t5", "qwen"], default="t5")
    parser.add_argument("--t5-path", type=Path, default=Path("ckpts/t5_base"))
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/data/manipulation/llm_models/Qwen3-8B"))
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=["success", "utility", "value"], default="utility")
    parser.add_argument("--step-lambda", type=float, default=0.05)
    parser.add_argument("--ep-len", type=float, default=160.0)
    parser.add_argument("--exclude-sequence-start", type=int, default=-1)
    parser.add_argument("--exclude-sequence-end", type=int, default=-1)
    parser.add_argument("--include-tasks", default="")
    parser.add_argument("--value-weight", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.roots)
    rows_before_filter = len(rows)
    rows = filter_rows_by_sequence(rows, args.exclude_sequence_start, args.exclude_sequence_end)
    include_tasks = parse_task_list(args.include_tasks)
    rows = filter_rows_by_task(rows, include_tasks)
    if not rows:
        raise ValueError("No Universal MoE rows found.")
    data = load_base_arrays(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, feature_layout = apply_conditioning(
        data,
        args.mode,
        args.ep_len,
        args.qwen_reflections,
        args.reflection_encoder_type,
        args.t5_path,
        args.qwen_path,
        args.text_batch_size,
        device,
    )
    model, scores_np, labels_np, target_score_np, (train_idx, val_idx) = train_model(data, args)
    metrics = {
        "mode": args.mode,
        "target": args.target,
        "step_lambda": float(args.step_lambda),
        "ep_len": float(args.ep_len),
        "num_samples": int(len(data["x"])),
        "num_samples_before_sequence_filter": int(rows_before_filter),
        "include_tasks": include_tasks,
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "feature_dim": int(data["x"].shape[-1]),
        "feature_layout": feature_layout,
        "qwen_reflections": str(args.qwen_reflections) if args.qwen_reflections is not None else None,
        "reflection_encoder_type": args.reflection_encoder_type,
        "target_label_counts": {
            str(expert): int(np.sum(labels_np == expert))
            for expert in range(data["x"].shape[1])
        },
        "target_score_mean": float(target_score_np.mean()),
        "train": evaluate(scores_np, data, train_idx, labels_np),
        "val": evaluate(scores_np, data, val_idx, labels_np),
        "all": evaluate(scores_np, data, np.arange(len(data["x"]), dtype=np.int64), labels_np),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_dim": int(data["x"].shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "num_experts": int(data["x"].shape[1]),
            "target": args.target,
            "step_lambda": float(args.step_lambda),
            "ep_len": float(args.ep_len),
            "feature_layout": feature_layout,
            "conditioning_mode": args.mode,
            "qwen_reflections": str(args.qwen_reflections) if args.qwen_reflections is not None else None,
            "reflection_encoder_type": args.reflection_encoder_type,
            "value_weight": float(args.value_weight),
        },
        args.output_dir / "reflection_conditioned_router.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
