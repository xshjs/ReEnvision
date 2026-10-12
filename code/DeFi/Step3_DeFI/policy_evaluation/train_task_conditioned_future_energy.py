from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from policy_evaluation.future_energy_model import (
    TaskConditionedFutureEnergy,
    build_energy_features_np,
)


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: str | Path, key: str) -> np.ndarray | None:
    try:
        data = np.load(str(path))
    except Exception:
        return None
    if key not in data.files:
        return None
    future = np.asarray(data[key], dtype=np.float32)
    return future if future.ndim == 2 else None


def load_examples(dataset_jsonl: Path, task_dim: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in iter_jsonl(dataset_jsonl):
        trace_path = row.get("trace_path")
        if not trace_path or not Path(str(trace_path)).exists():
            continue
        task = str(row.get("task", "unknown"))
        subtask_index = int(row.get("subtask_index", 0) or 0)
        base = load_future(trace_path, "base_future")
        target = load_future(trace_path, "target_proxy_future")
        final = load_future(trace_path, "final_future")
        if base is None:
            continue
        success = bool(row.get("success", False))
        examples.append(
            {
                "sequence_index": int(row.get("sequence_index", len(examples)) or 0),
                "subtask_index": subtask_index,
                "task": task,
                "success": success,
                "label_harm": bool(row.get("label_harm", False)),
                "label_improve": bool(row.get("label_improve", False)),
                "base": build_energy_features_np(base, task, subtask_index, task_dim),
                "target": build_energy_features_np(target, task, subtask_index, task_dim) if target is not None else None,
                "final": build_energy_features_np(final, task, subtask_index, task_dim) if final is not None else None,
            }
        )
    if not examples:
        raise ValueError("no usable examples")
    return examples


def split_by_sequence(examples: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    seqs = sorted({int(row["sequence_index"]) for row in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_seqs = set(seqs[:n_val])
    train, val = [], []
    for idx, row in enumerate(examples):
        (val if int(row["sequence_index"]) in val_seqs else train).append(idx)
    return train, val


def stack_features(examples: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.stack([np.asarray(row[key], dtype=np.float32) for row in examples]).astype(np.float32)


def make_pairs(examples: list[dict[str, Any]], indices: list[int], same_task: bool) -> list[tuple[int, int, str, str]]:
    success_by_task: dict[str, list[int]] = defaultdict(list)
    fail_by_task: dict[str, list[int]] = defaultdict(list)
    harm_by_task: dict[str, list[int]] = defaultdict(list)
    success_all, fail_all = [], []
    for idx in indices:
        row = examples[idx]
        if row["success"]:
            success_by_task[row["task"]].append(idx)
            success_all.append(idx)
        else:
            fail_by_task[row["task"]].append(idx)
            fail_all.append(idx)
        if row.get("label_harm") and row.get("final") is not None:
            harm_by_task[row["task"]].append(idx)
    pairs: list[tuple[int, int, str, str]] = []
    for task, fail_ids in fail_by_task.items():
        positives = success_by_task.get(task) if same_task else success_all
        if not positives:
            positives = success_all
        for fail_idx in fail_ids:
            for pos_idx in positives:
                pairs.append((pos_idx, fail_idx, task, "fail_base"))
    for task, harm_ids in harm_by_task.items():
        positives = success_by_task.get(task) if same_task else success_all
        if not positives:
            positives = success_all
        for harm_idx in harm_ids:
            for pos_idx in positives:
                pairs.append((pos_idx, harm_idx, task, "harm_final"))
    if not pairs:
        raise ValueError("no success/failure pairs")
    return pairs


def load_task_weights(path: Path | None) -> dict[str, float]:
    if path is None or not str(path):
        return {}
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, value in data.items():
        try:
            out[str(key)] = max(0.0, float(value))
        except Exception:
            continue
    return out


def compute_metrics(model, xz: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        e = model(torch.from_numpy(xz).to(device)).detach().cpu().numpy()
    y_fail = (y < 0.5).astype(np.int64)
    pos = e[y_fail == 1]
    neg = e[y_fail == 0]
    auc = 0.0
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    order = np.argsort(-e)
    top = {}
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(1, int(round(len(order) * frac)))
        ids = order[:k]
        top[f"top_{int(frac*100)}pct_fail_rate"] = float(y_fail[ids].mean())
        top[f"top_{int(frac*100)}pct_fail_count"] = int(y_fail[ids].sum())
        top[f"top_{int(frac*100)}pct_count"] = int(k)
    return {
        "energy_success_mean": float(e[y > 0.5].mean()) if (y > 0.5).any() else 0.0,
        "energy_fail_mean": float(e[y < 0.5].mean()) if (y < 0.5).any() else 0.0,
        "auc_fail_energy": auc,
        **top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train task-conditioned future energy model.")
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda-success", type=float, default=0.05)
    parser.add_argument("--lambda-fail", type=float, default=0.05)
    parser.add_argument("--fail-margin", type=float, default=1.0)
    parser.add_argument("--same-task-pairs", action="store_true")
    parser.add_argument("--hard-task-weights-json", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    examples = load_examples(args.dataset_jsonl, args.task_dim)
    train_idx, val_idx = split_by_sequence(examples, args.val_ratio, args.seed)
    pairs = make_pairs(examples, train_idx, args.same_task_pairs)
    task_weights = load_task_weights(args.hard_task_weights_json)

    x_base = stack_features(examples, "base")
    y = np.asarray([1.0 if row["success"] else 0.0 for row in examples], dtype=np.float32)
    mean = x_base[train_idx].mean(axis=0, keepdims=True)
    std = np.maximum(x_base[train_idx].std(axis=0, keepdims=True), 1e-6)
    xz_base = (x_base - mean) / std

    target_features = []
    final_features = []
    for row in examples:
        target_features.append(row["target"] if row["target"] is not None else row["base"])
        final_features.append(row["final"] if row["final"] is not None else row["base"])
    x_target = np.stack(target_features).astype(np.float32)
    xz_target = (x_target - mean) / std
    x_final = np.stack(final_features).astype(np.float32)
    xz_final = (x_final - mean) / std

    model = TaskConditionedFutureEnergy(x_base.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best = None
    best_state = None
    rng = np.random.default_rng(args.seed)

    train_base = torch.from_numpy(xz_base).to(device)
    train_target = torch.from_numpy(xz_target).to(device)
    train_final = torch.from_numpy(xz_final).to(device)
    pair_arr = np.asarray([(p, f, 1 if kind == "harm_final" else 0) for p, f, _, kind in pairs], dtype=np.int64)
    pair_weights = np.asarray([task_weights.get(task, 1.0) for _, _, task, _ in pairs], dtype=np.float64)
    pair_weights = pair_weights / max(float(pair_weights.sum()), 1e-12)

    for step in range(1, args.steps + 1):
        model.train()
        ids = rng.choice(len(pair_arr), size=max(1, args.batch_size), replace=True, p=pair_weights)
        pos_idx = torch.from_numpy(pair_arr[ids, 0]).to(device)
        neg_idx = torch.from_numpy(pair_arr[ids, 1]).to(device)
        neg_kind = torch.from_numpy(pair_arr[ids, 2]).to(device)
        e_pos = model(train_base[pos_idx])
        neg_base_features = train_base[neg_idx]
        neg_final_features = train_final[neg_idx]
        neg_features = torch.where((neg_kind > 0).view(-1, 1), neg_final_features, neg_base_features)
        e_fail = model(neg_features)
        loss_rank = F.relu(float(args.margin) + e_pos - e_fail).mean()
        e_target = model(train_target[pos_idx])
        loss_success = (e_pos.pow(2).mean() + e_target.pow(2).mean()) * 0.5
        loss_fail = F.relu(float(args.fail_margin) - e_fail).pow(2).mean()
        loss = loss_rank + float(args.lambda_success) * loss_success + float(args.lambda_fail) * loss_fail
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "rank_loss": float(loss_rank.detach().cpu()),
                "success_compactness": float(loss_success.detach().cpu()),
                "failure_margin_loss": float(loss_fail.detach().cpu()),
                **compute_metrics(model, xz_base[val_idx], y[val_idx], device),
            }
            history.append(record)
            score = record["auc_fail_energy"] + record["top_10pct_fail_rate"] - abs(record["energy_success_mean"])
            if best is None or score > best["score"]:
                best = {**record, "score": float(score)}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(record), flush=True)

    assert best is not None and best_state is not None
    ckpt_path = args.output_dir / "task_conditioned_future_energy.pt"
    torch.save(
        {
            "model_state": best_state,
            "input_dim": int(x_base.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "task_dim": int(args.task_dim),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "margin": float(args.margin),
        },
        ckpt_path,
    )
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "num_examples": len(examples),
        "success_count": int(y.sum()),
        "failure_count": int(len(y) - y.sum()),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "num_pairs": len(pairs),
        "same_task_pairs": bool(args.same_task_pairs),
        "task_counts": dict(Counter(row["task"] for row in examples).most_common()),
        "best": best,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
