from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "final_future" and "target_proxy_future" in data:
                arr = np.asarray(data["target_proxy_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def future_summary(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    first = arr[0]
    last = arr[-1]
    delta = last - first
    return np.concatenate([mean, std, first, delta], axis=0).astype(np.float32)


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row["sequence_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    val = set(seqs[: max(1, int(round(len(seqs) * val_ratio)))])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row["sequence_index"]) in val else train_idx).append(idx)
    return train_idx, val_idx


def build_rows(rows_path: Path, memory_npz: Path) -> list[dict[str, Any]]:
    with np.load(memory_npz, allow_pickle=True) as data:
        action_chunks = np.asarray(data["action_chunk"], dtype=np.float32)
    raw_rows = list(iter_jsonl(rows_path))
    tasks = sorted({str(row.get("task", "unknown")) for row in raw_rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}
    rows = []
    for row in raw_rows:
        trace_path = Path(str(row.get("trace_path", "") or ""))
        mem_idx = int(row.get("joint_pair_selector_memory_index", -1))
        if not trace_path.exists() or mem_idx < 0 or mem_idx >= len(action_chunks):
            continue
        base = load_future(trace_path, "base_future")
        final = load_future(trace_path, "final_future")
        if base is None or final is None:
            continue
        task_onehot = np.zeros((len(tasks),), dtype=np.float32)
        task_onehot[task_to_id[str(row.get("task", "unknown"))]] = 1.0
        stage = np.asarray([float(row.get("subtask_index", 0)) / 5.0], dtype=np.float32)
        meta = np.asarray(
            [
                float(row.get("joint_pair_selector_energy", 0.0)),
                float(row.get("joint_pair_selector_energy_margin", 0.0)),
                float(row.get("joint_pair_selector_fused_score", 0.0)),
                float(row.get("joint_pair_selector_fused_margin", 0.0)),
                float(row.get("joint_pair_selector_similarity", 0.0)),
                float(row.get("joint_pair_selector_dynamics_score", 0.0)),
                float(row.get("joint_pair_selector_dynamics_cos", 0.0)),
                float(row.get("joint_pair_selector_dynamics_l1", 0.0)),
                float(row.get("joint_pair_repair_future_shift_norm", 0.0)),
                float(row.get("joint_pair_repair_action_shift_norm", 0.0)),
                float(row.get("joint_pair_verify_c_exp", 0.0)),
                float(row.get("joint_pair_verify_c_latent", 0.0)),
                float(row.get("joint_pair_verify_c_physical", 0.0)),
                float(row.get("joint_pair_verify_delta_c_exp", 0.0)),
                float(row.get("joint_pair_verify_delta_c_latent", 0.0)),
                float(row.get("joint_pair_verify_delta_c_physical", 0.0)),
                float(int(bool(row.get("joint_pair_verify_passed", False)))),
                float(int(bool(row.get("joint_pair_repair_used", False)))),
            ],
            dtype=np.float32,
        )
        feat = np.concatenate(
            [
                future_summary(base),
                future_summary(final),
                action_chunks[mem_idx].reshape(-1).astype(np.float32),
                task_onehot,
                stage,
                meta,
            ],
            axis=0,
        ).astype(np.float32)
        rows.append(
            {
                "sequence_index": int(row.get("sequence_index", -1)),
                "x": feat,
                "y": float(bool(row.get("success", False))),
            }
        )
    if not rows:
        raise ValueError("no usable joint acceptance rows")
    return rows


class AcceptanceMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def tensorize(rows: list[dict[str, Any]], indices: list[int], device: torch.device):
    x = torch.from_numpy(np.stack([rows[i]["x"] for i in indices], axis=0)).to(device)
    y = torch.tensor([rows[i]["y"] for i in indices], dtype=torch.float32, device=device)
    return x, y


def evaluate(model: AcceptanceMLP, x: torch.Tensor, y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        acc = float((pred == y).float().mean().item())
        tp = int(((pred == 1) & (y == 1)).sum().item())
        fp = int(((pred == 1) & (y == 0)).sum().item())
        fn = int(((pred == 0) & (y == 1)).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        loss = float(F.binary_cross_entropy_with_logits(logits, y).item())
    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Train unified acceptance model for repaired joint hypothesis.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args.rows_jsonl, args.memory_npz)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = tensorize(rows, train_idx, device)
    x_val, y_val = tensorize(rows, val_idx, device)
    model = AcceptanceMLP(x_train.shape[-1], int(args.hidden_dim), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_ids = torch.randint(0, x_train.shape[0], (min(int(args.batch_size), x_train.shape[0]),), device=device)
        logits = model(x_train[batch_ids])
        loss = F.binary_cross_entropy_with_logits(logits, y_train[batch_ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == args.steps:
            metrics = {"step": step, "train_loss": float(loss.item()), **evaluate(model, x_val, y_val)}
            history.append(metrics)
            if best is None or metrics["f1"] > best["f1"]:
                best = metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": int(x_train.shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                    },
                    args.output_dir / "joint_acceptance_model.pt",
                )

    summary = {
        "num_rows": len(rows),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "joint_acceptance_model.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
