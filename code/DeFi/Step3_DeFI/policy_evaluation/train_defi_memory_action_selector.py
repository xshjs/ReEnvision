import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import T5TextEmbedder, iter_jsonl, split_sequence_indices


SELECTOR_CLASSES = ["continue", "recover", "retry_demo"]


class ActionSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_reflect_rows(path: Path) -> Dict[tuple[int, int], Dict[str, Any]]:
    rows = {}
    for row in iter_jsonl(path):
        rows[(int(row["sequence_index"]), int(row["subtask_index"]))] = row
    return rows


def load_oracle_examples(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def build_prev_context_text(prev_row: Dict[str, Any], current_task: str) -> str:
    return (
        f"current_task={current_task} ; "
        f"prev_task={prev_row.get('task', 'none')} ; "
        f"prev_trust={float(prev_row.get('trust_score', 0.5)):.3f} ; "
        f"prev_mismatch={prev_row.get('mismatch_type', 'unknown')} ; "
        f"prev_correction={prev_row.get('correction_direction', 'none')} ; "
        f"prev_recoverability={prev_row.get('recoverability', 'uncertain')} ; "
        f"prev_recent={prev_row.get('recent_summary', 'none')} ; "
        f"prev_key={prev_row.get('key_summary', 'none')}"
    )


def build_selector_input(
    state_feature: Sequence[float],
    scalars: Dict[str, float],
    context_embedding: np.ndarray,
    prev_row: Dict[str, Any],
) -> np.ndarray:
    scalar_vec = np.asarray([float(scalars[key]) for key in sorted(scalars)], dtype=np.float32)
    prev_numeric = np.asarray(
        [
            float(prev_row.get("trust_score", 0.5)),
            float(bool(prev_row.get("success", False))),
            float(prev_row.get("steps", 0)) / max(float(prev_row.get("ep_len", 1)), 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.asarray(state_feature, dtype=np.float32),
            scalar_vec,
            np.asarray(context_embedding, dtype=np.float32).reshape(-1),
            prev_numeric,
        ],
        axis=0,
    ).astype(np.float32)


def accuracy_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    acc = float((pred == labels).float().mean().item())
    per_class = {}
    for idx, name in enumerate(SELECTOR_CLASSES):
        mask = labels == idx
        if int(mask.sum().item()) == 0:
            per_class[name] = 0.0
        else:
            per_class[name] = float((pred[mask] == idx).float().mean().item())
    return {"acc": acc, "per_class_acc": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train next-step action selector conditioned on reflection memory.")
    parser.add_argument("--reflect-jsonl", type=Path, required=True)
    parser.add_argument("--oracle-selector-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t5-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    reflect_rows = load_reflect_rows(args.reflect_jsonl)
    oracle_rows = load_oracle_examples(args.oracle_selector_jsonl)

    kept_rows = []
    context_texts = []
    for row in oracle_rows:
        seq_idx = int(row["sequence_index"])
        sub_idx = int(row["subtask_index"])
        if sub_idx <= 0:
            continue
        prev_row = reflect_rows.get((seq_idx, sub_idx - 1))
        if prev_row is None:
            continue
        label = str(row["label"])
        if label not in SELECTOR_CLASSES:
            continue
        kept_rows.append((row, prev_row))
        context_texts.append(build_prev_context_text(prev_row, str(row["task"])))

    embedder = T5TextEmbedder(args.t5_path, device)
    context_embeddings = []
    batch = 32
    for start in range(0, len(context_texts), batch):
        encoded = embedder.encode(context_texts[start : start + batch]).detach().cpu().numpy().astype(np.float32)
        context_embeddings.append(encoded)
    context_embeddings_np = np.concatenate(context_embeddings, axis=0) if context_embeddings else np.zeros((0, 768), dtype=np.float32)

    x_rows = []
    y_rows = []
    seq_rows = []
    meta_rows = []
    for idx, (row, prev_row) in enumerate(kept_rows):
        x_rows.append(build_selector_input(row["state_feature"], row["scalars"], context_embeddings_np[idx], prev_row))
        y_rows.append(float(SELECTOR_CLASSES.index(str(row["label"]))))
        seq_rows.append(int(row["sequence_index"]))
        meta_rows.append(
            {
                "sequence_index": int(row["sequence_index"]),
                "subtask_index": int(row["subtask_index"]),
                "task": str(row["task"]),
                "label": str(row["label"]),
            }
        )

    x = np.stack(x_rows, axis=0).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    seq = np.asarray(seq_rows, dtype=np.int64)
    train_idx, val_idx = split_sequence_indices(seq)

    x_train_raw = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val_raw = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    mean = x_train_raw.mean(dim=0)
    std = x_train_raw.std(dim=0).clamp_min(1e-6)
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std

    model = ActionSelectorMLP(x.shape[1], args.hidden_dim, len(SELECTOR_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_counts = np.bincount(y_train.numpy(), minlength=len(SELECTOR_CLASSES)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_acc = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.to(device)).cpu()
        metrics = accuracy_metrics(val_logits, y_val)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_acc": metrics["acc"],
            }
        )
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "classes": SELECTOR_CLASSES,
        "mean": mean.numpy(),
        "std": std.numpy(),
        "t5_path": str(args.t5_path),
    }
    torch.save(checkpoint, args.output_dir / "action_selector.pt")

    with torch.no_grad():
        train_logits = model((x_train - mean) / std)
        val_logits = model((x_val - mean) / std)
    summary = {
        "train_num": int(len(train_idx)),
        "val_num": int(len(val_idx)),
        "train_metrics": accuracy_metrics(train_logits, y_train),
        "val_metrics": accuracy_metrics(val_logits, y_val),
        "history_tail": history[-5:],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "train_rows.jsonl").open("w") as handle:
        for row in meta_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
