import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import T5TextEmbedder, iter_jsonl, split_sequence_indices


class TrustHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_row_text(row: Dict[str, Any]) -> str:
    return (
        f"task={row.get('task', 'unknown')} ; "
        f"correction={row.get('correction_direction', 'none')} ; "
        f"recoverability={row.get('recoverability', 'uncertain')} ; "
        f"recent={row.get('recent_summary', 'none')} ; "
        f"key={row.get('key_summary', 'none')} ; "
        f"rule={row.get('rule_text', 'none')}"
    )


def build_numeric(row: Dict[str, Any], ep_len: int) -> np.ndarray:
    recoverability = str(row.get("recoverability", "uncertain"))
    return np.asarray(
        [
            float(row.get("trust_score", 0.5)),
            1.0 if recoverability == "repairable" else 0.0,
            1.0 if recoverability == "uncertain" else 0.0,
            1.0 if recoverability == "not_repairable" else 0.0,
        ],
        dtype=np.float32,
    )


def label_row(row: Dict[str, Any]) -> int:
    return 0 if bool(row.get("success", False)) else 1


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    labels_bool = labels >= 0.5
    tp = int((pred & labels_bool).sum().item())
    fp = int((pred & ~labels_bool).sum().item())
    fn = int((~pred & labels_bool).sum().item())
    tn = int((~pred & ~labels_bool).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(labels.float().mean().item()) if len(labels) else 0.0,
        "pred_positive_rate": float(pred.float().mean().item()) if len(labels) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train trust/risk head from DeFi reflect-only rollout rows.")
    parser.add_argument("--reflect-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t5-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows: List[Dict[str, Any]] = list(iter_jsonl(args.reflect_jsonl))
    texts = [build_row_text(row) for row in rows]
    embedder = T5TextEmbedder(args.t5_path, device)
    embeddings = []
    batch = 32
    for start in range(0, len(texts), batch):
        encoded = embedder.encode(texts[start : start + batch]).detach().cpu().numpy().astype(np.float32)
        embeddings.append(encoded)
    text_x = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 768), dtype=np.float32)
    numeric_x = np.stack([build_numeric(row, args.ep_len) for row in rows], axis=0).astype(np.float32)
    x = np.concatenate([text_x, numeric_x], axis=1).astype(np.float32)
    y = np.asarray([label_row(row) for row in rows], dtype=np.float32)
    seq = np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int64)

    train_idx, val_idx = split_sequence_indices(seq)
    x_train_raw = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val_raw = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    mean = x_train_raw.mean(dim=0)
    std = x_train_raw.std(dim=0).clamp_min(1e-6)
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std

    model = TrustHeadMLP(x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_count = float(y_train.sum().item())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_f1 = -1.0
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
        metrics = binary_metrics(val_logits, y_val)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_f1": metrics["f1"],
                "val_acc": metrics["acc"],
            }
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": int(x.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "mean": mean.numpy(),
        "std": std.numpy(),
        "t5_path": str(args.t5_path),
        "label": "risk = rollout failure",
    }
    torch.save(checkpoint, args.output_dir / "trust_head.pt")

    with torch.no_grad():
        train_logits = model((x_train_raw - mean) / std)
        val_logits = model((x_val_raw - mean) / std)
    summary = {
        "train_num": int(len(train_idx)),
        "val_num": int(len(val_idx)),
        "positive_train": int(y_train.sum().item()),
        "positive_val": int(y_val.sum().item()),
        "train_metrics": binary_metrics(train_logits, y_train),
        "val_metrics": binary_metrics(val_logits, y_val),
        "history_tail": history[-5:],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
