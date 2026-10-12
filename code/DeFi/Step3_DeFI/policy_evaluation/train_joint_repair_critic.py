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


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def split_by_sequence(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(row.get("sequence_index", -1)) for row in rows if int(row.get("sequence_index", -1)) >= 0})
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio))) if seqs else 1
    val = set(seqs[:n_val])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if int(row.get("sequence_index", -1)) in val else train_idx).append(idx)
    return train_idx, val_idx


def make_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    tasks = sorted({str(row.get("task", "unknown")) for row in rows})
    task_to_id = {task: i for i, task in enumerate(tasks)}
    max_stage = max(float(row.get("stage", 0)) for row in rows)
    max_stage = max(max_stage, 1.0)

    xs = []
    advantages = []
    labels = []
    for row in rows:
        task_onehot = np.zeros((len(tasks),), dtype=np.float32)
        task_onehot[task_to_id[str(row.get("task", "unknown"))]] = 1.0
        stage = np.asarray([float(row.get("stage", 0)) / max_stage], dtype=np.float32)
        x = np.concatenate(
            [
                np.asarray(row["state"], dtype=np.float32),
                np.asarray(row["future"], dtype=np.float32),
                np.asarray(row["action"], dtype=np.float32),
                np.asarray(row["delta_future"], dtype=np.float32),
                np.asarray(row["delta_action"], dtype=np.float32),
                np.asarray(row["pending_effect"], dtype=np.float32),
                task_onehot,
                stage,
            ],
            axis=0,
        ).astype(np.float32)
        xs.append(x)
        advantages.append(float(row.get("advantage", 0.0)))
        labels.append(float(row.get("label", 0)))
    adv = np.asarray(advantages, dtype=np.float32)
    scale = max(float(np.abs(adv).mean()), 1e-6)
    return {
        "x": np.stack(xs, axis=0).astype(np.float32),
        "advantage": adv.astype(np.float32),
        "advantage_norm": (adv / scale).astype(np.float32),
        "label": np.asarray(labels, dtype=np.float32),
        "scale": np.asarray([scale], dtype=np.float32),
        "task_names": np.asarray(tasks, dtype=object),
    }


class JointRepairCriticMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adv_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "advantage": self.adv_head(h).squeeze(-1),
            "logit": self.cls_head(h).squeeze(-1),
        }


def evaluate(model: JointRepairCriticMLP, x: torch.Tensor, adv: torch.Tensor, label: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred_adv = out["advantage"]
        pred_label = (torch.sigmoid(out["logit"]) > 0.5).float()
        mse = F.mse_loss(pred_adv, adv)
        mae = F.l1_loss(pred_adv, adv)
        acc = float((pred_label == label).float().mean().item())
        if adv.numel() > 1:
            corr = float(torch.corrcoef(torch.stack([pred_adv, adv], dim=0))[0, 1].item())
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
    return {"loss": float(mse.item()), "mse": float(mse.item()), "mae": float(mae.item()), "corr": corr, "cls_acc": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train critic A_t for repair gating.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = load_rows(args.rows_jsonl)
    arrays = make_arrays(rows)
    train_idx, val_idx = split_by_sequence(rows, args.val_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(arrays["x"]).to(device)
    adv = torch.from_numpy(arrays["advantage_norm"]).to(device)
    label = torch.from_numpy(arrays["label"]).to(device)

    x_train, adv_train, label_train = x[train_idx], adv[train_idx], label[train_idx]
    x_val, adv_val, label_val = x[val_idx], adv[val_idx], label[val_idx]

    model = JointRepairCriticMLP(int(x.shape[1]), int(args.hidden_dim), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_ids = torch.randint(0, x_train.shape[0], (min(int(args.batch_size), x_train.shape[0]),), device=device)
        out = model(x_train[batch_ids])
        adv_loss = F.mse_loss(out["advantage"], adv_train[batch_ids])
        cls_loss = F.binary_cross_entropy_with_logits(out["logit"], label_train[batch_ids])
        loss = float(args.lambda_adv) * adv_loss + float(args.lambda_cls) * cls_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == args.steps:
            metrics = {"step": step, "train_loss": float(loss.item()), **evaluate(model, x_val, adv_val, label_val)}
            history.append(metrics)
            if best is None or metrics["loss"] < best["loss"]:
                best = dict(metrics)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": int(x.shape[1]),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "advantage_scale": float(arrays["scale"][0]),
                        "task_names": arrays["task_names"].tolist(),
                    },
                    args.output_dir / "joint_repair_critic.pt",
                )
            print(json.dumps(metrics, ensure_ascii=False), flush=True)

    summary = {
        "rows_jsonl": str(args.rows_jsonl),
        "num_rows": int(len(rows)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "advantage_scale": float(arrays["scale"][0]),
        "best": best,
        "history": history,
        "output_ckpt": str(args.output_dir / "joint_repair_critic.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
