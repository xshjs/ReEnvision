import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


class PairwiseRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_groups(meta: np.ndarray) -> np.ndarray:
    rows = [json.loads(str(row)) for row in meta]
    return np.asarray([f'{row["sequence_index"]}:{row["subtask_index"]}' for row in rows])


def make_pairs(x: np.ndarray, y: np.ndarray, groups: np.ndarray, meta: np.ndarray, disagreements_only: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: List[int] = []
    right: List[int] = []
    labels: List[float] = []
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        if len(idx) < 2:
            continue
        router_best = idx[np.argmin([json.loads(str(meta[item]))["router_rank"] for item in idx])]
        for pos_a in range(len(idx)):
            for pos_b in range(pos_a + 1, len(idx)):
                a, b = int(idx[pos_a]), int(idx[pos_b])
                if float(y[a]) == float(y[b]):
                    continue
                best = a if y[a] > y[b] else b
                if disagreements_only and int(best) == int(router_best):
                    continue
                left.append(a)
                right.append(b)
                labels.append(1.0 if y[a] > y[b] else 0.0)
    return x[np.asarray(left)], x[np.asarray(right)], np.asarray(labels, dtype=np.float32)


def rank_metrics(model: PairwiseRanker, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    with torch.no_grad():
        scores = model(torch.from_numpy(x)).numpy()
    hits = []
    regrets = []
    useful = 0
    for group in sorted(set(groups.tolist())):
        idx = np.where(groups == group)[0]
        if len(idx) < 2:
            continue
        if float(y[idx[0]]) != float(y[idx[1]]):
            useful += 1
        chosen = idx[int(np.argmax(scores[idx]))]
        best = idx[int(np.argmax(y[idx]))]
        hits.append(float(chosen == best))
        regrets.append(float(y[best] - y[chosen]))
    return {
        "top1_match": float(np.mean(hits)) if hits else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "num_groups": int(len(hits)),
        "useful_groups": int(useful),
    }


def split_groups(groups: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(groups.tolist())))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_groups = set(unique[:n_val].tolist())
    val_idx = np.asarray([idx for idx, group in enumerate(groups) if group in val_groups], dtype=np.int64)
    train_idx = np.asarray([idx for idx, group in enumerate(groups) if group not in val_groups], dtype=np.int64)
    return train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise Reflection ranker for top2 future reranking.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--disagreements-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    meta = data["meta"]
    meta_global = meta
    groups = parse_groups(meta_global)
    train_idx, val_idx = split_groups(groups, args.val_fraction, args.seed)

    x_train, y_train, g_train = x[train_idx], y[train_idx], groups[train_idx]
    x_val, y_val, g_val = x[val_idx], y[val_idx], groups[val_idx]
    xa, xb, labels = make_pairs(x_train, y_train, g_train, meta_global[train_idx], args.disagreements_only)
    if len(labels) == 0:
        raise ValueError("No pairwise training pairs were produced.")

    model = PairwiseRanker(x.shape[1], args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ta = torch.from_numpy(xa)
    tb = torch.from_numpy(xb)
    tl = torch.from_numpy(labels)
    best_state = None
    best_val = -float("inf")
    for _ in range(args.epochs):
        model.train()
        margin = model(ta) - model(tb)
        loss = nn.functional.binary_cross_entropy_with_logits(margin, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        val_metric = rank_metrics(model, x_val, y_val, g_val)["top1_match"]
        if val_metric >= best_val:
            best_val = val_metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "num_examples": int(len(x)),
        "num_groups": int(len(set(groups.tolist()))),
        "num_train_pairs": int(len(labels)),
        "input_dim": int(x.shape[1]),
        "disagreements_only": bool(args.disagreements_only),
        "rank_train": rank_metrics(model, x_train, y_train, g_train),
        "rank_val": rank_metrics(model, x_val, y_val, g_val),
        "rank_all": rank_metrics(model, x, y, groups),
    }
    torch.save(
        {"model_state": model.state_dict(), "input_dim": int(x.shape[1]), "hidden_dim": args.hidden_dim},
        args.output_dir / "reflection_pairwise_ranker.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
