from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from policy_evaluation.summary_conditioned_future_adapter import SummaryConditionedFutureAdapter


_FUTURE_CACHE: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
_FUTURE_CACHE_MAX = 256


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_memory(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_future(trace_path: str | Path, key: str = "target_proxy_future") -> np.ndarray | None:
    cache_key = (str(trace_path), str(key))
    cached = _FUTURE_CACHE.get(cache_key)
    if cached is not None:
        _FUTURE_CACHE.move_to_end(cache_key)
        return cached
    try:
        with np.load(str(trace_path)) as data:
            if key in data.files:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "target_proxy_future" and "final_future" in data.files:
                arr = np.asarray(data["final_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    _FUTURE_CACHE[cache_key] = arr
    _FUTURE_CACHE.move_to_end(cache_key)
    while len(_FUTURE_CACHE) > _FUTURE_CACHE_MAX:
        _FUTURE_CACHE.popitem(last=False)
    return arr


def future_summary_exec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def load_examples(pairs_jsonl: Path, memory_npz: Path | None, max_examples: int) -> list[dict[str, Any]]:
    memory = load_memory(memory_npz) if memory_npz is not None else None
    if memory is not None and ("trace_path" not in memory or "target_summary_exec" not in memory):
        raise ValueError("memory npz must contain trace_path and target_summary_exec")
    rows = []
    for pair in iter_jsonl(pairs_jsonl):
        if "base_trace_path" in pair:
            base_trace = str(pair["base_trace_path"])
            target_trace = str(pair.get("target_trace_path", base_trace))
            base_key = str(pair.get("base_future_key", "target_proxy_future"))
            target_key = str(pair.get("target_future_key", "target_proxy_future"))
            neg_future = load_future(base_trace, key=base_key)
            pos_future = load_future(target_trace, key=target_key)
            if neg_future is None or pos_future is None or neg_future.shape != pos_future.shape:
                continue
            source_summary = np.asarray(
                pair.get("source_summary", future_summary_exec(neg_future)),
                dtype=np.float32,
            ).reshape(-1)
            target_summary = np.asarray(
                pair.get("target_summary", future_summary_exec(pos_future)),
                dtype=np.float32,
            ).reshape(-1)
            rows.append(
                {
                    "sequence_index": int(pair.get("sequence_index", len(rows))),
                    "task": str(pair.get("task", "unknown")),
                    "base_trace_path": base_trace,
                    "target_trace_path": target_trace,
                    "base_future_key": base_key,
                    "target_future_key": target_key,
                    "source_summary": source_summary,
                    "target_summary": target_summary,
                }
            )
            if max_examples > 0 and len(rows) >= max_examples:
                break
            continue
        if memory is None:
            raise ValueError("memory npz is required for index-based pairs")
        neg_idx = int(pair["memory_index_neg"])
        pos_idx = int(pair["memory_index_pos"])
        neg_trace = str(memory["trace_path"][neg_idx])
        pos_trace = str(memory["trace_path"][pos_idx])
        neg_future = load_future(neg_trace)
        pos_future = load_future(pos_trace)
        if neg_future is None or pos_future is None or neg_future.shape != pos_future.shape:
            continue
        source_summary = np.asarray(memory["target_summary_exec"][neg_idx], dtype=np.float32).reshape(-1)
        target_summary = np.asarray(memory["target_summary_exec"][pos_idx], dtype=np.float32).reshape(-1)
        # Skip rows if the saved summary no longer matches the trace shape.
        if source_summary.shape[0] != future_summary_exec(neg_future).shape[0]:
            continue
        rows.append(
            {
                "sequence_index": int(pair.get("sequence_index_neg", len(rows))),
                "task": str(pair.get("task", "unknown")),
                "base_trace_path": neg_trace,
                "target_trace_path": pos_trace,
                "base_future_key": "target_proxy_future",
                "target_future_key": "target_proxy_future",
                "source_summary": source_summary,
                "target_summary": target_summary,
            }
        )
        if max_examples > 0 and len(rows) >= max_examples:
            break
    if not rows:
        raise ValueError("no usable adapter examples")
    return rows


def split(rows: list[dict[str, Any]], val_ratio: float, seed: int):
    seqs = sorted({int(r["sequence_index"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_set = set(seqs[:n_val])
    train, val = [], []
    for i, row in enumerate(rows):
        (val if int(row["sequence_index"]) in val_set else train).append(i)
    if not train:
        train = list(range(max(0, len(rows) - 1)))
    if not val:
        val = [len(rows) - 1]
    return train, val


def make_batch(rows: list[dict[str, Any]], idxs: list[int], device: torch.device):
    chosen = [rows[i] for i in idxs]
    base_futures = []
    target_futures = []
    for row in chosen:
        base_future = load_future(row["base_trace_path"], key=str(row.get("base_future_key", "target_proxy_future")))
        target_future = load_future(row["target_trace_path"], key=str(row.get("target_future_key", "target_proxy_future")))
        if base_future is None or target_future is None:
            raise ValueError("failed to reload future trace during batching")
        base_futures.append(base_future)
        target_futures.append(target_future)
    base = torch.from_numpy(np.stack(base_futures).astype(np.float32)).to(device)
    target = torch.from_numpy(np.stack(target_futures).astype(np.float32)).to(device)
    source_summary = torch.from_numpy(np.stack([r["source_summary"] for r in chosen]).astype(np.float32)).to(device)
    target_summary = torch.from_numpy(np.stack([r["target_summary"] for r in chosen]).astype(np.float32)).to(device)
    return base, target, source_summary, target_summary


def summarize_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False), x[:, 0], x[:, -1] - x[:, 0]], dim=-1)


def evaluate(model: SummaryConditionedFutureAdapter, rows: list[dict[str, Any]], idxs: list[int], device: torch.device) -> dict[str, float]:
    model.eval()
    recon_losses = []
    summary_losses = []
    future_cos = []
    summary_cos = []
    with torch.no_grad():
        for start in range(0, len(idxs), 32):
            batch_ids = idxs[start : start + 32]
            base, target, source_summary, target_summary = make_batch(rows, batch_ids, device)
            corrected, _ = model(base, source_summary, target_summary)
            pred_summary = summarize_torch(corrected)
            recon_losses.append(F.smooth_l1_loss(corrected, target).cpu())
            summary_losses.append(F.smooth_l1_loss(pred_summary, target_summary).cpu())
            future_cos.append(F.cosine_similarity(corrected.flatten(1), target.flatten(1), dim=-1).mean().cpu())
            summary_cos.append(F.cosine_similarity(pred_summary, target_summary, dim=-1).mean().cpu())
    return {
        "loss": float(torch.stack(recon_losses).mean().item()) if recon_losses else 0.0,
        "summary_loss": float(torch.stack(summary_losses).mean().item()) if summary_losses else 0.0,
        "future_cos": float(torch.stack(future_cos).mean().item()) if future_cos else 0.0,
        "summary_cos": float(torch.stack(summary_cos).mean().item()) if summary_cos else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train summary-conditioned adapter that maps repaired summary targets back to raw future tokens.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-summary", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.25)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_examples(args.pairs_jsonl, args.memory_npz, int(args.max_examples))
    train_idx, val_idx = split(rows, float(args.val_ratio), int(args.seed))
    sample = rows[0]
    sample_future = load_future(sample["base_trace_path"], key=str(sample.get("base_future_key", "target_proxy_future")))
    if sample_future is None:
        raise ValueError("failed to load sample future from trace path")
    future_shape = tuple(int(x) for x in sample_future.shape)
    summary_dim = int(sample["source_summary"].shape[0])
    print(
        json.dumps(
            {
                "stage": "dataset_loaded",
                "num_examples": int(len(rows)),
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "future_shape": list(future_shape),
                "summary_dim": int(summary_dim),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model = SummaryConditionedFutureAdapter(
        future_dim=future_shape[-1],
        summary_dim=summary_dim,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    metrics_jsonl_path = args.output_dir / "metrics.jsonl"

    best = None
    history = []
    for step in range(1, int(args.steps) + 1):
        model.train()
        batch_ids = random.choices(train_idx, k=min(int(args.batch_size), len(train_idx)))
        base, target, source_summary, target_summary = make_batch(rows, batch_ids, device)
        corrected, aux = model(base, source_summary, target_summary)
        pred_summary = summarize_torch(corrected)
        target_delta = target - base
        pred_delta = corrected - base
        recon_loss = F.smooth_l1_loss(corrected, target)
        summary_loss = F.smooth_l1_loss(pred_summary, target_summary)
        delta_loss = F.smooth_l1_loss(pred_delta, target_delta)
        loss = recon_loss + float(args.lambda_summary) * summary_loss + float(args.lambda_delta) * delta_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        log_every = max(1, int(args.log_every))
        if step == 1 or step == int(args.steps) or step % log_every == 0:
            metrics = evaluate(model, rows, val_idx, device)
            rec = {
                "step": int(step),
                "train_loss": float(loss.item()),
                "recon_loss_train": float(recon_loss.item()),
                "summary_loss_train": float(summary_loss.item()),
                "delta_loss_train": float(delta_loss.item()),
                "gate_mean_train": float(aux["gate"].mean().item()),
                **metrics,
            }
            history.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with metrics_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            score = metrics["future_cos"] + metrics["summary_cos"] - metrics["loss"] - metrics["summary_loss"]
            if best is None or score > best["score"]:
                best = {"score": float(score), "metrics": rec}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "future_dim": int(future_shape[-1]),
                        "summary_dim": int(summary_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "future_token_shape": future_shape,
                        "best_metrics": rec,
                    },
                    args.output_dir / "summary_conditioned_future_adapter.pt",
                )

    summary = {
        "pairs_jsonl": str(args.pairs_jsonl),
        "memory_npz": str(args.memory_npz),
        "num_examples": int(len(rows)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "future_shape": list(future_shape),
        "summary_dim": int(summary_dim),
        "best": None if best is None else best["metrics"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if history:
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
