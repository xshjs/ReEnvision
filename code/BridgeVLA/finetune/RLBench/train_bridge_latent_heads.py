import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float_vector(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, list):
        if len(value) == 0:
            return None
        first = value[0]
        if isinstance(first, (list, np.ndarray)):
            value = first
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _match_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[0] > target_dim:
        return arr[:target_dim]
    if arr.shape[0] < target_dim:
        repeat = (target_dim + arr.shape[0] - 1) // arr.shape[0]
        return np.tile(arr, repeat)[:target_dim]
    return arr


def load_samples(rows_jsonl: Path, max_rows: int = 0):
    samples = []
    with rows_jsonl.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            row = json.loads(line)
            feature_path = row.get("feature_path", "")
            if not feature_path:
                continue
            npz_path = Path(feature_path)
            if not npz_path.is_file():
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            current = _to_float_vector(data.get("obs_current_feature"))
            future = _to_float_vector(data.get("end_obs_current_feature"))
            if current is None or future is None:
                continue
            samples.append(
                {
                    "current": current,
                    "future": future,
                    "success": float(bool(row.get("success", False))),
                    "task": str(row.get("task", "")),
                }
            )
            if max_rows > 0 and len(samples) >= max_rows:
                break
    return samples


class BridgeLatentHeads(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.future_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state_vec: torch.Tensor):
        s = torch.tanh(self.state_proj(state_vec))
        zf = torch.tanh(self.future_head(state_vec))
        za = torch.tanh(self.motion_head(torch.cat([s, zf], dim=-1)))
        return s, zf, za


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--output-ckpt", required=True)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = load_samples(Path(args.rows_jsonl), max_rows=args.max_rows)
    if not samples:
        raise RuntimeError("no usable samples found")

    input_dim = max(sample["current"].shape[0] for sample in samples)
    latent_dim = int(args.latent_dim)
    current_np = np.stack([_match_dim(s["current"], input_dim) for s in samples], axis=0)
    future_target_np = np.stack([_match_dim(s["future"], latent_dim) for s in samples], axis=0)
    motion_target_np = np.tanh(future_target_np - np.stack([_match_dim(s["current"], latent_dim) for s in samples], axis=0))
    success_np = np.asarray([s["success"] for s in samples], dtype=np.float32)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = BridgeLatentHeads(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    current = torch.from_numpy(current_np).to(device)
    future_target = torch.from_numpy(future_target_np).to(device)
    motion_target = torch.from_numpy(motion_target_np).to(device)
    success_weight = torch.from_numpy(0.5 + 0.5 * success_np).to(device)

    num_samples = current.shape[0]
    best_loss = None
    best_state = None

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, num_samples, (min(args.batch_size, num_samples),), device=device)
        batch_current = current[idx]
        batch_future = future_target[idx]
        batch_motion = motion_target[idx]
        batch_weight = success_weight[idx].unsqueeze(-1)

        _, pred_future, pred_motion = model(batch_current)
        future_loss = ((pred_future - batch_future) ** 2 * batch_weight).mean()
        motion_loss = ((pred_motion - batch_motion) ** 2 * batch_weight).mean()
        align_loss = (1.0 - F.cosine_similarity(pred_motion, batch_motion, dim=-1)).mean()
        loss = future_loss + motion_loss + 0.1 * align_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if step == 1 or step % 200 == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_val,
                        "future_loss": float(future_loss.item()),
                        "motion_loss": float(motion_loss.item()),
                        "align_loss": float(align_loss.item()),
                        "num_samples": num_samples,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bridge_latent_heads": best_state if best_state is not None else model.state_dict(),
            "meta": {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "hidden_dim": int(args.hidden_dim),
                "num_samples": int(num_samples),
                "best_loss": float(best_loss if best_loss is not None else 0.0),
                "rows_jsonl": str(args.rows_jsonl),
            },
        },
        output_path,
    )
    print(json.dumps({"saved_ckpt": str(output_path), "best_loss": best_loss}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
