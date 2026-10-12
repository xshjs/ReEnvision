import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float_vector(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], (list, np.ndarray)):
            value = value[0]
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


def _flatten_actions(actions: np.ndarray, action_chunk_len: int) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((action_chunk_len * 9,), dtype=np.float32)
    action_dim = arr.shape[1]
    if arr.shape[0] >= action_chunk_len:
        arr = arr[:action_chunk_len]
    else:
        pad = np.repeat(arr[-1:], action_chunk_len - arr.shape[0], axis=0) if arr.shape[0] else np.zeros((action_chunk_len, action_dim), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr.reshape(-1).astype(np.float32)


def load_samples(rows_jsonl: Path, max_rows: int = 0, action_chunk_len: int = 16):
    samples = []
    with rows_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            feature_path = row.get("feature_path", "")
            if not feature_path:
                continue
            npz_path = feature_path if Path(feature_path).is_absolute() else rows_jsonl.parent / feature_path
            npz_path = Path(npz_path)
            if not npz_path.is_file():
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            current = _to_float_vector(data.get("obs_current_feature"))
            future = _to_float_vector(data.get("end_obs_current_feature"))
            actions = data.get("actions")
            if current is None or future is None or actions is None:
                continue
            samples.append(
                {
                    "current": current,
                    "future": future,
                    "actions_flat": _flatten_actions(actions, action_chunk_len),
                    "success": float(bool(row.get("success", False))),
                    "task": str(row.get("task", "")),
                }
            )
            if max_rows > 0 and len(samples) >= max_rows:
                break
    return samples


class TaskConditionedBridgeLatentHeads(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, num_tasks: int, latent_dim: int = 768, hidden_dim: int = 1024):
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
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.task_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks),
        )
        self.success_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_vec: torch.Tensor, action_vec: torch.Tensor):
        s = torch.tanh(self.state_proj(state_vec))
        zf = torch.tanh(self.future_head(state_vec))
        za_teacher = torch.tanh(self.action_encoder(action_vec))
        za = torch.tanh(self.motion_head(torch.cat([s, zf], dim=-1)))
        task_logits = self.task_head(za)
        success_logit = self.success_head(za).squeeze(-1)
        return s, zf, za, za_teacher, task_logits, success_logit


def main():
    parser = argparse.ArgumentParser(description="Train task-conditioned RLBench latent heads with a more task-specific z_a.")
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--output-ckpt", required=True)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--action-chunk-len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--task-weight", type=float, default=0.3)
    parser.add_argument("--success-weight", type=float, default=0.2)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = load_samples(Path(args.rows_jsonl), max_rows=args.max_rows, action_chunk_len=args.action_chunk_len)
    if not samples:
        raise RuntimeError("no usable samples found")

    input_dim = max(sample["current"].shape[0] for sample in samples)
    latent_dim = int(args.latent_dim)
    action_dim = max(sample["actions_flat"].shape[0] for sample in samples)
    tasks = sorted({sample["task"] for sample in samples})
    task_to_id = {task: idx for idx, task in enumerate(tasks)}

    current_np = np.stack([_match_dim(sample["current"], input_dim) for sample in samples], axis=0)
    future_target_np = np.stack([_match_dim(sample["future"], latent_dim) for sample in samples], axis=0)
    action_np = np.stack([_match_dim(sample["actions_flat"], action_dim) for sample in samples], axis=0)
    success_np = np.asarray([sample["success"] for sample in samples], dtype=np.float32)
    task_np = np.asarray([task_to_id[sample["task"]] for sample in samples], dtype=np.int64)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = TaskConditionedBridgeLatentHeads(
        input_dim=input_dim,
        action_dim=action_dim,
        num_tasks=len(tasks),
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    current = torch.from_numpy(current_np).to(device)
    future_target = torch.from_numpy(future_target_np).to(device)
    action_target = torch.from_numpy(action_np).to(device)
    success_target = torch.from_numpy(success_np).to(device)
    task_target = torch.from_numpy(task_np).to(device)

    num_samples = current.shape[0]
    best_loss = None
    best_state = None

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, num_samples, (min(args.batch_size, num_samples),), device=device)
        batch_current = current[idx]
        batch_future = future_target[idx]
        batch_action = action_target[idx]
        batch_success = success_target[idx]
        batch_task = task_target[idx]

        _, pred_future, pred_motion, teacher_motion, task_logits, success_logit = model(batch_current, batch_action)
        future_loss = F.smooth_l1_loss(pred_future, batch_future)
        align_loss = F.smooth_l1_loss(pred_motion, teacher_motion.detach())
        align_loss = align_loss + 0.5 * (1.0 - F.cosine_similarity(pred_motion, teacher_motion.detach(), dim=-1)).mean()
        task_loss = F.cross_entropy(task_logits, batch_task)
        success_loss = F.binary_cross_entropy_with_logits(success_logit, batch_success)
        loss = future_loss + args.align_weight * align_loss + args.task_weight * task_loss + args.success_weight * success_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if step == 1 or step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                task_acc = float((task_logits.argmax(dim=-1) == batch_task).float().mean().item())
                success_acc = float(((torch.sigmoid(success_logit) > 0.5).float() == batch_success).float().mean().item())
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_val,
                        "future_loss": float(future_loss.item()),
                        "align_loss": float(align_loss.item()),
                        "task_loss": float(task_loss.item()),
                        "success_loss": float(success_loss.item()),
                        "task_acc": task_acc,
                        "success_acc": success_acc,
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
                "action_dim": action_dim,
                "num_samples": int(num_samples),
                "best_loss": float(best_loss if best_loss is not None else 0.0),
                "rows_jsonl": str(args.rows_jsonl),
                "tasks": tasks,
                "task_conditioned": True,
            },
        },
        output_path,
    )
    print(json.dumps({"saved_ckpt": str(output_path), "best_loss": best_loss, "num_tasks": len(tasks)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
