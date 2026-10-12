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


class BridgeLatentHeads(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.state_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.future_head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.motion_head = nn.Sequential(nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))


class JointLatentPolicy(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, num_tasks: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.task_head = nn.Linear(hidden_dim, num_tasks)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor, zf: torch.Tensor, za: torch.Tensor):
        h = self.backbone(torch.cat([s, zf, za], dim=-1))
        return self.action_head(h), self.task_head(h), self.success_head(h).squeeze(-1)


def load_rows(rows_jsonl: Path, max_rows: int = 0):
    rows = []
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
            actions = data.get("actions")
            if current is None or actions is None:
                continue
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] == 0:
                continue
            rows.append(
                {
                    "current": current,
                    "action": actions[0].astype(np.float32),
                    "success": float(bool(row.get("success", False))),
                    "task": str(row.get("task", "")),
                }
            )
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train a latent policy pi(s, z_f, z_a) on RLBench rollout rows.")
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--latent-head-ckpt", required=True)
    parser.add_argument("--output-ckpt", required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--task-weight", type=float, default=0.2)
    parser.add_argument("--success-weight", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    payload = torch.load(args.latent_head_ckpt, map_location="cpu")
    meta = payload["meta"]
    latent_dim = int(meta["latent_dim"])
    input_dim = int(meta["input_dim"])
    rows = load_rows(Path(args.rows_jsonl), max_rows=args.max_rows)
    if not rows:
        raise RuntimeError("no usable rows found")

    tasks = sorted({row["task"] for row in rows})
    task_to_id = {task: idx for idx, task in enumerate(tasks)}
    action_dim = max(row["action"].shape[0] for row in rows)

    current_np = np.stack([_match_dim(row["current"], input_dim) for row in rows], axis=0)
    action_np = np.stack([_match_dim(row["action"], action_dim) for row in rows], axis=0)
    success_np = np.asarray([row["success"] for row in rows], dtype=np.float32)
    task_np = np.asarray([task_to_id[row["task"]] for row in rows], dtype=np.int64)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    heads = BridgeLatentHeads(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=int(meta["hidden_dim"]))
    heads.load_state_dict(payload["bridge_latent_heads"], strict=False)
    heads.to(device).eval()
    for param in heads.parameters():
        param.requires_grad = False

    policy = JointLatentPolicy(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=args.hidden_dim, num_tasks=len(tasks)).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    current = torch.from_numpy(current_np).to(device)
    action_target = torch.from_numpy(action_np).to(device)
    success_target = torch.from_numpy(success_np).to(device)
    task_target = torch.from_numpy(task_np).to(device)

    num_samples = current.shape[0]
    best_loss = None
    best_state = None

    with torch.no_grad():
        s_all = torch.tanh(heads.state_proj(current))
        zf_all = torch.tanh(heads.future_head(current))
        za_all = torch.tanh(heads.motion_head(torch.cat([s_all, zf_all], dim=-1)))

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, num_samples, (min(args.batch_size, num_samples),), device=device)
        s = s_all[idx]
        zf = zf_all[idx]
        za = za_all[idx]
        target_action = action_target[idx]
        target_success = success_target[idx]
        target_task = task_target[idx]

        pred_action, task_logits, success_logit = policy(s, zf, za)
        action_loss = F.smooth_l1_loss(pred_action, target_action)
        task_loss = F.cross_entropy(task_logits, target_task)
        success_loss = F.binary_cross_entropy_with_logits(success_logit, target_success)
        loss = action_loss + args.task_weight * task_loss + args.success_weight * success_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}

        if step == 1 or step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                task_acc = float((task_logits.argmax(dim=-1) == target_task).float().mean().item())
                success_acc = float(((torch.sigmoid(success_logit) > 0.5).float() == target_success).float().mean().item())
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_val,
                        "action_loss": float(action_loss.item()),
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
            "joint_latent_policy": best_state if best_state is not None else policy.state_dict(),
            "meta": {
                "latent_dim": latent_dim,
                "action_dim": action_dim,
                "hidden_dim": int(args.hidden_dim),
                "num_tasks": len(tasks),
                "tasks": tasks,
                "num_samples": int(num_samples),
                "best_loss": float(best_loss if best_loss is not None else 0.0),
                "rows_jsonl": str(args.rows_jsonl),
                "latent_head_ckpt": str(args.latent_head_ckpt),
            },
        },
        output_path,
    )
    print(json.dumps({"saved_ckpt": str(output_path), "best_loss": best_loss, "num_tasks": len(tasks)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
