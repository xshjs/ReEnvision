from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Iterable

if "distutils.spawn" not in sys.modules:
    distutils_mod = sys.modules.setdefault("distutils", types.ModuleType("distutils"))
    spawn_mod = types.ModuleType("distutils.spawn")
    spawn_mod.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_mod
    setattr(distutils_mod, "spawn", spawn_mod)

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


STEP3_ROOT = Path(__file__).resolve().parents[1]
if str(STEP3_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP3_ROOT))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def replace_linear_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += replace_linear_with_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collect_rows(
    roots: list[Path],
    limit: int,
    include_tasks: set[str] | None = None,
    exclude_tasks: set[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        index = root / "collect_success_expert_policy_dataset" / "index.jsonl"
        base = root / "collect_success_expert_policy_dataset"
        if not index.exists():
            raise FileNotFoundError(index)
        for row in iter_jsonl(index):
            task = str(row.get("task", ""))
            if include_tasks is not None and task not in include_tasks:
                continue
            if exclude_tasks is not None and task in exclude_tasks:
                continue
            rows.append((base / str(row["path"]), row))
            if limit and len(rows) >= limit:
                return rows
    return rows


def parse_task_list(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def build_sample_weights(
    rows: list[tuple[Path, dict[str, Any]]],
    upweight_tasks: set[str],
    upweight_factor: float,
    rebalance_by_task: bool,
) -> list[float]:
    if not rows:
        return []
    task_counts: dict[str, int] = {}
    for _, row in rows:
        task = str(row.get("task", ""))
        task_counts[task] = task_counts.get(task, 0) + 1

    weights: list[float] = []
    for _, row in rows:
        task = str(row.get("task", ""))
        weight = 1.0
        if rebalance_by_task:
            weight *= 1.0 / math.sqrt(max(task_counts.get(task, 1), 1))
        if task in upweight_tasks:
            weight *= upweight_factor
        weights.append(weight)
    return weights


def image_to_tensor(array: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return (tensor - 0.5) / 0.5


def action_chunk(actions: np.ndarray, length: int) -> torch.Tensor:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape[0] >= length:
        arr = arr[:length]
    else:
        pad = np.repeat(arr[-1:], length - arr.shape[0], axis=0) if arr.shape[0] else np.zeros((length, 7), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return torch.from_numpy(arr.astype(np.float32))


class SuccessPolicyDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[Path, dict[str, Any]]],
        image_size: int,
        action_seq_len: int,
        action_intent_len: int,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.action_seq_len = action_seq_len
        self.action_intent_len = action_intent_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, row = self.rows[index]
        data = np.load(path)
        latent_tokens = torch.from_numpy(data["latent_tokens"].astype(np.float32)).reshape(-1, 768)
        return {
            "rgb_static": image_to_tensor(data["rgb_static"][0], self.image_size).unsqueeze(0),
            "rgb_gripper": image_to_tensor(data["rgb_gripper"][0], self.image_size).unsqueeze(0),
            "actions": action_chunk(data["actions"], self.action_seq_len),
            "action_intent": action_chunk(data["actions"], self.action_intent_len),
            "latent_tokens": latent_tokens,
            "lang_text": str(row["lang_text"]),
            "path": str(path),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rgb_obs": {
            "rgb_static": torch.stack([item["rgb_static"] for item in batch], dim=0),
            "rgb_gripper": torch.stack([item["rgb_gripper"] for item in batch], dim=0),
        },
        "actions": torch.stack([item["actions"] for item in batch], dim=0),
        "action_intent": torch.stack([item["action_intent"] for item in batch], dim=0),
        "latent_tokens": torch.stack([item["latent_tokens"] for item in batch], dim=0),
        "lang_text": [item["lang_text"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


def load_policy(args: argparse.Namespace, device: torch.device) -> Any:
    config_dir = STEP3_ROOT / "policy_conf"
    with initialize_config_dir(config_dir=str(config_dir)):
        cfg = compose(config_name="VPP_Calvinabc_train")
    with open_dict(cfg):
        cfg.model.pretrained_model_path = args.video_model_path
        cfg.model.text_encoder_path = args.clip_model_path
        cfg.model.t5_model_path = args.t5_model_path
        cfg.model.language_goal_path = args.language_goal_path
        cfg.root_data_dir = args.calvin_dir
        cfg.datamodule.root_data_dir = args.calvin_dir
        cfg.batch_size = args.batch_size
        cfg.max_epochs = 1

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.to(device)
    model.process_device()
    model.future_feature_mode = "current"
    for param in model.parameters():
        param.requires_grad = False
    if args.use_action_intent:
        for param in model.action_intent_proj.parameters():
            param.requires_grad = True
    replaced = replace_linear_with_lora(model.model.inner_model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.to(device)
    model.process_device()
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return model, replaced, trainable, cfg


def resolve_device(raw: str) -> torch.device:
    if raw.isdigit():
        raw = f"cuda:{raw}"
    if raw.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested with --device {raw}, but torch.cuda.is_available() is False")
        return torch.device(raw)
    return torch.device("cpu")


def save_lora_state(model: Any, path: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    lora_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name or "action_intent_proj." in name
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state": lora_state, "args": vars(args), "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA smoke finetune on successful MoE expert policy traces.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/policy_lora_success_smoke50"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--action-seq-len", type=int, default=10)
    parser.add_argument("--action-intent-len", type=int, default=16)
    parser.add_argument("--use-action-intent", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-model-path", default="/DeFI/ckpts/_hf_defi/step1_gfdm")
    parser.add_argument("--checkpoint", default="/DeFI/ckpts/step3_defi.pt")
    parser.add_argument("--clip-model-path", default="/DeFI/ckpts/openai_clip_vit_base_patch32")
    parser.add_argument("--t5-model-path", default="/DeFI/ckpts/t5_base")
    parser.add_argument("--language-goal-path", default="/DeFI/ckpts/ViT-B-32.pt")
    parser.add_argument("--calvin-dir", default="/DeFI/ckpts/task_ABC_D")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--include-tasks", default="")
    parser.add_argument("--exclude-tasks", default="")
    parser.add_argument("--upweight-tasks", default="")
    parser.add_argument("--upweight-factor", type=float, default=1.0)
    parser.add_argument("--rebalance-by-task", action="store_true")
    parser.add_argument("--loss-window", type=int, default=20)
    args = parser.parse_args()

    device = resolve_device(str(args.device))
    print(json.dumps({"requested_device": str(args.device), "resolved_device": str(device)}), flush=True)
    include_tasks = parse_task_list(args.include_tasks) if args.include_tasks else None
    exclude_tasks = parse_task_list(args.exclude_tasks) if args.exclude_tasks else None
    upweight_tasks = parse_task_list(args.upweight_tasks)
    rows = collect_rows(args.roots, args.limit, include_tasks=include_tasks, exclude_tasks=exclude_tasks)
    if not rows:
        raise ValueError("no policy trace rows found")
    dataset = SuccessPolicyDataset(
        rows,
        image_size=args.image_size,
        action_seq_len=args.action_seq_len,
        action_intent_len=args.action_intent_len,
    )
    sample_weights = build_sample_weights(
        rows,
        upweight_tasks=upweight_tasks,
        upweight_factor=args.upweight_factor,
        rebalance_by_task=args.rebalance_by_task,
    )
    sampler = None
    shuffle = True
    if sample_weights and (
        args.rebalance_by_task or (upweight_tasks and abs(args.upweight_factor - 1.0) > 1e-8)
    ):
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate, drop_last=False)

    model, replaced, trainable, _ = load_policy(args, device)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_counts: dict[str, int] = {}
    for _, row in rows:
        task = str(row.get("task", ""))
        task_counts[task] = task_counts.get(task, 0) + 1
    metrics: dict[str, Any] = {
        "num_examples": len(dataset),
        "lora_linear_modules": replaced,
        "trainable_params": trainable,
        "include_tasks": sorted(include_tasks) if include_tasks else [],
        "exclude_tasks": sorted(exclude_tasks) if exclude_tasks else [],
        "upweight_tasks": sorted(upweight_tasks),
        "upweight_factor": args.upweight_factor,
        "rebalance_by_task": bool(args.rebalance_by_task),
        "use_action_intent": bool(args.use_action_intent),
        "action_intent_len": int(args.action_intent_len),
        "loss_window": int(args.loss_window),
        "task_counts": task_counts,
        "losses": [],
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "losses"}, indent=2), flush=True)

    step = 0
    best_loss = float("inf")
    best_step = 0
    while step < args.steps:
        for batch in loader:
            model.model.train()
            batch["rgb_obs"]["rgb_static"] = batch["rgb_obs"]["rgb_static"].to(device)
            batch["rgb_obs"]["rgb_gripper"] = batch["rgb_obs"]["rgb_gripper"].to(device)
            actions = batch["actions"].to(device)
            if args.use_action_intent:
                batch["action_intent"] = batch["action_intent"].to(device)
            latent_tokens = batch["latent_tokens"].to(device)

            predictive_feature, latent_goal = model.extract_predictive_feature(batch)
            action_intent_cond = None
            if args.use_action_intent:
                action_intent_cond = model._project_action_intent_condition(
                    batch,
                    latent_goal.device,
                    latent_goal.dtype,
                )
            if model.use_Former == "linear":
                predictive_feature["state_images"] = predictive_feature["state_images"].reshape(
                    predictive_feature["state_images"].size(0), -1, predictive_feature["state_images"].size(-1)
                )
            loss, _, _ = model.diffusion_loss(
                predictive_feature,
                latent_goal.to(torch.float32),
                actions,
                latent_tokens,
                action_intent_cond=action_intent_cond,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 1.0)
            optimizer.step()

            step += 1
            loss_value = float(loss.detach().cpu())
            metrics["losses"].append(loss_value)
            window = max(1, int(args.loss_window))
            moving_avg = float(np.mean(metrics["losses"][-window:]))
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                save_lora_state(
                    model,
                    args.output_dir / "best.pt",
                    args,
                    {
                        **{k: v for k, v in metrics.items() if k != "losses"},
                        "best_step": best_step,
                        "best_loss": best_loss,
                        "current_loss": loss_value,
                        "moving_avg_loss": moving_avg,
                    },
                )
            if step == 1 or step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss_value,
                            "moving_avg_loss": moving_avg,
                            "best_loss": best_loss,
                            "best_step": best_step,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if step >= args.steps:
                break

    losses = metrics["losses"]
    metrics["initial_loss"] = losses[0]
    metrics["final_loss"] = losses[-1]
    metrics["mean_last_10"] = float(np.mean(losses[-10:]))
    metrics["best_loss"] = float(best_loss)
    metrics["best_step"] = int(best_step)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_lora_state(model, args.output_dir / "last.pt", args, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "losses"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
