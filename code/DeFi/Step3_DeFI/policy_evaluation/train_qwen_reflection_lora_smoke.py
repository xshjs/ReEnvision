from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Example:
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(int(rank), 1)
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_examples(path: Path, limit: int) -> list[Example]:
    rows: list[Example] = []
    for row in read_jsonl(path):
        rows.append(Example(messages=row["messages"], metadata=row.get("metadata", {})))
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def render_example(tokenizer: Any, example: Example) -> tuple[list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(example.messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(example.messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(full_text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    labels = [-100] * len(prompt_ids) + list(input_ids[len(prompt_ids) :])
    return input_ids, labels


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x[0]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def save_lora_state(model: nn.Module, path: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    lora_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if ".lora_a." in name or ".lora_b." in name}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state": lora_state, "args": vars(args), "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen reflection LoRA smoke SFT.")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/data/manipulation/llm_models/Qwen3-8B"))
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen_reflection_lora_smoke"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=torch.float32 if device.type == "cuda" else torch.float32,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    replaced = replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            param.requires_grad = True
    model.to(device)

    examples = load_examples(args.train_jsonl, args.limit)
    if not examples:
        raise ValueError("no examples loaded")
    tokenized = []
    for ex in examples:
        input_ids, labels = render_example(tokenizer, ex)
        if len(input_ids) > args.max_length:
            input_ids = input_ids[-args.max_length :]
            labels = labels[-args.max_length :]
        tokenized.append((input_ids, labels))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "num_examples": len(tokenized),
        "max_length": args.max_length,
        "lora_linear_modules": replaced,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "losses": [],
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "losses"}, indent=2), flush=True)

    step = 0
    while step < args.steps:
        for start in range(0, len(tokenized), args.batch_size):
            batch = tokenized[start : start + args.batch_size]
            batch_t = collate(batch, tokenizer.pad_token_id)
            batch_t = {k: v.to(device) for k, v in batch_t.items()}

            out = model(**batch_t)
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            step += 1
            loss_value = float(loss.detach().cpu())
            metrics["losses"].append(loss_value)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps({"step": step, "loss": loss_value}, ensure_ascii=False), flush=True)
            if step >= args.steps:
                break

    losses = metrics["losses"]
    metrics["initial_loss"] = float(losses[0])
    metrics["final_loss"] = float(losses[-1])
    metrics["mean_loss"] = float(sum(losses) / len(losses))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_lora_state(model, args.output_dir / "qwen_reflection_lora_smoke.pt", args, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "losses"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
