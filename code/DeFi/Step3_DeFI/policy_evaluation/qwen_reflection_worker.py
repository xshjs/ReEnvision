from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PosixPath
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / max(int(rank), 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(x).to(self.lora_a.weight.dtype)))
        residual = residual.to(self.base.weight.dtype)
        base_out = self.base(x.to(self.base.weight.dtype))
        return base_out + residual * self.scaling


def replace_linear_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += replace_linear_with_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return count


def load_model(model_path: Path, device: torch.device, lora_path: Path | None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    if lora_path is not None:
        try:
            try:
                torch.serialization.add_safe_globals([Path, PosixPath])
            except Exception:
                pass
            checkpoint = torch.load(lora_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(lora_path, map_location="cpu")
        except Exception:
            checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
        train_args = dict(checkpoint.get("args", {}))
        replace_linear_with_lora(
            model,
            rank=int(train_args.get("lora_rank", 8)),
            alpha=float(train_args.get("lora_alpha", 16.0)),
            dropout=float(train_args.get("lora_dropout", 0.05)),
        )
        model.load_state_dict(checkpoint.get("lora_state", {}), strict=False)
    model = model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def generate_reflection(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int) -> str:
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][tokens["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text or (
        '{"trust": "medium", "mismatch_type": "unknown", '
        '"correction_direction": "stabilize contact"}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen reflection worker over stdin/stdout JSONL.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_path, device, args.lora_path)
    print(json.dumps({"status": "ready"}, ensure_ascii=False), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            prompt = str(request.get("prompt", ""))
            max_new_tokens = int(request.get("max_new_tokens", args.max_new_tokens))
            text = generate_reflection(tokenizer, model, prompt, max_new_tokens)
            print(json.dumps({"ok": True, "text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": repr(exc)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
