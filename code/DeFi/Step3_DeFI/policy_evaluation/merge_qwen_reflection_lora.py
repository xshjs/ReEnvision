from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen reflection LoRA weights into a Hugging Face model directory.")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=None, help="Override alpha/rank scale. Defaults to checkpoint args.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = torch.load(str(args.lora_checkpoint), map_location="cpu")
    lora_state = payload.get("lora_state")
    if lora_state is None:
        raise ValueError(f"{args.lora_checkpoint} does not contain lora_state")
    lora_args = payload.get("args", {})
    scale = args.scale
    if scale is None:
        scale = float(lora_args.get("lora_alpha", 16.0)) / max(float(lora_args.get("lora_rank", 8.0)), 1.0)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if str(args.device).startswith("cuda") else torch.float32,
    )

    base_state = model.state_dict()
    prefixes = sorted({key.rsplit(".lora_", 1)[0] for key in lora_state})
    merged = 0
    missing = []
    for prefix in prefixes:
        weight_key = f"{prefix}.weight"
        a_key = f"{prefix}.lora_a.weight"
        b_key = f"{prefix}.lora_b.weight"
        if weight_key not in base_state:
            missing.append(weight_key)
            continue
        a = lora_state[a_key].float()
        b = lora_state[b_key].float()
        delta = (b @ a) * float(scale)
        base_state[weight_key] = base_state[weight_key].float() + delta.to(base_state[weight_key].dtype)
        merged += 1

    if missing:
        raise KeyError(f"Missing base weights for {len(missing)} LoRA prefixes; first few: {missing[:10]}")

    model.load_state_dict(base_state, strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "merge_summary.json").write_text(
        json.dumps(
            {
                "base_model": str(args.base_model),
                "lora_checkpoint": str(args.lora_checkpoint),
                "output_dir": str(args.output_dir),
                "merged_linear_layers": merged,
                "scale": float(scale),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"merged_linear_layers": merged, "scale": float(scale), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
