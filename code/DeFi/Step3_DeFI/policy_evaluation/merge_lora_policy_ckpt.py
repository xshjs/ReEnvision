from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge smoke LoRA weights into a DeFi policy checkpoint.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=None, help="Override alpha/rank. Defaults to checkpoint args.")
    args = parser.parse_args()

    base = torch.load(args.base_checkpoint, map_location="cpu")
    state = base["model"] if isinstance(base, dict) and "model" in base else base
    lora_ckpt = torch.load(args.lora_checkpoint, map_location="cpu")
    lora_state = lora_ckpt["lora_state"]
    lora_args = lora_ckpt.get("args", {})
    scale = args.scale
    if scale is None:
        scale = float(lora_args.get("lora_alpha", 8.0)) / float(lora_args.get("lora_rank", 4))

    prefixes = sorted({key.rsplit(".lora_", 1)[0] for key in lora_state})
    merged = 0
    for prefix in prefixes:
        weight_key = f"{prefix}.weight"
        a_key = f"{prefix}.lora_a.weight"
        b_key = f"{prefix}.lora_b.weight"
        if weight_key not in state:
            raise KeyError(f"missing base weight for {prefix}: {weight_key}")
        a = lora_state[a_key].float()
        b = lora_state[b_key].float()
        delta = (b @ a) * scale
        state[weight_key] = state[weight_key].float() + delta.to(state[weight_key].dtype)
        merged += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print({"merged_linear_layers": merged, "scale": scale, "output": str(args.output)})


if __name__ == "__main__":
    main()
