import argparse
import json
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if "distutils.spawn" not in sys.modules:
    distutils_mod = sys.modules.setdefault("distutils", types.ModuleType("distutils"))
    spawn_mod = types.ModuleType("distutils.spawn")
    spawn_mod.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_mod
    setattr(distutils_mod, "spawn", spawn_mod)

import torch
from torch import nn

from train_universal_moe_oracle_router import load_rows


SYSTEM_PROMPT = """You are auditing a robot policy after one attempted subtask.
Return one compact JSON object only:
{"progress_status": string, "failure_type": string, "prefer_next": string, "avoid_next": string, "correction_direction": string, "memory_query": string}
Use short phrases. Do not include explanations outside JSON.
progress_status should be one of: cold_start, progressing, stuck, recovered, uncertain.
failure_type should be one of: none, slow_success, wrong_expert, all_failed, unstable, uncertain."""


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


def load_lora_checkpoint(model: nn.Module, checkpoint: Path, device: torch.device, dtype: torch.dtype) -> None:
    payload = torch.load(str(checkpoint), map_location="cpu")
    lora_state = payload.get("lora_state")
    if lora_state is None:
        raise ValueError(f"{checkpoint} does not contain lora_state")
    args = payload.get("args", {})
    rank = int(args.get("lora_rank", 8))
    alpha = float(args.get("lora_alpha", 16.0))
    dropout = float(args.get("lora_dropout", 0.05))
    replaced = replace_linear_with_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    model.to(device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            param.requires_grad = False
    model.load_state_dict(lora_state, strict=False)
    model.to(device=device, dtype=dtype)
    print(json.dumps({"loaded_lora_modules": replaced, "lora_checkpoint": str(checkpoint)}, ensure_ascii=False), flush=True)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compact_branch(branch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "expert": int(branch.get("expert", -1)),
        "rank": int(branch.get("router_rank", -1)),
        "logit": round(float(branch.get("router_logit", 0.0)), 3),
        "vjepa": round(float(branch.get("vjepa_score", 0.0)), 6),
        "contradiction": round(float(branch.get("contradiction", 0.0)), 6),
        "success": bool(branch.get("success", False)),
        "steps": int(branch.get("steps", 160)),
    }


def row_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["_global_sequence_index"]), int(row["subtask_index"])


def make_prompt(row: Dict[str, Any], prev: Optional[Dict[str, Any]], ep_len: int) -> str:
    current_candidates = [compact_branch(branch) for branch in row.get("branches", [])]
    lines = [
        f"Current sequence index: {row['_global_sequence_index']}",
        f"Current subtask index: {row['subtask_index']}",
        f"Current goal: {row['task']}",
        "Current candidate experts:",
        json.dumps(current_candidates, ensure_ascii=False),
    ]
    if prev is None:
        lines.extend(
            [
                "Previous outcome: none. This is the first subtask in the chain.",
                "Generate a cold-start reflection for selecting among current experts.",
            ]
        )
    else:
        prev_branches = [compact_branch(branch) for branch in prev.get("branches", [])]
        chosen = int(prev.get("chosen_expert", -1))
        oracle = int(prev.get("oracle_expert", -1))
        chosen_branch = next((branch for branch in prev_branches if branch["expert"] == chosen), None)
        oracle_branch = next((branch for branch in prev_branches if branch["expert"] == oracle), None)
        lines.extend(
            [
                f"Previous goal: {prev['task']}",
                f"Previous selected expert: {chosen}",
                f"Previous oracle expert by success minus step cost: {oracle}",
                f"Previous selected outcome: success={bool(prev.get('success', False))}, steps={int(prev.get('steps', ep_len))}",
                "Previous selected branch:",
                json.dumps(chosen_branch, ensure_ascii=False),
                "Previous oracle branch:",
                json.dumps(oracle_branch, ensure_ascii=False),
                "Previous all expert outcomes:",
                json.dumps(prev_branches, ensure_ascii=False),
            ]
        )
    lines.append("Produce reflection that helps choose the best current expert, not a robot action.")
    return "\n".join(lines)


def extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


def normalize_reflection(row: Dict[str, Any], reflection: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    fields = [
        "progress_status",
        "failure_type",
        "prefer_next",
        "avoid_next",
        "correction_direction",
        "memory_query",
    ]
    clean = {field: str(reflection.get(field, "uncertain"))[:200] for field in fields}
    text = " | ".join(f"{field}: {clean[field]}" for field in fields)
    return {
        "global_sequence_index": int(row["_global_sequence_index"]),
        "sequence_index": int(row.get("sequence_index", -1)),
        "subtask_index": int(row["subtask_index"]),
        "task": str(row["task"]),
        **clean,
        "reflection_text": text,
        "raw_response": raw_text.strip()[:1000],
    }


@torch.no_grad()
def generate_one(model: Any, tokenizer: Any, prompt: str, device: torch.device, max_new_tokens: int) -> Tuple[Dict[str, Any], str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    output = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return extract_json(output), output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen reflections for reflection-conditioned expert selection.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/data/manipulation/llm_models/Qwen3-8B"))
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    rows = [row for _, row in load_rows(args.roots)]
    rows.sort(key=lambda item: (int(item["_global_sequence_index"]), int(item["subtask_index"])))
    if args.limit:
        rows = rows[: args.limit]
    by_key = {row_key(row): row for row in rows}

    done = set()
    if args.resume and args.output.exists():
        for item in iter_jsonl(args.output):
            done.add((int(item["global_sequence_index"]), int(item["subtask_index"])))
    rows = [row for row in rows if row_key(row) not in done]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    if args.lora_checkpoint is not None:
        load_lora_checkpoint(model, args.lora_checkpoint, device, dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pending": len(rows),
                "model_path": str(args.model_path),
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as out:
        for idx, row in enumerate(rows, start=1):
            prev = by_key.get((int(row["_global_sequence_index"]), int(row["subtask_index"]) - 1))
            prompt = make_prompt(row, prev, args.ep_len)
            try:
                reflection, raw = generate_one(model, tokenizer, prompt, device, args.max_new_tokens)
                item = normalize_reflection(row, reflection, raw)
            except Exception as exc:
                item = normalize_reflection(
                    row,
                    {
                        "progress_status": "uncertain",
                        "failure_type": "uncertain",
                        "prefer_next": "prefer high success and fewer steps",
                        "avoid_next": "avoid previously slow or failed choices",
                        "correction_direction": "use rollout utility",
                        "memory_query": str(row["task"]),
                    },
                    f"ERROR: {exc}",
                )
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            if args.log_every and (idx == 1 or idx % args.log_every == 0):
                print(
                    json.dumps(
                        {
                            "status": "processed",
                            "idx": idx,
                            "total": len(rows),
                            "global_sequence_index": int(row["_global_sequence_index"]),
                            "subtask_index": int(row["subtask_index"]),
                            "task": str(row["task"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
