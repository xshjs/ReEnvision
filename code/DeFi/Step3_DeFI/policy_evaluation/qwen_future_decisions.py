import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

SYSTEM_PROMPT = """You judge whether a robot future hypothesis is semantically consistent.
Return one compact JSON object only:
{"consistent": boolean, "confidence": number, "corrected_task": string|null, "corrected_future_description": string|null}
Use corrected_task only when the hypothesis clearly conflicts with the requested task.
When inconsistent is false, corrected_task must be null.
When consistent is false, corrected_task must equal the requested Task exactly.
The corrected_task must be one of the known snake_case CALVIN task names if possible."""

KNOWN_TASKS = [
    "close_drawer", "open_drawer", "move_slider_left", "move_slider_right",
    "turn_on_led", "turn_off_led", "turn_on_lightbulb", "turn_off_lightbulb",
    "lift_red_block_table", "lift_blue_block_table", "lift_pink_block_table",
    "lift_red_block_slider", "lift_blue_block_slider", "lift_pink_block_slider",
    "lift_red_block_drawer", "lift_blue_block_drawer", "lift_pink_block_drawer",
    "push_red_block_left", "push_red_block_right",
    "push_blue_block_left", "push_blue_block_right",
    "push_pink_block_left", "push_pink_block_right",
    "push_into_drawer", "place_in_drawer", "place_in_slider",
    "rotate_red_block_left", "rotate_red_block_right",
    "rotate_blue_block_left", "rotate_blue_block_right",
    "rotate_pink_block_left", "rotate_pink_block_right",
    "stack_block", "unstack_block",
]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_user_prompt(row: Dict[str, Any]) -> str:
    hyp = row["future_hypothesis"]
    return "\n".join(
        [
            f"Task: {row['task']}",
            f"Current observation summary: {row['current_observation_summary']}",
            f"Future hypothesis caption: {hyp['nearest_train_future_caption']}",
            f"Future hypothesis task token: {hyp['nearest_train_future_task']}",
            "Known task tokens: " + ", ".join(KNOWN_TASKS),
        ]
    )


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


def normalize_decision(row: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    consistent = bool(decision.get("consistent", True))
    corrected_task = decision.get("corrected_task")
    if consistent:
        corrected_task = None
    elif corrected_task != row["task"]:
        corrected_task = row["task"]
    return {
        "sample_index": int(row["sample_index"]),
        "expert": int(row["expert"]),
        "task": row["task"],
        "hypothesis_task": row["future_hypothesis"]["nearest_train_future_task"],
        "consistent": consistent,
        "confidence": float(decision.get("confidence", 0.0)),
        "corrected_task": corrected_task,
        "corrected_future_description": decision.get("corrected_future_description"),
    }


@torch.no_grad()
def generate_decision(model: Any, tokenizer: Any, row: Dict[str, Any], device: torch.device, max_new_tokens: int) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
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
    output_ids = generated[0, inputs.input_ids.shape[1] :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return normalize_decision(row, extract_json(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local Qwen decisions for future-refine prompts.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/data/manipulation/llm_models/Qwen3-8B"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

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
    print(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "num_rows_pending": None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rows = list(iter_jsonl(args.prompts))
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.resume and args.output.exists():
        for row in iter_jsonl(args.output):
            done.add((int(row["sample_index"]), int(row["expert"])))
    rows = [row for row in rows if (int(row["sample_index"]), int(row["expert"])) not in done]
    print(
        json.dumps(
            {
                "num_rows_pending": len(rows),
                "output": str(args.output),
                "max_new_tokens": int(args.max_new_tokens),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as out:
        for idx, row in enumerate(rows, start=1):
            if args.log_every and (idx == 1 or idx % args.log_every == 0):
                print(
                    json.dumps(
                        {
                            "status": "generating",
                            "idx": idx,
                            "total": len(rows),
                            "sample_index": int(row["sample_index"]),
                            "expert": int(row["expert"]),
                            "task": row["task"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            decision = generate_decision(model, tokenizer, row, device, args.max_new_tokens)
            out.write(json.dumps(decision, ensure_ascii=False) + "\n")
            out.flush()
            if args.log_every and (idx == 1 or idx % args.log_every == 0):
                print(
                    json.dumps({"status": "processed", "idx": idx, "total": len(rows)}, ensure_ascii=False),
                    flush=True,
                )


if __name__ == "__main__":
    main()
