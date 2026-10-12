from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

STEP3_ROOT = Path(__file__).resolve().parents[1]
if str(STEP3_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP3_ROOT))

from policy_evaluation.qwen_reflection_worker import generate_reflection, load_model


@dataclass
class Sample:
    idx: int
    success: bool
    metadata: dict[str, Any]
    messages: list[dict[str, str]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for idx, row in enumerate(read_jsonl(path)):
        metadata = dict(row.get("metadata", {}))
        if "success" in metadata:
            success = bool(metadata["success"])
        else:
            success = (
                str(metadata.get("failure_factor", "")) == "none"
                or str(metadata.get("mismatch_type", "")) == "none"
            )
        samples.append(
            Sample(
                idx=idx,
                success=success,
                metadata=metadata,
                messages=list(row["messages"]),
            )
        )
    return samples


def safe_json_load(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)


def generate_reflection_batch(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts: list[str] = []
    for idx in range(outputs.shape[0]):
        generated = outputs[idx][tokens["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        texts.append(
            text
            or (
                '{"trust": "medium", "mismatch_type": "unknown", '
                '"correction_direction": "stabilize contact"}'
            )
        )
    return texts


def score_sample(sample: Sample, pred: dict[str, Any] | None) -> dict[str, Any]:
    gold = sample.messages[-1]["content"]
    gold_obj = safe_json_load(gold) or {}
    pred_obj = pred or {}
    gold_trust = str(gold_obj.get("trust", ""))
    gold_mismatch = str(gold_obj.get("mismatch_type", ""))
    gold_correction = str(gold_obj.get("correction_direction", ""))
    pred_trust = str(pred_obj.get("trust", ""))
    pred_mismatch = str(pred_obj.get("mismatch_type", ""))
    pred_correction = str(pred_obj.get("correction_direction", ""))

    exact = (
        pred_trust == gold_trust
        and pred_mismatch == gold_mismatch
        and pred_correction == gold_correction
    )
    success_overcorrect = sample.success and (
        pred_trust != "high"
        or pred_mismatch != "none"
        or pred_correction != "preserve progress"
    )
    failure_reasonable = (not sample.success) and (
        pred_trust in {"low", "medium"}
        and pred_mismatch not in {"", "none"}
        and pred_correction not in {"", "preserve progress"}
    )
    return {
        "idx": sample.idx,
        "success": sample.success,
        "task": sample.metadata.get("task", ""),
        "lang_goal": sample.metadata.get("lang_goal", ""),
        "gold": gold_obj,
        "pred": pred_obj,
        "parse_ok": pred is not None,
        "exact_match": exact,
        "success_overcorrect": success_overcorrect,
        "failure_reasonable": failure_reasonable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen reflection LoRA on held-out success/failure samples.")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/data/manipulation/llm_models/Qwen3-8B"))
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen_reflection_lora_validation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-task", type=int, default=0, help="If > 0, sample up to this many rows per task.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-failures", type=int, default=16)
    parser.add_argument("--num-successes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = load_samples(args.samples_jsonl)
    failures = [s for s in samples if not s.success]
    successes = [s for s in samples if s.success]
    if not failures:
        raise ValueError("no failure samples found")
    if not successes:
        raise ValueError("no success samples found")

    if args.per_task > 0:
        by_task: dict[str, list[Sample]] = {}
        for sample in samples:
            task = str(sample.metadata.get("task", ""))
            by_task.setdefault(task, []).append(sample)
        picked = []
        for task in sorted(by_task):
            task_rows = by_task[task]
            if not task_rows:
                continue
            n_take = min(args.per_task, len(task_rows))
            picked.extend(rng.sample(task_rows, n_take))
    else:
        n_fail = min(args.num_failures, len(failures))
        n_succ = min(args.num_successes, len(successes))
        picked = rng.sample(failures, n_fail) + rng.sample(successes, n_succ)
    picked.sort(key=lambda s: (s.success, s.idx))

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_path, device, args.lora_path)

    results: list[dict[str, Any]] = []
    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(picked), batch_size):
        batch = picked[start : start + batch_size]
        print(
            json.dumps(
                {
                    "progress": "batch_start",
                    "batch_index": start // batch_size,
                    "batch_size": len(batch),
                    "done": start,
                    "total": len(picked),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        prompts = [render_prompt(tokenizer, sample.messages) for sample in batch]
        raws = generate_reflection_batch(tokenizer, model, prompts, args.max_new_tokens)
        for sample, raw in zip(batch, raws):
            pred = safe_json_load(raw)
            item = score_sample(sample, pred)
            item["raw"] = raw
            results.append(item)
        print(
            json.dumps(
                {
                    "progress": "batch_done",
                    "batch_index": start // batch_size,
                    "done": min(start + len(batch), len(picked)),
                    "total": len(picked),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "samples_jsonl": str(args.samples_jsonl),
        "model_path": str(args.model_path),
        "lora_path": str(args.lora_path),
        "num_samples": len(results),
        "num_failures": sum(1 for r in results if not r["success"]),
        "num_successes": sum(1 for r in results if r["success"]),
        "parse_rate": sum(1 for r in results if r["parse_ok"]) / max(len(results), 1),
        "exact_match_rate": sum(1 for r in results if r["exact_match"]) / max(len(results), 1),
        "failure_reasonable_rate": sum(1 for r in results if r["failure_reasonable"]) / max(sum(1 for r in results if not r["success"]), 1),
        "success_overcorrect_rate": sum(1 for r in results if r["success_overcorrect"]) / max(sum(1 for r in results if r["success"]), 1),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in results:
        tag = "SUCCESS" if row["success"] else "FAIL"
        print(
            json.dumps(
                {
                    "tag": tag,
                    "idx": row["idx"],
                    "task": row["task"],
                    "gold": row["gold"],
                    "pred": row["pred"],
                    "raw": row["raw"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
