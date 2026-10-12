import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = """You judge whether a robot future hypothesis is semantically consistent.
Return one compact JSON object only:
{"consistent": boolean, "confidence": number, "corrected_task": string|null, "corrected_future_description": string|null}
Use corrected_task only when the hypothesis clearly conflicts with the requested task.
The corrected_task must be one of the known snake_case CALVIN task names if possible."""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_user_prompt(row: Dict[str, Any]) -> str:
    known = [
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
    hyp = row["future_hypothesis"]
    return "\n".join(
        [
            f"Task: {row['task']}",
            f"Current observation summary: {row['current_observation_summary']}",
            f"Future hypothesis caption: {hyp['nearest_train_future_caption']}",
            f"Future hypothesis task token: {hyp['nearest_train_future_task']}",
            "Known task tokens: " + ", ".join(known),
        ]
    )


def request_decision(api_key: str, base_url: str, model: str, row: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    decision = json.loads(content)
    return {
        "sample_index": int(row["sample_index"]),
        "expert": int(row["expert"]),
        "task": row["task"],
        "hypothesis_task": row["future_hypothesis"]["nearest_train_future_task"],
        "consistent": bool(decision.get("consistent", True)),
        "confidence": float(decision.get("confidence", 0.0)),
        "corrected_task": decision.get("corrected_task"),
        "corrected_future_description": decision.get("corrected_future_description"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepSeek LLM decisions for future-refine prompts.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    rows = list(iter_jsonl(args.prompts))
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and args.output.exists():
        for row in iter_jsonl(args.output):
            done.add((int(row["sample_index"]), int(row["expert"])))
    rows = [row for row in rows if (int(row["sample_index"]), int(row["expert"])) not in done]

    def run_one(row: Dict[str, Any]) -> Dict[str, Any]:
        last_exc = None
        for attempt in range(3):
            try:
                if args.sleep:
                    time.sleep(args.sleep)
                return request_decision(api_key, args.base_url, args.model, row, args.timeout)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"failed on sample {row.get('sample_index')}: {last_exc}") from last_exc

    mode = "a" if args.resume else "w"
    processed = 0
    with args.output.open(mode) as out:
        if args.workers <= 1:
            for row in rows:
                decision = run_one(row)
                out.write(json.dumps(decision, ensure_ascii=False) + "\n")
                out.flush()
                processed += 1
                if processed % 25 == 0:
                    print(f"processed {processed}/{len(rows)}")
            return

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, row): row for row in rows}
            for fut in as_completed(futures):
                decision = fut.result()
                out.write(json.dumps(decision, ensure_ascii=False) + "\n")
                out.flush()
                processed += 1
                if processed % 25 == 0:
                    print(f"processed {processed}/{len(rows)}")

    """
    Serial implementation kept above via workers=1. The block below is intentionally
    unreachable after the early return, but left removed from execution to avoid
    duplicate writes.
    """
    if False:
        for idx, row in enumerate(rows, start=1):
            for attempt in range(3):
                try:
                    decision = request_decision(api_key, args.base_url, args.model, row, args.timeout)
                    out.write(json.dumps(decision, ensure_ascii=False) + "\n")
                    out.flush()
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f"failed on sample {row.get('sample_index')}: {exc}") from exc
                    time.sleep(1.5 * (attempt + 1))
            if args.sleep:
                time.sleep(args.sleep)
            if idx % 25 == 0:
                print(f"processed {idx}/{len(rows)}")


if __name__ == "__main__":
    main()
