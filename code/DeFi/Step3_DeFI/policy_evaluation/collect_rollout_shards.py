import argparse
import json
import subprocess
import sys
from pathlib import Path


def shard_name(prefix: str, start: int, end: int) -> str:
    return f"{prefix}_shard_{start:04d}_{end - 1:04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rollout data collection over sequence shards.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rollout-script", type=Path, default=Path("Step3_DeFI/policy_evaluation/oracle_hypothesis_rollout.py"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefix", default="rollout_collect")
    parser.add_argument("--eval-sequences-path", type=Path, required=True)
    parser.add_argument("--sequence-start", type=int, default=0)
    parser.add_argument("--sequence-end", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=["collect_moe_universal_self_dataset", "collect_success_expert_policy_dataset"])
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--demo-calvin-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--vjepa-moe", type=Path, required=True)
    parser.add_argument("--vjepa-root", type=Path, required=True)
    parser.add_argument("--vjepa-bank-cache", type=Path, required=True)
    parser.add_argument("--vpp-bank-per-task", type=int, default=8)
    parser.add_argument("--execution-signal-steps", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.sequence_end <= args.sequence_start:
        raise ValueError("--sequence-end must be greater than --sequence-start")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)

    for start in range(args.sequence_start, args.sequence_end, args.shard_size):
        end = min(start + args.shard_size, args.sequence_end)
        out_dir = args.output_root / shard_name(args.prefix, start, end)
        selector_summary = out_dir / "collect_moe_universal_self_dataset" / "summary.json"
        policy_summary = out_dir / "collect_success_expert_policy_dataset" / "summary.json"
        required_summaries = []
        if "collect_moe_universal_self_dataset" in args.variants:
            required_summaries.append(selector_summary)
        if "collect_success_expert_policy_dataset" in args.variants:
            required_summaries.append(policy_summary)
        if args.skip_existing and required_summaries and all(summary.exists() for summary in required_summaries):
            print(
                json.dumps(
                    {
                        "status": "skip_existing",
                        "start": start,
                        "end": end,
                        "output_dir": str(out_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        cmd = [
            args.python,
            str(args.rollout_script),
            "--video-model-path",
            args.video_model_path,
            "--checkpoint",
            str(args.checkpoint),
            "--clip-model-path",
            args.clip_model_path,
            "--t5-model-path",
            args.t5_model_path,
            "--language-goal-path",
            args.language_goal_path,
            "--calvin-abc-dir",
            args.calvin_abc_dir,
            "--demo-calvin-dir",
            str(args.demo_calvin_dir),
            "--output-dir",
            str(out_dir),
            "--eval-sequences-path",
            str(args.eval_sequences_path),
            "--num-sequences",
            str(end - start),
            "--sequence-offset",
            str(start),
            "--ep-len",
            str(args.ep_len),
            "--device",
            args.device,
            "--variants",
            *args.variants,
            "--vjepa-moe",
            str(args.vjepa_moe),
            "--vjepa-root",
            str(args.vjepa_root),
            "--vjepa-bank-cache",
            str(args.vjepa_bank_cache),
            "--vpp-bank-per-task",
            str(args.vpp_bank_per_task),
            "--execution-signal-steps",
            str(args.execution_signal_steps),
        ]
        print(
            json.dumps(
                {
                    "status": "start",
                    "start": start,
                    "end": end,
                    "output_dir": str(out_dir),
                    "cmd": cmd,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        subprocess.run(cmd, check=True)
        print(
            json.dumps(
                {
                    "status": "done",
                    "start": start,
                    "end": end,
                    "output_dir": str(out_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
