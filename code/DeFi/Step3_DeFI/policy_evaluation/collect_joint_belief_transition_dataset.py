from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def ensure_egl_vendor_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        """{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
""",
        encoding="utf-8",
    )


def newest_log_dir(logs_dir: Path, started_at: float) -> Path:
    candidates = [p for p in logs_dir.iterdir() if p.is_dir() and p.stat().st_mtime >= started_at - 2.0]
    if not candidates:
        candidates = [p for p in logs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no log directory found under {logs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-step joint-belief transition data from CALVIN rollout and "
            "build a compact T/D training memory."
        )
    )
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--ep-len", type=int, default=360)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder/saved_models/step3_defi.pt"))
    parser.add_argument("--action-model-folder", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/outputs/collect_step3_action_chunk_phi_1000_sanitize/train_folder"))
    parser.add_argument("--video-model-path", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/ckpts/_hf_defi/step1_gfdm"))
    parser.add_argument("--clip-model-path", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/ckpts/openai_clip_vit_base_patch32"))
    parser.add_argument("--t5-model-path", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/ckpts/t5_base"))
    parser.add_argument("--language-goal-path", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt"))
    parser.add_argument("--joint-pair-action-generator-ckpt", type=Path, default=Path("/mnt/workspace/manipulation/BridgeVLA/eval/joint_action_generator_mlp_d300_abc700_abc1500_v2/joint_action_generator_mlp.pt"))
    parser.add_argument("--calvin-abc-dir", type=Path, default=Path("/mnt/workspace/calvin/task_ABC_D"))
    parser.add_argument("--eval-sequences-path", type=Path, default=Path("/mnt/workspace/manipulation/DeFi/outputs/splits/outcome_router_clean_1k_train1000.json"))
    parser.add_argument("--python-bin", type=Path, default=Path("/mnt/data/xiyin/manipulation/DeFi/.venv/bin/python"))
    parser.add_argument("--summary-dim", type=int, default=1024)
    parser.add_argument("--task-dim", type=int, default=128)
    parser.add_argument("--keep-delta-norm", type=float, default=0.02)
    parser.add_argument("--replan-delta-norm", type=float, default=8.5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--debug-action", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    step3 = repo / "Step3_DeFI"
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "train_folder" / "saved_models").mkdir(parents=True, exist_ok=True)
    (args.output_root / "train_folder" / "logs").mkdir(parents=True, exist_ok=True)

    # The evaluator expects checkpoint and train folder to be paired.
    local_ckpt = args.output_root / "train_folder" / "saved_models" / args.checkpoint_path.name
    if not local_ckpt.exists():
        local_ckpt.symlink_to(args.checkpoint_path)
    action_model_folder = args.output_root / "train_folder"

    egl_vendor_json = Path("/tmp/nvidia470199/10_nvidia_470199.json")
    ensure_egl_vendor_json(egl_vendor_json)
    env = os.environ.copy()
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env.update(
        {
            "DEFI_T5_SANITIZE": env.get("DEFI_T5_SANITIZE", "1"),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "EGL_VISIBLE_DEVICE": "0",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_GL": "egl",
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(egl_vendor_json),
            "LD_LIBRARY_PATH": (
                "/tmp/nvidia470199:/usr/lib/x86_64-linux-gnu:"
                "/usr/local/nvidia/lib64:/usr/local/cuda/lib64:"
                "/tmp/nvidia_470199:/usr/local/nvidia/lib:/root/CoppeliaSim"
                + (f":{ld_library_path}" if ld_library_path else "")
            ),
            "MPLCONFIGDIR": env.get("MPLCONFIGDIR", "/tmp/mpl_defi"),
            "PYTHONPATH": str(step3),
        }
    )
    env.pop("EGL_VISIBLE_DEVICES", None)
    env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    env.pop("MESA_LOADER_DRIVER_OVERRIDE", None)

    eval_cmd = [
        str(args.python_bin),
        str(step3 / "policy_evaluation" / "calvin_evaluate_with_memory_reflection.py"),
        "--video_model_path",
        str(args.video_model_path),
        "--action_model_folder",
        str(action_model_folder),
        "--checkpoint_path",
        str(local_ckpt),
        "--clip_model_path",
        str(args.clip_model_path),
        "--t5_model_path",
        str(args.t5_model_path),
        "--language_goal_path",
        str(args.language_goal_path),
        "--calvin_abc_dir",
        str(args.calvin_abc_dir),
        "--eval_sequences_path",
        str(args.eval_sequences_path),
        "--num_sequences",
        str(args.num_sequences),
        "--ep_len",
        str(args.ep_len),
        "--no_qwen_reflection",
        "--disable_retry_after_first_pass",
        "--collect_joint_belief_transition",
    ]
    if args.joint_pair_action_generator_ckpt:
        eval_cmd += [
            "--dynamic_coupling_generate_action_intent",
            "--joint_pair_action_generator_ckpt",
            str(args.joint_pair_action_generator_ckpt),
        ]
    if args.debug_action:
        eval_cmd.append("--debug_action_before_env_step")

    started_at = time.time()
    command_json = args.output_root / "collect_command.json"
    command_json.write_text(json.dumps({"eval_cmd": eval_cmd, "env_overrides": {k: env[k] for k in ["DEFI_T5_SANITIZE", "CUDA_VISIBLE_DEVICES", "EGL_VISIBLE_DEVICE", "PYOPENGL_PLATFORM", "MUJOCO_GL", "__GLX_VENDOR_LIBRARY_NAME", "__EGL_VENDOR_LIBRARY_FILENAMES", "LD_LIBRARY_PATH", "MPLCONFIGDIR", "PYTHONPATH"]}}, indent=2), encoding="utf-8")
    subprocess.run(eval_cmd, cwd=str(step3), env=env, check=True)

    log_dir = newest_log_dir(action_model_folder / "logs", started_at)
    memory_npz = args.output_root / "joint_belief_transition_memory_s1024.npz"
    summary_json = args.output_root / "joint_belief_transition_memory_s1024.json"
    build_cmd = [
        str(args.python_bin),
        str(step3 / "policy_evaluation" / "build_joint_belief_memory_from_rollout.py"),
        "--log-dir",
        str(log_dir),
        "--output-npz",
        str(memory_npz),
        "--summary-json",
        str(summary_json),
        "--summary-dim",
        str(args.summary_dim),
        "--task-dim",
        str(args.task_dim),
        "--keep-delta-norm",
        str(args.keep_delta_norm),
        "--replan-delta-norm",
        str(args.replan_delta_norm),
    ]
    if args.max_rows:
        build_cmd += ["--max-rows", str(args.max_rows)]
    subprocess.run(build_cmd, cwd=str(repo), env=env, check=True)

    print(
        json.dumps(
            {
                "log_dir": str(log_dir),
                "rows_jsonl": str(log_dir / "joint_belief_transition_rows.jsonl"),
                "trace_dir": str(log_dir / "joint_belief_transition"),
                "memory_npz": str(memory_npz),
                "summary_json": str(summary_json),
                "command_json": str(command_json),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
