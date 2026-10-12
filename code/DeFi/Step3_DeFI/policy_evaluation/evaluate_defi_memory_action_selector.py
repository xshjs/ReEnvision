import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import T5TextEmbedder
from policy_evaluation.oracle_hypothesis_rollout import (
    RULE_MEMORY,
    build_demo_bank,
    build_memory_reflection_prompt,
    defi_future_feature,
    demo_future_feature,
    ensure_train_folder,
    get_default_beso_and_env,
    get_env_state_for_initial_condition,
    goal_to_device,
    infer_mismatch_type,
    load_eval_sequences,
    load_key_memory_rows,
    load_model,
    parse_memory_reflection_output,
    raw_env_obs,
    reset_raw,
    retrieve_key_memory,
    rollout_once,
    summarize_key_memory,
    summarize_recent_memory,
)
from policy_evaluation.train_defi_memory_action_selector import (
    SELECTOR_CLASSES,
    ActionSelectorMLP,
    build_prev_context_text,
    build_selector_input,
)


def choose_branch(
    logits: torch.Tensor,
    available: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    probs = torch.softmax(logits, dim=0)
    ranked = torch.argsort(probs, descending=True).tolist()
    for idx in ranked:
        name = SELECTOR_CLASSES[idx]
        if name in available:
            return name, {SELECTOR_CLASSES[i]: float(probs[i]) for i in range(len(SELECTOR_CLASSES))}
    return "continue", {SELECTOR_CLASSES[i]: float(probs[i]) for i in range(len(SELECTOR_CLASSES))}


def write_summary(
    output_dir: Path,
    name: str,
    results: List[int],
    logs: List[Dict[str, Any]],
    task_total: Counter,
    task_success: Counter,
) -> Dict[str, Any]:
    summary = {
        "variant": name,
        "num_sequences": len(results),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": {str(idx + 1): value for idx, value in enumerate([(np.asarray(results) >= (idx + 1)).mean() if results else 0.0 for idx in range(10)])},
        "results": list(results),
        "task_info": {
            task: {"success": int(task_success[task]), "total": int(total)}
            for task, total in sorted(task_total.items())
        },
    }
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (variant_dir / "memory_rollout_rows.jsonl").open("w") as handle:
        for log in logs:
            for row in log.get("subtasks", []):
                handle.write(json.dumps(row) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeFi with reflection-memory-conditioned next-step action selector.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", required=True)
    parser.add_argument("--language-goal-path", required=True)
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--demo-calvin-dir", type=Path, required=True)
    parser.add_argument("--eval-sequences-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--selector-ckpt", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=110)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--key-topk", type=int, default=3)
    parser.add_argument("--recent-k", type=int, default=4)
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    with initialize(config_path="../policy_conf", job_name="evaluate_defi_memory_action_selector"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.model.t5_model_path = args.t5_model_path
    cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.train_folder = str(ensure_train_folder(args.output_dir, args.checkpoint))
    cfg.ep_len = args.ep_len
    cfg.num_sequences = args.num_sequences
    cfg.log_wandb = False
    with open_dict(cfg):
        cfg.future_feature_mode = "gfdm"
        cfg.future_frame_idx = 5

    env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder,
        cfg.root_data_dir,
        args.checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=device.index if device.type == "cuda" and device.index is not None else "cpu",
        cfg=cfg,
    )
    model = load_model(cfg, args.checkpoint, device)
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    sequences = load_eval_sequences(args.num_sequences, args.eval_sequences_path, None)
    key_memory_rows = load_key_memory_rows(args.memory_jsonl)
    demo_bank = build_demo_bank(args.demo_calvin_dir)

    from policy_evaluation.oracle_hypothesis_rollout import QwenReflectionEncoder, selector_feature_parts

    qwen = QwenReflectionEncoder(args.qwen_model_path, device)
    selector_ckpt = torch.load(args.selector_ckpt, map_location="cpu")
    selector = ActionSelectorMLP(
        int(selector_ckpt["input_dim"]),
        int(selector_ckpt["hidden_dim"]),
        len(selector_ckpt["classes"]),
    )
    selector.load_state_dict(selector_ckpt["model_state"])
    selector = selector.to(device).eval()
    selector_mean = torch.as_tensor(selector_ckpt["mean"], dtype=torch.float32)
    selector_std = torch.as_tensor(selector_ckpt["std"], dtype=torch.float32)
    t5_embedder = T5TextEmbedder(Path(selector_ckpt.get("t5_path", args.t5_model_path)), device)

    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    feature_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="defi_memory_action_selector")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        recent_memory: List[Dict[str, Any]] = []
        prev_row: Optional[Dict[str, Any]] = None

        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            decision_obs = env.get_obs()
            _ = defi_future_feature(model, decision_obs, lang_text)
            next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
            recent_summary = summarize_recent_memory(recent_memory, args.recent_k)
            key_rows = retrieve_key_memory(key_memory_rows, subtask, args.key_topk)
            key_summary = summarize_key_memory(key_rows)
            rule_text = RULE_MEMORY.get(subtask, "Prefer conservative future calibration and preserve verified progress.")
            prompt = build_memory_reflection_prompt(subtask, next_task, recent_summary, key_summary, rule_text)
            current_decision = parse_memory_reflection_output(qwen.generate_reflection(prompt, max_new_tokens=192))

            selector_choice = "continue"
            selector_probs = {name: 0.0 for name in SELECTOR_CLASSES}
            selector_probs["continue"] = 1.0
            branch_mode = "gfdm"
            branch_override = None

            if prev_row is not None:
                feature_parts, selector_scalars = selector_feature_parts(model, decision_obs, goal, [])
                context_text = build_prev_context_text(prev_row, subtask)
                context_embedding = t5_embedder.encode([context_text]).squeeze(0).detach().cpu().numpy().astype(np.float32)
                selector_input = build_selector_input(feature_parts["state"], selector_scalars, context_embedding, prev_row)
                selector_tensor = torch.from_numpy(selector_input).to(device)
                selector_tensor = (selector_tensor - selector_mean.to(device)) / selector_std.to(device)
                with torch.no_grad():
                    logits = selector(selector_tensor.unsqueeze(0)).squeeze(0)
                available = ["continue", "recover"]
                if subtask in demo_bank:
                    available.append("retry_demo")
                selector_choice, selector_probs = choose_branch(logits, available)
                if selector_choice == "recover":
                    branch_mode = "current"
                elif selector_choice == "retry_demo":
                    branch_mode = "override"
                    key = (subtask, lang_text)
                    if key not in feature_cache:
                        feature_cache[key] = demo_future_feature(model, demo_bank[subtask], lang_text)
                    branch_override = feature_cache[key]

            success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, branch_mode, branch_override)
            mismatch_type = infer_mismatch_type(subtask, success, steps, cfg.ep_len)
            row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "success": bool(success),
                "steps": int(steps),
                "selector_choice": selector_choice,
                "selector_probs": selector_probs,
                "trust_score": float(current_decision.trust_score),
                "mismatch_type": mismatch_type if mismatch_type != "none" else current_decision.mismatch_type,
                "correction_direction": current_decision.correction_direction,
                "recoverability": current_decision.recoverability,
                "reflection_text": current_decision.raw_text,
                "used_future": "original",
                "rule_text": rule_text,
                "recent_summary": recent_summary,
                "key_summary": key_summary,
                "previous_reflection_used": prev_row is not None,
            }
            recent_memory.append(row)
            recent_memory = recent_memory[-args.recent_k :]
            prev_row = row
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(row)
            if success:
                success_counter += 1
            else:
                break

        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})

    summary = write_summary(args.output_dir, "defi_memory_action_selector", results, logs, task_total, task_success)
    (args.output_dir / "defi_memory_action_selector" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
