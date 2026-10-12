import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.defi_memory_models import (
    GateHead,
    RepairAdapter,
    T5TextEmbedder,
    build_gate_input,
    build_repair_input,
    load_memory_arrays,
    retrieve_memory_topk,
)
from policy_evaluation.oracle_hypothesis_rollout import (
    QwenReflectionEncoder,
    build_memory_reflection_prompt,
    count_success,
    defi_future_feature,
    ensure_train_folder,
    get_default_beso_and_env,
    get_env_state_for_initial_condition,
    goal_to_device,
    infer_mismatch_type,
    latent_tokens_from_future_feature,
    load_eval_sequences,
    load_key_memory_rows,
    load_model,
    obs_current_feature,
    parse_memory_reflection_output,
    rollout_once,
    summarize_key_memory,
    summarize_recent_memory,
)


def load_rule_memory(path: Path) -> Dict[str, Dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else {}


def _module_expected_input_dim(module: torch.nn.Module | None) -> int | None:
    if module is None:
        return None
    for attr in ("net", "model", "mlp"):
        obj = getattr(module, attr, None)
        if obj is None:
            continue
        if isinstance(obj, torch.nn.Sequential):
            for layer in obj:
                if hasattr(layer, "in_features"):
                    return int(layer.in_features)
    for layer in module.modules():
        if hasattr(layer, "in_features"):
            return int(layer.in_features)
    return None


def _project_to_reference_shape(reference: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    ref = reference
    src = source.to(device=ref.device, dtype=ref.dtype)
    if src.shape == ref.shape:
        return src
    ref_flat = ref.reshape(-1)
    src_flat = src.reshape(-1)
    if src_flat.numel() == 0:
        return torch.zeros_like(ref)
    if src_flat.numel() == ref_flat.numel():
        return src_flat.view_as(ref)
    if src_flat.numel() > ref_flat.numel():
        src_flat = src_flat[: ref_flat.numel()]
    else:
        repeat = (ref_flat.numel() + src_flat.numel() - 1) // src_flat.numel()
        src_flat = src_flat.repeat(repeat)[: ref_flat.numel()]
    return src_flat.view_as(ref)


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
        "chain_sr": {str(idx + 1): value for idx, value in enumerate(count_success(results))},
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
    parser = argparse.ArgumentParser(description="Evaluate DeFi with offline memory + learned gate + repair adapter.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", required=True)
    parser.add_argument("--language-goal-path", required=True)
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--demo-calvin-dir", type=Path, default=None)
    parser.add_argument("--eval-sequences-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--rule-memory-json", type=Path, required=True)
    parser.add_argument("--gate-ckpt", type=Path, required=True)
    parser.add_argument("--repair-ckpt", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=110)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--recent-k", type=int, default=4)
    parser.add_argument("--gate-threshold", type=float, default=-1.0)
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    with initialize(config_path="../policy_conf", job_name="evaluate_defi_memory_pipeline"):
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
    key_memory_arrays = load_memory_arrays(args.memory_npz)
    rule_memory = load_rule_memory(args.rule_memory_json)
    qwen = QwenReflectionEncoder(args.qwen_model_path, device)
    gate_ckpt = torch.load(args.gate_ckpt, map_location="cpu")
    repair_ckpt = torch.load(args.repair_ckpt, map_location="cpu")
    t5_embedder = T5TextEmbedder(Path(gate_ckpt.get("t5_path", args.t5_model_path)), device)

    gate = GateHead(int(gate_ckpt["input_dim"]), int(gate_ckpt["hidden_dim"]))
    gate.load_state_dict(gate_ckpt["model_state"])
    gate = gate.to(device).eval()
    repair = RepairAdapter(int(repair_ckpt["input_dim"]), int(repair_ckpt["future_dim"]), int(repair_ckpt["hidden_dim"]))
    repair.load_state_dict(repair_ckpt["model_state"])
    repair = repair.to(device).eval()
    repair_future_shape = tuple(int(x) for x in repair_ckpt.get("future_shape", []))
    gate_threshold = float(gate_ckpt["threshold"]) if args.gate_threshold < 0 else float(args.gate_threshold)

    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()

    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="defi_memory_gate_repair")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        recent_memory: List[Dict[str, Any]] = []
        subtasks: List[Dict[str, Any]] = []

        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            decision_obs = env.get_obs()
            current_feature = obs_current_feature(model, decision_obs, lang_text).detach().cpu().numpy().astype(np.float32)
            future_feature = defi_future_feature(model, decision_obs, lang_text).detach().cpu().numpy().astype(np.float32)
            next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None

            memory_future, memory_delta, memory_row_ids, memory_scores = retrieve_memory_topk(
                subtask,
                future_feature,
                key_memory_arrays,
                args.topk,
                exclude_row_id=None,
            )
            key_rows = [row for row in key_memory_rows if int(row["row_id"]) in set(int(item) for item in memory_row_ids.tolist())]
            recent_summary = summarize_recent_memory(recent_memory, args.recent_k)
            key_summary = summarize_key_memory(key_rows)
            rule_text = rule_memory.get(subtask, {}).get(
                "rule_text",
                "Prefer conservative future calibration and preserve verified progress.",
            )
            prompt = build_memory_reflection_prompt(subtask, next_task, recent_summary, key_summary, rule_text)
            decision = parse_memory_reflection_output(qwen.generate_reflection(prompt, max_new_tokens=192))
            reflection_text = (
                f"mismatch_type: {decision.mismatch_type} | "
                f"correction_direction: {decision.correction_direction} | "
                f"recoverability: {decision.recoverability} | "
                f"explanation: {decision.explanation}"
            )
            reflection_embedding = t5_embedder.encode([reflection_text]).squeeze(0).detach().cpu().numpy().astype(np.float32)
            gate_input = build_gate_input(
                current_feature,
                future_feature,
                memory_future,
                memory_delta,
                reflection_embedding,
                memory_scores,
            )
            expected_gate_dim = _module_expected_input_dim(gate)
            expected_repair_dim = _module_expected_input_dim(repair)
            if expected_gate_dim is None or gate_input.shape[-1] != expected_gate_dim:
                gate_prob = 0.0
                use_repair = False
                used_future = "heuristic"
            else:
                with torch.no_grad():
                    gate_prob = torch.sigmoid(gate(torch.from_numpy(gate_input).unsqueeze(0).to(device))).item()
                use_repair = gate_prob >= gate_threshold and memory_row_ids.size > 0 and expected_repair_dim is not None
                used_future = "original"

            if use_repair:
                repair_input = build_repair_input(future_feature, memory_future, memory_delta, reflection_embedding)
                if repair_input.shape[-1] == expected_repair_dim:
                    with torch.no_grad():
                        delta = repair(torch.from_numpy(repair_input).unsqueeze(0).to(device)).squeeze(0)
                    if repair_future_shape:
                        delta = delta.reshape(*repair_future_shape)
                    corrected_future = torch.from_numpy(future_feature).to(device) + _project_to_reference_shape(
                        torch.from_numpy(future_feature).to(device),
                        delta,
                    )
                    latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, corrected_future)
                    success, steps, _ = rollout_once(
                        env,
                        model,
                        task_oracle,
                        cfg,
                        subtask,
                        goal,
                        start_info,
                        "current",
                        None,
                        latent_tokens,
                    )
                    used_future = "repaired"
                else:
                    success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")
                    used_future = "heuristic"
            else:
                success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")

            mismatch_type = infer_mismatch_type(subtask, success, steps, cfg.ep_len)
            row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "success": bool(success),
                "steps": int(steps),
                "gate_prob": float(gate_prob),
                "gate_threshold": float(gate_threshold),
                "used_future": used_future,
                "mismatch_type": mismatch_type,
                "reflection_text": decision.raw_text,
                "correction_direction": decision.correction_direction,
                "recoverability": decision.recoverability,
                "recent_summary": recent_summary,
                "key_summary": key_summary,
                "memory_row_ids": [int(item) for item in memory_row_ids.tolist()],
                "memory_scores": [float(item) for item in memory_scores.tolist()],
                "repair_helped": bool(use_repair and success),
            }
            recent_memory.append(row)
            recent_memory = recent_memory[-args.recent_k :]
            subtasks.append(row)
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            if success:
                success_counter += 1
            else:
                break

        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})

    summary = write_summary(args.output_dir, "defi_memory_gate_repair", results, logs, task_total, task_success)
    summary["gate_threshold"] = float(gate_threshold)
    (args.output_dir / "defi_memory_gate_repair" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
