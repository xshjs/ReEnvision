import argparse
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from hydra import compose, initialize
from omegaconf import open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.calvin_evaluate import count_success
from policy_evaluation.calvin_evaluate import evaluate_sequence as calvin_evaluate_sequence
from policy_evaluation.calvin_evaluate import evaluate_policy as calvin_evaluate_policy
from policy_evaluation.calvin_evaluate import rollout as calvin_baseline_rollout
from policy_evaluation.defi_memory_models import load_memory_arrays
from policy_evaluation.defi_memory_models import feature_signature
from policy_evaluation.memory_state_guided_adapter import MemoryStateGuidedAdapter
from policy_evaluation.memory_weight_calibrator import MemoryWeightCalibrator, pad_weight_inputs

RULE_MEMORY = {
    "default": "Prefer conservative future calibration and preserve verified progress.",
}


def ensure_train_folder(output_dir: Path, checkpoint: Path) -> Path:
    train_folder = output_dir / "train_folder"
    saved_models = train_folder / "saved_models"
    saved_models.mkdir(parents=True, exist_ok=True)
    target = saved_models / checkpoint.name
    if not target.exists():
        target.symlink_to(checkpoint)
    return train_folder


def load_eval_sequences(num_sequences: int, eval_sequences_path: Path, _: Any = None) -> List[Any]:
    with eval_sequences_path.open("r") as handle:
        sequences = json.load(handle)
    return sequences[:num_sequences]


def load_key_memory_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open("r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_baseline_counts(path: Optional[Path]) -> Optional[List[int]]:
    if path is None:
        return None
    with path.open("r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [int(item) for item in data]
    if isinstance(data, dict):
        if "results" in data:
            return [int(item) for item in data["results"]]
        if "step3_defi" in data and isinstance(data["step3_defi"], dict) and "results" in data["step3_defi"]:
            return [int(item) for item in data["step3_defi"]["results"]]
    raise ValueError(f"Unsupported baseline counts format: {path}")


def retrieve_key_memory(rows: Sequence[Dict[str, Any]], task: str, topk: int) -> List[Dict[str, Any]]:
    task_rows = [row for row in rows if row.get("task") == task]
    return (task_rows or list(rows))[:topk]


def make_state_memory_row(
    memory_id: str,
    seq_idx: int,
    sub_idx: int,
    task: str,
    success: bool,
    steps: int,
    source: str,
) -> Dict[str, Any]:
    return {
        "memory_id": memory_id,
        "memory_type": source,
        "sequence_index": int(seq_idx),
        "subtask_index": int(sub_idx),
        "task": task,
        "success": bool(success),
        "steps": int(steps),
        "state_semantics": {
            "completed_actions": [task] if success else [],
            "object_state_summary": "state_verified" if success else "state_unverified",
            "relation_summary": "relation_verified" if success else "relation_unverified",
        },
    }


def update_runtime_state_memory(
    recent_memory: List[Dict[str, Any]],
    key_state_memory: List[Dict[str, Any]],
    row: Dict[str, Any],
    recent_k: int,
    key_max: int,
) -> None:
    recent_memory.append(row)
    del recent_memory[:-recent_k]
    if row.get("success"):
        key_state_memory.append(dict(row, memory_type="key_state"))
        if key_max > 0:
            del key_state_memory[:-key_max]


def _short_text(value: Any, limit: int = 96) -> str:
    return str(value).replace("\n", " ")[:limit]


def summarize_key_diagnostics(rows: Sequence[Dict[str, Any]], topk: int = 2) -> str:
    if not rows:
        return "none"
    parts = []
    for row in rows[:topk]:
        parts.append(
            "task={task}; mismatch={mismatch}; fix={fix}".format(
                task=_short_text(row.get("task", "unknown"), 32),
                mismatch=_short_text(row.get("mismatch_type", "unknown"), 48),
                fix=_short_text(row.get("correction_direction", row.get("reflection", "none")), 72),
            )
        )
    return " ; ".join(parts)


def summarize_key_memory(rows: Sequence[Dict[str, Any]], topk: int = 3) -> str:
    if not rows:
        return "none"
    parts = []
    for row in rows[:topk]:
        state = row.get("state_semantics", {})
        completed = state.get("completed_actions", [])
        parts.append(
            "id={id}; task={task}; success={success}; completed={completed}; state={state_text}".format(
                id=_short_text(row.get("memory_id", "unknown"), 32),
                task=_short_text(row.get("task", "unknown"), 32),
                success=bool(row.get("success")),
                completed=_short_text(completed, 48),
                state_text=_short_text(state.get("object_state_summary", "state_unknown"), 72),
            )
        )
    return " ; ".join(parts)


def summarize_recent_memory(rows: Sequence[Dict[str, Any]], topk: int = 4) -> str:
    if not rows:
        return "none"
    return " ; ".join(
        "id={id}; task={task}; success={success}; completed={completed}".format(
            id=_short_text(row.get("memory_id", "unknown"), 32),
            task=_short_text(row.get("task", "unknown"), 32),
            success=bool(row.get("success")),
            completed=_short_text(row.get("state_semantics", {}).get("completed_actions", []), 48),
        )
        for row in rows[-topk:]
    )


def trust_score_to_label(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def infer_mismatch_type(task: str, success: bool, steps: int, ep_len: int) -> str:
    del task
    if success:
        return "none"
    if steps >= ep_len:
        return "goal_progress_hallucinated"
    return "future_outcome_mismatch"


def mean_pool_feature(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)


def recoverability_to_score(recoverability: str) -> float:
    label = str(recoverability).strip().lower()
    if "repair" in label:
        return 0.85
    if "uncertain" in label:
        return 0.45
    if "not" in label or "none" in label:
        return 0.1
    return 0.35


def counterfactual_residual_gate(
    decision: Any,
    base_future: np.ndarray,
    proposal_future: np.ndarray,
    memory_sims: np.ndarray,
    memory_ids: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    base_np = np.asarray(base_future, dtype=np.float32).reshape(-1)
    proposal_np = np.asarray(proposal_future, dtype=np.float32).reshape(-1)
    if base_np.size == 0 or proposal_np.size == 0:
        return 0.0, {
            "trust_score": float(getattr(decision, "trust_score", 0.5)),
            "recoverability_score": recoverability_to_score(getattr(decision, "recoverability", "uncertain")),
            "memory_support": 0.0,
            "alignment": 0.0,
            "gate": 0.0,
        }
    trust_score = float(np.clip(getattr(decision, "trust_score", 0.5), 0.0, 1.0))
    recoverability_score = float(recoverability_to_score(getattr(decision, "recoverability", "uncertain")))
    sim_arr = np.asarray(memory_sims, dtype=np.float32).reshape(-1)
    valid_len = min(len(memory_ids), int(sim_arr.shape[0])) if memory_ids else int(sim_arr.shape[0])
    memory_support = float(np.clip(np.mean(np.clip(sim_arr[:valid_len], 0.0, 1.0)), 0.0, 1.0)) if valid_len > 0 else 0.0
    alignment = float(np.clip((cosine_np(base_np, proposal_np) + 1.0) / 2.0, 0.0, 1.0))
    mismatch = str(getattr(decision, "mismatch_type", "uncertain")).strip().lower()
    if mismatch == "none":
        mismatch_bonus = 0.0
    elif "hallucin" in mismatch or "contact" in mismatch or "overestimated" in mismatch:
        mismatch_bonus = 0.35
    elif "slow_success" in mismatch:
        mismatch_bonus = 0.12
    else:
        mismatch_bonus = 0.2

    gate = (
        0.02
        + 0.08 * trust_score
        + 0.10 * recoverability_score
        + 0.10 * memory_support
        + 0.06 * alignment
        + 0.05 * mismatch_bonus
    )
    gate *= float(np.clip((alignment - 0.18) / 0.55, 0.0, 1.0))
    if mismatch == "none":
        gate = min(gate, 0.06)
    elif "slow_success" in mismatch:
        gate = min(gate, 0.10)
    gate = float(np.clip(gate, 0.0, 0.30))
    if memory_support < 0.15 and trust_score < 0.4:
        gate *= 0.5
    trace = {
        "trust_score": trust_score,
        "recoverability_score": recoverability_score,
        "memory_support": memory_support,
        "alignment": alignment,
        "gate": gate,
    }
    return gate, trace


def blend_counterfactual_future(
    base_future: torch.Tensor,
    proposal_future: torch.Tensor,
    decision: Any,
    memory_sims: np.ndarray,
    memory_ids: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    gate, trace = counterfactual_residual_gate(
        decision,
        base_future.detach().cpu().numpy(),
        proposal_future.detach().cpu().numpy(),
        memory_sims,
        memory_ids,
    )
    corrected = base_future + gate * (proposal_future - base_future)
    trace["residual_norm"] = float(torch.norm((proposal_future - base_future).reshape(-1), p=2).item())
    trace["corrected_shift_norm"] = float(torch.norm((corrected - base_future).reshape(-1), p=2).item())
    return corrected, trace


def retrieve_adapter_memory_topk(
    task: str,
    future_feature: np.ndarray,
    memory_arrays: Dict[str, np.ndarray],
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    row_ids = memory_arrays["row_ids"].astype(np.int64)
    tasks = [str(item) for item in memory_arrays["tasks"].tolist()]
    future_bank = memory_arrays["future_features"].astype(np.float32)
    delta_bank = memory_arrays["repair_deltas"].astype(np.float32)
    future_pooled = mean_pool_feature(future_feature)
    candidate_idx = [idx for idx, item in enumerate(tasks) if item == str(task)]
    if not candidate_idx:
        candidate_idx = list(range(len(row_ids)))
    if not candidate_idx:
        dim = future_pooled.shape[0]
        return (
            np.zeros((topk, dim), dtype=np.float32),
            np.zeros((topk,), dtype=np.float32),
            [],
            np.zeros((topk,), dtype=bool),
        )
    scored = [(cosine_np(future_pooled, mean_pool_feature(future_bank[idx])), idx) for idx in candidate_idx]
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[:topk]
    memory_dim = mean_pool_feature(delta_bank[0]).shape[0]
    memory_features = np.zeros((topk, memory_dim), dtype=np.float32)
    sims = np.zeros((topk,), dtype=np.float32)
    mask = np.zeros((topk,), dtype=bool)
    memory_ids: List[str] = []
    for out_idx, (score, bank_idx) in enumerate(chosen):
        memory_features[out_idx] = mean_pool_feature(delta_bank[bank_idx])
        sims[out_idx] = float(score)
        mask[out_idx] = True
        memory_ids.append(f"KEY_{int(row_ids[bank_idx])}")
    return memory_features, sims, memory_ids, mask


def load_oracle_runtime() -> Tuple[Any, Any, Any, Any, Any]:
    from policy_evaluation.oracle_hypothesis_rollout import (
        QwenReflectionEncoder,
        defi_future_feature,
        parse_memory_reflection_output,
        raw_env_obs,
        reset_raw,
        rollout_once,
    )

    return QwenReflectionEncoder, parse_memory_reflection_output, raw_env_obs, reset_raw, rollout_once, defi_future_feature


def load_model_like_calvin_evaluate(cfg: Any, checkpoint: Path, acc: Accelerator) -> torch.nn.Module:
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(state_dict["model"], strict=False)
    model.freeze()
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model = acc.prepare(model, device_placement=[True])
    model.process_device()
    model.eval()
    return model


def baseline_rollout_like_calvin(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    lang_embeddings: Any,
    val_annotations: Dict[str, Sequence[str]],
) -> Tuple[bool, int]:
    success = calvin_baseline_rollout(
        env,
        model,
        task_oracle,
        cfg,
        subtask,
        lang_embeddings,
        val_annotations,
        record=False,
        rollout_video=None,
    )
    return bool(success), -1 if success else int(cfg.ep_len)


def rollout_goal_like_calvin(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    start_info: Optional[Dict[str, Any]] = None,
    override_future_feature: Optional[torch.Tensor] = None,
) -> Tuple[bool, int]:
    """Same rollout loop as calvin_evaluate.rollout, with an already-built goal."""
    old_mode = getattr(model, "future_feature_mode", None)
    old_override = getattr(model, "override_future_feature", None)
    try:
        if override_future_feature is not None:
            model.future_feature_mode = "override"
            model.override_future_feature = override_future_feature
        obs = env.get_obs()
        model.reset()
        if start_info is None:
            start_info = env.get_info()
        for step in range(cfg.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                return True, int(step + 1)
        return False, int(cfg.ep_len)
    finally:
        if old_mode is not None:
            model.future_feature_mode = old_mode
        if old_override is not None:
            model.override_future_feature = old_override
        elif hasattr(model, "override_future_feature"):
            model.override_future_feature = None


def snapshot_env_raw(env: Any) -> Dict[str, np.ndarray]:
    return env.env.get_obs()


def reset_env_raw(env: Any, raw: Dict[str, np.ndarray]) -> None:
    env.reset(robot_obs=raw["robot_obs"], scene_obs=raw["scene_obs"])


def summarize_important_key_memory(rows: Sequence[Dict[str, Any]], topk: int = 2) -> str:
    if not rows:
        return "none"
    return summarize_key_memory(rows, topk=topk)


def build_postexec_reflection_prompt(
    task: str,
    next_task: Optional[str],
    success: bool,
    steps: int,
    mismatch_type: str,
    recent_summary: str,
    key_summary: str,
    important_summary: str,
    rule_text: str,
) -> str:
    next_text = next_task or "none"
    return (
        "You audit the executed robot step and write a compact repair hint for the next imagined future.\n"
        "Return one compact JSON object only with keys: "
        '{"trust": "high|medium|low", "mismatch_type": string, "correction_direction": string}.\n'
        "Use short phrases.\n"
        f"task: {task}\n"
        f"next_task: {next_text}\n"
        f"execution_success: {bool(success)}\n"
        f"execution_steps: {int(steps)}\n"
        f"observed_mismatch: {mismatch_type}\n"
        f"recent_memory: {recent_summary}\n"
        f"key_memory: {key_summary}\n"
        f"important_events: {important_summary}\n"
        f"repair_rule: {rule_text}\n"
    )


def build_counterfactual_reflection_prompt(
    task: str,
    next_task: Optional[str],
    base_text: str,
    recent_summary: str,
    key_summary: str,
    important_summary: str,
    rule_text: str,
) -> str:
    next_text = next_task or "none"
    return (
        "You are doing a pre-execution counterfactual audit.\n"
        "No real failure has happened yet. Imagine the most likely failure if this future were executed as-is, "
        "then describe the counterfactual future that should replace it.\n"
        "Return one compact JSON object only with keys: "
        '{"trust": "high|medium|low", "mismatch_type": string, "hypothetical_failure": string, '
        '"counterfactual_future": string, "correction_direction": string, "failure_factor": string, '
        '"intervention": string, "recoverability": string, "explanation": string}.\n'
        "failure_factor must be one of: contact, object_displacement, object_identity, "
        "drawer_slider_progress, goal_completion, none.\n"
        "Use short phrases.\n"
        f"task: {task}\n"
        f"next_task: {next_text}\n"
        f"base_instruction: {base_text}\n"
        f"recent_memory: {recent_summary}\n"
        f"key_memory: {key_summary}\n"
        f"important_events: {important_summary}\n"
        f"repair_rule: {rule_text}\n"
    )


def build_augmented_instruction(base_text: str, prev_row: Dict[str, Any], important_summary: str) -> str:
    short_fix = str(prev_row.get('correction_direction', 'stabilize contact')).replace(" ", "_")[:24]
    short_mismatch = str(prev_row.get('mismatch_type', 'unknown')).replace(" ", "_")[:24]
    short_trust = str(prev_row.get("trust", trust_score_to_label(float(prev_row.get("trust_score", 0.5))))).replace(" ", "_")[:8]
    short_events = important_summary.replace("task=", "").replace("mismatch=", "m=").replace("fix=", "f=")
    short_events = short_events.replace(" ; ", " | ")[:80]
    return (
        f"{base_text}. "
        f"Hint: trust={short_trust}; "
        f"m={short_mismatch}; "
        f"fix={short_fix}; "
        f"events={short_events}."
    )


def build_counterfactual_instruction(base_text: str, prev_row: Dict[str, Any], important_summary: str) -> str:
    del important_summary
    short_failure = str(prev_row.get("hypothetical_failure", "possible drift")).replace(" ", "_")[:18]
    short_cf = str(prev_row.get("counterfactual_future", "preserve_progress")).replace(" ", "_")[:18]
    short_fix = str(prev_row.get("correction_direction", "stabilize_contact")).replace(" ", "_")[:18]
    short_mismatch = str(prev_row.get("mismatch_type", "unknown")).replace(" ", "_")[:18]
    short_trust = str(prev_row.get("trust", trust_score_to_label(float(prev_row.get("trust_score", 0.5))))).replace(" ", "_")[:6]
    prompt = (
        f"{base_text}. "
        f"cf={short_cf}; "
        f"fail={short_failure}; "
        f"fix={short_fix}; "
        f"m={short_mismatch}; "
        f"t={short_trust}"
    )
    # Keep the future prompt short enough for the text encoder context window.
    if len(prompt) > 180:
        prompt = f"{base_text}. cf={short_cf}; fix={short_fix}"
    return prompt[:180]


def make_reflection_row(
    seq_idx: int,
    sub_idx: int,
    subtask: str,
    success: bool,
    steps: int,
    lang_text: str,
    decision: Any,
    mismatch_type: str,
    recent_summary: str,
    key_summary: str,
    important_summary: str,
    rule_text: str,
    baseline_success: Optional[bool] = None,
    baseline_steps: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "sequence_index": seq_idx,
        "subtask_index": sub_idx,
        "task": subtask,
        "success": bool(success),
        "steps": int(steps),
        "state_semantics": {
            "completed_actions": [subtask] if success else [],
            "object_state_summary": "state_verified" if success else "state_unverified_after_retry",
            "relation_summary": "relation_verified" if success else "relation_needs_repair",
        },
        "trust_score": float(decision.trust_score),
        "trust": trust_score_to_label(float(decision.trust_score)),
        "mismatch_type": mismatch_type if mismatch_type != "none" else decision.mismatch_type,
        "correction_direction": decision.correction_direction,
        "hypothetical_failure": getattr(decision, "hypothetical_failure", ""),
        "counterfactual_future": getattr(decision, "counterfactual_future", ""),
        "reflection_text": decision.raw_text,
        "lang_text": lang_text,
        "rule_text": rule_text,
        "recent_summary": recent_summary,
        "key_summary": key_summary,
        "important_summary": important_summary,
        "baseline_success": None if baseline_success is None else bool(baseline_success),
        "baseline_steps": None if baseline_steps is None else int(baseline_steps),
    }


def write_summary(
    output_dir: Path,
    name: str,
    results: List[int],
    logs: List[Dict[str, Any]],
    task_total: Counter,
    task_success: Counter,
    key_state_memory: Optional[Sequence[Dict[str, Any]]] = None,
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
    if key_state_memory is not None:
        with (variant_dir / "runtime_key_state_memory.jsonl").open("w") as handle:
            for row in key_state_memory:
                handle.write(json.dumps(row) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeFi with online failure-triggered memory/reflection retry.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-model-folder", type=Path, default=None, help="Optional original calvin_evaluate.py action_model_folder. If set, use it as cfg.train_folder.")
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", required=True)
    parser.add_argument("--language-goal-path", required=True)
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--eval-sequences-path", type=Path, required=True)
    parser.add_argument("--use-calvin-default-sequences", action="store_true", help="Match calvin_evaluate.py rollout behavior: ignore eval JSON and use get_sequences(num_sequences).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, default=None)
    parser.add_argument("--calibrator-ckpt", type=Path, default=None)
    parser.add_argument("--future-adapter-ckpt", type=Path, default=None)
    parser.add_argument("--baseline-counts-path", type=Path, default=None, help="Deprecated: ignored. Reflection is triggered only by online failure.")
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-lora-path", type=Path, default=None)
    parser.add_argument("--qwen-python-bin", type=Path, default=None)
    parser.add_argument("--num-sequences", type=int, default=110)
    parser.add_argument("--ep-len", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--key-topk", type=int, default=3)
    parser.add_argument("--recent-k", type=int, default=4)
    parser.add_argument("--runtime-key-memory-max", type=int, default=5000)
    parser.add_argument("--disable-reflection", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../policy_conf", job_name="evaluate_defi_memory_language_conditioned"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.model.t5_model_path = args.t5_model_path
    cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_abc_dir
    if args.action_model_folder is not None:
        cfg.train_folder = str(args.action_model_folder)
        eval_checkpoint = Path(cfg.train_folder) / "saved_models" / args.checkpoint.name
    else:
        cfg.train_folder = str(ensure_train_folder(args.output_dir, args.checkpoint))
        eval_checkpoint = Path(cfg.train_folder) / "saved_models" / args.checkpoint.name
    if args.ep_len is not None:
        cfg.ep_len = args.ep_len
    cfg.num_sequences = args.num_sequences
    cfg.log_wandb = False

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    seed_everything(0, workers=True)
    print("train_folder", cfg.train_folder)
    print(device)
    print(f"Processing checkpoint: {str(eval_checkpoint)}")

    env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder,
        cfg.root_data_dir,
        eval_checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=device.index if device.type == "cuda" and device.index is not None else 0,
        cfg=cfg,
    )
    print(f"Loading model from {str(eval_checkpoint)}")
    print(f"✓ Successfully loaded checkpoint from: {str(eval_checkpoint)}")
    model = load_model_like_calvin_evaluate(cfg, eval_checkpoint, acc)
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    if args.use_calvin_default_sequences:
        sequences = get_sequences(args.num_sequences)
        print("[INFO] Using calvin_evaluate.py default get_sequences; --eval-sequences-path is ignored for rollout.")
    else:
        sequences = load_eval_sequences(args.num_sequences, args.eval_sequences_path, None)
    key_memory_rows = load_key_memory_rows(args.memory_jsonl)
    if args.baseline_counts_path is not None:
        print(f"[INFO] Ignoring --baseline-counts-path={args.baseline_counts_path}; using online actual-failure trigger.")
    qwen = None
    oracle_runtime = None
    future_adapter_bundle = None

    future_adapter_enabled = args.future_adapter_ckpt is not None
    if future_adapter_enabled and (args.memory_npz is None or args.calibrator_ckpt is None):
        raise ValueError("--memory-npz and --calibrator-ckpt are required when --future-adapter-ckpt is set")

    variant_dir = args.output_dir / "defi_memory_language_conditioned"
    variant_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_reflection:
        print("[INFO] Reflection disabled: delegating rollout to calvin_evaluate.evaluate_policy exactly.")
        results = calvin_evaluate_policy(
            model,
            env,
            lang_embeddings,
            cfg,
            num_procs=1,
            procs_id=0,
            checkpoint=eval_checkpoint,
            save_dir=variant_dir,
        )
        sequences_for_stats = get_sequences(cfg.num_sequences)
        summary = {
            "variant": "defi_memory_language_conditioned",
            "num_sequences": len(results),
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": {str(i + 1): float(sr) for i, sr in enumerate(count_success(results))},
            "results": [int(item) for item in results],
            "trigger_policy": "disabled_exact_calvin_evaluate_policy",
        }
        task_total_exact = Counter()
        task_success_exact = Counter()
        with (variant_dir / "memory_rollout_rows.jsonl").open("w") as handle:
            for seq_idx, (result, (_, eval_sequence)) in enumerate(zip(results, sequences_for_stats)):
                for sub_idx, subtask in enumerate(eval_sequence):
                    success = sub_idx < int(result)
                    task_total_exact[subtask] += 1
                    task_success_exact[subtask] += int(success)
                    row = {
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": bool(success),
                        "steps": -1,
                        "used_reflection_retry": False,
                        "language_augmented": False,
                        "lang_text": cfg.annotations[subtask][0],
                        "trigger_policy": "disabled_exact_calvin_evaluate_policy",
                    }
                    handle.write(json.dumps(row) + "\n")
                    if not success:
                        break
        summary["task_info"] = {
            task: {"success": int(task_success_exact[task]), "total": int(total)}
            for task, total in sorted(task_total_exact.items())
        }
        with (variant_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return

    def get_runtime() -> Tuple[Any, Any, Any, Any, Any, Any]:
        nonlocal oracle_runtime
        if oracle_runtime is None:
            oracle_runtime = load_oracle_runtime()
        return oracle_runtime

    def get_qwen() -> Any:
        nonlocal qwen
        if qwen is None:
            QwenReflectionEncoder, _, _, _, _, _ = get_runtime()
            qwen = QwenReflectionEncoder(
                args.qwen_model_path,
                device,
                lora_path=args.qwen_lora_path,
                python_bin=args.qwen_python_bin,
            )
        return qwen

    def get_future_adapter_bundle() -> Optional[Dict[str, Any]]:
        nonlocal future_adapter_bundle
        if not future_adapter_enabled:
            return None
        if future_adapter_bundle is None:
            assert args.memory_npz is not None
            assert args.calibrator_ckpt is not None
            assert args.future_adapter_ckpt is not None
            memory_arrays = load_memory_arrays(args.memory_npz)
            calibrator_ckpt = torch.load(args.calibrator_ckpt, map_location="cpu")
            calibrator = MemoryWeightCalibrator(
                future_dim=int(calibrator_ckpt["future_dim"]),
                memory_dim=int(calibrator_ckpt["memory_dim"]),
                max_memories=int(calibrator_ckpt["max_memories"]),
                hidden_dim=int(calibrator_ckpt["hidden_dim"]),
            )
            calibrator.load_state_dict(calibrator_ckpt["model_state"])
            calibrator = calibrator.to(device).eval()

            adapter_ckpt = torch.load(args.future_adapter_ckpt, map_location="cpu")
            adapter = MemoryStateGuidedAdapter(
                future_dim=int(adapter_ckpt["future_dim"]),
                memory_state_dim=int(adapter_ckpt["memory_state_dim"]),
                hidden_dim=int(adapter_ckpt["hidden_dim"]),
            )
            adapter.load_state_dict(adapter_ckpt["model_state"])
            adapter = adapter.to(device).eval()
            future_adapter_bundle = {
                "memory_arrays": memory_arrays,
                "calibrator": calibrator,
                "adapter": adapter,
                "token_level": bool(adapter_ckpt.get("token_level", False)),
                "future_token_shape": tuple(int(x) for x in adapter_ckpt.get("future_token_shape", ())),
                "topk": int(calibrator_ckpt["max_memories"]),
            }
            print(
                "[INFO] Loaded future adapter: "
                f"token_level={future_adapter_bundle['token_level']}, "
                f"future_token_shape={future_adapter_bundle['future_token_shape']}"
            )
        return future_adapter_bundle

    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    runtime_key_state_memory: List[Dict[str, Any]] = []

    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="defi_memory_language_conditioned")):
        success_counter = 0
        subtasks = []
        recent_memory: List[Dict[str, Any]] = []

        if args.disable_reflection:
            success_counter = int(
                calvin_evaluate_sequence(
                    env,
                    model,
                    task_oracle,
                    initial_state,
                    eval_sequence,
                    lang_embeddings,
                    cfg.annotations,
                    cfg,
                    False,
                    None,
                    seq_idx,
                )
            )
            for sub_idx, subtask in enumerate(eval_sequence):
                success = sub_idx < success_counter
                task_total[subtask] += 1
                task_success[subtask] += int(success)
                subtasks.append(
                    {
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": bool(success),
                        "steps": -1,
                        "used_reflection_retry": False,
                        "language_augmented": False,
                        "lang_text": cfg.annotations[subtask][0],
                        "trigger_policy": "disabled_exact_calvin_evaluate_sequence",
                    }
                )
                if not success:
                    break
            results.append(success_counter)
            logs.append(
                {
                    "sequence_index": seq_idx,
                    "eval_sequence": list(eval_sequence),
                    "success_counter": success_counter,
                    "trigger_policy": "disabled_exact_calvin_evaluate_sequence",
                    "subtasks": subtasks,
                }
            )
            continue

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        for sub_idx, subtask in enumerate(eval_sequence):
            base_lang_text = cfg.annotations[subtask][0]
            step_start_raw = snapshot_env_raw(env)
            step_start_info = env.get_info()
            success, steps = baseline_rollout_like_calvin(
                env,
                model,
                task_oracle,
                cfg,
                subtask,
                lang_embeddings,
                cfg.annotations,
            )
            if success:
                state_row = make_state_memory_row(
                    f"R_seq{seq_idx}_sub{sub_idx}",
                    seq_idx,
                    sub_idx,
                    subtask,
                    True,
                    int(steps),
                    "recent_state",
                )
                update_runtime_state_memory(
                    recent_memory,
                    runtime_key_state_memory,
                    state_row,
                    args.recent_k,
                    args.runtime_key_memory_max,
                )
                row = {
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": True,
                    "steps": int(steps),
                    "state_semantics": {
                        "completed_actions": [subtask],
                        "object_state_summary": "state_verified",
                        "relation_summary": "relation_verified",
                    },
                    "used_reflection_retry": False,
                    "language_augmented": False,
                    "lang_text": base_lang_text,
                    "trigger_policy": "online_actual_failure",
                    "recent_state_memory_ids": [item.get("memory_id") for item in recent_memory],
                    "runtime_key_state_memory_size": len(runtime_key_state_memory),
                }
            else:
                if args.disable_reflection:
                    row = {
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": False,
                        "steps": int(steps),
                        "state_semantics": {
                            "completed_actions": [],
                            "object_state_summary": "state_unverified_after_baseline",
                            "relation_summary": "relation_unverified_after_baseline",
                        },
                        "used_reflection_retry": False,
                        "language_augmented": False,
                        "lang_text": base_lang_text,
                        "initial_attempt_success": False,
                        "initial_attempt_steps": int(steps),
                        "trigger_policy": "online_actual_failure",
                    }
                else:
                    baseline_steps = int(steps)
                    repair_sub_idx = sub_idx
                    repair_task = subtask
                    repair_lang_text = base_lang_text
                    _, parse_memory_reflection_output, raw_env_obs, reset_raw, _, defi_future_feature = get_runtime()
                    del raw_env_obs, reset_raw
                    reset_env_raw(env, step_start_raw)
                    mismatch_type = infer_mismatch_type(repair_task, False, baseline_steps, cfg.ep_len)
                    next_task = eval_sequence[repair_sub_idx + 1] if repair_sub_idx + 1 < len(eval_sequence) else None
                    runtime_key_rows = retrieve_key_memory(runtime_key_state_memory, repair_task, args.key_topk)
                    historical_key_rows = retrieve_key_memory(key_memory_rows, repair_task, max(0, args.key_topk - len(runtime_key_rows)))
                    key_rows = (runtime_key_rows + historical_key_rows)[: args.key_topk]
                    recent_summary = summarize_recent_memory(recent_memory, args.recent_k)
                    key_summary = summarize_key_memory(key_rows)
                    important_summary = summarize_important_key_memory(key_rows)
                    rule_text = RULE_MEMORY.get(repair_task, "Prefer conservative future calibration and preserve verified progress.")
                    prompt = build_postexec_reflection_prompt(
                        repair_task,
                        next_task,
                        False,
                        int(steps),
                        mismatch_type,
                        recent_summary,
                        key_summary,
                        important_summary,
                        rule_text,
                    )
                    qwen_encoder = get_qwen()
                    decision = parse_memory_reflection_output(qwen_encoder.generate_reflection(prompt, max_new_tokens=192))
                    reflection_row = make_reflection_row(
                        seq_idx,
                        repair_sub_idx,
                        repair_task,
                        False,
                        baseline_steps,
                        repair_lang_text,
                        decision,
                        mismatch_type,
                        recent_summary,
                        key_summary,
                        important_summary,
                        rule_text,
                        baseline_success=False,
                        baseline_steps=baseline_steps,
                    )
                    retry_lang_text = build_augmented_instruction(repair_lang_text, reflection_row, important_summary)
                    retry_goal = lang_embeddings.get_lang_goal(repair_lang_text)
                    retry_goal["lang_text"] = retry_lang_text
                    retry_start_info = step_start_info
                    adapter_info: Dict[str, Any] = {"future_adapter_enabled": bool(future_adapter_enabled), "future_adapter_used": False}
                    adapter_bundle = get_future_adapter_bundle()
                    if adapter_bundle is not None and bool(adapter_bundle.get("token_level", False)):
                        decision_obs = env.get_obs()
                        original_future = defi_future_feature(model, decision_obs, retry_lang_text).detach().to(model.device)
                        original_future_np = original_future.detach().cpu().numpy().astype(np.float32)
                        memory_features, memory_sims, memory_ids, memory_mask = retrieve_adapter_memory_topk(
                            repair_task,
                            original_future_np,
                            adapter_bundle["memory_arrays"],
                            int(adapter_bundle["topk"]),
                        )
                        if memory_ids:
                            qwen_weight_map = {memory_id: 1.0 / float(len(memory_ids)) for memory_id in memory_ids}
                            similarity_map = {memory_id: float(memory_sims[idx]) for idx, memory_id in enumerate(memory_ids)}
                            qwen_weights, sims, mask = pad_weight_inputs(
                                qwen_weight_map,
                                similarity_map,
                                memory_ids,
                                int(adapter_bundle["topk"]),
                            )
                            future_pooled = mean_pool_feature(original_future_np)
                            with torch.no_grad():
                                _, memory_state, cal_aux = adapter_bundle["calibrator"](
                                    torch.from_numpy(future_pooled).unsqueeze(0).to(device),
                                    torch.from_numpy(memory_features).unsqueeze(0).to(device),
                                    qwen_weights.unsqueeze(0).to(device),
                                    sims.unsqueeze(0).to(device),
                                    mask.unsqueeze(0).to(device),
                                )
                                corrected_future, _, adapter_aux = adapter_bundle["adapter"](
                                    original_future.unsqueeze(0).to(device),
                                    memory_state,
                                )
                            corrected_future = corrected_future.squeeze(0).detach().to(model.device)
                            adapter_info.update(
                                {
                                    "future_adapter_used": True,
                                    "adapter_memory_ids": list(memory_ids),
                                    "adapter_memory_scores": [float(x) for x in memory_sims[: len(memory_ids)].tolist()],
                                    "adapter_weight_mean": float(cal_aux["aggregated_memory"].detach().mean().item()),
                                    "adapter_gate_mean": float(adapter_aux["gate"].detach().mean().item()),
                                    "adapter_delta_norm": float(torch.norm((corrected_future - original_future).reshape(-1), p=2).item()),
                                }
                            )
                            success, steps = rollout_goal_like_calvin(
                                env,
                                model,
                                task_oracle,
                                cfg,
                                repair_task,
                                retry_goal,
                                retry_start_info,
                                corrected_future,
                            )
                        else:
                            adapter_info["future_adapter_skip_reason"] = "no_memory_candidates"
                            success, steps = rollout_goal_like_calvin(env, model, task_oracle, cfg, repair_task, retry_goal, retry_start_info)
                    else:
                        if adapter_bundle is not None:
                            adapter_info["future_adapter_skip_reason"] = "adapter_checkpoint_not_token_level"
                        success, steps = rollout_goal_like_calvin(env, model, task_oracle, cfg, repair_task, retry_goal, retry_start_info)
                    row = make_reflection_row(
                        seq_idx,
                        repair_sub_idx,
                        repair_task,
                        success,
                        steps,
                        retry_lang_text,
                        decision,
                        infer_mismatch_type(subtask, success, steps, cfg.ep_len),
                        recent_summary,
                        key_summary,
                        important_summary,
                        rule_text,
                        baseline_success=False,
                        baseline_steps=baseline_steps,
                    )
                    row["used_reflection_retry"] = True
                    row["language_augmented"] = True
                    row["trigger_policy"] = "online_actual_failure"
                    row["triggered_after_initial_failure"] = True
                    row["initial_attempt_steps"] = baseline_steps
                    row.update(adapter_info)
                    state_row = make_state_memory_row(
                        f"R_seq{seq_idx}_repair{repair_sub_idx}",
                        seq_idx,
                        repair_sub_idx,
                        repair_task,
                        bool(success),
                        int(steps),
                        "recent_state",
                    )
                    update_runtime_state_memory(
                        recent_memory,
                        runtime_key_state_memory,
                        state_row,
                        args.recent_k,
                        args.runtime_key_memory_max,
                    )
                    row["recent_state_memory_ids"] = [item.get("memory_id") for item in recent_memory]
                    row["selected_key_state_memory_ids"] = [item.get("memory_id") for item in runtime_key_rows]
                    row["runtime_key_state_memory_size"] = len(runtime_key_state_memory)
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(row)
            if success:
                success_counter += 1
            else:
                break

        results.append(success_counter)
        logs.append(
            {
                "sequence_index": seq_idx,
                "eval_sequence": list(eval_sequence),
                "success_counter": success_counter,
                "trigger_policy": "online_actual_failure",
                "subtasks": subtasks,
            }
        )

    summary = write_summary(
        args.output_dir,
        "defi_memory_language_conditioned",
        results,
        logs,
        task_total,
        task_success,
        runtime_key_state_memory,
    )
    (args.output_dir / "defi_memory_language_conditioned" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if qwen is not None:
        qwen.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
