import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRIDGE_FINETUNE = ROOT.parents[1] / "BridgeVLA" / "finetune"
BRIDGE_RLBENCH_DIR = BRIDGE_FINETUNE / "RLBench"
BRIDGE_LIBS = BRIDGE_FINETUNE / "bridgevla" / "libs"
for path in [
    BRIDGE_FINETUNE,
    BRIDGE_RLBENCH_DIR,
    BRIDGE_LIBS / "PyRep",
    BRIDGE_LIBS / "RLBench",
    BRIDGE_LIBS / "YARR",
    BRIDGE_LIBS / "point-renderer",
]:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from scipy.spatial.transform import Rotation as R

from eval import load_agent as load_bridge_agent  # type: ignore
from eval import eval as bridge_eval  # type: ignore
from policy_models.RLBench_gidm_mvt_policy import RLBenchGIDMMVTPolicy
from policy_evaluation.defi_memory_models import (
    GateHead,
    RepairAdapter,
    T5TextEmbedder,
    build_gate_input,
    build_repair_input,
    iter_jsonl,
    load_memory_arrays,
    retrieve_memory_topk,
)
from policy_evaluation.oracle_hypothesis_rollout import (
    QwenReflectionEncoder,
    build_augmented_future_instruction,
    build_postexec_reflection_prompt,
    build_memory_reflection_prompt,
    parse_memory_reflection_output,
    summarize_important_key_memory,
    summarize_recent_memory,
    summarize_key_memory,
    template_postexec_reflection_decision,
)
from policy_evaluation.evaluate_defi_memory_language_conditioned import make_reflection_row

try:
    from policy_evaluation.label_defi_future_mismatch import infer_mismatch_type
except Exception:  # pragma: no cover - optional fallback
    infer_mismatch_type = None


def _quat_from_euler(euler_xyz: torch.Tensor) -> torch.Tensor:
    euler_np = euler_xyz.detach().cpu().numpy()
    quat_np = R.from_euler("xyz", euler_np, degrees=False).as_quat().astype("float32")
    quat = torch.from_numpy(quat_np).to(euler_xyz.device)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _extract_scalar(observation: dict, keys: list[str], default: float = 0.0) -> torch.Tensor:
    for key in keys:
        if key not in observation:
            continue
        value = observation[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.float()
        if value.ndim == 0:
            return value.reshape(1, 1)
        if value.ndim == 1:
            return value.reshape(-1, 1)
        return value.reshape(value.shape[0], -1)[:, :1]
    return torch.tensor([[default]], dtype=torch.float32)


def filter_state_dict_by_shape(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append((key, f"{tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue
        filtered[key] = value
    return filtered, skipped


def _pool_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).mean(dim=1)


def _ensure_future_shape(reference: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 1:
        delta = delta.view(1, 1, -1)
    elif delta.ndim == 2:
        delta = delta.unsqueeze(1)
    if reference.ndim == 2 and delta.ndim == 3:
        delta = delta.mean(dim=1)
    return delta.to(device=reference.device, dtype=reference.dtype)


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


def _load_jsonl_rows(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    for row in iter_jsonl(path):
        rows.append(row)
    return rows


def _load_rule_memory(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _path_or_none(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


_TASK_STOPWORDS = {
    "to",
    "of",
    "in",
    "the",
    "and",
    "with",
    "on",
    "at",
    "for",
}


def _task_keywords(task_name: str) -> list[str]:
    tokens = [tok for tok in re.split(r"[_\s]+", str(task_name).strip().lower()) if tok]
    return [tok for tok in tokens if tok not in _TASK_STOPWORDS and len(tok) > 1]


def _language_matches_task(text: str, task_name: str) -> bool:
    text_lower = str(text).lower()
    keywords = _task_keywords(task_name)
    if not keywords:
        return True
    return any(keyword in text_lower for keyword in keywords)


def _module_expected_input_dim(module: torch.nn.Module | None) -> Optional[int]:
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


def _infer_task_hint(lang_text: str, task_names: Sequence[str] | None = None) -> str:
    text = lang_text.lower()
    aliases = [
        ("close_jar", ("jar", "close the jar")),
        ("open_drawer", ("drawer", "open the drawer")),
        ("insert_onto_square_peg", ("square peg", "peg")),
        ("light_bulb_in", ("bulb", "light bulb")),
        ("meat_off_grill", ("grill", "meat")),
        ("place_cups", ("cups", "cup")),
        ("place_shape_in_shape_sorter", ("shape sorter", "shape")),
        ("place_wine_at_rack_location", ("wine", "rack")),
        ("push_buttons", ("button", "buttons")),
        ("put_groceries_in_cupboard", ("groceries", "cupboard")),
        ("put_item_in_drawer", ("item", "drawer")),
        ("put_money_in_safe", ("money", "safe")),
        ("reach_and_drag", ("drag", "stick")),
        ("slide_block_to_color_target", ("slide", "block", "target")),
        ("stack_blocks", ("stack blocks", "blocks")),
        ("stack_cups", ("stack cups", "cups")),
        ("sweep_to_dustpan_of_size", ("dustpan", "sweep")),
        ("turn_tap", ("tap", "faucet")),
    ]
    for task_name, needles in aliases:
        if all(needle in text for needle in needles):
            return task_name
    if task_names:
        for task_name in task_names:
            if task_name.replace("_", " ") in text:
                return task_name
    return text.replace(" ", "_")


@dataclass
class MemoryReflectionBundle:
    enabled: bool = False
    key_rows: list[dict[str, Any]] | None = None
    memory_arrays: dict[str, np.ndarray] | None = None
    rule_memory: dict[str, dict[str, Any]] | None = None
    qwen: Any | None = None
    gate: torch.nn.Module | None = None
    repair: torch.nn.Module | None = None
    t5_embedder: Any | None = None
    memory_topk: int = 3
    gate_threshold: float = 0.5
    fallback_scale: float = 0.1
    task_names: tuple[str, ...] = ()

    def has_learned_modules(self) -> bool:
        return self.qwen is not None and self.gate is not None and self.repair is not None and self.t5_embedder is not None


class BridgeDeFiSidecarAgent:
    def __init__(
        self,
        bridge_agent,
        sidecar=None,
        sidecar_log_dir: str = "",
        memory_bundle: MemoryReflectionBundle | None = None,
        gidm_gate: float = 0.1,
        online_failure_trigger: bool = True,
        online_failure_trigger_steps: int = 2,
        online_failure_gate: float = 1.0,
        calvin_reflection_mode: bool = False,
    ):
        self.bridge_agent = bridge_agent
        self.sidecar = sidecar
        self.sidecar_log_dir = Path(sidecar_log_dir) if sidecar_log_dir else None
        self.memory_bundle = memory_bundle if memory_bundle is not None else MemoryReflectionBundle()
        self._gidm_gate = float(gidm_gate)
        self._online_failure_trigger = bool(online_failure_trigger)
        self._online_failure_trigger_steps = int(online_failure_trigger_steps)
        self._online_failure_gate = float(online_failure_gate)
        self._calvin_reflection_mode = bool(calvin_reflection_mode)
        self._episode_idx = -1
        self._step_idx = 0
        self._last_lang = ""
        self._language_override = None
        self._pending_language_override = None
        self._pending_reflection_row = None
        self._last_motion_latent = None
        self._motion_latent_history = []
        self._pred_future_history = []
        self._current_token_history = []
        self._trigger_steps_remaining = 0
        self._trigger_reason = ""
        self._repair_mode_next_episode = False
        self._repair_mode_active = False
        self._repair_mode_reason = ""
        self._step_rows = []
        self._episode_result: dict[str, Any] = {}
        self._episode_initial_log_path: Path | None = None
        if self.sidecar_log_dir:
            self.sidecar_log_dir.mkdir(parents=True, exist_ok=True)
            (self.sidecar_log_dir / "step_features").mkdir(parents=True, exist_ok=True)
            (self.sidecar_log_dir / "retry_reflections").mkdir(parents=True, exist_ok=True)

    def eval(self):
        self.bridge_agent.eval()
        if self.sidecar is not None:
            self.sidecar.eval()

    def reset(self):
        self._episode_idx += 1
        self._step_idx = 0
        self._last_motion_latent = None
        self._motion_latent_history = []
        self._pred_future_history = []
        self._current_token_history = []
        self._trigger_steps_remaining = 0
        self._trigger_reason = ""
        self._repair_mode_active = bool(self._repair_mode_next_episode)
        if self._repair_mode_active and not self._repair_mode_reason:
            self._repair_mode_reason = "forced_repair_retry"
        self._repair_mode_next_episode = False
        self._language_override = self._pending_language_override if self._repair_mode_active else None
        if self._repair_mode_active:
            self._pending_language_override = None
        self._step_rows = []
        self._episode_result = {}
        self._episode_initial_log_path = None
        self.bridge_agent.reset()
        if self.sidecar is not None and hasattr(self.sidecar, "reset"):
            self.sidecar.reset()

    def set_episode_result(
        self,
        *,
        task_name: str,
        lang_goal: str,
        baseline_success: bool,
        repair_success: bool | None,
        final_success: bool,
        baseline_reward: float,
        repair_reward: float | None,
        final_reward: float,
        baseline_length: int,
        repair_length: int | None,
        final_length: int,
    ) -> None:
        self._episode_result = {
            "task_name": str(task_name),
            "lang_goal": str(lang_goal),
            "baseline_success": bool(baseline_success),
            "repair_success": None if repair_success is None else bool(repair_success),
            "final_success": bool(final_success),
            "baseline_reward": float(baseline_reward),
            "repair_reward": None if repair_reward is None else float(repair_reward),
            "final_reward": float(final_reward),
            "baseline_length": int(baseline_length),
            "repair_length": None if repair_length is None else int(repair_length),
            "final_length": int(final_length),
        }
        if not self.sidecar_log_dir:
            return
        try:
            summary_path = self.sidecar_log_dir / "episode_summary_rows.jsonl"
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"episode": int(self._episode_idx), **self._episode_result}, ensure_ascii=False)
                    + "\n"
                )
        except Exception:
            pass
        self._update_episode_initial_log()

    def _update_episode_initial_log(self) -> None:
        if not self.sidecar_log_dir or not self._episode_result:
            return
        path = self._episode_initial_log_path
        if path is None:
            candidate = self.sidecar_log_dir / f"episode_{self._episode_idx:03d}_step_-01.json"
            if candidate.exists():
                path = candidate
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
            payload.update(self._episode_result)
            path.write_text(json.dumps(payload, ensure_ascii=True))
        except Exception:
            pass

    def _write_retry_reflection_row(self, row: dict[str, Any]):
        if not self.sidecar_log_dir:
            return
        try:
            path = self.sidecar_log_dir / "memory_rollout_rows.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _infer_last_task_name(self) -> str:
        if self._pending_reflection_row and self._pending_reflection_row.get("task"):
            return str(self._pending_reflection_row["task"])
        if self._step_rows:
            tail = self._step_rows[-1]
            if tail.get("task_hint"):
                return str(tail["task_hint"])
        return _infer_task_hint(self._last_lang, self.memory_bundle.task_names)

    def _build_retry_reflection(
        self,
        task_name: str,
        lang_goal: str,
        success: bool,
        steps: int,
        ep_len: int,
    ) -> tuple[str, dict[str, Any]]:
        task_hint = str(task_name or "").strip()
        if not task_hint or task_hint not in self.memory_bundle.task_names:
            task_hint = _infer_task_hint(task_name or lang_goal, self.memory_bundle.task_names)
        mismatch_type = "future_outcome_mismatch"
        if infer_mismatch_type is not None:
            try:
                mismatch_type = infer_mismatch_type(task_hint, bool(success), int(steps), int(ep_len))
            except Exception:
                mismatch_type = "future_outcome_mismatch"

        recent_summary = summarize_recent_memory(self._step_rows, 5) if self._step_rows else "none"
        key_rows = self.memory_bundle.key_rows or []
        key_rows = [row for row in key_rows if str(row.get("task")) == task_hint][: self.memory_bundle.memory_topk]
        key_summary = summarize_key_memory(key_rows)
        important_summary = summarize_important_key_memory(key_rows)
        rule_text = "Prefer conservative future calibration and preserve verified progress."
        if self.memory_bundle.rule_memory:
            rule_text = self.memory_bundle.rule_memory.get(task_hint, {}).get(
                "rule_text",
                self.memory_bundle.rule_memory.get("default", rule_text),
            )

        decision = template_postexec_reflection_decision(task_hint, bool(success), int(steps), int(ep_len), mismatch_type)
        if self.memory_bundle.qwen is not None:
            try:
                prompt = build_postexec_reflection_prompt(
                    task_hint,
                    None,
                    bool(success),
                    int(steps),
                    mismatch_type,
                    recent_summary,
                    key_summary,
                    important_summary,
                    rule_text,
                )
                raw_text = self.memory_bundle.qwen.generate_reflection(prompt, max_new_tokens=192)
                decision = parse_memory_reflection_output(raw_text)
            except Exception as exc:
                decision = template_postexec_reflection_decision(task_hint, bool(success), int(steps), int(ep_len), mismatch_type)
                decision.raw_text = (
                    f'{{"trust_score":{decision.trust_score},"mismatch_type":"{decision.mismatch_type}",'
                    f'"correction_direction":"{decision.correction_direction}",'
                    f'"recoverability":"{decision.recoverability}",'
                    f'"explanation":"qwen_fallback:{str(exc)[:120]}"}}'
                )

        reflection_row = make_reflection_row(
            seq_idx=self._episode_idx,
            sub_idx=-1,
            subtask=task_hint,
            success=bool(success),
            steps=int(steps),
            lang_text=lang_goal,
            decision=decision,
            mismatch_type=mismatch_type,
            recent_summary=recent_summary,
            key_summary=key_summary,
            important_summary=important_summary,
            rule_text=rule_text,
            baseline_success=bool(success),
            baseline_steps=int(steps),
        )
        retry_hint = str(reflection_row.get("correction_direction", "")).strip()
        retry_hint = retry_hint.replace("\n", " ").replace("  ", " ")
        base_lang = str(lang_goal or "").strip()
        if not base_lang or not _language_matches_task(base_lang, task_hint):
            print(
                json.dumps(
                    {
                        "repair_retry": True,
                        "task": task_hint,
                        "warning": "fallback_to_task_canonical_language",
                        "incoming_lang_goal": base_lang,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            base_lang = task_hint.replace("_", " ")
        if retry_hint:
            retry_lang_text = f"{base_lang}. Retry hint: {retry_hint}."
        else:
            retry_lang_text = f"{base_lang}. Retry once more with conservative motion."
        reflection_row["retry_lang_goal"] = retry_lang_text
        reflection_row["task_name"] = task_name
        reflection_row["task_hint"] = task_hint
        return retry_lang_text, reflection_row

    def prepare_repair_retry(
        self,
        reason: str = "forced_repair_retry",
        task_name: str | None = None,
        lang_goal: str | None = None,
        success: bool = False,
        steps: int = 0,
        ep_len: int = 0,
    ):
        self._repair_mode_next_episode = True
        self._repair_mode_reason = str(reason or "forced_repair_retry")
        if not self._calvin_reflection_mode:
            return
        task_name = str(task_name or self._infer_last_task_name())
        lang_goal = str(lang_goal or self._last_lang or task_name)
        if not _language_matches_task(lang_goal, task_name):
            lang_goal = task_name.replace("_", " ")
        try:
            retry_lang_text, reflection_row = self._build_retry_reflection(task_name, lang_goal, success, steps, ep_len)
            self._pending_language_override = retry_lang_text
            self._pending_reflection_row = reflection_row
            self._write_retry_reflection_row(reflection_row)
        except Exception as exc:
            self._pending_language_override = lang_goal
            self._pending_reflection_row = {
                "task": task_name,
                "lang_text": lang_goal,
                "retry_lang_goal": lang_goal,
                "reflection_error": str(exc)[:200],
            }
            self._write_retry_reflection_row(self._pending_reflection_row)

    def extract_pooled_feature(self, observation: dict, language_goal=None):
        if self._language_override is not None:
            language_goal = self._language_override
        elif language_goal is None:
            language_goal = observation.get("language_goal", None)
        return self.bridge_agent.extract_pooled_feature(observation, language_goal=language_goal)

    def extract_defi_future_feature(self, observation: dict | None = None, language_goal=None):
        if self._last_motion_latent is None:
            return None
        return self._last_motion_latent

    def get_defi_future_feature_sequence(self):
        if not self._motion_latent_history:
            return None
        try:
            return torch.stack([t.detach().cpu() for t in self._motion_latent_history], dim=0)
        except Exception:
            return None

    def _update_online_trigger(self, prev_transition_info: dict | None):
        if not self._online_failure_trigger:
            return
        info = prev_transition_info or {}
        error_type = str(info.get("error_type", "") or "")
        error_message = str(info.get("error_message", "") or "")
        failure_flag = bool(info.get("needs_reset", False))
        trigger_reason = ""
        if error_type in {"InvalidActionError", "IKError", "ConfigurationPathError"}:
            trigger_reason = error_type
        elif "outside of workspace" in error_message.lower():
            trigger_reason = "outside_workspace"
        elif "could not perform ik" in error_message.lower():
            trigger_reason = "ik_failed"
        elif "path could not be found" in error_message.lower():
            trigger_reason = "path_not_found"
        elif failure_flag:
            trigger_reason = "needs_reset"

        if trigger_reason:
            self._trigger_steps_remaining = max(self._trigger_steps_remaining, self._online_failure_trigger_steps)
            self._trigger_reason = trigger_reason

    def _current_gidm_gate(self) -> float:
        if self._repair_mode_active or self._trigger_steps_remaining > 0:
            return max(self._gidm_gate, self._online_failure_gate)
        return self._gidm_gate

    def _build_sidecar_obs(self, observation: dict):
        device = next(self.sidecar.parameters()).device
        rgb_obs = {
            "rgb_static": observation["front_rgb"].to(device).float(),
            "rgb_left_shoulder": observation["left_shoulder_rgb"].to(device).float(),
            "rgb_right_shoulder": observation["right_shoulder_rgb"].to(device).float(),
            "rgb_gripper": observation["wrist_rgb"].to(device).float(),
        }
        point_cloud_obs = {
            "pcd_front": observation["front_point_cloud"].to(device).float(),
            "pcd_left_shoulder": observation["left_shoulder_point_cloud"].to(device).float(),
            "pcd_right_shoulder": observation["right_shoulder_point_cloud"].to(device).float(),
            "pcd_wrist": observation["wrist_point_cloud"].to(device).float(),
        }

        grip_pose = observation["gripper_pose"].to(device).float()
        if grip_pose.ndim == 3:
            pose = grip_pose[:, -1]
        else:
            pose = grip_pose
        pos = pose[:, :3]
        quat = pose[:, 3:7]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        euler = torch.from_numpy(R.from_quat(quat.detach().cpu().numpy()).as_euler("xyz", degrees=False).astype("float32")).to(device)

        grip_open = _extract_scalar(observation, ["gripper_open", "low_dim_state", "robot_obs"], default=0.0).to(device)
        if grip_open.shape[0] != pos.shape[0]:
            grip_open = grip_open.expand(pos.shape[0], -1)
        if grip_open.shape[1] > 1:
            grip_open = grip_open[:, :1]

        ignore_collisions = _extract_scalar(observation, ["ignore_collisions", "low_dim_state", "robot_obs"], default=0.0).to(device)
        if ignore_collisions.shape[0] != pos.shape[0]:
            ignore_collisions = ignore_collisions.expand(pos.shape[0], -1)
        if ignore_collisions.shape[1] > 1:
            ignore_collisions = ignore_collisions[:, :1]
        robot_obs = torch.cat([pos, euler, grip_open, ignore_collisions], dim=-1)

        return {
            "rgb_obs": rgb_obs,
            "point_cloud_obs": point_cloud_obs,
            "robot_obs": robot_obs,
        }

    def _apply_memory_reflection(
        self,
        lang_text: str,
        current_tokens: torch.Tensor,
        pred_future: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bundle = self.memory_bundle
        if not bundle.enabled:
            return pred_future, {"memory_enabled": False}

        debug: dict[str, Any] = {"memory_enabled": True}
        task_hint = _infer_task_hint(lang_text, bundle.task_names)
        current_feat = _pool_last_dim(current_tokens.float())
        future_feat = _pool_last_dim(pred_future.float())
        if future_feat.ndim == 1:
            future_feat = future_feat.unsqueeze(0)
        future_np = future_feat.detach().cpu().numpy().astype(np.float32)
        current_np = current_feat.detach().cpu().numpy().astype(np.float32)

        memory_future = np.zeros_like(future_np[0] if future_np.ndim > 1 else future_np)
        memory_delta = np.zeros_like(memory_future)
        memory_row_ids = np.asarray([], dtype=np.int64)
        memory_scores = np.asarray([], dtype=np.float32)

        if bundle.memory_arrays:
            try:
                memory_future, memory_delta, memory_row_ids, memory_scores = retrieve_memory_topk(
                    task_hint,
                    future_np,
                    bundle.memory_arrays,
                    bundle.memory_topk,
                    exclude_row_id=None,
                )
            except Exception as exc:
                debug["memory_retrieve_error"] = str(exc)[:200]

        key_rows = []
        if bundle.key_rows and memory_row_ids.size > 0:
            chosen = {int(item) for item in memory_row_ids.tolist()}
            key_rows = [row for row in bundle.key_rows if int(row.get("row_id", -1)) in chosen]

        rule_text = "Prefer conservative future calibration and preserve verified progress."
        if bundle.rule_memory:
            rule_text = bundle.rule_memory.get(task_hint, {}).get("rule_text", bundle.rule_memory.get("default", rule_text))

        reflection_text = "memory_reflection_disabled"
        trust_score = 0.5
        corrected_future = pred_future

        if bundle.has_learned_modules() and memory_row_ids.size > 0:
            try:
                prompt = build_memory_reflection_prompt(
                    task_hint,
                    None,
                    f"lang={lang_text}; step={effective_step}",
                    summarize_key_memory(key_rows),
                    rule_text,
                )
                raw_text = bundle.qwen.generate_reflection(prompt, max_new_tokens=96)
                decision = parse_memory_reflection_output(raw_text)
                trust_score = float(decision.trust_score)
                reflection_text = decision.raw_text
                reflection_embedding = bundle.t5_embedder.encode(
                    [f"{decision.mismatch_type} | {decision.correction_direction} | {decision.recoverability} | {decision.explanation}"]
                ).squeeze(0)
                gate_input = build_gate_input(
                    current_np.reshape(-1),
                    future_np.reshape(-1),
                    memory_future.reshape(-1),
                    memory_delta.reshape(-1),
                    reflection_embedding.detach().cpu().numpy().astype(np.float32).reshape(-1),
                    memory_scores,
                )
                expected_gate_dim = _module_expected_input_dim(bundle.gate)
                expected_repair_dim = _module_expected_input_dim(bundle.repair)
                if expected_gate_dim is not None and gate_input.shape[-1] != expected_gate_dim:
                    debug["memory_mode"] = "heuristic_delta"
                    debug["memory_gate_input_mismatch"] = {
                        "actual": int(gate_input.shape[-1]),
                        "expected": int(expected_gate_dim),
                    }
                else:
                    with torch.no_grad():
                        gate_prob = torch.sigmoid(bundle.gate(torch.from_numpy(gate_input).unsqueeze(0).to(current_tokens.device))).item()
                    debug["gate_prob"] = float(gate_prob)
                    debug["trust_score"] = trust_score
                    debug["reflection_text"] = reflection_text[:1000]
                    if gate_prob >= bundle.gate_threshold and expected_repair_dim is not None:
                        repair_input = build_repair_input(
                            future_np.reshape(-1),
                            memory_future.reshape(-1),
                            memory_delta.reshape(-1),
                            reflection_embedding.detach().cpu().numpy().astype(np.float32).reshape(-1),
                        )
                        if repair_input.shape[-1] == expected_repair_dim:
                            with torch.no_grad():
                                delta = bundle.repair(torch.from_numpy(repair_input).unsqueeze(0).to(current_tokens.device)).squeeze(0)
                            delta = _ensure_future_shape(pred_future, delta)
                            corrected_future = pred_future + delta
                            debug["memory_mode"] = "learned_repair"
                        else:
                            debug["memory_mode"] = "heuristic_delta"
                            debug["memory_repair_input_mismatch"] = {
                                "actual": int(repair_input.shape[-1]),
                                "expected": int(expected_repair_dim),
                            }
                    else:
                        corrected_future = pred_future
                        debug["memory_mode"] = "learned_gate_skip"
            except Exception as exc:
                debug["memory_reflection_error"] = str(exc)[:200]

        if corrected_future is pred_future and bundle.memory_arrays and memory_row_ids.size > 0:
            memory_delta_t = torch.from_numpy(memory_delta).to(pred_future.device, dtype=pred_future.dtype)
            memory_delta_t = _project_to_reference_shape(pred_future, memory_delta_t)
            memory_score = float(memory_scores.mean()) if memory_scores.size > 0 else 0.0
            scale = float(bundle.fallback_scale) * (0.5 + 0.5 * max(0.0, memory_score))
            corrected_future = pred_future + memory_delta_t * scale
            debug["memory_mode"] = "heuristic_delta"
            debug["memory_score_mean"] = memory_score
            debug["memory_scale"] = scale

        debug["task_hint"] = task_hint
        debug["memory_row_ids"] = [int(x) for x in memory_row_ids.tolist()] if memory_row_ids.size > 0 else []
        debug["memory_scores"] = [float(x) for x in memory_scores.tolist()] if memory_scores.size > 0 else []
        return corrected_future, debug

    def _run_sidecar(self, step: int, observation: dict):
        if self.sidecar is None:
            return
        effective_step = int(step if step >= 0 else self._step_idx)
        lang_goal = observation.get("language_goal", [[[""]]])
        if isinstance(lang_goal, list):
            lang_text = lang_goal[0][0][0]
        else:
            lang_text = str(lang_goal)
        if self._language_override is not None:
            lang_text = self._language_override
        self._last_lang = lang_text
        obs = self._build_sidecar_obs(observation)
        with torch.no_grad():
            with self.sidecar._float32_context():
                token_out, latent_goal = self.sidecar._render_token_dict(
                    obs["rgb_obs"], obs["point_cloud_obs"], [lang_text]
                )
                current_tokens = token_out["head_tokens"].float()
                pred_future = self.sidecar.gfdm(current_tokens, latent_goal)
                if self._calvin_reflection_mode:
                    memory_debug = {
                        "memory_mode": "calvin_retry_only",
                        "memory_enabled": bool(self.memory_bundle.enabled),
                        "memory_row_ids": [],
                        "memory_scores": [],
                    }
                else:
                    pred_future, memory_debug = self._apply_memory_reflection(lang_text, current_tokens, pred_future.float())
                lam_out, motion_latent = self.sidecar._build_motion_latent(current_tokens, pred_future, [lang_text])
                if not torch.isfinite(motion_latent).all():
                    print(
                        f"[warn] non-finite motion_latent detected at step {effective_step}; "
                        "falling back to zeros to protect Bridge action head"
                    )
                    motion_latent = torch.zeros_like(motion_latent)
                self._last_motion_latent = motion_latent.detach().cpu()
                self._current_token_history.append(current_tokens.detach().cpu())
                self._pred_future_history.append(pred_future.detach().cpu())
                self._motion_latent_history.append(self._last_motion_latent)
                bridge_feature = {"state_images_for_videoformer": current_tokens}
                native_pred = self.sidecar.native_head(bridge_feature, latent_goal)
                trans_idx = native_pred["trans_logits"].argmax(dim=-1)
                rot_idx = native_pred["rot_logits"].argmax(dim=-1)
                grip_idx = native_pred["gripper_logits"].argmax(dim=-1)
                coll_idx = native_pred["collision_logits"].argmax(dim=-1)
                sidecar_target_pos = self.sidecar._decode_translation(trans_idx)
                sidecar_target_quat = self.sidecar._decode_quaternion(rot_idx.reshape(-1, 3)).reshape(
                    rot_idx.shape[0], rot_idx.shape[1], 4
                )
                sidecar_action = torch.cat(
                    [
                        sidecar_target_pos,
                        sidecar_target_quat,
                        torch.where(grip_idx > 0, torch.ones_like(grip_idx), -torch.ones_like(grip_idx)).float().unsqueeze(-1),
                        coll_idx.float().unsqueeze(-1),
                    ],
                    dim=-1,
                )

                gate_prob = None
                if memory_debug.get("gate_prob") is not None:
                    gate_prob = float(memory_debug["gate_prob"])

        if self.sidecar_log_dir:
            try:
                bridge_obs = {k: v for k, v in observation.items() if k != "prev_transition_info"}
                bridge_obs_feature = self.bridge_agent.extract_pooled_feature(bridge_obs, language_goal=lang_text)
                if isinstance(bridge_obs_feature, torch.Tensor):
                    bridge_obs_feature = bridge_obs_feature.detach().cpu().numpy()
                step_feature_path = self.sidecar_log_dir / "step_features" / f"episode_{self._episode_idx:03d}_step_{effective_step:03d}.npz"
                np.savez_compressed(
                    step_feature_path,
                    obs_current_feature=bridge_obs_feature,
                    defi_future_feature=self._last_motion_latent.detach().cpu().numpy() if self._last_motion_latent is not None else np.zeros((1, 768), dtype=np.float32),
                    current_tokens=current_tokens.detach().cpu().numpy(),
                    pred_future=pred_future.detach().cpu().numpy(),
                    motion_latent=self._last_motion_latent.detach().cpu().numpy() if self._last_motion_latent is not None else np.zeros((1, 768), dtype=np.float32),
                    gate_prob=np.asarray([gate_prob if gate_prob is not None else -1.0], dtype=np.float32),
                )
            except Exception as exc:
                step_feature_path = None
                memory_debug["step_feature_error"] = str(exc)[:200]

            payload = {
                "episode": self._episode_idx,
                "step": effective_step,
                "attempt_phase": "repair" if self._repair_mode_active else "baseline",
                "repair_mode_active": bool(self._repair_mode_active),
                "repair_mode_reason": self._repair_mode_reason,
                "lang_goal": lang_text,
                "motion_latent_norm": float(motion_latent.norm(dim=-1).mean().item()),
                "future_token_norm": float(pred_future.norm(dim=-1).mean().item()),
                "current_token_norm": float(current_tokens.norm(dim=-1).mean().item()),
                "online_trigger_steps_remaining": int(self._trigger_steps_remaining),
                "online_trigger_reason": self._trigger_reason,
                "gate_label_hint": int(self._trigger_steps_remaining > 0 or memory_debug.get("memory_mode") == "learned_repair"),
                "step_feature_path": str(step_feature_path) if step_feature_path is not None else "",
                "trans_idx": trans_idx[0].detach().cpu().tolist(),
                "rot_idx": rot_idx[0].detach().cpu().tolist(),
                "sidecar_action": sidecar_action[0].detach().cpu().tolist(),
                "vq_loss_proxy": float(((lam_out["emb"].detach() - lam_out["z"]) ** 2).mean().item()),
            }
            payload.update(memory_debug)
            self._step_rows.append(payload)
            out_path = self.sidecar_log_dir / f"episode_{self._episode_idx:03d}_step_{effective_step:03d}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=True))
            with (self.sidecar_log_dir / "sidecar_step_rows.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if effective_step == -1:
                self._episode_initial_log_path = out_path
                self._update_episode_initial_log()

    def act(self, step: int, observation: dict, **kwargs):
        prev_transition_info = observation.get("prev_transition_info", {})
        if isinstance(prev_transition_info, dict):
            self._update_online_trigger(prev_transition_info)
        else:
            self._update_online_trigger({})
        self._run_sidecar(step, observation)
        self._step_idx += 1
        effective_gidm_gate = self._current_gidm_gate()
        if self._trigger_steps_remaining > 0:
            self._trigger_steps_remaining -= 1
        effective_observation = observation
        if self._language_override is not None:
            effective_observation = dict(observation)
            effective_observation["language_goal"] = self._language_override
            effective_observation["lang_goal"] = self._language_override
        return self.bridge_agent.act(
            step,
            effective_observation,
            gidm_latent=self._last_motion_latent,
            gidm_gate=effective_gidm_gate,
            **kwargs,
        )


def load_sidecar(args):
    if not args.enable_sidecar:
        return None

    with initialize(config_path="../policy_conf", job_name="eval_bridge_with_defi_sidecar"):
        cfg = compose(config_name=args.defi_config_name)

    with open_dict(cfg):
        cfg.model.bridge_ckpt_path = args.bridge_model_path
        cfg.model.freeze_bridge_mvt = True
        if args.defi_future_mode:
            cfg.model.future_mode = args.defi_future_mode
        if args.defi_t5_model_path:
            cfg.model.t5_model_path = args.defi_t5_model_path
        if args.defi_language_goal_path:
            cfg.model.language_goal_path = args.defi_language_goal_path

    sidecar = hydra.utils.instantiate(cfg.model)
    if args.defi_ckpt:
        ckpt = torch.load(args.defi_ckpt, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
        state, skipped = filter_state_dict_by_shape(sidecar, state)
        if skipped:
            print(f"[sidecar] skipped {len(skipped)} incompatible checkpoint keys")
            for key, reason in skipped[:10]:
                print(f"[sidecar] skip key={key} reason={reason}")
        load_result = sidecar.load_state_dict(state, strict=False)
        print(f"[sidecar] loaded DeFi ckpt: {args.defi_ckpt}")
        if getattr(load_result, "missing_keys", None):
            print(f"[sidecar] missing keys: {len(load_result.missing_keys)}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"[sidecar] unexpected keys: {len(load_result.unexpected_keys)}")
    else:
        print("[sidecar] no DeFi ckpt provided; GFDM/GIDM/native head remain untrained. Bridge action is unaffected.")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    sidecar = sidecar.to(device)
    sidecar.process_device()
    sidecar.eval()
    return sidecar


def load_memory_bundle(args) -> MemoryReflectionBundle:
    bundle = MemoryReflectionBundle()
    memory_jsonl = _path_or_none(args.memory_jsonl)
    memory_npz = _path_or_none(args.memory_npz)
    rule_memory_json = _path_or_none(args.rule_memory_json)
    gate_ckpt = _path_or_none(args.gate_ckpt)
    repair_ckpt = _path_or_none(args.repair_ckpt)
    qwen_model_path = _path_or_none(args.qwen_model_path)
    qwen_lora_path = _path_or_none(args.qwen_lora_path)

    bundle.key_rows = _load_jsonl_rows(memory_jsonl)
    bundle.memory_arrays = load_memory_arrays(memory_npz) if memory_npz is not None and memory_npz.exists() else {}
    bundle.rule_memory = _load_rule_memory(rule_memory_json)
    bundle.task_names = tuple(args.tasks)
    bundle.memory_topk = int(args.memory_topk)
    bundle.gate_threshold = float(args.memory_gate_threshold)
    bundle.fallback_scale = float(args.memory_fallback_scale)
    bundle.enabled = bool(args.enable_memory_reflection)

    if not bundle.enabled:
        return bundle

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if qwen_model_path is not None and qwen_model_path.exists():
        try:
            bundle.qwen = QwenReflectionEncoder(
                qwen_model_path,
                device,
                lora_path=qwen_lora_path if qwen_lora_path is not None and qwen_lora_path.exists() else None,
            )
        except Exception as exc:
            print(f"[memory-reflection] failed to load Qwen model: {exc}")
    else:
        print("[memory-reflection] no qwen model path provided; reflection text will be heuristic only")

    if gate_ckpt is not None and gate_ckpt.exists():
        try:
            gate_payload = torch.load(gate_ckpt, map_location="cpu")
            gate = GateHead(int(gate_payload["input_dim"]), int(gate_payload["hidden_dim"]))
            gate.load_state_dict(gate_payload["model_state"])
            bundle.gate = gate.to(device).eval()
            t5_path = _path_or_none(gate_payload.get("t5_path", args.memory_t5_model_path))
            if t5_path.exists():
                bundle.t5_embedder = T5TextEmbedder(t5_path, device)
            bundle.gate_threshold = float(gate_payload.get("threshold", bundle.gate_threshold))
        except Exception as exc:
            print(f"[memory-reflection] failed to load gate ckpt: {exc}")

    if repair_ckpt is not None and repair_ckpt.exists():
        try:
            repair_payload = torch.load(repair_ckpt, map_location="cpu")
            repair = RepairAdapter(
                int(repair_payload["input_dim"]),
                int(repair_payload["future_dim"]),
                int(repair_payload["hidden_dim"]),
            )
            repair.load_state_dict(repair_payload["model_state"])
            bundle.repair = repair.to(device).eval()
        except Exception as exc:
            print(f"[memory-reflection] failed to load repair ckpt: {exc}")

    if bundle.qwen is None or bundle.gate is None or bundle.repair is None or bundle.t5_embedder is None:
        print("[memory-reflection] running in fallback mode; Bridge action still works, memory only nudges future features.")

    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-model-path", required=True)
    parser.add_argument("--exp-cfg-path", default=None)
    parser.add_argument("--mvt-cfg-path", default=None)
    parser.add_argument("--eval-datafolder", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--model-name", default="bridge_with_sidecar")
    parser.add_argument("--enable-sidecar", action="store_true")
    parser.add_argument("--defi-ckpt", default="")
    parser.add_argument("--defi-config-name", default="VPP_RLBench_gidm_mvt_train")
    parser.add_argument("--defi-future-mode", default="pred", choices=["pred", "gt", "none"])
    parser.add_argument("--defi-t5-model-path", default="/mnt/workspace/manipulation/DeFi/ckpts/t5_base")
    parser.add_argument("--defi-language-goal-path", default="/mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt")
    parser.add_argument("--enable-memory-reflection", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--memory-jsonl", default="")
    parser.add_argument("--memory-npz", default="")
    parser.add_argument("--rule-memory-json", default="")
    parser.add_argument("--gate-ckpt", default="")
    parser.add_argument("--repair-ckpt", default="")
    parser.add_argument("--qwen-model-path", default="")
    parser.add_argument(
        "--qwen-lora-path",
        default="/mnt/workspace/manipulation/BridgeVLA/eval/qwen_reflection_lora_rlbench_balanced_success_failure_preserve/qwen_reflection_lora_smoke.pt",
    )
    parser.add_argument("--memory-t5-model-path", default="/mnt/workspace/manipulation/DeFi/ckpts/t5_base")
    parser.add_argument("--memory-topk", type=int, default=3)
    parser.add_argument("--memory-gate-threshold", type=float, default=0.5)
    parser.add_argument("--memory-fallback-scale", type=float, default=0.1)
    parser.add_argument("--gidm-gate", type=float, default=0.1)
    parser.add_argument("--online-failure-trigger", action="store_true")
    parser.add_argument("--online-failure-trigger-steps", type=int, default=2)
    parser.add_argument("--online-failure-gidm-gate", type=float, default=1.0)
    parser.add_argument("--collect-repair-after-failure", action="store_true")
    parser.add_argument("--calvin-reflection-mode", action="store_true")
    parser.add_argument("--save-rollout-dir", default="")
    parser.add_argument("--sidecar-log-dir", default="")
    args = parser.parse_args()

    if not args.log_dir:
        args.log_dir = str(Path(args.bridge_model_path).resolve().parent / "eval" / args.model_name)
    if not args.save_rollout_dir:
        args.save_rollout_dir = str(Path(args.log_dir) / "rollouts")
    if not args.sidecar_log_dir and args.save_rollout_dir:
        args.sidecar_log_dir = str(Path(args.save_rollout_dir) / "sidecar_logs")

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    bridge_agent = load_bridge_agent(
        model_path=args.bridge_model_path,
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=log_dir,
        device=args.device,
    )
    if args.baseline_only:
        print("[baseline-only] running Bridge baseline only; sidecar, memory, reflection, and gidm_latent are disabled.")
        sidecar = None
        memory_bundle = MemoryReflectionBundle()
        agent = bridge_agent
    else:
        sidecar = load_sidecar(args)
        memory_bundle = load_memory_bundle(args)
        agent = BridgeDeFiSidecarAgent(
            bridge_agent,
            sidecar=sidecar,
            sidecar_log_dir=args.sidecar_log_dir,
            memory_bundle=memory_bundle,
            gidm_gate=args.gidm_gate,
            online_failure_trigger=args.online_failure_trigger,
            online_failure_trigger_steps=args.online_failure_trigger_steps,
            online_failure_gate=args.online_failure_gidm_gate,
            calvin_reflection_mode=args.calvin_reflection_mode,
        )

    scores = bridge_eval(
        agent=agent,
        tasks=args.tasks,
        eval_datafolder=args.eval_datafolder,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        device=args.device,
        headless=args.headless,
        logging=False,
        log_dir=log_dir,
        verbose=True,
        save_video=args.save_video,
        model_name=args.model_name,
        save_rollout_dir=args.save_rollout_dir,
        collect_repair_after_failure=False if args.baseline_only else args.collect_repair_after_failure,
    )
    print(f"[sidecar-eval] scores={scores}")


if __name__ == "__main__":
    main()
