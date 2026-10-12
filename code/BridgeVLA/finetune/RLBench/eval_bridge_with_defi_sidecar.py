import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_LIBS = ROOT / "bridgevla" / "libs"
for path in [
    ROOT,
    BRIDGE_LIBS / "PyRep",
    BRIDGE_LIBS / "RLBench",
    BRIDGE_LIBS / "YARR",
    BRIDGE_LIBS / "peract_colab",
    BRIDGE_LIBS / "point-renderer",
]:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFI_ROOT_DEFAULT = Path("/mnt/workspace/manipulation/DeFi/Step3_DeFI")
if DEFI_ROOT_DEFAULT.is_dir() and str(DEFI_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(DEFI_ROOT_DEFAULT))


def _force_local_pretrained_loads() -> None:
    try:
        import transformers

        patched = getattr(transformers, "_codex_local_only_patch", False)
        if not patched:
            original_auto = transformers.AutoTokenizer.from_pretrained

            def _auto_tokenizer_local(*args, **kwargs):
                kwargs.setdefault("local_files_only", True)
                return original_auto(*args, **kwargs)

            transformers.AutoTokenizer.from_pretrained = _auto_tokenizer_local

            for cls_name in [
                "T5Tokenizer",
                "T5EncoderModel",
                "CLIPTextModelWithProjection",
                "CLIPVisionModelWithProjection",
                "CLIPProcessor",
                "PaliGemmaProcessor",
                "PaliGemmaForConditionalGeneration",
            ]:
                cls = getattr(transformers, cls_name, None)
                if cls is None or not hasattr(cls, "from_pretrained"):
                    continue
                original = cls.from_pretrained

                def _wrap_local(orig):
                    def _local_only(*args, **kwargs):
                        kwargs.setdefault("local_files_only", True)
                        return orig(*args, **kwargs)

                    return _local_only

                cls.from_pretrained = _wrap_local(original)

            transformers._codex_local_only_patch = True
    except Exception:
        pass

    try:
        import diffusers

        patched = getattr(diffusers, "_codex_local_only_patch", False)
        if not patched:
            for cls_name in ["StableVideoDiffusionPipeline"]:
                cls = getattr(diffusers, cls_name, None)
                if cls is None or not hasattr(cls, "from_pretrained"):
                    continue
                original = cls.from_pretrained

                def _wrap_local(orig):
                    def _local_only(*args, **kwargs):
                        kwargs.setdefault("local_files_only", True)
                        return orig(*args, **kwargs)

                    return _local_only

                cls.from_pretrained = _wrap_local(original)

            diffusers._codex_local_only_patch = True
    except Exception:
        pass


_force_local_pretrained_loads()

from eval import eval as bridge_eval  # type: ignore
from eval import load_agent as load_bridge_agent  # type: ignore
from policy_models.RLBench_gidm_mvt_policy import RLBenchGIDMMVTPolicy

try:
    from policy_evaluation.oracle_hypothesis_rollout import (
        QwenReflectionEncoder,
        build_augmented_future_instruction,
        build_postexec_reflection_prompt,
        infer_mismatch_type as defi_infer_mismatch_type,
        load_key_memory_rows,
        parse_memory_reflection_output,
        retrieve_key_memory,
        summarize_important_key_memory,
        summarize_key_diagnostics,
        summarize_key_memory,
        summarize_recent_memory,
        template_postexec_reflection_decision,
    )
except Exception:  # pragma: no cover - keep bridge-only mode working
    QwenReflectionEncoder = None
    build_augmented_future_instruction = None
    build_postexec_reflection_prompt = None
    defi_infer_mismatch_type = None
    load_key_memory_rows = None
    parse_memory_reflection_output = None
    retrieve_key_memory = None
    summarize_important_key_memory = None
    summarize_key_diagnostics = None
    summarize_key_memory = None
    summarize_recent_memory = None
    template_postexec_reflection_decision = None


def _quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    quat_np = quat.detach().cpu().numpy()
    euler_np = R.from_quat(quat_np).as_euler("xyz", degrees=False).astype("float32")
    return torch.from_numpy(euler_np).to(quat.device)


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


class BridgeWithGidmSidecarAgent:
    def __init__(
        self,
        bridge_agent,
        sidecar=None,
        use_sidecar_action: bool = True,
        shadow_sidecar: bool = False,
        repair_reflection: Optional[Any] = None,
        key_memory_rows: Optional[list[dict[str, Any]]] = None,
        key_topk: int = 3,
        recent_k: int = 4,
        repair_max_new_tokens: int = 192,
        bridge_feature_gidm: bool = False,
        pooled_feature_gidm: bool = False,
        bridge_gfdm_gidm: bool = False,
    ):
        self.bridge_agent = bridge_agent
        self.sidecar = sidecar
        self.use_sidecar_action = use_sidecar_action and sidecar is not None
        self.shadow_sidecar = shadow_sidecar and sidecar is not None
        self.repair_reflection = repair_reflection
        self.key_memory_rows = key_memory_rows or []
        self.key_topk = int(key_topk)
        self.recent_k = int(recent_k)
        self.repair_max_new_tokens = int(repair_max_new_tokens)
        self.bridge_feature_gidm = bool(bridge_feature_gidm)
        self.pooled_feature_gidm = bool(pooled_feature_gidm)
        self.bridge_gfdm_gidm = bool(bridge_gfdm_gidm)
        self._repair_mode_active = False
        self._repair_mode_reason = ""
        self._language_override: Optional[str] = None
        self._repair_context: dict[str, Any] = {}
        self._recent_memory: list[dict[str, Any]] = []

    def eval(self):
        self.bridge_agent.eval()
        if self.sidecar is not None:
            self.sidecar.eval()

    def reset(self):
        self.bridge_agent.reset()
        if self.sidecar is not None and hasattr(self.sidecar, "reset"):
            self.sidecar.reset()
        self._repair_context = {}
        self._recent_memory = []
        if not self._repair_mode_active:
            self._language_override = None
            self._repair_mode_reason = ""

    def finish_repair_retry(self):
        self._repair_mode_active = False
        self._repair_mode_reason = ""
        self._repair_context = {}
        self._language_override = None
        self._recent_memory = []

    def extract_pooled_feature(self, observation: dict, language_goal=None):
        return self.bridge_agent.extract_pooled_feature(observation, language_goal=language_goal)

    def _extract_lang_text(self, observation: dict) -> str:
        lang_goal = observation.get("language_goal")
        if isinstance(lang_goal, list):
            try:
                return str(lang_goal[0][0][0])
            except Exception:
                return str(lang_goal)
        if lang_goal is not None:
            return str(lang_goal)
        return str(observation.get("lang_goal", observation.get("lang", "")))

    def _build_sidecar_obs(self, observation: dict):
        device = next(self.sidecar.parameters()).device

        def _to_tensor(x):
            if torch.is_tensor(x):
                return x.to(device)
            return torch.as_tensor(x, device=device)

        def _norm_rgb(x):
            x = x.to(device).float()
            if x.max() > 1.5:
                x = x / 255.0
            return x * 2.0 - 1.0

        rgb_obs = {
            "rgb_static": _norm_rgb(_to_tensor(observation["front_rgb"])),
            "rgb_left_shoulder": _norm_rgb(_to_tensor(observation["left_shoulder_rgb"])),
            "rgb_right_shoulder": _norm_rgb(_to_tensor(observation["right_shoulder_rgb"])),
            "rgb_gripper": _norm_rgb(_to_tensor(observation["wrist_rgb"])),
        }
        point_cloud_obs = {
            "pcd_front": _to_tensor(observation["front_point_cloud"]).float(),
            "pcd_left_shoulder": _to_tensor(observation["left_shoulder_point_cloud"]).float(),
            "pcd_right_shoulder": _to_tensor(observation["right_shoulder_point_cloud"]).float(),
            "pcd_wrist": _to_tensor(observation["wrist_point_cloud"]).float(),
        }

        low_dim = _to_tensor(observation["low_dim_state"]).float()
        if low_dim.ndim == 1:
            low_dim = low_dim.unsqueeze(0)
        low_dim = low_dim.reshape(-1, low_dim.shape[-1])

        gripper_open = low_dim[:, 0:1]
        if "gripper_pose" in observation and observation["gripper_pose"] is not None:
            gripper_pose = _to_tensor(observation["gripper_pose"]).float()
            if gripper_pose.ndim == 1:
                gripper_pose = gripper_pose.unsqueeze(0)
            if gripper_pose.ndim > 2:
                gripper_pose = gripper_pose.reshape(-1, gripper_pose.shape[-1])
            if gripper_pose.numel() >= 7 and gripper_pose.shape[-1] >= 7 and gripper_pose.shape[0] > 0:
                gripper_pose = gripper_pose[:, :7]
                pos = gripper_pose[:, :3]
                quat = gripper_pose[:, 3:7]
            else:
                gripper_pose = None
        else:
            gripper_pose = None

        if gripper_pose is None:
            if low_dim.shape[-1] >= 22:
                pose_start = 15
                pose_end = 22
                pose = low_dim[:, pose_start:pose_end]
                if pose.shape[-1] == 7 and pose.shape[0] > 0:
                    pos = pose[:, :3]
                    quat = pose[:, 3:7]
                else:
                    pos = torch.zeros((low_dim.shape[0], 3), device=device, dtype=torch.float32)
                    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32).repeat(low_dim.shape[0], 1)
            else:
                pos = torch.zeros((low_dim.shape[0], 3), device=device, dtype=torch.float32)
                quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32).repeat(low_dim.shape[0], 1)
        else:
            pass
        euler = _quat_to_euler(quat)
        ignore_collisions = _to_tensor(observation["ignore_collisions"]).float().reshape(-1, 1)
        ignore_collisions = ignore_collisions[: pos.shape[0]]
        if ignore_collisions.shape[0] != pos.shape[0]:
            ignore_collisions = torch.zeros((pos.shape[0], 1), device=device, dtype=torch.float32)
        robot_obs = torch.cat([pos, euler, gripper_open, ignore_collisions], dim=-1)

        if getattr(self, "_debug_step_state", False):
            print(
                "[sidecar-debug] obs shapes:",
                {k: tuple(v.shape) for k, v in observation.items() if hasattr(v, "shape")},
                "-> pos",
                tuple(pos.shape),
                "quat",
                tuple(quat.shape),
                "robot_obs",
                tuple(robot_obs.shape),
            )

        return {
            "rgb_obs": rgb_obs,
            "point_cloud_obs": point_cloud_obs,
            "robot_obs": robot_obs,
        }

    def _sidecar_action(self, observation: dict):
        if self.sidecar is None:
            return None
        lang_text = self._extract_lang_text(observation)
        obs = self._build_sidecar_obs(observation)
        with torch.no_grad():
            action = self.sidecar.eval_forward(obs, {"lang_text": lang_text})
        return action

    def _sidecar_latent(self, observation: dict):
        if self.sidecar is None:
            return None
        lang_text = self._extract_lang_text(observation)
        obs = self._build_sidecar_obs(observation)
        with torch.no_grad():
            token_out, latent_goal = self.sidecar._render_token_dict(obs["rgb_obs"], obs["point_cloud_obs"], [lang_text])
            current_tokens = token_out["head_tokens"]
            pred_future = self.sidecar.gfdm(current_tokens, latent_goal)
            lam_out, motion_latent = self.sidecar._build_motion_latent(current_tokens, pred_future, [lang_text])
        if getattr(self, "_debug_step_state", False):
            print(
                "[sidecar-latent] current_tokens",
                tuple(current_tokens.shape),
                "pred_future",
                tuple(pred_future.shape),
                "motion_latent",
                tuple(motion_latent.shape),
                "lam_z",
                tuple(lam_out["z"].shape),
            )
        return {
            "current_tokens": current_tokens,
            "pred_future": pred_future,
            "motion_latent": motion_latent,
            "lam_out": lam_out,
        }

    def _bridge_feature_gidm_latent(self, observation: dict):
        lang_text = self._extract_lang_text(observation)
        try:
            pooled = self.bridge_agent.extract_pooled_feature(observation, language_goal=lang_text)
        except Exception as exc:
            print(f"[bridge-feature-gidm] pooled feature extraction failed: {exc}")
            return None
        if pooled is None:
            return None
        if not torch.is_tensor(pooled):
            pooled = torch.as_tensor(pooled)
        pooled = pooled.float()
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        pooled = pooled.reshape(pooled.shape[0], -1)
        if pooled.shape[1] == 0:
            return None
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.tanh(pooled)
        target_dim = 768
        if pooled.shape[1] > target_dim:
            pooled = pooled[:, :target_dim]
        elif pooled.shape[1] < target_dim:
            repeat = (target_dim + pooled.shape[1] - 1) // pooled.shape[1]
            pooled = pooled.repeat(1, repeat)[:, :target_dim]
        try:
            device = next(self.bridge_agent._network.parameters()).device
        except Exception:
            device = torch.device("cpu")
        return pooled.to(device=device, dtype=torch.float32)

    def _task_key_memory(self, task_name: str) -> list[dict[str, Any]]:
        if not self.key_memory_rows:
            return []
        if retrieve_key_memory is not None:
            try:
                rows = retrieve_key_memory(self.key_memory_rows, task_name, self.key_topk)
                if rows:
                    return list(rows)
            except Exception:
                pass
        exact = [row for row in self.key_memory_rows if str(row.get("task")) == str(task_name)]
        if exact:
            return exact[: self.key_topk]
        return self.key_memory_rows[: self.key_topk]

    def prepare_repair_retry(
        self,
        reason: str,
        task_name: Optional[str] = None,
        lang_goal: Optional[str] = None,
        success: bool = False,
        steps: int = 0,
        ep_len: int = 25,
        next_task: Optional[str] = None,
    ):
        self._repair_mode_active = True
        self._repair_mode_reason = str(reason)
        self._repair_context = {
            "reason": reason,
            "task_name": task_name,
            "lang_goal": lang_goal,
            "success": bool(success),
            "steps": int(steps),
            "ep_len": int(ep_len),
            "next_task": next_task,
        }
        self._language_override = None
        if self.repair_reflection is None or build_postexec_reflection_prompt is None or parse_memory_reflection_output is None:
            return

        task = str(task_name or "unknown_task")
        base_lang = str(lang_goal or task.replace("_", " "))
        mismatch_type = "future_outcome_mismatch"
        if defi_infer_mismatch_type is not None:
            try:
                mismatch_type = defi_infer_mismatch_type(task, bool(success), int(steps), int(ep_len))
            except Exception:
                pass
        elif template_postexec_reflection_decision is not None:
            try:
                mismatch_type = template_postexec_reflection_decision(
                    task, bool(success), int(steps), int(ep_len), "future_outcome_mismatch"
                ).mismatch_type
            except Exception:
                pass

        recent_summary = summarize_recent_memory(self._recent_memory, self.recent_k) if summarize_recent_memory is not None else "none"
        key_rows = self._task_key_memory(task)
        key_summary = summarize_key_memory(key_rows) if summarize_key_memory is not None else "none"
        important_summary = summarize_important_key_memory(key_rows) if summarize_important_key_memory is not None else key_summary
        diagnostic_summary = summarize_key_diagnostics(key_rows) if summarize_key_diagnostics is not None else "none"
        rule_text = "Prefer conservative future calibration and preserve verified progress."
        prompt = build_postexec_reflection_prompt(
            task=task,
            next_task=next_task,
            success=bool(success),
            steps=int(steps),
            mismatch_type=str(mismatch_type),
            recent_summary=f"{recent_summary} ; diagnostics={diagnostic_summary}",
            key_summary=key_summary,
            important_summary=important_summary,
            rule_text=rule_text,
        )
        try:
            raw = self.repair_reflection.generate_reflection(prompt, max_new_tokens=self.repair_max_new_tokens)
            decision = parse_memory_reflection_output(raw)
            reflection_row = {
                "trust_score": float(decision.trust_score),
                "trust": "high" if decision.trust_score >= 0.67 else "medium" if decision.trust_score >= 0.34 else "low",
                "mismatch_type": decision.mismatch_type if decision.mismatch_type != "none" else str(mismatch_type),
                "correction_direction": decision.correction_direction,
                "recoverability": getattr(decision, "recoverability", "uncertain"),
                "reflection_text": getattr(decision, "raw_text", raw),
            }
            if build_augmented_future_instruction is not None:
                candidate_override = build_augmented_future_instruction(base_lang, reflection_row, important_summary)
                if _language_matches_task(candidate_override, task):
                    self._language_override = candidate_override
                else:
                    self._language_override = (
                        f"{base_lang}. Hint: trust={reflection_row['trust']}; "
                        f"m={reflection_row['mismatch_type']}; fix={reflection_row['correction_direction']}; "
                        f"r={reflection_row['recoverability']}."
                    )
                    print(
                        json.dumps(
                            {
                                "repair_retry": True,
                                "task": task,
                                "reason": reason,
                                "warning": "rejected_language_override_task_mismatch",
                                "candidate_override": candidate_override,
                                "fallback_override": self._language_override,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            else:
                self._language_override = (
                    f"{base_lang}. Hint: trust={reflection_row['trust']}; "
                    f"m={reflection_row['mismatch_type']}; fix={reflection_row['correction_direction']}."
                )
            if hasattr(self.bridge_agent, "_language_override"):
                self.bridge_agent._language_override = self._language_override
            print(
                json.dumps(
                    {
                        "repair_retry": True,
                        "task": task,
                        "reason": reason,
                        "reflection": reflection_row,
                        "augmented_lang_goal": self._language_override,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(f"[repair] reflection generation failed for task={task}: {exc}", flush=True)
            self._language_override = base_lang
        return self._language_override

    def act(self, step: int, observation: dict, **kwargs):
        gidm_latent = None
        if self.sidecar is not None and not self.use_sidecar_action:
            try:
                sidecar_out = self._sidecar_latent(observation)
                gidm_latent = sidecar_out["motion_latent"]
            except Exception as exc:
                print(f"[sidecar-latent] latent pass failed at step {step}: {exc}")
        elif self.bridge_feature_gidm:
            gidm_latent = self._bridge_feature_gidm_latent(observation)
        adjusted_observation = dict(observation)
        if self._language_override:
            adjusted_observation["language_goal"] = [[[self._language_override]]]
            adjusted_observation["lang_goal"] = self._language_override
            adjusted_observation["lang"] = self._language_override
        bridge_result = self.bridge_agent.act(
            step,
            adjusted_observation,
            gidm_latent=gidm_latent,
            gidm_gate=getattr(self, "_gidm_gate", 1.0),
            gidm_from_pooled=self.pooled_feature_gidm and gidm_latent is None,
            gidm_from_bridge_gfdm_gidm=self.bridge_gfdm_gidm and gidm_latent is None,
            **kwargs,
        )
        if self.use_sidecar_action:
            try:
                sidecar_action = self._sidecar_action(adjusted_observation)
                if sidecar_action is not None:
                    bridge_result.action = (
                        sidecar_action[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
                    )
            except Exception as exc:
                print(f"[sidecar] action replacement failed at step {step}: {exc}")
        return bridge_result


def _load_sidecar(args):
    if not args.enable_sidecar:
        return None

    defi_root = Path(args.defi_root)
    if str(defi_root) not in sys.path:
        sys.path.insert(0, str(defi_root))

    with initialize_config_dir(version_base=None, config_dir=str(defi_root / "policy_conf")):
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
    if args.defi_ckpt and Path(args.defi_ckpt).is_file():
        ckpt = torch.load(args.defi_ckpt, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
        model_state = sidecar.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in state.items():
            if key not in model_state:
                skipped.append((key, "missing_in_model"))
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                skipped.append((key, f"{tuple(value.shape)} != {tuple(model_state[key].shape)}"))
                continue
            filtered_state[key] = value
        state = filtered_state
        load_result = sidecar.load_state_dict(state, strict=False)
        print(f"[sidecar] loaded DeFi ckpt: {args.defi_ckpt}")
        if skipped:
            print(f"[sidecar] skipped incompatible keys: {len(skipped)}")
            for key, reason in skipped[:10]:
                print(f"[sidecar] skip key={key} reason={reason}")
        if getattr(load_result, "missing_keys", None):
            print(f"[sidecar] missing keys: {len(load_result.missing_keys)}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"[sidecar] unexpected keys: {len(load_result.unexpected_keys)}")
    else:
        if args.defi_ckpt:
            print(f"[sidecar] skip missing DeFi ckpt: {args.defi_ckpt}")
        else:
            print("[sidecar] no DeFi ckpt provided; sidecar is untrained.")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    sidecar = sidecar.to(device)
    sidecar.process_device()
    sidecar.eval()
    return sidecar


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
    parser.add_argument("--save-rollout-dir", default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--model-name", default="bridge_with_sidecar")
    parser.add_argument("--enable-sidecar", action="store_true")
    parser.add_argument("--action-source", choices=["sidecar", "bridge"], default="sidecar")
    parser.add_argument("--defi-root", default=str(DEFI_ROOT_DEFAULT))
    parser.add_argument("--defi-ckpt", default="")
    parser.add_argument("--defi-config-name", default="VPP_RLBench_gidm_mvt_train")
    parser.add_argument("--defi-future-mode", default="pred", choices=["pred", "gt", "none"])
    parser.add_argument("--defi-t5-model-path", default="/mnt/workspace/manipulation/DeFi/ckpts/t5_base")
    parser.add_argument("--defi-language-goal-path", default="/mnt/workspace/manipulation/DeFi/ckpts/ViT-B-32.pt")
    parser.add_argument("--log-sidecar-choices", action="store_true")
    parser.add_argument("--debug-step-state", action="store_true")
    parser.add_argument("--shadow-sidecar", action="store_true")
    parser.add_argument("--bridge-feature-gidm", action="store_true")
    parser.add_argument("--pooled-feature-gidm", action="store_true")
    parser.add_argument("--bridge-gfdm-gidm", action="store_true")
    parser.add_argument("--bridge-latent-head-ckpt", default="")
    parser.add_argument("--bridge-latent-dim", type=int, default=768)
    parser.add_argument("--bridge-latent-hidden-dim", type=int, default=1024)
    parser.add_argument("--bridge-standalone-gidm-ckpt", default="")
    parser.add_argument("--bridge-standalone-gidm-t5-path", default="/mnt/workspace/manipulation/DeFi/ckpts/t5_base")
    parser.add_argument("--joint-pair-memory-npz", default="")
    parser.add_argument("--joint-energy-rerank-ckpt", default="")
    parser.add_argument("--joint-pair-repair-ckpt", default="")
    parser.add_argument("--joint-pair-acceptance-ckpt", default="")
    parser.add_argument("--joint-pair-topk", type=int, default=8)
    parser.add_argument("--joint-pair-same-task-only", action="store_true")
    parser.add_argument("--joint-pair-success-only-memory", action="store_true")
    parser.add_argument("--joint-pair-min-similarity", type=float, default=0.20)
    parser.add_argument("--joint-pair-min-energy-improve", type=float, default=0.05)
    parser.add_argument("--joint-pair-max-future-shift", type=float, default=1.50)
    parser.add_argument("--joint-pair-max-motion-shift", type=float, default=1.50)
    parser.add_argument("--joint-pair-future-alpha", type=float, default=0.35)
    parser.add_argument("--joint-pair-motion-alpha", type=float, default=0.50)
    parser.add_argument("--joint-pair-verify-exp-max", type=float, default=1e9)
    parser.add_argument("--joint-pair-verify-latent-max", type=float, default=1e9)
    parser.add_argument("--joint-pair-verify-physical-max", type=float, default=1e9)
    parser.add_argument("--joint-pair-acceptance-threshold", type=float, default=0.5)
    parser.add_argument("--debug-gfdm-gidm", action="store_true")
    parser.add_argument("--gidm-gate", type=float, default=1.0)
    parser.add_argument("--collect-repair-after-failure", action="store_true")
    parser.add_argument("--qwen-model-path", default="")
    parser.add_argument("--qwen-lora-path", default="")
    parser.add_argument("--qwen-python-bin", default="")
    parser.add_argument("--memory-jsonl", default="")
    parser.add_argument("--key-topk", type=int, default=3)
    parser.add_argument("--recent-k", type=int, default=4)
    parser.add_argument("--repair-max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    if args.joint_pair_memory_npz:
        args.bridge_gfdm_gidm = True
        if not args.bridge_latent_head_ckpt:
            raise ValueError("--bridge-latent-head-ckpt is required when --joint-pair-memory-npz is set")
        print("[info] joint pair mode enabled; forcing --bridge-gfdm-gidm")

    if args.shadow_sidecar:
        print("[warn] --shadow-sidecar is deprecated; sidecar gidm injection is now the default when --enable-sidecar --action-source bridge.")
    if args.bridge_feature_gidm:
        print("[warn] --bridge-feature-gidm triggers an extra Bridge render path and is not recommended for RLBench stability.")
    if args.pooled_feature_gidm:
        print("[info] using Bridge pooled feature from the same forward pass to synthesize gidm_latent.")
    if args.bridge_gfdm_gidm:
        print("[info] using Bridge single-forward GFDM->GIDM proxy to synthesize z_f, z_a, and gidm_latent.")
    if args.bridge_standalone_gidm_ckpt:
        print(f"[info] using standalone GIDM/LAM for z_a: {args.bridge_standalone_gidm_ckpt}")
    if args.joint_pair_memory_npz:
        print(f"[info] using joint pair memory: {args.joint_pair_memory_npz}")

    effective_log_dir = args.log_dir or os.path.join(Path(args.bridge_model_path).parent, "eval", args.model_name)
    Path(effective_log_dir).mkdir(parents=True, exist_ok=True)
    (Path(effective_log_dir) / "run_config.json").write_text(
        json.dumps(
            {
                "date": "2026-08-14",
                "bridge_gfdm_gidm": bool(args.bridge_gfdm_gidm),
                "bridge_latent_head_ckpt": args.bridge_latent_head_ckpt,
                "bridge_standalone_gidm_ckpt": args.bridge_standalone_gidm_ckpt,
                "joint_pair_memory_npz": args.joint_pair_memory_npz,
                "joint_energy_rerank_ckpt": args.joint_energy_rerank_ckpt,
                "joint_pair_repair_ckpt": args.joint_pair_repair_ckpt,
                "joint_pair_acceptance_ckpt": args.joint_pair_acceptance_ckpt,
                "joint_pair_topk": int(args.joint_pair_topk),
                "joint_pair_same_task_only": bool(args.joint_pair_same_task_only),
                "joint_pair_success_only_memory": bool(args.joint_pair_success_only_memory),
                "joint_pair_min_similarity": float(args.joint_pair_min_similarity),
                "joint_pair_min_energy_improve": float(args.joint_pair_min_energy_improve),
                "joint_pair_max_future_shift": float(args.joint_pair_max_future_shift),
                "joint_pair_max_motion_shift": float(args.joint_pair_max_motion_shift),
                "joint_pair_future_alpha": float(args.joint_pair_future_alpha),
                "joint_pair_motion_alpha": float(args.joint_pair_motion_alpha),
                "joint_pair_verify_exp_max": float(args.joint_pair_verify_exp_max),
                "joint_pair_verify_latent_max": float(args.joint_pair_verify_latent_max),
                "joint_pair_verify_physical_max": float(args.joint_pair_verify_physical_max),
                "joint_pair_acceptance_threshold": float(args.joint_pair_acceptance_threshold),
                "gidm_gate": float(args.gidm_gate),
                "tasks": list(args.tasks),
                "eval_episodes": int(args.eval_episodes),
                "episode_length": int(args.episode_length),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[info] wrote run config: {Path(effective_log_dir) / 'run_config.json'}")

    bridge_agent = load_bridge_agent(
        model_path=args.bridge_model_path,
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=effective_log_dir,
        device=args.device,
        use_input_place_with_mean=False,
    )
    if args.bridge_latent_head_ckpt:
        bridge_agent.init_bridge_latent_heads(
            latent_dim=args.bridge_latent_dim,
            hidden_dim=args.bridge_latent_hidden_dim,
            ckpt_path=args.bridge_latent_head_ckpt,
        )
    if args.bridge_standalone_gidm_ckpt:
        bridge_agent.init_bridge_standalone_gidm(
            ckpt_path=args.bridge_standalone_gidm_ckpt,
            t5_model_path=args.bridge_standalone_gidm_t5_path,
        )
    if args.joint_pair_memory_npz:
        bridge_agent.init_joint_pair_selector(
            memory_npz=args.joint_pair_memory_npz,
            energy_ckpt=args.joint_energy_rerank_ckpt,
            repair_ckpt=args.joint_pair_repair_ckpt,
            acceptance_ckpt=args.joint_pair_acceptance_ckpt,
            topk=args.joint_pair_topk,
            same_task_only=args.joint_pair_same_task_only,
            success_only_memory=args.joint_pair_success_only_memory,
            min_similarity=args.joint_pair_min_similarity,
            min_energy_improve=args.joint_pair_min_energy_improve,
            max_future_shift=args.joint_pair_max_future_shift,
            max_motion_shift=args.joint_pair_max_motion_shift,
            future_refine_alpha=args.joint_pair_future_alpha,
            motion_refine_alpha=args.joint_pair_motion_alpha,
            verify_exp_max=args.joint_pair_verify_exp_max,
            verify_latent_max=args.joint_pair_verify_latent_max,
            verify_physical_max=args.joint_pair_verify_physical_max,
            acceptance_threshold=args.joint_pair_acceptance_threshold,
        )

    sidecar = _load_sidecar(args)
    repair_reflection = None
    key_memory_rows: list[dict[str, Any]] = []
    if args.collect_repair_after_failure:
        if QwenReflectionEncoder is None:
            raise RuntimeError("Qwen reflection helpers are unavailable in this environment")
        if not args.qwen_model_path:
            raise ValueError("--qwen-model-path is required when --collect-repair-after-failure is set")
        qwen_device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        qwen_kwargs = {}
        if args.qwen_lora_path:
            qwen_kwargs["lora_path"] = Path(args.qwen_lora_path)
        if args.qwen_python_bin:
            qwen_kwargs["python_bin"] = Path(args.qwen_python_bin)
        repair_reflection = QwenReflectionEncoder(Path(args.qwen_model_path), qwen_device, **qwen_kwargs)
        if args.memory_jsonl and load_key_memory_rows is not None:
            key_memory_rows = load_key_memory_rows(Path(args.memory_jsonl))
    agent = BridgeWithGidmSidecarAgent(
        bridge_agent=bridge_agent,
        sidecar=sidecar,
        use_sidecar_action=args.action_source == "sidecar",
        shadow_sidecar=args.shadow_sidecar,
        repair_reflection=repair_reflection,
        key_memory_rows=key_memory_rows,
        key_topk=args.key_topk,
        recent_k=args.recent_k,
        repair_max_new_tokens=args.repair_max_new_tokens,
        bridge_feature_gidm=args.bridge_feature_gidm,
        pooled_feature_gidm=args.pooled_feature_gidm,
        bridge_gfdm_gidm=args.bridge_gfdm_gidm,
    )
    agent._debug_step_state = args.debug_step_state
    agent._gidm_gate = args.gidm_gate
    bridge_agent._debug_gfdm_gidm = bool(args.debug_gfdm_gidm)

    scores = bridge_eval(
        agent=agent,
        tasks=args.tasks,
        eval_datafolder=args.eval_datafolder,
        start_episode=0,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        replay_ground_truth=False,
        collect_repair_after_failure=args.collect_repair_after_failure,
        device=args.device,
        headless=args.headless,
        logging=False,
        log_dir=effective_log_dir,
        verbose=True,
        save_video=args.save_video,
        save_rollout_dir=args.save_rollout_dir,
        model_name=args.model_name,
        visualize=False,
        visualize_root_dir="",
    )
    print(scores)


if __name__ == "__main__":
    main()
