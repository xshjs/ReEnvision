import argparse
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

if "distutils.spawn" not in sys.modules:
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    spawn_module.spawn = lambda cmd, search_path=1, verbose=0, dry_run=0, env=None: subprocess.check_call(
        cmd,
        env=env,
    )
    sys.modules["distutils.spawn"] = spawn_module

try:
    import huggingface_hub as _hf_hub

    if not hasattr(_hf_hub, "cached_download") and hasattr(_hf_hub, "hf_hub_download"):
        def _cached_download(*args, **kwargs):
            return _hf_hub.hf_hub_download(*args, **kwargs)

        _hf_hub.cached_download = _cached_download
except Exception:
    pass

try:
    import transformers.utils as _tf_utils

    if not hasattr(_tf_utils, "FLAX_WEIGHTS_NAME"):
        _tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if hasattr(_tf_utils, "check_torch_load_is_safe"):
        _tf_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import hydra
import cv2
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from policy_evaluation.calvin_evaluate import count_success
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
from policy_models.module.diffusion_extract_outvideo import Diffusion_feature_extractor as GFDMVideoExtractor


CONTACT_RICH_TASKS = [
    "push_red_block_right",
    "push_blue_block_right",
    "push_pink_block_right",
    "push_into_drawer",
    "stack_block",
]

CONTROL_TASKS = [
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "open_drawer",
    "close_drawer",
]

BLOCK_BY_TASK = {
    "push_red_block_right": "block_red",
    "push_blue_block_right": "block_blue",
    "push_pink_block_right": "block_pink",
}

BLOCK_COLORS = ["red", "blue", "pink"]


def parse_task_list(value: str) -> List[str]:
    if value == "contact_rich":
        return CONTACT_RICH_TASKS
    if value == "controls":
        return CONTROL_TASKS
    if value == "all":
        return CONTACT_RICH_TASKS + CONTROL_TASKS
    return [item.strip() for item in value.split(",") if item.strip()]


def init_counts(tasks: Sequence[str]) -> Dict[str, Dict[str, int]]:
    return {task: {"success": 0, "failure": 0, "total": 0} for task in tasks}


def quota_reached(counts: Dict[str, Dict[str, int]], task: str, success: bool, max_per_task: int, max_success: Optional[int], max_failure: Optional[int]) -> bool:
    if max_success is not None or max_failure is not None:
        if success and max_success is not None:
            return counts[task]["success"] >= max_success
        if not success and max_failure is not None:
            return counts[task]["failure"] >= max_failure
    return counts[task]["total"] >= max_per_task


def all_quotas_reached(counts: Dict[str, Dict[str, int]], max_per_task: int, max_success: Optional[int], max_failure: Optional[int]) -> bool:
    for task, task_counts in counts.items():
        if max_success is not None or max_failure is not None:
            if max_success is not None and task_counts["success"] < max_success:
                return False
            if max_failure is not None and task_counts["failure"] < max_failure:
                return False
        elif task_counts["total"] < max_per_task:
            return False
    return True


def update_counts(counts: Dict[str, Dict[str, int]], task: str, success: bool) -> None:
    counts[task]["total"] += 1
    counts[task]["success" if success else "failure"] += 1


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def invert_body_links(info: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[Tuple[int, int], str]]:
    uid_to_name = {}
    link_to_name = {}
    scene_info = info.get("scene_info", {})
    for group in ["fixed_objects", "movable_objects", "doors", "buttons", "switches", "lights"]:
        for name, obj_info in scene_info.get(group, {}).items():
            uid = obj_info.get("uid")
            if uid is None:
                continue
            uid_to_name[int(uid)] = name
            link_to_name[(int(uid), -1)] = "base"
            for link_name, link_id in obj_info.get("links", {}).items():
                link_to_name[(int(uid), int(link_id))] = link_name
    robot_uid = info.get("robot_info", {}).get("uid")
    if robot_uid is not None:
        uid_to_name[int(robot_uid)] = "robot"
        link_to_name[(int(robot_uid), -1)] = "base"
    return uid_to_name, link_to_name


def pybullet_contact_to_dict(contact: Sequence[Any], uid_to_name: Dict[int, str], link_to_name: Dict[Tuple[int, int], str]) -> Dict[str, Any]:
    body_a = int(contact[1])
    body_b = int(contact[2])
    link_a = int(contact[3])
    link_b = int(contact[4])
    normal = np.asarray(contact[7], dtype=np.float64)
    normal_force = float(contact[9])
    return {
        "body_a": uid_to_name.get(body_a, str(body_a)),
        "link_a": link_to_name.get((body_a, link_a), str(link_a)),
        "body_b": uid_to_name.get(body_b, str(body_b)),
        "link_b": link_to_name.get((body_b, link_b), str(link_b)),
        "position": np.asarray(contact[5], dtype=np.float64).tolist(),
        "normal": normal.tolist(),
        "force_magnitude": normal_force,
        "force": (normal * normal_force).tolist(),
    }


def observed_movable_contacts(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    uid_to_name, link_to_name = invert_body_links(info)
    contacts = []
    for obj_name, obj_info in info.get("scene_info", {}).get("movable_objects", {}).items():
        seen = set()
        for contact in obj_info.get("contacts", []):
            item = pybullet_contact_to_dict(contact, uid_to_name, link_to_name)
            key = (item["body_a"], item["link_a"], item["body_b"], item["link_b"])
            if key in seen:
                continue
            seen.add(key)
            contacts.append(item)
    return contacts


def _is_block(name: str) -> bool:
    return name.startswith("block_")


def _contact_entity(body: str, link: str) -> str:
    if body == "robot" or _is_block(body):
        return body
    if body == "table":
        if "drawer" in link:
            return "drawer"
        return "table"
    if "drawer" in body or "drawer" in link:
        return "drawer"
    return body


def _contact_category(entity_a: str, entity_b: str) -> Optional[str]:
    entities = {entity_a, entity_b}
    if "robot" in entities and any(_is_block(entity) for entity in entities):
        return "robot_block"
    if "table" in entities and any(_is_block(entity) for entity in entities):
        return "block_table"
    if "drawer" in entities and any(_is_block(entity) for entity in entities):
        return "block_drawer"
    if _is_block(entity_a) and _is_block(entity_b):
        return "block_block"
    return None


def _canonical_contact_key(contact: Dict[str, Any]) -> Tuple[str, str, str, str]:
    side_a = (contact["body_a"], contact["link_a"])
    side_b = (contact["body_b"], contact["link_b"])
    return tuple(side_a + side_b) if side_a <= side_b else tuple(side_b + side_a)


def observed_contact_sets(info: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return contact_sets_from_contacts(observed_movable_contacts(info))


def contact_sets_from_contacts(contacts: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    contact_sets: Dict[str, List[Dict[str, Any]]] = {
        "robot_block": [],
        "block_table": [],
        "block_drawer": [],
        "block_block": [],
    }
    seen = {name: set() for name in contact_sets}
    for contact in contacts:
        entity_a = _contact_entity(contact["body_a"], contact["link_a"])
        entity_b = _contact_entity(contact["body_b"], contact["link_b"])
        category = _contact_category(entity_a, entity_b)
        if category is None:
            continue
        key = _canonical_contact_key(contact)
        if key in seen[category]:
            continue
        seen[category].add(key)
        item = dict(contact)
        item["entity_a"] = entity_a
        item["entity_b"] = entity_b
        item["pair"] = sorted([entity_a, entity_b])
        contact_sets[category].append(item)
    return contact_sets


def _target_block_from_task(task: str) -> Optional[str]:
    for color in ["red", "blue", "pink"]:
        if f"_{color}_block_" in task:
            return f"block_{color}"
    return BLOCK_BY_TASK.get(task)


def expected_object_motion_prediction(task: str, info: Dict[str, Any]) -> Dict[str, Any]:
    movable = info.get("scene_info", {}).get("movable_objects", {})
    all_blocks = {
        block: {
            "role": "non_target",
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "condition": "should remain near start pose unless it is the interacted object",
        }
        for block in movable
    }
    target_block = _target_block_from_task(task)

    if task.startswith("push_") and task.endswith("_right") and target_block in all_blocks:
        all_blocks[target_block].update(
            {
                "role": "target",
                "delta_x": 0.1,
                "condition": "target block delta_x > 0.1 and support contact preserved",
            }
        )
        return {"source": "task_expected_oracle_proxy", "target_blocks": [target_block], "all_blocks": all_blocks}

    if task.startswith("push_") and task.endswith("_left") and target_block in all_blocks:
        all_blocks[target_block].update(
            {
                "role": "target",
                "delta_x": -0.1,
                "condition": "target block delta_x < -0.1 and support contact preserved",
            }
        )
        return {"source": "task_expected_oracle_proxy", "target_blocks": [target_block], "all_blocks": all_blocks}

    if task == "push_into_drawer":
        for block in all_blocks:
            all_blocks[block].update(
                {
                    "role": "candidate_target",
                    "condition": "one block changes contact from table/base_link to table/drawer_link",
                }
            )
        return {"source": "task_expected_oracle_proxy", "target_blocks": list(all_blocks), "all_blocks": all_blocks}

    if task == "stack_block":
        for block in all_blocks:
            all_blocks[block].update(
                {
                    "role": "candidate_stack_block",
                    "condition": "new block-block contact and final contacts only movable blocks",
                }
            )
        return {"source": "task_expected_oracle_proxy", "target_blocks": list(all_blocks), "all_blocks": all_blocks}

    if task.startswith("lift_") and target_block in all_blocks:
        threshold = 0.03 if task.endswith("_slider") else 0.05
        all_blocks[target_block].update(
            {
                "role": "target",
                "delta_z": threshold,
                "condition": f"target block delta_z > {threshold} and end contact only with robot",
            }
        )
        return {"source": "task_expected_oracle_proxy", "target_blocks": [target_block], "all_blocks": all_blocks}

    return {"source": "task_expected_oracle_proxy", "target_blocks": [], "all_blocks": all_blocks}


def all_object_motion(start_info: Dict[str, Any], end_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    motions = {}
    start_objects = start_info.get("scene_info", {}).get("movable_objects", {})
    end_objects = end_info.get("scene_info", {}).get("movable_objects", {})
    for obj_name, start_obj in start_objects.items():
        end_obj = end_objects.get(obj_name)
        if end_obj is None:
            continue
        start_pos = np.asarray(start_obj.get("current_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        end_pos = np.asarray(end_obj.get("current_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        delta = end_pos - start_pos
        motions[obj_name] = {
            "start_pos": start_pos.tolist(),
            "end_pos": end_pos.tolist(),
            "delta": delta.tolist(),
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
            "distance": float(np.linalg.norm(delta)),
            "current_lin_vel": to_jsonable(end_obj.get("current_lin_vel", [])),
            "current_ang_vel": to_jsonable(end_obj.get("current_ang_vel", [])),
        }
    return motions


def subtask_diagnostics(task: str, start_info: Dict[str, Any], end_info: Dict[str, Any]) -> Dict[str, Any]:
    expected_contacts = expected_contact_prediction(task, start_info)
    return {
        "contact_sets": {
            "pred": contact_sets_from_contacts(expected_contacts),
            "start": observed_contact_sets(start_info),
            "end": observed_contact_sets(end_info),
        },
        "predicted_all_object_motion": expected_object_motion_prediction(task, start_info),
        "all_object_motion": all_object_motion(start_info, end_info),
    }


def first_non_robot_contact(obj_info: Dict[str, Any], info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    uid_to_name, link_to_name = invert_body_links(info)
    robot_uid = info.get("robot_info", {}).get("uid")
    for contact in obj_info.get("contacts", []):
        if robot_uid is not None and int(contact[2]) == int(robot_uid):
            continue
        return pybullet_contact_to_dict(contact, uid_to_name, link_to_name)
    return None


def expected_contact_prediction(task: str, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    scene_info = info.get("scene_info", {})
    movable = scene_info.get("movable_objects", {})
    if task in BLOCK_BY_TASK:
        block = BLOCK_BY_TASK[task]
        obj_info = movable.get(block)
        if obj_info is None:
            return []
        base_contact = first_non_robot_contact(obj_info, info)
        if base_contact is None:
            return []
        start_pos = np.asarray(obj_info.get("current_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        predicted = dict(base_contact)
        predicted["position"] = (start_pos + np.asarray([0.1, 0.0, 0.0])).tolist()
        predicted["force"] = [1.0, 0.0, 0.0]
        predicted["normal"] = [1.0, 0.0, 0.0]
        predicted["force_magnitude"] = 1.0
        return [predicted]

    if task == "push_into_drawer":
        for block, obj_info in movable.items():
            current_pos = np.asarray(obj_info.get("current_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
            if first_non_robot_contact(obj_info, info) is None:
                continue
            return [
                {
                    "body_a": block,
                    "link_a": "base",
                    "body_b": "table",
                    "link_b": "drawer_link",
                    "position": (current_pos + np.asarray([0.0, -0.08, -0.02])).tolist(),
                    "normal": [0.0, 0.0, 1.0],
                    "force": [0.0, 0.0, 1.0],
                    "force_magnitude": 1.0,
                }
            ]
        return []

    if task == "stack_block":
        blocks = list(movable.items())
        if len(blocks) < 2:
            return []
        positions = [(name, np.asarray(obj.get("current_pos", [0.0, 0.0, 0.0]), dtype=np.float64)) for name, obj in blocks]
        best_pair = None
        best_dist = float("inf")
        for i, (name_a, pos_a) in enumerate(positions):
            for name_b, pos_b in positions[i + 1 :]:
                dist = float(np.linalg.norm(pos_a[:2] - pos_b[:2]))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (name_a, name_b, (pos_a + pos_b) / 2.0 + np.asarray([0.0, 0.0, 0.04]))
        if best_pair is None:
            return []
        name_a, name_b, position = best_pair
        return [
            {
                "body_a": name_a,
                "link_a": "base",
                "body_b": name_b,
                "link_b": "base",
                "position": position.tolist(),
                "normal": [0.0, 0.0, 1.0],
                "force": [0.0, 0.0, 1.0],
                "force_magnitude": 1.0,
            }
        ]

    return observed_movable_contacts(info)


@torch.no_grad()
def visual_signature(model, obs: Dict[str, Any], lang_text: str, frame_idx: int) -> List[float]:
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    features = model.TVP_encoder(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    frame_idx = max(0, min(int(frame_idx), features.shape[1] - 1))
    selected = features[:, frame_idx]
    if selected.ndim == 4:
        selected = selected.mean(dim=(-1, -2))
    selected = selected.reshape(selected.shape[0], -1).float().mean(dim=0)
    return selected.detach().cpu().numpy().astype(np.float32).tolist()


def load_model(cfg: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
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
    model = model.to(device)
    model.process_device()
    model.eval()
    return model


def enable_gfdm_video_extractor(model: torch.nn.Module) -> None:
    if hasattr(model, "GFDM_video_extractor"):
        return
    base = model.TVP_encoder
    model.GFDM_video_extractor = GFDMVideoExtractor(
        pipeline=base.pipeline,
        tokenizer=base.tokenizer,
        text_encoder=base.text_encoder,
        position_encoding=base.position_encoding,
    ).to(model.device)
    model.GFDM_video_extractor.eval()


def _rgb_tensor_to_uint8(image: Any) -> Optional[np.ndarray]:
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        while image.ndim > 3:
            image = image[0]
        array = image.numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
    else:
        array = np.asarray(image)
    if array.size == 0:
        return None
    if array.dtype != np.uint8:
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] > 3:
        array = array[..., :3]
    return array


def _block_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    rgb = rgb.astype(np.uint8)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    if color == "red":
        mask = (red > 110) & (red > green * 1.25) & (red > blue * 1.25)
    elif color == "blue":
        mask = (blue > 90) & (blue > red * 1.15) & (blue > green * 1.15)
    else:
        mask = (red > 120) & (blue > 85) & (red > green * 1.2) & (blue > green * 1.05)
    mask = mask.astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


def _blob_features(rgb: np.ndarray, color: str) -> Dict[str, Any]:
    mask = _block_mask(rgb, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"visible": False, "area": 0.0}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 5.0:
        return {"visible": False, "area": area}
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        center = np.asarray(contour.reshape(-1, 2), dtype=np.float64).mean(axis=0)
    else:
        center = np.asarray([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float64)
    rect = cv2.minAreaRect(contour)
    return {
        "visible": True,
        "area": area,
        "center_px": center.tolist(),
        "angle_deg": float(rect[-1]),
    }


def video_outcome_from_frames(frames: Sequence[Any], task: str) -> Dict[str, Any]:
    if not frames:
        return {"source": "rgb_video_cv", "error": "no_frames"}
    first = _rgb_tensor_to_uint8(frames[0])
    last = _rgb_tensor_to_uint8(frames[-1])
    if first is None or last is None:
        return {"source": "rgb_video_cv", "error": "invalid_frames"}
    height, width = first.shape[:2]
    blocks = {}
    for color in BLOCK_COLORS:
        start = _blob_features(first, color)
        end = _blob_features(last, color)
        block_name = f"block_{color}"
        item = {"start": start, "end": end}
        if start.get("visible") and end.get("visible"):
            start_center = np.asarray(start["center_px"], dtype=np.float64)
            end_center = np.asarray(end["center_px"], dtype=np.float64)
            delta_px = end_center - start_center
            item.update(
                {
                    "delta_px": delta_px.tolist(),
                    "delta_x_px": float(delta_px[0]),
                    "delta_y_px": float(delta_px[1]),
                    "delta_x_norm": float(delta_px[0] / max(width, 1)),
                    "delta_y_norm": float(delta_px[1] / max(height, 1)),
                    "rotation_delta_deg_proxy": float(end.get("angle_deg", 0.0) - start.get("angle_deg", 0.0)),
                }
            )
        blocks[block_name] = item
    target_block = _target_block_from_task(task)
    stack_distance = None
    visible_centers = [np.asarray(item["end"]["center_px"], dtype=np.float64) for item in blocks.values() if item["end"].get("visible")]
    if len(visible_centers) >= 2:
        stack_distance = float(min(np.linalg.norm(a - b) for i, a in enumerate(visible_centers) for b in visible_centers[i + 1 :]))
    return {
        "source": "rgb_video_cv",
        "frame_count": len(frames),
        "frame_shape": [int(height), int(width), 3],
        "task": task,
        "target_block": target_block,
        "blocks": blocks,
        "target": blocks.get(target_block) if target_block else None,
        "stack_xy_distance_min_px": stack_distance,
        "notes": "CV proxy extracted from RGB video; pixel deltas are camera-space proxies, not metric simulator state.",
    }


def observed_video_outcome(start_obs: Dict[str, Any], end_obs: Dict[str, Any], task: str) -> Dict[str, Any]:
    return video_outcome_from_frames(
        [start_obs.get("rgb_obs", {}).get("rgb_static"), end_obs.get("rgb_obs", {}).get("rgb_static")],
        task,
    )


def gfdm_predicted_video_outcome(model: torch.nn.Module, obs: Dict[str, Any], lang_text: str, task: str) -> Dict[str, Any]:
    enable_gfdm_video_extractor(model)
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    _, videos = model.GFDM_video_extractor(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
        return_video=True,
        save_video=False,
    )
    static_video = videos[0] if videos else []
    return video_outcome_from_frames(static_video, task)


def rollout_subtask_with_recording(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    lang_embeddings: Any,
    val_annotations: Any,
    future_offset: int,
    record_stride: int,
    max_record_steps: int,
    skip_visual_signatures: bool,
    gfdm_video_outcome: bool,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    obs = env.get_obs()
    start_obs = obs
    final_obs = obs
    lang_text = val_annotations[subtask][0]
    goal = lang_embeddings.get_lang_goal(lang_text)
    goal["lang_text"] = lang_text
    model.reset()
    start_info = env.get_info()
    final_info = start_info
    task_steps: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    obs_history: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {0: (obs, start_info)}
    gfdm_outcome = None
    if gfdm_video_outcome and max_record_steps > 0:
        try:
            gfdm_outcome = gfdm_predicted_video_outcome(model, obs, lang_text, subtask)
        except Exception as exc:
            gfdm_outcome = {"source": "gfdm_future_video", "error": str(exc)}
            print(f"[WARN] GFDM predicted video outcome failed for {subtask}: {exc}")

    for step in range(cfg.ep_len):
        should_record = step % record_stride == 0 and len(task_steps) < max_record_steps
        if should_record:
            current_info = env.get_info()
            expected_contacts = expected_contact_prediction(subtask, current_info)
            if skip_visual_signatures:
                pred_visual = []
            else:
                try:
                    pred_visual = visual_signature(model, obs, lang_text, frame_idx=future_offset)
                except Exception as exc:
                    pred_visual = []
                    print(f"[WARN] visual prediction failed at step {step}: {exc}")
            pending.append(
                {
                    "target_t": step + future_offset,
                    "record": {
                        "t": step,
                        "visual": {"pred": pred_visual},
                        "contact": {"pred": expected_contacts},
                        "contact_sets": {
                            "pred": contact_sets_from_contacts(expected_contacts),
                            "current": observed_contact_sets(current_info),
                        },
                        "predicted_all_object_motion": expected_object_motion_prediction(subtask, current_info),
                        "all_object_motion": {"current": all_object_motion(start_info, current_info)},
                    },
                }
            )

        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        final_obs = obs
        final_info = current_info
        obs_history[step + 1] = (obs, current_info)

        ready = [item for item in pending if item["target_t"] <= step + 1]
        pending = [item for item in pending if item["target_t"] > step + 1]
        for item in ready:
            future_obs, future_info = obs_history[item["target_t"]]
            if skip_visual_signatures:
                item["record"]["visual"]["obs"] = []
            else:
                try:
                    item["record"]["visual"]["obs"] = visual_signature(model, future_obs, lang_text, frame_idx=0)
                except Exception as exc:
                    item["record"]["visual"]["obs"] = []
                    print(f"[WARN] visual observation encoding failed at step {step}: {exc}")
            item["record"]["contact"]["obs"] = observed_movable_contacts(future_info)
            item["record"]["contact_sets"]["obs"] = observed_contact_sets(future_info)
            item["record"]["all_object_motion"]["obs"] = all_object_motion(start_info, future_info)
            task_steps.append(to_jsonable(item["record"]))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            diagnostics = subtask_diagnostics(subtask, start_info, current_info)
            if gfdm_outcome is not None:
                diagnostics["gfdm_predicted_outcome"] = gfdm_outcome
                diagnostics["observed_video_outcome"] = observed_video_outcome(start_obs, final_obs, subtask)
            return True, task_steps, to_jsonable(diagnostics)

    diagnostics = subtask_diagnostics(subtask, start_info, final_info)
    if gfdm_outcome is not None:
        diagnostics["gfdm_predicted_outcome"] = gfdm_outcome
        diagnostics["observed_video_outcome"] = observed_video_outcome(start_obs, final_obs, subtask)
    return False, task_steps, to_jsonable(diagnostics)


def ensure_local_eval_dir(output_dir: Path, checkpoint: Path) -> Path:
    train_folder = output_dir / "eval_train_folder"
    saved_models = train_folder / "saved_models"
    saved_models.mkdir(parents=True, exist_ok=True)
    link_path = saved_models / checkpoint.name
    if not link_path.exists():
        try:
            link_path.symlink_to(checkpoint.resolve())
        except OSError:
            import shutil

            shutil.copy2(checkpoint, link_path)
    return train_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small CALVIN rollouts and save visual/contact future mismatch trajectories.")
    parser.add_argument("--config-name", default="calvin_evaluate_all.yaml")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--max-success-per-task", type=int, default=None)
    parser.add_argument("--max-failure-per-task", type=int, default=None)
    parser.add_argument("--num-sequences", type=int, default=80)
    parser.add_argument("--ep-len", type=int, default=240)
    parser.add_argument("--future-offset", type=int, default=5)
    parser.add_argument("--record-stride", type=int, default=10)
    parser.add_argument("--max-record-steps", type=int, default=24)
    parser.add_argument("--skip-visual-signatures", action="store_true")
    parser.add_argument("--gfdm-video-outcome", action="store_true")
    parser.add_argument("--future-feature-mode", choices=["gfdm", "current", "zero", "reflection"], default="gfdm")
    parser.add_argument("--future-frame-idx", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_task_list(args.tasks)
    counts = init_counts(tasks)

    with initialize(config_path="../policy_conf", job_name="failure_warning_rollout"):
        cfg = compose(config_name=args.config_name)
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.train_folder = str(ensure_local_eval_dir(args.output_dir, args.checkpoint))
    cfg.ep_len = args.ep_len
    cfg.num_sequences = args.num_sequences
    with open_dict(cfg):
        cfg.future_feature_mode = args.future_feature_mode
        cfg.future_frame_idx = args.future_frame_idx
    cfg.log_wandb = False

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
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
    model.future_feature_mode = cfg.future_feature_mode
    model.future_frame_idx = cfg.future_frame_idx
    print(f"future_feature_mode={model.future_feature_mode}, future_frame_idx={model.future_frame_idx}")
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    trajectories_path = args.output_dir / "trajectories.jsonl"
    metadata = {
        "tasks": tasks,
        "max_per_task": args.max_per_task,
        "max_success_per_task": args.max_success_per_task,
        "max_failure_per_task": args.max_failure_per_task,
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "future_offset": args.future_offset,
        "record_stride": args.record_stride,
        "max_record_steps": args.max_record_steps,
        "skip_visual_signatures": args.skip_visual_signatures,
        "gfdm_video_outcome": args.gfdm_video_outcome,
        "future_feature_mode": args.future_feature_mode,
        "future_frame_idx": args.future_frame_idx,
        "checkpoint": str(args.checkpoint),
        "contact_prediction_note": "contact.pred is a task-expected contact proxy because no CA-GFDM contact predictor/checkpoint exists in this workspace.",
        "diagnostic_logs": {
            "contact_sets": "Task-expected pred plus observed robot-block, block-table, block-drawer, and block-block contact sets at subtask start/end and sampled current/future steps.",
            "predicted_all_object_motion": "Task-expected oracle-proxy motion for every movable block; this is not a learned dynamics prediction.",
            "gfdm_predicted_outcome": "Outcome extracted from GFDM predicted future RGB video with a color-blob CV proxy.",
            "observed_video_outcome": "Outcome extracted from observed start/end RGB frames with the same color-blob CV proxy.",
            "all_object_motion": "Observed delta for every movable block from subtask start to subtask end and sampled current/future steps.",
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    results = []
    sequences = get_sequences(args.num_sequences)
    with trajectories_path.open("w") as handle:
        for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="rollout sequences")):
            if all_quotas_reached(counts, args.max_per_task, args.max_success_per_task, args.max_failure_per_task):
                break
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success_counter = 0
            for sub_idx, subtask in enumerate(eval_sequence):
                candidate_record = subtask in counts
                success, steps, diagnostics = rollout_subtask_with_recording(
                    env,
                    model,
                    task_oracle,
                    cfg,
                    subtask,
                    lang_embeddings,
                    val_annotations,
                    args.future_offset,
                    args.record_stride,
                    args.max_record_steps if candidate_record else 0,
                    args.skip_visual_signatures,
                    args.gfdm_video_outcome,
                )
                record = candidate_record and not quota_reached(
                    counts,
                    subtask,
                    success,
                    args.max_per_task,
                    args.max_success_per_task,
                    args.max_failure_per_task,
                )
                if record:
                    update_counts(counts, subtask, success)
                    row = {
                        "trajectory_id": f"seq{seq_idx:04d}_sub{sub_idx:02d}",
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": success,
                        "steps": steps,
                        **diagnostics,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"[REC] {row['trajectory_id']} {subtask} success={success} steps={len(steps)} counts={counts}")
                if success:
                    success_counter += 1
                else:
                    break
            results.append(success_counter)
            if results:
                rates = count_success(results)
                print(" ".join([f"{i + 1}/5={rate * 100:.1f}%" for i, rate in enumerate(rates)]))

    print(f"Wrote {trajectories_path}")
    print(f"Final counts: {counts}")


if __name__ == "__main__":
    os.environ.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")
    main()
