from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.agents.agent import ActResult

from bridgevla.libs.peract.helpers import utils
from bridgevla.utils.rvt_utils import RLBENCH_TASKS
from eval import load_agent
from utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv
from utils.peract_utils_rlbench import CAMERAS, IMAGE_SIZE
from utils.rlbench_planning import EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning


TASKS_18 = [
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]


def _tensor_obs(obs: dict, device: str) -> dict:
    data = {}
    for k, v in obs.items():
        if k in {"language_goal_text", "prev_transition_info"}:
            continue
        arr = np.asarray(v)
        if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
            continue
        # Match YARR RolloutGenerator with timesteps=1:
        # obs_history[k] is [obs[k]], then np.array([obs_history[k]]).
        data[k] = torch.tensor(np.array([[v]]), device=device)
    data["language_goal"] = [[[obs.get("language_goal_text", "unknown goal")]]]
    data["prev_transition_info"] = obs.get("prev_transition_info", {})
    return data


def _copy_persistent(agent) -> tuple:
    fut = getattr(agent, "_suffix_td_persistent_future", None)
    mot = getattr(agent, "_suffix_td_persistent_motion", None)
    return (
        None if fut is None else fut.clone(),
        None if mot is None else mot.clone(),
    )


def _restore_persistent(agent, state: tuple) -> None:
    agent._suffix_td_persistent_future = None if state[0] is None else state[0].clone()
    agent._suffix_td_persistent_motion = None if state[1] is None else state[1].clone()


def _snapshot_env(env) -> dict:
    scene = env._task._scene
    return {
        "task_state": env._task._task.get_state(),
        "arm_state": scene.robot.arm.get_configuration_tree(),
        "gripper_state": scene.robot.gripper.get_configuration_tree(),
        "i": int(getattr(env, "_i", 0)),
        "previous_obs_dict": deepcopy(getattr(env, "_previous_obs_dict", None)),
        "recorded_images": list(getattr(env, "_recorded_images", [])),
    }


def _restore_env(env, snap: dict, obs: dict) -> None:
    scene = env._task._scene
    scene.pyrep.set_configuration_tree(snap["arm_state"])
    scene.pyrep.set_configuration_tree(snap["gripper_state"])
    env._task._task.restore_state(snap["task_state"])
    env._i = int(snap["i"])
    env._previous_obs_dict = deepcopy(obs)
    if hasattr(env, "_recorded_images"):
        env._recorded_images = list(snap.get("recorded_images", []))


def _make_env(task_name: str, eval_datafolder: str, episode_length: int, headless: bool):
    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]
    if task_name not in task_files:
        raise ValueError(f"Task {task_name} not recognised.")
    obs_config = utils.create_obs_config(CAMERAS, [IMAGE_SIZE, IMAGE_SIZE], method_name="")
    action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
    env = CustomMultiTaskRLBenchEnv(
        task_classes=[task_file_to_task_class(task_name)],
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=10**9,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=-1,
    )
    env.eval = True
    env.launch()
    return env


def _step_agent(agent, obs: dict, prev_info: dict, lang_goal: str, device: str, apply_enabled: bool):
    if hasattr(agent, "set_suffix_td_apply_enabled"):
        agent.set_suffix_td_apply_enabled(apply_enabled)
    obs_for_agent = dict(obs)
    obs_for_agent["language_goal_text"] = lang_goal
    obs_for_agent["prev_transition_info"] = prev_info
    return agent.act(0, _tensor_obs(obs_for_agent, device))


def _continue_from_action(
    env,
    agent,
    snap: dict,
    obs: dict,
    agent_state: tuple,
    first_action: np.ndarray,
    first_info: dict,
    lang_goal: str,
    device: str,
    episode_length: int,
    start_step: int,
    apply_enabled: bool,
) -> dict:
    _restore_env(env, snap, obs)
    _restore_persistent(agent, agent_state)
    if hasattr(agent, "set_suffix_td_apply_enabled"):
        agent.set_suffix_td_apply_enabled(apply_enabled)

    reward = 0.0
    terminal = False
    steps = 0
    prev_info = dict(first_info)
    transition = env.step(ActResult(np.asarray(first_action, dtype=np.float32), info=prev_info))
    reward = float(transition.reward)
    terminal = bool(transition.terminal)
    steps += 1
    obs_cur = dict(transition.observation)
    prev_info = dict(transition.info)

    for step in range(start_step + 1, episode_length):
        if terminal or prev_info.get("needs_reset", False):
            break
        act_result = _step_agent(agent, obs_cur, prev_info, lang_goal, device, apply_enabled)
        transition = env.step(act_result)
        reward = float(transition.reward)
        terminal = bool(transition.terminal)
        steps += 1
        obs_cur = dict(transition.observation)
        prev_info = dict(transition.info)
        if terminal or prev_info.get("needs_reset", False):
            break
    return {
        "success": bool(reward > 99),
        "reward": reward,
        "steps_after_snapshot": int(steps),
        "terminal": bool(terminal),
    }


def _select_trigger(info: dict, low_prob: float, high_prob: float, max_delta: float, require_requested: bool) -> bool:
    if not info.get("suffix_td_active"):
        return False
    prob = float(info.get("suffix_td_patch_prob", -1.0))
    delta = float(info.get("suffix_td_delta_norm", 1e9))
    if prob < low_prob or prob > high_prob or delta > max_delta:
        return False
    if require_requested and not info.get("suffix_td_requested"):
        return False
    return True


def run_episode(
    env,
    agent,
    task_name: str,
    ep: int,
    device: str,
    episode_length: int,
    low_prob: float,
    high_prob: float,
    max_delta: float,
    require_requested: bool,
) -> dict | None:
    obs = env.reset_to_demo(ep)
    lang_goal = env._lang_goal
    agent.reset()
    if hasattr(agent, "set_suffix_td_apply_enabled"):
        agent.set_suffix_td_apply_enabled(True)
    prev_info = {}

    for step in range(episode_length):
        snap = _snapshot_env(env)
        act_result = _step_agent(agent, obs, prev_info, lang_goal, device, apply_enabled=True)
        info = dict(getattr(act_result, "info", {}) or {})
        agent_state_after_act = _copy_persistent(agent)
        original_action = np.asarray(info.get("original_action", act_result.action), dtype=np.float32)
        patched_action = np.asarray(act_result.action, dtype=np.float32)
        if _select_trigger(info, low_prob, high_prob, max_delta, require_requested):
            keep = _continue_from_action(
                env,
                agent,
                snap,
                obs,
                agent_state_after_act,
                original_action,
                {**info, "action": original_action.copy(), "suffix_td_applied": False, "suffix_td_requested": False},
                lang_goal,
                device,
                episode_length,
                step,
                apply_enabled=False,
            )
            replan_act = _step_agent(agent, obs, {}, lang_goal, device, apply_enabled=False)
            replan = _continue_from_action(
                env,
                agent,
                snap,
                obs,
                (None, None),
                np.asarray(replan_act.action, dtype=np.float32),
                dict(getattr(replan_act, "info", {}) or {}),
                lang_goal,
                device,
                episode_length,
                step,
                apply_enabled=False,
            )
            patch = _continue_from_action(
                env,
                agent,
                snap,
                obs,
                agent_state_after_act,
                patched_action,
                info,
                lang_goal,
                device,
                episode_length,
                step,
                apply_enabled=False,
            )
            return {
                "task": task_name,
                "episode": int(ep),
                "step": int(step),
                "lang_goal": lang_goal,
                "patch_prob": float(info.get("suffix_td_patch_prob", 0.0)),
                "replan_prob": float(info.get("suffix_td_replan_prob", 0.0)),
                "delta_norm": float(info.get("suffix_td_delta_norm", 0.0)),
                "trigger_reason": str(info.get("suffix_td_trigger_reason", "none")),
                "action_patch_l2": float(np.linalg.norm(patched_action - original_action)),
                "keep": keep,
                "replan": replan,
                "patch": patch,
            }

        transition = env.step(act_result)
        obs = dict(transition.observation)
        prev_info = dict(transition.info)
        if transition.terminal or prev_info.get("needs_reset", False):
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay keep/replan/patch from the same moderate-deviation RLBench snapshot.")
    parser.add_argument("--model-folder", default="/mnt/workspace/manipulation/BridgeVLA/checkpoints/RLBench")
    parser.add_argument("--model-name", default="model_80.pth")
    parser.add_argument("--eval-datafolder", default="/mnt/workspace/manipulation/BridgeVLA/finetune/rlbench_eval_all18")
    parser.add_argument("--tasks", nargs="+", default=TASKS_18)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--low-prob", type=float, default=0.01)
    parser.add_argument("--high-prob", type=float, default=0.20)
    parser.add_argument("--max-delta", type=float, default=0.30)
    parser.add_argument("--require-requested", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    if args.tasks == ["all"]:
        args.tasks = RLBENCH_TASKS

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(args.model_folder, args.model_name)
    agent = load_agent(model_path=model_path, eval_log_dir=str(args.output_jsonl.parent), device=args.device)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for task_name in args.tasks:
        env = _make_env(task_name, args.eval_datafolder, args.episode_length, headless=True)
        try:
            for ep in range(args.start_episode, args.start_episode + args.episodes_per_task):
                print(f"[snapshot-replay] task={task_name} ep={ep}", flush=True)
                row = run_episode(
                    env,
                    agent,
                    task_name,
                    ep,
                    device,
                    args.episode_length,
                    args.low_prob,
                    args.high_prob,
                    args.max_delta,
                    args.require_requested,
                )
                if row is None:
                    continue
                results.append(row)
                with args.output_jsonl.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    "[snapshot-replay][case] "
                    f"task={task_name} ep={ep} step={row['step']} prob={row['patch_prob']:.4f} "
                    f"keep={row['keep']['success']} replan={row['replan']['success']} patch={row['patch']['success']}",
                    flush=True,
                )
        finally:
            env.shutdown()

    if results:
        for name in ["keep", "replan", "patch"]:
            vals = [float(r[name]["success"]) for r in results]
            print(f"[summary] {name}: n={len(vals)} success={float(np.mean(vals)):.3f}", flush=True)
        patch_l2 = [float(r["action_patch_l2"]) for r in results]
        print(f"[summary] action_patch_l2_mean={float(np.mean(patch_l2)):.6f}", flush=True)
    else:
        print("[summary] no moderate snapshots found", flush=True)


if __name__ == "__main__":
    main()
