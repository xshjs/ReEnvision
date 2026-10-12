#!/usr/bin/env python3
"""Convert DeFi normalized 7D EE relative actions into X5 joint deltas.

This is an offline post-processing utility for real-robot rollout integration.
It uses URDF forward kinematics plus a numerical Jacobian and damped least
 squares to map:

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

into:

    [dq1, dq2, dq3, dq4, dq5, dq6, gripper]

The first 6 output dims are joint deltas in radians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from policy_models.utils.x5_action_conversion import (
    convert_rel_action_to_x5_joint,
    load_action_array,
    load_action_array_json,
    load_joint_state_from_dataset,
    load_x5_fk_chain,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action_input", type=Path, default=None, help="DeFi action array (.npy/.npz/.json).")
    parser.add_argument("--action_json", type=str, default=None, help="Inline JSON array, e.g. '[0,0,0,0,0,0,-1]'.")
    parser.add_argument("--joint_state_input", type=Path, default=None, help="Current joint state array (.npy/.npz/.json).")
    parser.add_argument("--joint_state_json", type=str, default=None, help="Inline JSON joint state array.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Raw LeRobot dataset root for auto-loading joint state.")
    parser.add_argument("--episode_index", type=int, default=None, help="Used with --dataset_root.")
    parser.add_argument("--frame_index", type=int, default=None, help="Used with --dataset_root.")
    parser.add_argument("--urdf", type=Path, required=True, help="X5 URDF path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output .npy path.")
    parser.add_argument("--max_pos", type=float, default=0.05)
    parser.add_argument("--max_orn", type=float, default=0.50)
    parser.add_argument("--max_joint_delta", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--jac_eps", type=float, default=1e-4)
    parser.add_argument("--binary_gripper", action="store_true")
    parser.add_argument("--gripper_close_value", type=float, default=-1.0)
    parser.add_argument("--gripper_open_value", type=float, default=1.0)
    args = parser.parse_args()

    if args.action_input is not None:
        action = load_action_array(args.action_input)
    elif args.action_json is not None:
        action = load_action_array_json(args.action_json)
    else:
        raise ValueError("One of --action_input or --action_json is required")

    if args.joint_state_input is not None:
        joints = load_action_array(args.joint_state_input)
    elif args.joint_state_json is not None:
        joints = load_action_array_json(args.joint_state_json)
    elif args.dataset_root is not None and args.episode_index is not None and args.frame_index is not None:
        joints = load_joint_state_from_dataset(args.dataset_root, args.episode_index, args.frame_index)
    else:
        raise ValueError(
            "Provide joint state via --joint_state_input or --joint_state_json or "
            "(--dataset_root + --episode_index + --frame_index)"
        )

    chain = load_x5_fk_chain(args.urdf)
    converted = convert_rel_action_to_x5_joint(
        actions=action,
        joints=joints,
        fk_chain=chain,
        max_pos=args.max_pos,
        max_orn=args.max_orn,
        damping=args.damping,
        jac_eps=args.jac_eps,
        max_joint_delta=args.max_joint_delta,
        binary_gripper=args.binary_gripper,
        gripper_close_value=args.gripper_close_value,
        gripper_open_value=args.gripper_open_value,
    )

    summary = {
        "action_input": str(args.action_input) if args.action_input is not None else None,
        "action_json": bool(args.action_json is not None),
        "joint_state_input": str(args.joint_state_input) if args.joint_state_input is not None else None,
        "joint_state_json": bool(args.joint_state_json is not None),
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
        "episode_index": args.episode_index,
        "frame_index": args.frame_index,
        "urdf": str(args.urdf),
        "shape": list(converted.shape),
        "binary_gripper": bool(args.binary_gripper),
        "gripper_close_value": args.gripper_close_value,
        "gripper_open_value": args.gripper_open_value,
        "sample_first": converted.reshape(-1, 7)[0].round(6).tolist(),
        "joint_abs_max_rad": np.abs(converted[..., :6]).max(axis=tuple(range(converted.ndim - 1))).round(6).tolist(),
        "gripper_unique_rounded": np.unique(np.round(converted[..., 6].reshape(-1), 4)).tolist(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, converted)
        summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
