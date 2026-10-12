#!/usr/bin/env python3
"""Convert DeFi normalized 7D relative actions into X5 EE-delta commands.

Input convention:
- first 3 dims: normalized delta xyz in [-1, 1]
- next 3 dims: normalized delta euler xyz in [-1, 1]
- last dim: gripper command, typically signed open/close

Output convention:
- delta_xyz_m: meters
- delta_euler_xyz_rad: radians
- gripper: signed command, optionally binarized
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from policy_models.utils.x5_action_conversion import (
    convert_rel_action_to_x5_ee,
    load_action_array,
    load_action_array_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Path to .npy/.npz/.json action array.")
    parser.add_argument("--action_json", type=str, default=None, help="Inline JSON array, e.g. '[0,0,0,0,0,0,-1]'.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output .npy path.")
    parser.add_argument("--max_pos", type=float, default=0.05, help="Meters per normalized xyz unit.")
    parser.add_argument("--max_orn", type=float, default=0.50, help="Radians per normalized rot unit.")
    parser.add_argument("--binary_gripper", action="store_true", help="Force gripper to {-1, 1}.")
    parser.add_argument("--gripper_close_value", type=float, default=-1.0, help="Physical closed-gripper value used when binary_gripper is off.")
    parser.add_argument("--gripper_open_value", type=float, default=1.0, help="Physical open-gripper value used when binary_gripper is off.")
    args = parser.parse_args()

    if args.input is not None:
        actions = load_action_array(args.input)
    elif args.action_json is not None:
        actions = load_action_array_json(args.action_json)
    else:
        raise ValueError("One of --input or --action_json is required")
    converted = convert_rel_action_to_x5_ee(
        actions,
        args.max_pos,
        args.max_orn,
        args.binary_gripper,
        args.gripper_close_value,
        args.gripper_open_value,
    )

    summary = {
        "input": str(args.input) if args.input is not None else None,
        "action_json": bool(args.action_json is not None),
        "shape": list(converted.shape),
        "max_pos": args.max_pos,
        "max_orn": args.max_orn,
        "binary_gripper": bool(args.binary_gripper),
        "gripper_close_value": args.gripper_close_value,
        "gripper_open_value": args.gripper_open_value,
        "sample_first": converted.reshape(-1, 7)[0].round(6).tolist(),
        "xyz_abs_max_m": np.abs(converted[..., :3]).max(axis=tuple(range(converted.ndim - 1))).round(6).tolist(),
        "rot_abs_max_rad": np.abs(converted[..., 3:6]).max(axis=tuple(range(converted.ndim - 1))).round(6).tolist(),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, converted)
        summary["output"] = str(args.output)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
