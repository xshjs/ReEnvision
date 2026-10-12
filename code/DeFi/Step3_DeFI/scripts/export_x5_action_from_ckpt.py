#!/usr/bin/env python3
"""Export DeFi real-robot action predictions from a checkpoint.

This script runs `eval_forward()` on one sample from a converted DeFi dataset and
optionally exports:

- raw normalized action chunk
- EE delta chunk
- joint delta chunk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize
import hydra
from omegaconf import open_dict

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_models.utils.x5_action_conversion import (
    convert_rel_action_to_x5_ee,
    convert_rel_action_to_x5_joint,
    load_joint_state_from_dataset,
    load_x5_fk_chain,
    _pose_vec,
)


def _log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[EXPORT_X5][{now}] {message}", flush=True)


def _load_checkpoint_weights(path: str):
    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            return state_dict["state_dict"], state_dict
        if "model" in state_dict:
            return state_dict["model"], state_dict
    return state_dict, state_dict


def _take_sample(loader, sample_index: int):
    seen = 0
    for batch in loader:
        batch_size = batch["actions"].shape[0]
        if seen + batch_size <= sample_index:
            seen += batch_size
            continue
        local_idx = sample_index - seen
        sample = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                sample[key] = {k: v[local_idx : local_idx + 1] for k, v in value.items()}
            elif torch.is_tensor(value):
                sample[key] = value[local_idx : local_idx + 1]
            elif isinstance(value, list):
                sample[key] = [value[local_idx]]
            else:
                sample[key] = value
        return sample
    raise IndexError(f"sample_index={sample_index} out of range")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root_data_dir", required=True)
    parser.add_argument("--config_name", default="VPP_Calvinabc_train")
    parser.add_argument("--video_model_path", required=True)
    parser.add_argument("--text_encoder_path", required=True)
    parser.add_argument("--t5_model_path", required=True)
    parser.add_argument("--language_goal_path", required=True)
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--export_mode", choices=["raw", "ee", "joint", "all"], default="all")
    parser.add_argument("--max_pos", type=float, default=0.05)
    parser.add_argument("--max_orn", type=float, default=0.50)
    parser.add_argument("--max_joint_delta", type=float, default=0.05)
    parser.add_argument("--binary_gripper", action="store_true")
    parser.add_argument("--gripper_close_value", type=float, default=-1.0)
    parser.add_argument("--gripper_open_value", type=float, default=1.0)
    parser.add_argument("--joint_state_input", default="")
    parser.add_argument("--raw_dataset_root", default="")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--urdf", default="")
    parser.add_argument(
        "--action_format",
        choices=["rel_ee", "joint_absolute"],
        default="rel_ee",
        help="rel_ee: DeFi relative EE action output. joint_absolute: model output is already X5 joint+gripper sequence.",
    )
    args = parser.parse_args()

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    _log(f"start ckpt={args.ckpt} split={args.split} sample_index={args.sample_index} export_mode={args.export_mode}")

    with initialize(config_path="../policy_conf", job_name="export_x5_action_from_ckpt"):
        cfg = compose(config_name=args.config_name)
    cfg.root_data_dir = args.root_data_dir
    cfg.datamodule.root_data_dir = args.root_data_dir
    cfg.batch_size = 1
    cfg.num_workers = args.num_workers
    cfg.datamodule.datasets.lang_dataset.num_workers = args.num_workers
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.text_encoder_path
    cfg.model.t5_model_path = args.t5_model_path
    cfg.model.language_goal_path = args.language_goal_path
    with open_dict(cfg):
        cfg.val_num_batches = 1

    _log("instantiate datamodule")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    _log("datamodule.setup begin")
    datamodule.setup()
    _log("datamodule.setup done")
    loader = datamodule.val_dataloader()["lang"] if args.split == "validation" else datamodule.train_dataloader()["lang"]
    # Export uses one dataset sample as the visual/language context for one chunk prediction.
    _log("take sample begin")
    sample = _take_sample(loader, args.sample_index)
    _log("take sample done")

    _log("instantiate model begin")
    model = hydra.utils.instantiate(cfg.model)
    _log("instantiate model done")
    _log("load checkpoint begin")
    weights, checkpoint = _load_checkpoint_weights(args.ckpt)
    load_result = model.load_state_dict(weights, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(f"move model to device={device}")
    model = model.to(device)
    model.process_device()
    model.eval()
    _log("model ready")

    obs = {"rgb_obs": {k: v.to(device) for k, v in sample["rgb_obs"].items()}}
    goal = {"lang_text": sample["lang_text"]}
    _log(
        "eval_forward begin "
        f"rgb_static_shape={tuple(obs['rgb_obs']['rgb_static'].shape)} "
        f"rgb_gripper_shape={tuple(obs['rgb_obs']['rgb_gripper'].shape)}"
    )
    with torch.no_grad():
        pred = model.eval_forward(obs, goal).detach().cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _log(f"eval_forward done pred_shape={tuple(pred.shape)}")

    output_dir = Path(args.output_dir) if args.output_dir else (Path(args.ckpt).resolve().parent / "x5_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"write outputs to {output_dir}")

    raw_path = output_dir / f"{args.split}_sample_{args.sample_index:05d}_raw.npy"
    ee_path = output_dir / f"{args.split}_sample_{args.sample_index:05d}_ee.npy"
    joint_path = output_dir / f"{args.split}_sample_{args.sample_index:05d}_joint.npy"

    summary = {
        "ckpt": args.ckpt,
        "split": args.split,
        "sample_index": args.sample_index,
        "lang_text": sample["lang_text"][0] if isinstance(sample["lang_text"], list) else sample["lang_text"],
        "pred_shape": list(pred.shape),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }

    if args.export_mode in {"raw", "all"}:
        npy = pred
        import numpy as np
        np.save(raw_path, npy)
        summary["raw_output"] = str(raw_path)

    if args.export_mode in {"ee", "all"}:
        import numpy as np
        if args.action_format == "joint_absolute":
            if not args.urdf:
                raise ValueError("--urdf is required for joint_absolute EE export")
            fk_chain = load_x5_fk_chain(Path(args.urdf))
            joint = np.asarray(pred[0], dtype=np.float32) if pred.ndim == 3 else np.asarray(pred, dtype=np.float32)
            ee = np.zeros((joint.shape[0], 7), dtype=np.float32)
            for idx in range(joint.shape[0]):
                ee[idx, :6] = _pose_vec(fk_chain, joint[idx, :6]).astype(np.float32)
                ee[idx, 6] = joint[idx, 6]
            if pred.ndim == 3:
                ee = ee[None, ...]
        else:
            # EE export only denormalizes the DeFi relative action chunk.
            ee = convert_rel_action_to_x5_ee(
                pred,
                max_pos=args.max_pos,
                max_orn=args.max_orn,
                binary_gripper=args.binary_gripper,
                gripper_close_value=args.gripper_close_value,
                gripper_open_value=args.gripper_open_value,
            )
        np.save(ee_path, ee)
        summary["ee_output"] = str(ee_path)

    if args.export_mode in {"joint", "all"}:
        import numpy as np
        if args.action_format == "joint_absolute":
            joint = pred
        else:
            if not args.urdf:
                raise ValueError("--urdf is required for joint export")
            if args.joint_state_input:
                joint_state = np.load(args.joint_state_input)
            elif args.raw_dataset_root:
                # When no explicit joint-state file is provided, use one state from the raw shared dataset.
                joint_state = load_joint_state_from_dataset(Path(args.raw_dataset_root), args.episode_index, args.frame_index)
            else:
                raise ValueError("Provide --joint_state_input or (--raw_dataset_root + --episode_index + --frame_index)")
            fk_chain = load_x5_fk_chain(Path(args.urdf))
            joint = convert_rel_action_to_x5_joint(
                pred,
                joint_state,
                fk_chain=fk_chain,
                max_pos=args.max_pos,
                max_orn=args.max_orn,
                max_joint_delta=args.max_joint_delta,
                binary_gripper=args.binary_gripper,
                gripper_close_value=args.gripper_close_value,
                gripper_open_value=args.gripper_open_value,
            )
        np.save(joint_path, joint)
        summary["joint_output"] = str(joint_path)

    summary_path = output_dir / f"{args.split}_sample_{args.sample_index:05d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _log(f"done summary={summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
