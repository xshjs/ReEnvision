import argparse
import json
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm import tqdm

from policy_evaluation.oracle_hypothesis_rollout import (
    build_vjepa_vpp_bank,
    latent_tokens_from_future_feature,
    load_model,
    load_universal_moe_oracle_router,
    load_vjepa2_encoder,
    load_vjepa_moe,
    retrieve_moe_vjepa_expert_futures,
    universal_moe_oracle_router_features,
)


def batchify_sample(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    obs = {
        "rgb_obs": {
            "rgb_static": sample["rgb_obs"]["rgb_static"].unsqueeze(0).to(device),
            "rgb_gripper": sample["rgb_obs"]["rgb_gripper"].unsqueeze(0).to(device),
        },
        "robot_obs": sample["robot_obs"].unsqueeze(0).to(device),
    }
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline outer-selected LAM token cache for policy fine-tuning.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-dir", required=True)
    parser.add_argument("--demo-calvin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vjepa-root", type=Path, default=Path("/mnt/data/manipulation/vjepa2"))
    parser.add_argument("--vjepa-moe", type=Path, required=True)
    parser.add_argument("--universal-moe-oracle-router", type=Path, required=True)
    parser.add_argument("--vjepa-bank-cache", type=Path, default=None)
    parser.add_argument("--vpp-bank-per-task", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    with initialize(config_path="../policy_conf", job_name="build_outer_token_cache"):
        cfg = compose(config_name="VPP_Calvinabc_train")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_dir
    with open_dict(cfg):
        cfg.future_feature_mode = "gfdm"
        cfg.future_frame_idx = 5
        cfg.num_sampling_steps = cfg.model.num_sampling_steps
        cfg.sampler_type = cfg.model.sampler_type
        cfg.multistep = cfg.model.multistep
        cfg.sigma_min = cfg.model.sigma_min
        cfg.sigma_max = cfg.model.sigma_max
        cfg.noise_scheduler = cfg.model.noise_scheduler

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    dataset = datamodule.train_datasets["lang"] if args.split == "train" else datamodule.val_datasets["lang"]

    model = load_model(cfg, args.checkpoint, device)
    vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
    vjepa_moe, moe_k = load_vjepa_moe(args.vjepa_moe, device)
    outer_router = load_universal_moe_oracle_router(args.universal_moe_oracle_router, device)
    vjepa_bank = build_vjepa_vpp_bank(
        model,
        vjepa_encoder,
        args.demo_calvin_dir,
        args.vpp_bank_per_task,
        cache_path=args.vjepa_bank_cache,
    )

    requested_indices = [
        args.start_index + i * args.sample_stride
        for i in range(args.num_samples)
        if args.start_index + i * args.sample_stride < len(dataset)
    ]
    end = requested_indices[-1] + 1 if requested_indices else args.start_index
    rows = []
    tokens = []
    skipped = []
    drift_memory = []

    for sample_idx in tqdm(requested_indices, desc="outer token cache"):
        sample = dataset[sample_idx]
        lang_ann_idx = int(dataset.lang_lookup[sample_idx])
        task = str(dataset.lang_ann["task"][lang_ann_idx]) if isinstance(dataset.lang_ann, dict) else ""
        if not task:
            # auto_lang_ann stores task separately on current CALVIN dumps.
            task = str(__import__("numpy").load(
                Path(dataset.abs_datasets_dir) / dataset.lang_folder / "auto_lang_ann.npy",
                allow_pickle=True,
            ).item()["language"]["task"][lang_ann_idx])
        lang_text = str(sample["lang_text"])
        obs = batchify_sample(sample, device)
        goal = {"lang_text": lang_text}
        expert_items, _ = retrieve_moe_vjepa_expert_futures(
            model,
            vjepa_encoder,
            vjepa_moe,
            obs,
            task,
            vjepa_bank,
            moe_k,
        )
        if not expert_items:
            skipped.append({"sample_idx": sample_idx, "task": task, "reason": "no_expert_items"})
            continue

        latent_tokens_by_expert = {
            int(item["expert"]): latent_tokens_from_future_feature(model, obs, goal, item["future"]).detach().cpu()
            for item in expert_items
        }
        features = universal_moe_oracle_router_features(
            expert_items,
            latent_tokens_by_expert,
            drift_memory,
            cfg,
            getattr(outer_router, "feature_layout", None),
        )
        with torch.no_grad():
            scores = outer_router(features.unsqueeze(0).to(device)).squeeze(0)
        sorted_items = sorted(expert_items, key=lambda row: row["expert"])
        best_pos = int(torch.argmax(scores).detach().cpu())
        best_item = sorted_items[best_pos]
        best_expert = int(best_item["expert"])
        token = latent_tokens_by_expert[best_expert].squeeze(0).to(torch.float16)
        tokens.append(token)
        rows.append(
            {
                "sample_idx": int(sample_idx),
                "episode_idx": int(dataset.episode_lookup[sample_idx]),
                "lang_ann_idx": lang_ann_idx,
                "task": task,
                "lang_text": lang_text,
                "selected_expert": best_expert,
                "scores": [float(x) for x in scores.detach().cpu().tolist()],
                "demo_index": int(best_item["demo_index"]),
                "router_logit": float(best_item["router_logit"]),
                "vjepa_score": float(best_item["vjepa_score"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "start_index": args.start_index,
            "end_index": end,
            "num_requested": args.num_samples,
            "sample_stride": args.sample_stride,
            "num_cached": len(rows),
            "num_skipped": len(skipped),
            "calvin_dir": args.calvin_dir,
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "vjepa_moe": str(args.vjepa_moe),
            "universal_moe_oracle_router": str(args.universal_moe_oracle_router),
        },
        "rows": rows,
        "tokens": torch.stack(tokens, dim=0) if tokens else torch.empty(0),
        "skipped": skipped,
    }
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({**payload["metadata"], "skipped": skipped[:20]}, indent=2) + "\n")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
