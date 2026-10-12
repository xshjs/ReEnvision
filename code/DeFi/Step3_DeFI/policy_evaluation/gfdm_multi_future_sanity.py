import argparse
import json
import os
import random
import shutil
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

if "distutils.spawn" not in sys.modules:
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_module

try:
    import pyhash  # noqa: F401
except ModuleNotFoundError:
    pyhash_module = types.ModuleType("pyhash")

    class _Fnv1_32:
        def __call__(self, value: Any) -> int:
            data = str(value).encode("utf-8")
            hval = 0x811C9DC5
            for byte in data:
                hval = (hval * 0x01000193) & 0xFFFFFFFF
                hval ^= byte
            return hval

    pyhash_module.fnv1_32 = _Fnv1_32
    sys.modules["pyhash"] = pyhash_module

try:
    import git  # noqa: F401
except ModuleNotFoundError:
    git_module = types.ModuleType("git")

    class _InvalidGitRepositoryError(Exception):
        pass

    class _Repo:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _InvalidGitRepositoryError()

    git_module.Repo = _Repo
    git_module.exc = types.SimpleNamespace(InvalidGitRepositoryError=_InvalidGitRepositoryError)
    sys.modules["git"] = git_module

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

torch.backends.cudnn.enabled = False

if "pytorch_lightning" not in sys.modules:
    pl_module = types.ModuleType("pytorch_lightning")

    def _seed_everything(seed: int, workers: bool = False) -> int:
        del workers
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    pl_module.__version__ = "shim"
    pl_module.LightningModule = torch.nn.Module
    pl_module.LightningDataModule = object
    pl_module.Callback = object
    pl_module.Trainer = object
    pl_module.seed_everything = _seed_everything

    utilities_module = types.ModuleType("pytorch_lightning.utilities")
    utilities_module.rank_zero_only = lambda fn=None, *args, **kwargs: (lambda inner: inner) if fn is None else fn
    cloud_io_module = types.ModuleType("pytorch_lightning.utilities.cloud_io")
    cloud_io_module.load = torch.load

    loggers_module = types.ModuleType("pytorch_lightning.loggers")
    loggers_module.TensorBoardLogger = object
    loggers_module.WandbLogger = object

    sys.modules["pytorch_lightning"] = pl_module
    sys.modules["pytorch_lightning.utilities"] = utilities_module
    sys.modules["pytorch_lightning.utilities.cloud_io"] = cloud_io_module
    sys.modules["pytorch_lightning.loggers"] = loggers_module

if "pytorch_lightning.utilities.cloud_io" not in sys.modules:
    cloud_io_module = types.ModuleType("pytorch_lightning.utilities.cloud_io")
    cloud_io_module.load = torch.load
    sys.modules["pytorch_lightning.utilities.cloud_io"] = cloud_io_module

from pytorch_lightning import seed_everything

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition
from policy_models.module.diffusion_extract_outvideo import Diffusion_feature_extractor as GFDMVideoExtractor


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


def ensure_local_eval_dir(output_dir: Path, checkpoint: Path) -> Path:
    train_folder = output_dir / "eval_train_folder"
    saved_models = train_folder / "saved_models"
    saved_models.mkdir(parents=True, exist_ok=True)
    link_path = saved_models / checkpoint.name
    if not link_path.exists():
        try:
            link_path.symlink_to(checkpoint.resolve())
        except OSError:
            shutil.copy2(checkpoint, link_path)
    return train_folder


def load_gfdm_only(cfg: Any, video_model_path: str, clip_model_path: str, device: torch.device) -> Any:
    from diffusers import StableVideoDiffusionPipeline
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    pipeline = StableVideoDiffusionPipeline.from_pretrained(video_model_path, torch_dtype=torch.float16)
    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path, use_fast=False)

    for module in [pipeline.image_encoder, pipeline.vae, pipeline.unet, text_encoder]:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    pipeline = pipeline.to(device)
    text_encoder = text_encoder.to(device).eval()
    extractor = GFDMVideoExtractor(
        pipeline=pipeline,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        position_encoding=bool(cfg.model.use_position_encoding),
    ).to(device)
    extractor.eval()
    return SimpleNamespace(
        device=device,
        timestep=int(cfg.model.timestep),
        extract_layer_idx=int(cfg.model.extract_layer_idx),
        use_all_layer=bool(cfg.model.use_all_layer),
        max_length=int(cfg.model.get("max_length", 20)),
        future_frame_idx=5,
        GFDM_video_extractor=extractor,
    )


def set_sample_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def find_fixed_case(
    env: Any,
    sequences: List[Tuple[Any, List[str]]],
    target_task: str,
) -> Tuple[int, int, str, Dict[str, Any]]:
    for seq_idx, (initial_state, eval_sequence) in enumerate(sequences):
        if target_task and target_task not in eval_sequence:
            continue
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        subtask = eval_sequence[0]
        if target_task and eval_sequence[0] != target_task:
            continue
        return seq_idx, 0, subtask, env.get_obs()

    if target_task:
        raise ValueError(
            f"Could not find target task {target_task!r} as the first subtask. "
            "Use --task '' to sample the first available initial obs/goal."
        )

    initial_state, eval_sequence = sequences[0]
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    return 0, 0, eval_sequence[0], env.get_obs()


def image_to_model_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    import cv2

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0
    return tensor.unsqueeze(0).unsqueeze(0)


def load_obs_from_npz(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    episode = np.load(path)
    return {
        "rgb_obs": {
            "rgb_static": image_to_model_tensor(episode["rgb_static"]),
            "rgb_gripper": image_to_model_tensor(episode["rgb_gripper"]),
        }
    }


@torch.no_grad()
def sample_gfdm_future_feature(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    lang_text: str,
    frame_idx: int,
    seed: int,
    include_gripper: bool,
) -> torch.Tensor:
    set_sample_seed(seed)
    enable_gfdm_video_extractor(model)
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    features = model.GFDM_video_extractor(
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
    if include_gripper:
        selected = selected.reshape(selected.shape[0], -1).float().mean(dim=0)
    else:
        selected = selected[0].reshape(-1).float()
    return selected.detach().cpu()


def pairwise_stats(features: List[torch.Tensor]) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            lhs = features[i]
            rhs = features[j]
            diff = lhs - rhs
            l2 = float(torch.linalg.vector_norm(diff).item())
            rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
            cosine = float(F.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0)).item())
            rows.append({"i": i, "j": j, "cosine": cosine, "l2": l2, "rmse": rmse})
    summary = {
        "num_features": len(features),
        "num_pairs": len(rows),
        "feature_dim": int(features[0].numel()) if features else 0,
        "cosine": summarize([row["cosine"] for row in rows]),
        "l2": summarize([row["l2"] for row in rows]),
        "rmse": summarize([row["rmse"] for row in rows]),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample K GFDM futures for one fixed obs/goal and measure pairwise diversity.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-npz", type=Path, default=None)
    parser.add_argument("--lang-text", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-sequences", type=int, default=200)
    parser.add_argument("--future-frame-idx", type=int, default=5)
    parser.add_argument("--include-gripper", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../policy_conf", job_name="gfdm_multi_future_sanity"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.train_folder = str(ensure_local_eval_dir(args.output_dir, args.checkpoint))
    cfg.log_wandb = False
    with open_dict(cfg):
        cfg.future_frame_idx = args.future_frame_idx

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_gfdm_only(cfg, args.video_model_path, args.clip_model_path, device)
    model.future_frame_idx = args.future_frame_idx

    if args.obs_npz is not None:
        seq_idx = -1
        sub_idx = -1
        subtask = args.task or "stack_block"
        obs = load_obs_from_npz(args.obs_npz)
        lang_text = args.lang_text or cfg.annotations[subtask][0]
    else:
        env, _, _ = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            args.checkpoint,
            env=None,
            lang_embeddings=None,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=device.index if device.type == "cuda" and device.index is not None else "cpu",
            cfg=cfg,
        )
        sequences = get_sequences(args.num_sequences)
        seq_idx, sub_idx, subtask, obs = find_fixed_case(env, sequences, args.task)
        lang_text = args.lang_text or cfg.annotations[subtask][0]
    print(f"fixed_case sequence={seq_idx} subtask={sub_idx} task={subtask} lang={lang_text!r}")

    features = []
    sample_rows = []
    for sample_idx in tqdm(range(args.k), desc="GFDM future samples"):
        sample_seed = args.seed + sample_idx
        feature = sample_gfdm_future_feature(
            model,
            obs,
            lang_text,
            args.future_frame_idx,
            sample_seed,
            args.include_gripper,
        )
        features.append(feature)
        sample_rows.append(
            {
                "sample": sample_idx,
                "seed": sample_seed,
                "feature_norm": float(torch.linalg.vector_norm(feature).item()),
            }
        )

    pairs, summary = pairwise_stats(features)
    summary.update(
        {
            "task": subtask,
            "lang_text": lang_text,
            "sequence_index": seq_idx,
            "subtask_index": sub_idx,
            "k": args.k,
            "seed": args.seed,
            "future_frame_idx": args.future_frame_idx,
            "include_gripper": args.include_gripper,
            "obs_npz": str(args.obs_npz) if args.obs_npz is not None else None,
            "samples": sample_rows,
        }
    )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "pairs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")
    main()
