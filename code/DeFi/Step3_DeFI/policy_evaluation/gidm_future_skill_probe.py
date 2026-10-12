import argparse
import json
import importlib.machinery
import math
import random
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).absolute().parents[2]
STEP2_ROOT = REPO_ROOT / "Step2_GIDM"
STEP3_ROOT = REPO_ROOT / "Step3_DeFI"
sys.path.insert(0, STEP2_ROOT.as_posix())
sys.path.insert(0, STEP3_ROOT.as_posix())


@dataclass
class SkillSegment:
    index: int
    skill: str
    language: str
    start_idx: int
    end_idx: int


class SkillProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def install_distutils_spawn_shim() -> None:
    if "distutils.spawn" in sys.modules:
        return
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_module


def install_timm_dino_torchhub_shim() -> None:
    try:
        import timm
    except ModuleNotFoundError:
        system_site = "/usr/local/lib/python3.12/site-packages"
        sys.path.append(system_site)
        try:
            import timm
        finally:
            try:
                sys.path.remove(system_site)
            except ValueError:
                pass

    original_torchhub_load = torch.hub.load

    def torchhub_load(repo_or_dir: str, model: str, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2" and model == "dinov2_vitb14_reg":
            dino = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=False, img_size=224)
            dino.register_tokens = dino.reg_token
            original_forward_features = dino.forward_features

            def forward_features(images: torch.Tensor) -> Dict[str, torch.Tensor]:
                tokens = original_forward_features(images)
                return {"x_norm_patchtokens": tokens[:, dino.num_prefix_tokens :]}

            dino.forward_features = forward_features
            return dino
        return original_torchhub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = torchhub_load


def disable_transformers_apex() -> None:
    import os

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    apex_module = types.ModuleType("apex")
    normalization_module = types.ModuleType("apex.normalization")
    apex_module.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
    normalization_module.__spec__ = importlib.machinery.ModuleSpec("apex.normalization", loader=None)
    normalization_module.FusedRMSNorm = nn.LayerNorm
    normalization_module.FusedLayerNorm = nn.LayerNorm
    apex_module.normalization = normalization_module
    sys.modules["apex"] = apex_module
    sys.modules["apex.normalization"] = normalization_module
    try:
        import transformers.utils.import_utils as import_utils

        import_utils._apex_available = False
    except Exception:
        pass


def resize_patch_pos_embed(pos_embed: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if pos_embed.shape[1] == target_tokens:
        return pos_embed
    patch_pos = pos_embed[:, 1:] if pos_embed.shape[1] != target_tokens else pos_embed
    source_side = int(math.sqrt(patch_pos.shape[1]))
    target_side = int(math.sqrt(target_tokens))
    if source_side * source_side != patch_pos.shape[1] or target_side * target_side != target_tokens:
        raise ValueError(f"Cannot resize pos_embed from {tuple(pos_embed.shape)} to {target_tokens} tokens")
    patch_pos = patch_pos.reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(target_side, target_side), mode="bicubic", align_corners=False)
    return patch_pos.permute(0, 2, 3, 1).reshape(1, target_tokens, -1)


def adapt_timm_dino_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    adapted = dict(state_dict)
    pos_key = "dino_encoder.pos_embed"
    if pos_key in adapted and hasattr(model.dino_encoder, "pos_embed"):
        target_tokens = model.dino_encoder.pos_embed.shape[1]
        adapted[pos_key] = resize_patch_pos_embed(adapted[pos_key], target_tokens)
    reg_key = "dino_encoder.register_tokens"
    if reg_key in adapted and "dino_encoder.reg_token" not in adapted:
        adapted["dino_encoder.reg_token"] = adapted[reg_key]
    return adapted


def load_gidm(args: argparse.Namespace) -> nn.Module:
    install_distutils_spawn_shim()
    disable_transformers_apex()
    if args.dino_backend == "timm":
        install_timm_dino_torchhub_shim()
    from genie.modules.lam import UncontrolledDINOLatentActionModel

    model = UncontrolledDINOLatentActionModel(
        in_dim=3,
        model_dim=args.gidm_model_dim,
        latent_dim=args.gidm_latent_dim,
        num_latents=args.gidm_num_latents,
        patch_size=args.gidm_patch_size,
        enc_blocks=args.gidm_enc_blocks,
        dec_blocks=args.gidm_dec_blocks,
        num_heads=args.gidm_num_heads,
        dropout=0.0,
    )
    checkpoint = torch.load(args.gidm_ckpt, map_location="cpu")
    raw_state = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key[len("lam.") :]: value
        for key, value in raw_state.items()
        if key.startswith("lam.")
    }
    if args.dino_backend == "timm":
        state_dict = adapt_timm_dino_state_dict(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded GIDM with {len(incompatible.missing_keys)} missing and {len(incompatible.unexpected_keys)} unexpected keys")
    device = torch.device(args.gidm_device if args.gidm_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


class GfdmFutureGenerator:
    def __init__(self, video_model_path: str, text_encoder_path: str, device: str, frame_index: int, timestep: int):
        install_distutils_spawn_shim()
        from diffusers import StableVideoDiffusionPipeline
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        from policy_models.module.diffusion_extract_outvideo import Diffusion_feature_extractor

        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.frame_index = frame_index
        self.timestep = timestep
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pipeline = StableVideoDiffusionPipeline.from_pretrained(video_model_path, torch_dtype=dtype).to(self.device)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path).to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)
        self.encoder = Diffusion_feature_extractor(
            pipeline=pipeline,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            position_encoding=True,
        ).to(self.device).eval()

    def preprocess_images(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
        tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return tensor.unsqueeze(1).to(self.device)

    @torch.no_grad()
    def generate_batch(self, static_images: Sequence[np.ndarray], texts: Sequence[str]) -> List[np.ndarray]:
        _, videos = self.encoder(
            self.preprocess_images(static_images),
            list(texts),
            self.timestep,
            1,
            all_layer=False,
            step_time=1,
            max_length=20,
            return_video=True,
        )
        futures = []
        for video in videos:
            index = min(self.frame_index, len(video) - 1)
            futures.append(np.asarray(video[index], dtype=np.uint8))
        return futures


def load_annotations(split_dir: Path) -> List[SkillSegment]:
    data = np.load(split_dir / "lang_clip_resnet50" / "auto_lang_ann.npy", allow_pickle=True).item()
    return [
        SkillSegment(idx, str(task), str(ann), int(interval[0]), int(interval[1]))
        for idx, (task, ann, interval) in enumerate(zip(data["language"]["task"], data["language"]["ann"], data["info"]["indx"]))
    ]


def load_episode(split_dir: Path, episode_idx: int) -> Optional[Dict[str, np.ndarray]]:
    path = split_dir / f"episode_{episode_idx:07d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def collect_rows(split_dir: Path, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    for segment in load_annotations(split_dir):
        start_ep = load_episode(split_dir, segment.start_idx)
        end_ep = load_episode(split_dir, segment.end_idx)
        if start_ep is None or end_ep is None:
            continue
        rows.append({"split": split, "segment": segment, "start_ep": start_ep, "end_ep": end_ep})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def preprocess_gidm_images(images: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    return tensor.div(255.0).to(device)


def token_feature_from_output(output: Dict[str, torch.Tensor], mode: str, num_latents: int) -> np.ndarray:
    if mode == "indices_onehot":
        indices = output["indices"].detach().long().cpu()
        features = F.one_hot(indices, num_classes=num_latents).reshape(indices.shape[0], -1).float()
        return features.numpy().astype(np.float32)
    if mode == "indices":
        return output["indices"].detach().float().cpu().numpy().astype(np.float32)
    if mode == "zq":
        return output["z_q"].detach().reshape(output["z_q"].shape[0], -1).float().cpu().numpy().astype(np.float32)
    if mode == "patches":
        patches = output["video_action_patches"].detach()
        return patches.reshape(patches.shape[0], -1).float().cpu().numpy().astype(np.float32)
    raise ValueError(mode)


@torch.no_grad()
def gidm_token_features(
    rows: List[Dict[str, Any]],
    model: nn.Module,
    future_source: str,
    args: argparse.Namespace,
    gfdm: Optional[GfdmFutureGenerator],
) -> np.ndarray:
    device = next(model.parameters()).device
    features = []
    for start in range(0, len(rows), args.gidm_batch_size):
        batch_rows = rows[start : start + args.gidm_batch_size]
        obs_images = [row["start_ep"]["rgb_static"] for row in batch_rows]
        if future_source == "gt_future":
            future_images = [row["end_ep"]["rgb_static"] for row in batch_rows]
        elif future_source == "gfdm_future":
            if gfdm is None:
                raise ValueError("gfdm_future requires GFDM generator")
            future_images = gfdm.generate_batch(obs_images, [row["segment"].language for row in batch_rows])
        else:
            raise ValueError(future_source)
        videos = torch.stack(
            [preprocess_gidm_images(obs_images, device), preprocess_gidm_images(future_images, device)],
            dim=1,
        )
        output = model({"videos": videos, "task_instruction": [row["segment"].language for row in batch_rows]})
        features.append(token_feature_from_output(output, args.token_feature, args.gidm_num_latents))
    return np.concatenate(features, axis=0)


def standardize(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (val - mean) / std


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.probe_device if args.probe_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = SkillProbe(x_train.shape[1], int(max(y_train.max(), y_val.max())) + 1, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    rng = np.random.default_rng(args.seed)
    model.train()
    for _ in range(args.epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            loss = criterion(model(x_tensor[batch]), y_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_val, dtype=torch.float32, device=device)).cpu().numpy()
    order = np.argsort(-logits, axis=1)
    top1 = np.mean(order[:, 0] == y_val)
    top3 = np.mean([label in order[idx, :3] for idx, label in enumerate(y_val)])
    return {"top1": float(top1), "top3": float(top3), "n_val": int(len(y_val))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe skill predictability from GIDM latent action tokens using GT vs GFDM futures.")
    parser.add_argument("--calvin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-split", type=int, default=200)
    parser.add_argument("--future-sources", default="gt_future,gfdm_future")
    parser.add_argument("--gidm-ckpt", type=Path, default=Path("ckpts/IDM/lam_idm.ckpt"))
    parser.add_argument("--dino-backend", choices=["torchhub", "timm"], default="timm")
    parser.add_argument("--gidm-device", default="cuda:0")
    parser.add_argument("--gidm-batch-size", type=int, default=8)
    parser.add_argument("--gidm-model-dim", type=int, default=768)
    parser.add_argument("--gidm-latent-dim", type=int, default=128)
    parser.add_argument("--gidm-num-latents", type=int, default=16)
    parser.add_argument("--gidm-patch-size", type=int, default=14)
    parser.add_argument("--gidm-enc-blocks", type=int, default=12)
    parser.add_argument("--gidm-dec-blocks", type=int, default=12)
    parser.add_argument("--gidm-num-heads", type=int, default=12)
    parser.add_argument("--token-feature", choices=["indices_onehot", "indices", "zq", "patches"], default="indices_onehot")
    parser.add_argument("--gfdm-video-model-path", default="ckpts/svd/checkpoint-100000")
    parser.add_argument("--gfdm-text-encoder-path", default="ckpts/openai_clip_vit_base_patch32")
    parser.add_argument("--gfdm-device", default="cuda:0")
    parser.add_argument("--gfdm-frame-index", type=int, default=5)
    parser.add_argument("--gfdm-timestep", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = collect_rows(args.calvin_dir / "training", "training", args.max_samples_per_split)
    val_rows = collect_rows(args.calvin_dir / "validation", "validation", args.max_samples_per_split)
    skills = sorted({row["segment"].skill for row in train_rows + val_rows})
    skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
    y_train = np.asarray([skill_to_idx[row["segment"].skill] for row in train_rows], dtype=np.int64)
    y_val = np.asarray([skill_to_idx[row["segment"].skill] for row in val_rows], dtype=np.int64)

    future_sources = [source.strip() for source in args.future_sources.split(",") if source.strip()]
    gidm = load_gidm(args)
    gfdm = None
    if "gfdm_future" in future_sources:
        gfdm = GfdmFutureGenerator(
            args.gfdm_video_model_path,
            args.gfdm_text_encoder_path,
            args.gfdm_device,
            args.gfdm_frame_index,
            args.gfdm_timestep,
        )

    results = {}
    for source in future_sources:
        print(f"Extracting GIDM token features for {source}")
        x_train = gidm_token_features(train_rows, gidm, source, args, gfdm)
        x_val = gidm_token_features(val_rows, gidm, source, args, gfdm)
        x_train, x_val = standardize(x_train, x_val)
        results[source] = train_probe(x_train, y_train, x_val, y_val, args)

    metrics = {
        "results": results,
        "skills": skills,
        "config": {
            "max_samples_per_split": args.max_samples_per_split,
            "token_feature": args.token_feature,
            "gfdm_frame_index": args.gfdm_frame_index,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = ["# GIDM Future Skill Probe", "", "| Future source | Top-1 | Top-3 | N val |", "|---|---:|---:|---:|"]
    for source, row in results.items():
        lines.append(f"| {source} | {row['top1']:.3f} | {row['top3']:.3f} | {row['n_val']} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
