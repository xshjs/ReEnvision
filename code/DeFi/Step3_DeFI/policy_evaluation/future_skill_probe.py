import argparse
import json
import random
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())


GROUPS = ["obs", "obs_gt_future", "gt_future_delta", "obs_gfdm_future"]


@dataclass
class SkillSegment:
    index: int
    skill: str
    language: str
    start_idx: int
    end_idx: int


class ClipFeatureEncoder:
    def __init__(self, model_path: str, device: str):
        if "distutils.spawn" not in sys.modules:
            spawn_module = types.ModuleType("distutils.spawn")
            spawn_module.find_executable = shutil.which
            sys.modules["distutils.spawn"] = spawn_module
        from transformers import CLIPProcessor, CLIPVisionModelWithProjection

        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self.model(**inputs).image_embeds
        features = features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-8)
        return features.detach().cpu().numpy().astype(np.float32)


class GfdmFeatureEncoder:
    def __init__(self, video_model_path: str, text_encoder_path: str, device: str):
        if "distutils.spawn" not in sys.modules:
            spawn_module = types.ModuleType("distutils.spawn")
            spawn_module.find_executable = shutil.which
            sys.modules["distutils.spawn"] = spawn_module
        from diffusers import StableVideoDiffusionPipeline
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        from policy_models.module.diffusion_extract import Diffusion_feature_extractor

        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pipeline = StableVideoDiffusionPipeline.from_pretrained(video_model_path, torch_dtype=dtype).to(self.device)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path).to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)
        self.encoder = Diffusion_feature_extractor(pipeline=pipeline, tokenizer=tokenizer, text_encoder=text_encoder, position_encoding=True).to(self.device).eval()

    def preprocess_images(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
        tensor = torch.nn.functional.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
        tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return tensor.unsqueeze(1).to(self.device)

    @torch.no_grad()
    def encode_batch(self, static_images: Sequence[np.ndarray], gripper_images: Sequence[np.ndarray], texts: Sequence[str]) -> np.ndarray:
        static_tensor = self.preprocess_images(static_images)
        gripper_tensor = self.preprocess_images(gripper_images)
        input_rgb = torch.cat([static_tensor, gripper_tensor], dim=0)
        features = self.encoder(
            input_rgb,
            list(texts) + list(texts),
            20,
            1,
            all_layer=False,
            step_time=1,
            max_length=20,
        )
        selected = features[:, min(5, features.shape[1] - 1)]
        if selected.ndim == 4:
            selected = selected.mean(dim=(-1, -2))
        static_feat, gripper_feat = torch.split(selected.reshape(selected.shape[0], -1).float(), [len(texts), len(texts)], dim=0)
        output = 0.5 * (static_feat + gripper_feat)
        return output.detach().cpu().numpy().astype(np.float32)


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


class SkillProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def clip_batch_features(rows: List[Dict[str, Any]], encoder: ClipFeatureEncoder, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    obs_features = []
    future_features = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        obs_features.append(encoder.encode_batch([row["start_ep"]["rgb_static"] for row in batch]))
        future_features.append(encoder.encode_batch([row["end_ep"]["rgb_static"] for row in batch]))
    return np.concatenate(obs_features, axis=0), np.concatenate(future_features, axis=0)


def gfdm_batch_features(rows: List[Dict[str, Any]], encoder: GfdmFeatureEncoder, batch_size: int) -> np.ndarray:
    features = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        features.append(
            encoder.encode_batch(
                [row["start_ep"]["rgb_static"] for row in batch],
                [row["start_ep"]["rgb_gripper"] for row in batch],
                [row["segment"].language for row in batch],
            )
        )
    return np.concatenate(features, axis=0)


def make_group_features(obs: np.ndarray, future: np.ndarray, gfdm: Optional[np.ndarray], group: str) -> np.ndarray:
    if group == "obs":
        return obs
    if group == "obs_gt_future":
        return np.concatenate([obs, future], axis=1)
    if group == "gt_future_delta":
        return np.concatenate([obs, future - obs], axis=1)
    if group == "obs_gfdm_future":
        if gfdm is None:
            raise ValueError("obs_gfdm_future requires GFDM features")
        return np.concatenate([obs, gfdm], axis=1)
    raise ValueError(group)


def standardize(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (val - mean) / std


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = SkillProbe(x_train.shape[1], int(y_train.max()) + 1, args.hidden_dim).to(device)
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
    parser = argparse.ArgumentParser(description="Skill probe for obs, GT future, and GFDM future features.")
    parser.add_argument("--calvin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-split", type=int, default=200)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--clip-model-path", default="ckpts/openai_clip_vit_base_patch32")
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--gfdm-video-model-path", default="ckpts/svd/checkpoint-100000")
    parser.add_argument("--gfdm-text-encoder-path", default="ckpts/openai_clip_vit_base_patch32")
    parser.add_argument("--gfdm-device", default="cuda:0")
    parser.add_argument("--gfdm-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    train_rows = collect_rows(args.calvin_dir / "training", "training", args.max_samples_per_split)
    val_rows = collect_rows(args.calvin_dir / "validation", "validation", args.max_samples_per_split)
    skills = sorted({row["segment"].skill for row in train_rows + val_rows})
    skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
    y_train = np.asarray([skill_to_idx[row["segment"].skill] for row in train_rows], dtype=np.int64)
    y_val = np.asarray([skill_to_idx[row["segment"].skill] for row in val_rows], dtype=np.int64)

    clip_encoder = ClipFeatureEncoder(args.clip_model_path, args.clip_device)
    train_obs, train_future = clip_batch_features(train_rows, clip_encoder, args.clip_batch_size)
    val_obs, val_future = clip_batch_features(val_rows, clip_encoder, args.clip_batch_size)
    train_gfdm = val_gfdm = None
    if "obs_gfdm_future" in groups:
        gfdm_encoder = GfdmFeatureEncoder(
            args.gfdm_video_model_path,
            args.gfdm_text_encoder_path,
            args.gfdm_device,
        )
        train_gfdm = gfdm_batch_features(train_rows, gfdm_encoder, args.gfdm_batch_size)
        val_gfdm = gfdm_batch_features(val_rows, gfdm_encoder, args.gfdm_batch_size)

    results = {}
    for group in groups:
        x_train = make_group_features(train_obs, train_future, train_gfdm, group)
        x_val = make_group_features(val_obs, val_future, val_gfdm, group)
        x_train, x_val = standardize(x_train, x_val)
        results[group] = train_probe(x_train, y_train, x_val, y_val, args)

    (args.output_dir / "metrics.json").write_text(json.dumps({"results": results, "skills": skills}, indent=2) + "\n")
    lines = ["# Future Skill Probe", "", "| Group | Top-1 | Top-3 | N val |", "|---|---:|---:|---:|"]
    for group, row in results.items():
        lines.append(f"| {group} | {row['top1']:.3f} | {row['top3']:.3f} | {row['n_val']} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
