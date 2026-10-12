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
import torch.nn.functional as F
from torch import nn

torch.backends.cudnn.enabled = False

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

if "distutils.spawn" not in sys.modules:
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_module


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


def patch_transformers_torch_load_check() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils

        import_utils.check_torch_load_is_safe = lambda: None
        modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


def load_annotations(split_dir: Path) -> List[SkillSegment]:
    data = np.load(split_dir / "lang_clip_resnet50" / "auto_lang_ann.npy", allow_pickle=True).item()
    return [
        SkillSegment(idx, str(task), str(ann), int(interval[0]), int(interval[1]))
        for idx, (task, ann, interval) in enumerate(zip(data["language"]["task"], data["language"]["ann"], data["info"]["indx"]))
    ]


def load_episode(split_dir: Path, episode_idx: int) -> Dict[str, np.ndarray]:
    path = split_dir / f"episode_{episode_idx:07d}.npz"
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def collect_rows(split_dir: Path, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    for segment in load_annotations(split_dir):
        start_path = split_dir / f"episode_{segment.start_idx:07d}.npz"
        end_path = split_dir / f"episode_{segment.end_idx:07d}.npz"
        if not start_path.exists() or not end_path.exists():
            continue
        rows.append(
            {
                "segment": segment,
                "start_ep": load_episode(split_dir, segment.start_idx),
                "end_ep": load_episode(split_dir, segment.end_idx),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def find_target_row(split_dir: Path, task: str, occurrence: int) -> Dict[str, Any]:
    seen = 0
    for segment in load_annotations(split_dir):
        if segment.skill != task:
            continue
        start_path = split_dir / f"episode_{segment.start_idx:07d}.npz"
        end_path = split_dir / f"episode_{segment.end_idx:07d}.npz"
        if not start_path.exists() or not end_path.exists():
            continue
        if seen == occurrence:
            return {
                "segment": segment,
                "start_ep": load_episode(split_dir, segment.start_idx),
                "end_ep": load_episode(split_dir, segment.end_idx),
            }
        seen += 1
    raise ValueError(f"Could not find task={task!r} occurrence={occurrence} in {split_dir}")


class ClipEncoder:
    def __init__(self, model_path: str, device: str):
        patch_transformers_torch_load_check()
        from transformers import CLIPProcessor, CLIPVisionModelWithProjection

        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def encode(self, images: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model(**inputs).image_embeds
            features = features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-8)
            outputs.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


class GfdmRgbGenerator:
    def __init__(self, video_model_path: str, text_encoder_path: str, device: str, frame_index: int, timestep: int):
        patch_transformers_torch_load_check()
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
            position_encoding=False,
        ).to(self.device).eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
        tensor = F.interpolate(tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)
        tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return tensor.unsqueeze(1).to(self.device)

    @torch.no_grad()
    def generate_one(self, image: np.ndarray, text: str, seed: int) -> np.ndarray:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _, videos = self.encoder(
            self.preprocess(image),
            [text],
            self.timestep,
            1,
            all_layer=True,
            step_time=1,
            max_length=20,
            return_video=True,
            save_video=False,
        )
        video = videos[0]
        return np.asarray(video[min(self.frame_index, len(video) - 1)], dtype=np.uint8)


def train_probe(x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> SkillProbe:
    device = torch.device(args.probe_device if args.probe_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
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
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K GFDM futures through a CLIP/RGB skill probe and compute best-of-K GT gap.")
    parser.add_argument("--calvin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="stack_block")
    parser.add_argument("--target-split", choices=["training", "validation"], default="validation")
    parser.add_argument("--target-occurrence", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--gfdm-video-model-path", required=True)
    parser.add_argument("--gfdm-text-encoder-path", required=True)
    parser.add_argument("--gfdm-device", default="cuda:0")
    parser.add_argument("--gfdm-frame-index", type=int, default=5)
    parser.add_argument("--gfdm-timestep", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = collect_rows(args.calvin_dir / "training", args.max_train_samples)
    target_row = find_target_row(args.calvin_dir / args.target_split, args.task, args.target_occurrence)
    skills = sorted({row["segment"].skill for row in train_rows})
    skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
    idx_to_skill = {idx: skill for skill, idx in skill_to_idx.items()}

    clip = ClipEncoder(args.clip_model_path, args.clip_device)
    train_obs = clip.encode([row["start_ep"]["rgb_static"] for row in train_rows], args.clip_batch_size)
    train_future = clip.encode([row["end_ep"]["rgb_static"] for row in train_rows], args.clip_batch_size)
    x_train = np.concatenate([train_obs, train_future], axis=1)
    y_train = np.asarray([skill_to_idx[row["segment"].skill] for row in train_rows], dtype=np.int64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    probe = train_probe((x_train - mean) / std, y_train, args)

    gfdm = GfdmRgbGenerator(
        args.gfdm_video_model_path,
        args.gfdm_text_encoder_path,
        args.gfdm_device,
        args.gfdm_frame_index,
        args.gfdm_timestep,
    )
    obs_image = target_row["start_ep"]["rgb_static"]
    gt_future_image = target_row["end_ep"]["rgb_static"]
    language = target_row["segment"].language
    future_images = [gfdm.generate_one(obs_image, language, args.seed + i) for i in range(args.k)]

    obs_feat = clip.encode([obs_image], args.clip_batch_size)
    gt_future_feat = clip.encode([gt_future_image], args.clip_batch_size)
    gen_future_feat = clip.encode(future_images, args.clip_batch_size)
    x_gen = np.concatenate([np.repeat(obs_feat, args.k, axis=0), gen_future_feat], axis=1)
    x_gen = (x_gen - mean) / std
    device = next(probe.parameters()).device
    with torch.no_grad():
        logits = probe(torch.tensor(x_gen, dtype=torch.float32, device=device)).cpu().numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    top_idx = np.argsort(-logits, axis=1)

    cosine_to_gt = (gen_future_feat @ gt_future_feat[0]).astype(np.float64)
    l2_to_gt = np.linalg.norm(gen_future_feat - gt_future_feat, axis=1).astype(np.float64)
    single_l2 = float(l2_to_gt[0])
    best_i = int(np.argmin(l2_to_gt))
    best_l2 = float(l2_to_gt[best_i])
    single_cosine = float(cosine_to_gt[0])
    best_cosine_i = int(np.argmax(cosine_to_gt))
    best_cosine = float(cosine_to_gt[best_cosine_i])

    samples = []
    for i in range(args.k):
        top1 = idx_to_skill[int(top_idx[i, 0])]
        samples.append(
            {
                "sample": i,
                "seed": args.seed + i,
                "top1": top1,
                "top1_prob": float(probs[i, top_idx[i, 0]]),
                "top3": [idx_to_skill[int(item)] for item in top_idx[i, :3]],
                "cosine_to_gt_future": float(cosine_to_gt[i]),
                "l2_to_gt_future": float(l2_to_gt[i]),
            }
        )

    report = {
        "target": {
            "split": args.target_split,
            "task": target_row["segment"].skill,
            "language": language,
            "start_idx": target_row["segment"].start_idx,
            "end_idx": target_row["segment"].end_idx,
        },
        "probe": {
            "train_samples": len(train_rows),
            "feature": "concat(CLIP(obs_rgb_static), CLIP(future_rgb_static))",
            "skills": skills,
        },
        "semantic_diversity": {
            "unique_top1": sorted({row["top1"] for row in samples}),
            "num_unique_top1": len({row["top1"] for row in samples}),
            "all_top1_same": len({row["top1"] for row in samples}) == 1,
        },
        "oracle_of_10_gap": {
            "metric": "CLIP future embedding distance to GT future",
            "single_sample_index": 0,
            "single_l2": single_l2,
            "best_l2": best_l2,
            "best_l2_sample": best_i,
            "l2_improvement_abs": single_l2 - best_l2,
            "l2_improvement_pct": (single_l2 - best_l2) / max(single_l2, 1e-8),
            "single_cosine": single_cosine,
            "best_cosine": best_cosine,
            "best_cosine_sample": best_cosine_i,
            "cosine_improvement_abs": best_cosine - single_cosine,
        },
        "samples": samples,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
