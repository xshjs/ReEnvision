import argparse
import json
import random
import shutil
import sys
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

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
    with np.load(split_dir / f"episode_{episode_idx:07d}.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def collect_rows(split_dir: Path, max_samples: Optional[int], task_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    allowed = None if not task_filter else {item.strip() for item in task_filter.split(",") if item.strip()}
    for segment in load_annotations(split_dir):
        if allowed is not None and segment.skill not in allowed:
            continue
        start_path = split_dir / f"episode_{segment.start_idx:07d}.npz"
        end_path = split_dir / f"episode_{segment.end_idx:07d}.npz"
        if not start_path.exists() or not end_path.exists():
            continue
        rows.append({"segment": segment, "start_ep": load_episode(split_dir, segment.start_idx), "end_ep": load_episode(split_dir, segment.end_idx)})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


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
            inputs = self.processor(images=list(images[start : start + batch_size]), return_tensors="pt")
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

    def preprocess(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
        tensor = tensor.div(255.0).sub(0.5).div(0.5)
        return tensor.unsqueeze(1).to(self.device)

    @torch.no_grad()
    def generate_batch(self, images: Sequence[np.ndarray], texts: Sequence[str], seed: int) -> List[np.ndarray]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _, videos = self.encoder(
            self.preprocess(images),
            list(texts),
            self.timestep,
            1,
            all_layer=True,
            step_time=1,
            max_length=20,
            return_video=True,
            save_video=False,
        )
        futures = []
        for video in videos:
            futures.append(np.asarray(video[min(self.frame_index, len(video) - 1)], dtype=np.uint8))
        return futures


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


def predict_probe(model: SkillProbe, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation goal skill with skill probe prediction from GFDM future.")
    parser.add_argument("--calvin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--gfdm-video-model-path", required=True)
    parser.add_argument("--gfdm-text-encoder-path", required=True)
    parser.add_argument("--gfdm-device", default="cuda:0")
    parser.add_argument("--gfdm-frame-index", type=int, default=5)
    parser.add_argument("--gfdm-timestep", type=int, default=20)
    parser.add_argument("--gfdm-batch-size", type=int, default=1)
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
    val_rows = collect_rows(args.calvin_dir / "validation", args.max_val_samples, args.task_filter)
    skills = sorted({row["segment"].skill for row in train_rows})
    skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
    idx_to_skill = {idx: skill for skill, idx in skill_to_idx.items()}

    clip = ClipEncoder(args.clip_model_path, args.clip_device)
    train_future = clip.encode([row["end_ep"]["rgb_static"] for row in train_rows], args.clip_batch_size)
    y_train = np.asarray([skill_to_idx[row["segment"].skill] for row in train_rows], dtype=np.int64)
    mean = train_future.mean(axis=0, keepdims=True)
    std = train_future.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    probe = train_probe((train_future - mean) / std, y_train, args)

    # Sanity-check the probe on GT validation futures for the same rows.
    gt_val_future = clip.encode([row["end_ep"]["rgb_static"] for row in val_rows], args.clip_batch_size)
    gt_logits = predict_probe(probe, (gt_val_future - mean) / std)
    gt_top1 = np.argmax(gt_logits, axis=1)
    y_val = np.asarray([skill_to_idx[row["segment"].skill] for row in val_rows], dtype=np.int64)
    gt_probe_acc = float(np.mean(gt_top1 == y_val)) if len(y_val) else 0.0

    gfdm = GfdmRgbGenerator(
        args.gfdm_video_model_path,
        args.gfdm_text_encoder_path,
        args.gfdm_device,
        args.gfdm_frame_index,
        args.gfdm_timestep,
    )

    rows_out = []
    correct = 0
    per_goal = defaultdict(lambda: {"match": 0, "total": 0, "future_top1": Counter()})
    for start in tqdm(range(0, len(val_rows), args.gfdm_batch_size), desc="GFDM validation futures"):
        batch = val_rows[start : start + args.gfdm_batch_size]
        future_images = gfdm.generate_batch(
            [row["start_ep"]["rgb_static"] for row in batch],
            [row["segment"].language for row in batch],
            args.seed + start,
        )
        future_feat = clip.encode(future_images, args.clip_batch_size)
        logits = predict_probe(probe, (future_feat - mean) / std)
        order = np.argsort(-logits, axis=1)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        for i, row in enumerate(batch):
            goal_skill = row["segment"].skill
            pred_skill = idx_to_skill[int(order[i, 0])]
            match = goal_skill == pred_skill
            correct += int(match)
            per_goal[goal_skill]["match"] += int(match)
            per_goal[goal_skill]["total"] += 1
            per_goal[goal_skill]["future_top1"][pred_skill] += 1
            rows_out.append(
                {
                    "segment_index": row["segment"].index,
                    "start_idx": row["segment"].start_idx,
                    "end_idx": row["segment"].end_idx,
                    "goal_skill": goal_skill,
                    "language": row["segment"].language,
                    "future_skill_top1": pred_skill,
                    "match": match,
                    "top1_prob": float(probs[i, order[i, 0]]),
                    "top3": [idx_to_skill[int(item)] for item in order[i, :3]],
                }
            )

    summary = {
        "task_filter": args.task_filter or None,
        "n_val": len(val_rows),
        "matches": correct,
        "mismatches": len(val_rows) - correct,
        "match_rate": correct / len(val_rows) if val_rows else 0.0,
        "gt_future_probe_acc_on_same_rows": gt_probe_acc,
        "per_goal": {
            goal: {
                "match": item["match"],
                "total": item["total"],
                "match_rate": item["match"] / item["total"] if item["total"] else 0.0,
                "future_top1_counts": dict(item["future_top1"].most_common()),
            }
            for goal, item in sorted(per_goal.items())
        },
        "config": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "gfdm_frame_index": args.gfdm_frame_index,
            "gfdm_timestep": args.gfdm_timestep,
            "gfdm_batch_size": args.gfdm_batch_size,
            "probe_feature": "CLIP(GT future rgb_static) -> skill; evaluated on CLIP(GFDM future rgb_static)",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_out))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
