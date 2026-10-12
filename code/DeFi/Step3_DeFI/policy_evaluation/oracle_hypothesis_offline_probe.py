import argparse
import json
import random
import shutil
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
    allowed = None if not task_filter else {item.strip() for item in task_filter.split(",") if item.strip()}
    rows = []
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
        return [np.asarray(video[min(self.frame_index, len(video) - 1)], dtype=np.uint8) for video in videos]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline oracle selection over Continue/Retry/Recover future hypotheses.")
    parser.add_argument("--calvin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-filter", default="stack_block,push_into_drawer,push_blue_block_right,push_pink_block_right,push_red_block_right")
    parser.add_argument("--max-val-samples", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=1000)
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = collect_rows(args.calvin_dir / "training", args.max_train_samples, args.task_filter)
    val_rows = collect_rows(args.calvin_dir / "validation", args.max_val_samples, args.task_filter)
    clip = ClipEncoder(args.clip_model_path, args.clip_device)

    train_obs = clip.encode([row["start_ep"]["rgb_static"] for row in train_rows], args.clip_batch_size)
    train_future = clip.encode([row["end_ep"]["rgb_static"] for row in train_rows], args.clip_batch_size)
    val_obs = clip.encode([row["start_ep"]["rgb_static"] for row in val_rows], args.clip_batch_size)
    val_gt = clip.encode([row["end_ep"]["rgb_static"] for row in val_rows], args.clip_batch_size)

    gfdm = GfdmRgbGenerator(args.gfdm_video_model_path, args.gfdm_text_encoder_path, args.gfdm_device, args.gfdm_frame_index, args.gfdm_timestep)
    continue_images = []
    for start in tqdm(range(0, len(val_rows), args.gfdm_batch_size), desc="GFDM continue"):
        batch = val_rows[start : start + args.gfdm_batch_size]
        continue_images.extend(gfdm.generate_batch([row["start_ep"]["rgb_static"] for row in batch], [row["segment"].language for row in batch], args.seed + start))
    continue_feat = clip.encode(continue_images, args.clip_batch_size)

    records = []
    selected = Counter()
    for i, row in enumerate(val_rows):
        same_task = [idx for idx, train_row in enumerate(train_rows) if train_row["segment"].skill == row["segment"].skill]
        if same_task:
            sims = train_obs[same_task] @ val_obs[i]
            nearest = same_task[int(np.argmax(sims))]
        else:
            nearest = int(np.argmax(train_obs @ val_obs[i]))
        h = {
            "continue": continue_feat[i],
            "retry": train_future[nearest],
            "recover": val_obs[i],
        }
        distances = {name: float(np.linalg.norm(feat - val_gt[i])) for name, feat in h.items()}
        cosines = {name: float(np.dot(feat, val_gt[i])) for name, feat in h.items()}
        best = min(distances, key=distances.get)
        selected[best] += 1
        records.append(
            {
                "index": i,
                "task": row["segment"].skill,
                "language": row["segment"].language,
                "start_idx": row["segment"].start_idx,
                "end_idx": row["segment"].end_idx,
                "nearest_train_start_idx": train_rows[nearest]["segment"].start_idx,
                "nearest_train_end_idx": train_rows[nearest]["segment"].end_idx,
                "distances": distances,
                "cosines": cosines,
                "oracle_choice": best,
                "oracle_l2": distances[best],
                "defi_continue_l2": distances["continue"],
                "improvement_abs": distances["continue"] - distances[best],
                "improvement_pct": (distances["continue"] - distances[best]) / max(distances["continue"], 1e-8),
            }
        )

    improvements = np.asarray([row["improvement_abs"] for row in records], dtype=np.float64)
    continue_l2 = np.asarray([row["defi_continue_l2"] for row in records], dtype=np.float64)
    oracle_l2 = np.asarray([row["oracle_l2"] for row in records], dtype=np.float64)
    summary = {
        "n": len(records),
        "task_filter": args.task_filter,
        "hypotheses": {
            "continue": "GFDM future from current obs and goal",
            "retry": "GT future of nearest same-task training demo state by CLIP(obs)",
            "recover": "current obs as rewind/recover proxy",
        },
        "oracle_choice_counts": dict(selected.most_common()),
        "continue_l2_mean": float(continue_l2.mean()),
        "oracle_l2_mean": float(oracle_l2.mean()),
        "improvement_abs_mean": float(improvements.mean()),
        "improvement_pct_mean": float(np.mean([row["improvement_pct"] for row in records])),
        "improved_fraction": float(np.mean(improvements > 1e-8)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
