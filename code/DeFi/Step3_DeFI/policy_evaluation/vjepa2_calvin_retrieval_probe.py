import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Segment:
    task: str
    language: str
    start_idx: int
    end_idx: int


def parse_task_list(raw: str) -> List[str]:
    return [task.strip() for task in str(raw).split(",") if task.strip()]


def load_annotations(split_dir: Path) -> List[Segment]:
    data = np.load(split_dir / "lang_clip_resnet50" / "auto_lang_ann.npy", allow_pickle=True).item()
    return [
        Segment(str(task), str(ann), int(interval[0]), int(interval[1]))
        for task, ann, interval in zip(data["language"]["task"], data["language"]["ann"], data["info"]["indx"])
    ]


def load_episode(split_dir: Path, episode_idx: int) -> Optional[Dict[str, np.ndarray]]:
    path = split_dir / f"episode_{episode_idx:07d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def collect_rows(
    split_dir: Path,
    max_per_task: Optional[int],
    max_total: Optional[int],
    include_tasks: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    counts = defaultdict(int)
    selected = set(include_tasks or [])
    for seg in load_annotations(split_dir):
        if selected and seg.task not in selected:
            continue
        if max_per_task is not None and counts[seg.task] >= max_per_task:
            continue
        start_ep = load_episode(split_dir, seg.start_idx)
        end_ep = load_episode(split_dir, seg.end_idx)
        if start_ep is None or end_ep is None:
            continue
        rows.append({"segment": seg, "start_ep": start_ep, "end_ep": end_ep})
        counts[seg.task] += 1
        if max_total is not None and len(rows) >= max_total:
            break
    return rows


def preprocess_images(images: Sequence[np.ndarray], size: int, device: torch.device) -> torch.Tensor:
    tensors = []
    for image in images:
        image = cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    x = torch.stack(tensors, dim=0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


@torch.no_grad()
def encode_vjepa(model: torch.nn.Module, images: Sequence[np.ndarray], size: int, device: torch.device, batch_size: int) -> np.ndarray:
    out = []
    for start in tqdm(range(0, len(images), batch_size), desc="encode_vjepa", leave=False):
        x = preprocess_images(images[start : start + batch_size], size, device).unsqueeze(2)
        feat = model(x).mean(dim=1)
        feat = F.normalize(feat.float(), dim=-1)
        out.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )


def retrieve_same_task(
    val_start: np.ndarray,
    val_tasks: Sequence[str],
    train_start: np.ndarray,
    train_tasks: Sequence[str],
) -> np.ndarray:
    train_by_task: Dict[str, np.ndarray] = {}
    for task in sorted(set(train_tasks)):
        train_by_task[task] = np.asarray([idx for idx, item in enumerate(train_tasks) if item == task], dtype=np.int64)
    retrieved = []
    for feat, task in zip(val_start, val_tasks):
        idx = train_by_task.get(task)
        if idx is None or len(idx) == 0:
            retrieved.append(0)
            continue
        sims = train_start[idx] @ feat
        retrieved.append(int(idx[int(np.argmax(sims))]))
    return np.asarray(retrieved, dtype=np.int64)


def prototype_same_task(val_tasks: Sequence[str], train_tasks: Sequence[str]) -> np.ndarray:
    first = {}
    for idx, task in enumerate(train_tasks):
        first.setdefault(task, idx)
    return np.asarray([first.get(task, 0) for task in val_tasks], dtype=np.int64)


def summarize(name: str, retrieved: np.ndarray, val_end: np.ndarray, train_end: np.ndarray) -> Dict[str, float]:
    sims = cosine(train_end[retrieved], val_end)
    l2 = np.linalg.norm(train_end[retrieved] - val_end, axis=1)
    return {
        f"{name}_cosine_mean": float(sims.mean()),
        f"{name}_cosine_median": float(np.median(sims)),
        f"{name}_l2_mean": float(l2.mean()),
        f"{name}_l2_median": float(np.median(l2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V-JEPA2 CALVIN future retrieval sanity check.")
    parser.add_argument("--calvin-dir", type=Path, required=True)
    parser.add_argument("--vjepa-root", type=Path, default=Path("/mnt/data/manipulation/vjepa2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=5)
    parser.add_argument("--val-max", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--include-tasks", default="")
    args = parser.parse_args()

    sys.path.insert(0, args.vjepa_root.as_posix())
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    encoder, _ = torch.hub.load(args.vjepa_root.as_posix(), "vjepa2_1_vit_base_384", source="local")
    encoder = encoder.to(device).eval()

    include_tasks = parse_task_list(args.include_tasks)
    train_rows = collect_rows(args.calvin_dir / "training", args.train_per_task, None, include_tasks)
    val_rows = collect_rows(args.calvin_dir / "validation", None, args.val_max, include_tasks)
    train_tasks = [row["segment"].task for row in train_rows]
    val_tasks = [row["segment"].task for row in val_rows]

    train_start = encode_vjepa(encoder, [row["start_ep"]["rgb_static"] for row in train_rows], args.image_size, device, args.batch_size)
    train_end = encode_vjepa(encoder, [row["end_ep"]["rgb_static"] for row in train_rows], args.image_size, device, args.batch_size)
    val_start = encode_vjepa(encoder, [row["start_ep"]["rgb_static"] for row in val_rows], args.image_size, device, args.batch_size)
    val_end = encode_vjepa(encoder, [row["end_ep"]["rgb_static"] for row in val_rows], args.image_size, device, args.batch_size)

    proto_idx = prototype_same_task(val_tasks, train_tasks)
    nn_idx = retrieve_same_task(val_start, val_tasks, train_start, train_tasks)
    current_to_future = cosine(val_start, val_end)

    metrics = {
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "include_tasks": include_tasks,
        "train_per_task": args.train_per_task,
        "model": "vjepa2_1_vit_base_384 encoder",
        "encoder_params": int(sum(p.numel() for p in encoder.parameters())),
        "current_to_gt_future_cosine_mean": float(current_to_future.mean()),
        "current_to_gt_future_cosine_median": float(np.median(current_to_future)),
    }
    metrics.update(summarize("prototype_same_task", proto_idx, val_end, train_end))
    metrics.update(summarize("vjepa_nn_same_task", nn_idx, val_end, train_end))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "features.npz",
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        train_tasks=np.asarray(train_tasks),
        val_tasks=np.asarray(val_tasks),
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
