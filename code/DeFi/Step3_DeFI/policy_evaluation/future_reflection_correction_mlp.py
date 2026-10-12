import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_vjepa2_moe_future import MoeFuture, STAGE_NAMES, infer_stage, topk_same_task


def load_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def parse_task_list(raw: str) -> List[str]:
    return [task.strip() for task in str(raw).split(",") if task.strip()]


def filter_feature_split(
    start: np.ndarray,
    end: np.ndarray,
    tasks: np.ndarray,
    include_tasks: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not include_tasks:
        return start, end, tasks
    keep = np.asarray([str(task) in set(include_tasks) for task in tasks], dtype=bool)
    return start[keep], end[keep], tasks[keep]


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    )


def load_moe(path: Path, device: torch.device) -> Tuple[MoeFuture, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    model = MoeFuture(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["heads"]),
        int(checkpoint["layers"]),
        int(checkpoint["num_experts"]),
        bool(checkpoint.get("future_aware_router", False)),
        len(checkpoint.get("task_vocab", {})),
        len(checkpoint.get("stage_names", STAGE_NAMES)),
        bool(checkpoint.get("condition_router", False)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.task_vocab = checkpoint.get("task_vocab", {})
    return model.to(device).eval(), checkpoint


def ids_for_tasks(tasks: Sequence[str], checkpoint: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if not bool(checkpoint.get("condition_router", False)):
        return None, None
    vocab = checkpoint.get("task_vocab", {})
    task_ids = torch.tensor([int(vocab.get(str(task), 0)) for task in tasks], dtype=torch.long, device=device)
    stage_ids = torch.tensor([infer_stage(str(task)) for task in tasks], dtype=torch.long, device=device)
    return task_ids, stage_ids


@torch.no_grad()
def predict_split(
    model: MoeFuture,
    checkpoint: Dict[str, Any],
    train_start: np.ndarray,
    train_end: np.ndarray,
    train_tasks: np.ndarray,
    split_start: np.ndarray,
    split_tasks: np.ndarray,
    split_name: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = int(checkpoint.get("k", 8))
    exclude_self = split_name == "train" and len(split_start) == len(train_start)
    knn = topk_same_task(split_start, split_tasks, train_start, train_tasks, k, exclude_self)
    task_ids, stage_ids = ids_for_tasks(split_tasks, checkpoint, device)
    futures, logits = model(
        torch.from_numpy(split_start).to(device),
        torch.from_numpy(train_start[knn]).to(device),
        torch.from_numpy(train_end[knn]).to(device),
        task_ids,
        stage_ids,
    )
    return futures.cpu().numpy(), logits.cpu().numpy(), knn


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def reflection_text(row: Dict[str, Any]) -> str:
    consistent = bool(row.get("consistent", True))
    corrected_task = row.get("corrected_task")
    desc = row.get("corrected_future_description")
    hyp = row.get("hypothesis_task") or row.get("future_hypothesis", {}).get("nearest_train_future_task")
    task = row.get("task")
    if consistent:
        return f"The future hypothesis {hyp} is consistent with task {task}."
    if corrected_task:
        return f"The future hypothesis {hyp} is inconsistent with task {task}; correct it toward {corrected_task}. {desc or ''}".strip()
    return f"The future hypothesis {hyp} is inconsistent with task {task}; correct it toward the requested task."


RANDOM_REFLECTION_TEMPLATES = [
    "The future hypothesis needs review.",
    "This future description may require adjustment.",
    "The predicted future is not fully aligned.",
    "Consider revising the future hypothesis.",
    "The current future statement is uncertain.",
    "The predicted next state should be checked.",
    "This reflection is intentionally generic.",
    "The future proposal might be inconsistent.",
]


def random_reflection_text(key: Tuple[int, int], task: str, mode: str) -> str:
    payload = f"{mode}|{key[0]}|{key[1]}|{task}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    idx = int.from_bytes(digest[:4], "little") % len(RANDOM_REFLECTION_TEMPLATES)
    if mode == "random":
        return RANDOM_REFLECTION_TEMPLATES[idx]
    if mode == "random_task":
        return f"{RANDOM_REFLECTION_TEMPLATES[idx]} Task: {task}."
    raise ValueError(f"unknown random reflection mode: {mode}")


def load_reflections(path: Path) -> Dict[Tuple[int, int], str]:
    rows = {}
    for row in iter_jsonl(path):
        rows[(int(row["sample_index"]), int(row["expert"]))] = reflection_text(row)
    return rows


def make_default_reflections(tasks: Sequence[str], experts: Sequence[int]) -> Dict[Tuple[int, int], str]:
    rows = {}
    for sample_idx, task in enumerate(tasks):
        for expert in experts:
            rows[(sample_idx, expert)] = f"Reflect on whether the future is consistent with task {task}; correct toward {task} if needed."
    return rows


@torch.no_grad()
def encode_reflections_t5(texts: Sequence[str], t5_path: Path, device: torch.device, batch_size: int) -> np.ndarray:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(str(t5_path))
    encoder = T5EncoderModel.from_pretrained(str(t5_path)).to(device).eval()
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        hidden = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        outs.append(F.normalize(pooled.float(), dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def encode_reflections_qwen(texts: Sequence[str], qwen_path: Path, device: torch.device, batch_size: int) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    encoder = AutoModelForCausalLM.from_pretrained(
        str(qwen_path),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device).eval()
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        outputs = encoder(
            **tokens,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        outs.append(F.normalize(pooled.float(), dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


class CorrectionMLP(nn.Module):
    def __init__(self, future_dim: int, reflection_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(future_dim + reflection_dim),
            nn.Linear(future_dim + reflection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, future: torch.Tensor, reflection: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat([future, reflection], dim=-1)), dim=-1)


def build_examples(
    futures: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    tasks: Sequence[str],
    reflections: Dict[Tuple[int, int], str],
    mode: str,
    reflection_text_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple[int, int]]]:
    if mode == "router":
        experts_by_sample = np.argmax(logits, axis=1)[:, None]
    elif mode == "all":
        experts_by_sample = np.tile(np.arange(futures.shape[1])[None, :], (len(futures), 1))
    else:
        raise ValueError(f"unknown mode: {mode}")
    future_rows, target_rows, texts, keys = [], [], [], []
    for sample_idx, expert_row in enumerate(experts_by_sample):
        for expert in expert_row:
            key = (int(sample_idx), int(expert))
            if reflection_text_mode == "prompt":
                text = reflections.get(key) or f"Correct this future toward task {tasks[sample_idx]}."
            elif reflection_text_mode in {"random", "random_task"}:
                text = random_reflection_text(key, str(tasks[sample_idx]), reflection_text_mode)
            elif reflection_text_mode == "shuffle":
                text = reflections.get(key) or f"Correct this future toward task {tasks[sample_idx]}."
            else:
                raise ValueError(f"unknown reflection_text_mode: {reflection_text_mode}")
            future_rows.append(futures[sample_idx, int(expert)])
            target_rows.append(targets[sample_idx])
            texts.append(text)
            keys.append(key)
    return (
        np.asarray(future_rows, dtype=np.float32),
        np.asarray(target_rows, dtype=np.float32),
        cosine_rows(np.asarray(future_rows, dtype=np.float32), np.asarray(target_rows, dtype=np.float32)),
        texts,
        keys,
    )


def summarize(name: str, values: np.ndarray) -> Dict[str, float]:
    return {
        f"{name}_mean": float(values.mean()),
        f"{name}_median": float(np.median(values)),
        f"{name}_min": float(values.min()),
        f"{name}_max": float(values.max()),
    }


def router_input_from_futures(
    moe: MoeFuture,
    current: np.ndarray,
    futures: np.ndarray,
    tasks: Sequence[str],
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    current_t = torch.from_numpy(current.astype(np.float32)).to(device)
    futures_t = torch.from_numpy(futures.astype(np.float32)).to(device)
    expanded = current_t.unsqueeze(1).expand(-1, futures_t.shape[1], -1)
    parts = [
        current_t,
        futures_t.reshape(futures_t.shape[0], -1),
        (futures_t - expanded).reshape(futures_t.shape[0], -1),
    ]
    if bool(checkpoint.get("condition_router", False)):
        task_ids, stage_ids = ids_for_tasks(tasks, checkpoint, device)
        assert task_ids is not None and stage_ids is not None
        parts.extend([moe.task_embed(task_ids), moe.stage_embed(stage_ids)])
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def frozen_router_cosine(
    moe: MoeFuture,
    checkpoint: Dict[str, Any],
    current: np.ndarray,
    futures: np.ndarray,
    targets: np.ndarray,
    tasks: Sequence[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if not bool(checkpoint.get("future_aware_router", False)):
        raise ValueError("frozen corrected router eval requires a future-aware router")
    router_in = router_input_from_futures(moe, current, futures, tasks, checkpoint, device)
    logits = moe.router(router_in).detach().cpu().numpy()
    idx = np.argmax(logits, axis=1)
    chosen = futures[np.arange(len(futures)), idx]
    return cosine_rows(chosen, targets), idx


def train_new_router(
    moe: MoeFuture,
    checkpoint: Dict[str, Any],
    train_current: np.ndarray,
    train_futures: np.ndarray,
    train_targets: np.ndarray,
    train_tasks: Sequence[str],
    val_current: np.ndarray,
    val_futures: np.ndarray,
    val_targets: np.ndarray,
    val_tasks: Sequence[str],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not bool(checkpoint.get("future_aware_router", False)):
        raise ValueError("new router eval requires a future-aware router")
    train_input = router_input_from_futures(moe, train_current, train_futures, train_tasks, checkpoint, device).detach()
    val_input = router_input_from_futures(moe, val_current, val_futures, val_tasks, checkpoint, device).detach()
    train_sims = np.einsum("bed,bd->be", train_futures, train_targets)
    labels = torch.from_numpy(np.argmax(train_sims, axis=1).astype(np.int64)).to(device)
    router = nn.Sequential(
        nn.LayerNorm(train_input.shape[1]),
        nn.Linear(train_input.shape[1], int(checkpoint["hidden_dim"])),
        nn.GELU(),
        nn.Linear(int(checkpoint["hidden_dim"]), train_futures.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -float("inf")
    yv = torch.from_numpy(val_targets.astype(np.float32)).to(device)
    vf = torch.from_numpy(val_futures.astype(np.float32)).to(device)
    for _ in range(epochs):
        router.train()
        logits = router(train_input)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        router.eval()
        with torch.no_grad():
            val_idx = router(val_input).argmax(dim=1)
            val_chosen = vf[torch.arange(len(vf), device=device), val_idx]
            val_cos = torch.sum(val_chosen * yv, dim=-1).mean().item()
        if val_cos > best_val:
            best_val = val_cos
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
    if best_state is not None:
        router.load_state_dict(best_state)
    router.eval()
    with torch.no_grad():
        idx = router(val_input).argmax(dim=1).detach().cpu().numpy()
    chosen = val_futures[np.arange(len(val_futures)), idx]
    return cosine_rows(chosen, val_targets), idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CorrectionMLP([Future_i, reflection]) -> Future_i'.")
    parser.add_argument("--features", type=Path, default=Path("outputs/vjepa2_retrieval_probe_n200/features.npz"))
    parser.add_argument("--moe", type=Path, default=Path("outputs/vjepa2_moe_future_aware_router_k8_e4_balanced/vjepa2_moe_future.pt"))
    parser.add_argument("--train-reflections", type=Path, required=True)
    parser.add_argument("--val-reflections", type=Path, required=True)
    parser.add_argument("--t5-path", type=Path, default=Path("ckpts/t5_base"))
    parser.add_argument("--qwen-path", type=Path, default=None)
    parser.add_argument("--reflection-encoder-type", choices=["t5", "qwen"], default="t5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["router", "all"], default="router")
    parser.add_argument("--reflection-text-mode", choices=["prompt", "shuffle", "random", "random_task"], default="prompt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--router-epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=0)
    parser.add_argument("--include-tasks", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    features = load_features(args.features)
    train_start = normalize_rows(features["train_start"].astype(np.float32))
    train_end = normalize_rows(features["train_end"].astype(np.float32))
    val_start = normalize_rows(features["val_start"].astype(np.float32))
    val_end = normalize_rows(features["val_end"].astype(np.float32))
    train_tasks = features["train_tasks"].astype(str)
    val_tasks = features["val_tasks"].astype(str)
    include_tasks = parse_task_list(args.include_tasks)
    train_start, train_end, train_tasks = filter_feature_split(train_start, train_end, train_tasks, include_tasks)
    val_start, val_end, val_tasks = filter_feature_split(val_start, val_end, val_tasks, include_tasks)
    if len(train_start) == 0 or len(val_start) == 0:
        raise ValueError("No train/val samples remain after --include-tasks filtering")

    moe, checkpoint = load_moe(args.moe, device)
    train_futures, train_logits, _ = predict_split(moe, checkpoint, train_start, train_end, train_tasks, train_start, train_tasks, "train", device)
    val_futures, val_logits, _ = predict_split(moe, checkpoint, train_start, train_end, train_tasks, val_start, val_tasks, "val", device)

    train_reflections = load_reflections(args.train_reflections)
    val_reflections = load_reflections(args.val_reflections)
    x_train, y_train, base_train_cos, train_texts, _ = build_examples(
        train_futures,
        train_logits,
        train_end,
        train_tasks,
        train_reflections,
        args.mode,
        args.reflection_text_mode,
    )
    x_val, y_val, base_val_cos, val_texts, val_keys = build_examples(
        val_futures,
        val_logits,
        val_end,
        val_tasks,
        val_reflections,
        args.mode,
        args.reflection_text_mode,
    )
    if args.reflection_text_mode == "shuffle":
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_texts)
        rng.shuffle(val_texts)
    if args.max_train_examples:
        keep = min(int(args.max_train_examples), len(x_train))
        x_train = x_train[:keep]
        y_train = y_train[:keep]
        base_train_cos = base_train_cos[:keep]
        train_texts = train_texts[:keep]
    if args.max_val_examples:
        keep = min(int(args.max_val_examples), len(x_val))
        x_val = x_val[:keep]
        y_val = y_val[:keep]
        base_val_cos = base_val_cos[:keep]
        val_texts = val_texts[:keep]
        val_keys = val_keys[:keep]

    if args.reflection_encoder_type == "qwen":
        if args.qwen_path is None:
            raise ValueError("--qwen-path is required when --reflection-encoder-type=qwen")
        r_train = encode_reflections_qwen(train_texts, args.qwen_path, device, args.text_batch_size)
        r_val = encode_reflections_qwen(val_texts, args.qwen_path, device, args.text_batch_size)
        reflection_encoder_desc = f"AutoModelForCausalLM:{args.qwen_path}"
    else:
        r_train = encode_reflections_t5(train_texts, args.t5_path, device, args.text_batch_size)
        r_val = encode_reflections_t5(val_texts, args.t5_path, device, args.text_batch_size)
        reflection_encoder_desc = f"T5EncoderModel:{args.t5_path}"

    model = CorrectionMLP(x_train.shape[1], r_train.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    rt = torch.from_numpy(r_train).to(device)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    rv = torch.from_numpy(r_val).to(device)

    best_state = None
    best_val = -float("inf")
    for _ in range(args.epochs):
        order = torch.randperm(len(xt), device=device)
        model.train()
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pred = model(xt[idx], rt[idx])
            loss = 1.0 - torch.sum(pred * yt[idx], dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(xv, rv)
            val_cos = torch.sum(val_pred * yv, dim=-1).mean().item()
        if val_cos > best_val:
            best_val = val_cos
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        corrected_val = model(xv, rv).cpu().numpy()
        corrected_train = model(xt, rt).cpu().numpy()
    corrected_val_cos = cosine_rows(corrected_val, y_val)
    corrected_train_cos = cosine_rows(corrected_train, y_train)
    delta_val = corrected_val_cos - base_val_cos

    metrics = {
        "mode": args.mode,
        "include_tasks": include_tasks,
        "num_train_examples": int(len(x_train)),
        "num_val_examples": int(len(x_val)),
        "reflection_encoder": reflection_encoder_desc,
        **summarize("train_base_cosine", base_train_cos),
        **summarize("train_corrected_cosine", corrected_train_cos),
        **summarize("val_base_cosine", base_val_cos),
        **summarize("val_corrected_cosine", corrected_val_cos),
        **summarize("val_delta_cosine", delta_val),
        "val_improved_rate": float(np.mean(delta_val > 0)),
        "val_degraded_rate": float(np.mean(delta_val < 0)),
    }

    if args.mode == "all" and not args.max_train_examples and not args.max_val_examples:
        num_train = len(train_start)
        num_val = len(val_start)
        num_experts = train_futures.shape[1]
        corrected_train_all = corrected_train.reshape(num_train, num_experts, -1)
        corrected_val_all = corrected_val.reshape(num_val, num_experts, -1)
        base_train_all = train_futures
        base_val_all = val_futures
        oracle4_base_val = np.max(np.einsum("bed,bd->be", base_val_all, val_end), axis=1)
        base_val_sims = np.einsum("bed,bd->be", base_val_all, val_end)
        corrected_val_sims = np.einsum("bed,bd->be", corrected_val_all, val_end)
        oracle4_corrected_val = np.max(corrected_val_sims, axis=1)
        base_oracle_idx = np.argmax(base_val_sims, axis=1)
        corrected_oracle_idx = np.argmax(corrected_val_sims, axis=1)
        frozen_corrected_cos, frozen_corrected_idx = frozen_router_cosine(
            moe, checkpoint, val_start, corrected_val_all, val_end, val_tasks, device
        )
        original_router_idx = np.argmax(val_logits, axis=1)
        original_router_cos = cosine_rows(base_val_all[np.arange(num_val), original_router_idx], val_end)
        new_router_cos, new_router_idx = train_new_router(
            moe,
            checkpoint,
            train_start,
            corrected_train_all,
            train_end,
            train_tasks,
            val_start,
            corrected_val_all,
            val_end,
            val_tasks,
            device,
            args.router_epochs,
            args.lr,
            args.weight_decay,
        )
        metrics.update(
            {
                **summarize("oracle4_base_val_cosine", oracle4_base_val),
                **summarize("oracle4_corrected_val_cosine", oracle4_corrected_val),
                **summarize("oracle4_corrected_minus_base_val_cosine", oracle4_corrected_val - oracle4_base_val),
                **summarize("original_router_base_val_cosine", original_router_cos),
                **summarize("frozen_router_corrected_val_cosine", frozen_corrected_cos),
                **summarize("frozen_router_corrected_minus_original_val_cosine", frozen_corrected_cos - original_router_cos),
                **summarize("new_router_corrected_val_cosine", new_router_cos),
                **summarize("new_router_corrected_minus_original_val_cosine", new_router_cos - original_router_cos),
                "original_router_matches_base_oracle_expert": float(np.mean(original_router_idx == base_oracle_idx)),
                "original_router_matches_corrected_oracle_expert": float(np.mean(original_router_idx == corrected_oracle_idx)),
                "frozen_router_matches_corrected_oracle_expert": float(np.mean(frozen_corrected_idx == corrected_oracle_idx)),
                "new_router_matches_corrected_oracle_expert": float(np.mean(new_router_idx == corrected_oracle_idx)),
                "frozen_router_changed_expert_rate": float(np.mean(frozen_corrected_idx != original_router_idx)),
                "new_router_changed_expert_rate": float(np.mean(new_router_idx != original_router_idx)),
                "new_router_epochs": int(args.router_epochs),
            }
        )
    elif args.mode == "all":
        metrics["subset_eval_note"] = "oracle/router all-expert metrics skipped because max-train/val-examples was used"

    torch.save(
        {
            "model_state": model.state_dict(),
            "future_dim": int(x_train.shape[1]),
            "reflection_dim": int(r_train.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "mode": args.mode,
        },
        args.output_dir / "correction_mlp.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "val_predictions.npz",
        base_future=x_val,
        corrected_future=corrected_val.astype(np.float32),
        gt_future=y_val,
        base_cosine=base_val_cos,
        corrected_cosine=corrected_val_cos,
        keys=np.asarray(val_keys, dtype=np.int64),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
