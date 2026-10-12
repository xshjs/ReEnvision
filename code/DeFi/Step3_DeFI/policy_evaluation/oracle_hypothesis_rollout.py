import argparse
import importlib.machinery
import importlib.util
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

if "distutils.spawn" not in sys.modules:
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    spawn_module.spawn = lambda cmd, search_path=1, verbose=0, dry_run=0, env=None: subprocess.check_call(
        cmd,
        env=env,
    )
    sys.modules["distutils.spawn"] = spawn_module

try:
    import huggingface_hub as _hf_hub

    if not hasattr(_hf_hub, "cached_download") and hasattr(_hf_hub, "hf_hub_download"):
        def _cached_download(*args, **kwargs):
            return _hf_hub.hf_hub_download(*args, **kwargs)

        _hf_hub.cached_download = _cached_download
except Exception:
    pass

try:
    import transformers.utils as _tf_utils

    if not hasattr(_tf_utils, "FLAX_WEIGHTS_NAME"):
        _tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if hasattr(_tf_utils, "check_torch_load_is_safe"):
        _tf_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import cv2
import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict
from tqdm.auto import tqdm
from policy_evaluation.defi_memory_models import iter_jsonl

if os.environ.get("DEFI_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False

apex_module = types.ModuleType("apex")
normalization_module = types.ModuleType("apex.normalization")
apex_module.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
normalization_module.__spec__ = importlib.machinery.ModuleSpec("apex.normalization", loader=None)
normalization_module.FusedRMSNorm = torch.nn.LayerNorm
normalization_module.FusedLayerNorm = torch.nn.LayerNorm
apex_module.normalization = normalization_module
sys.modules["apex"] = apex_module
sys.modules["apex.normalization"] = normalization_module

if "pytorch_lightning" not in sys.modules:
    pl_module = types.ModuleType("pytorch_lightning")

    class _LightningModule(torch.nn.Module):
        @property
        def device(self) -> torch.device:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def freeze(self) -> None:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.eval()

        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    def _seed_everything(seed: int, workers: bool = False) -> int:
        del workers
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    pl_module.__version__ = "shim"
    pl_module.LightningModule = _LightningModule
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

if "pyhash" not in sys.modules:
    pyhash_module = types.ModuleType("pyhash")

    class _Fnv1_32:
        def __call__(self, value: Any) -> int:
            hval = 0x811C9DC5
            for byte in str(value).encode("utf-8"):
                hval = (hval * 0x01000193) & 0xFFFFFFFF
                hval ^= byte
            return hval

    pyhash_module.fnv1_32 = _Fnv1_32
    sys.modules["pyhash"] = pyhash_module

if "gym" not in sys.modules:
    gym_module = types.ModuleType("gym")
    gym_utils_module = types.ModuleType("gym.utils")
    gym_seeding_module = types.ModuleType("gym.utils.seeding")

    def _np_random(seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        return rng, seed if seed is not None else 0

    class _Wrapper:
        def __init__(self, env: Any):
            self.env = env

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)

    class _Env:
        pass

    gym_seeding_module.np_random = _np_random
    gym_utils_module.seeding = gym_seeding_module
    gym_module.Env = _Env
    gym_module.Wrapper = _Wrapper
    gym_module.utils = gym_utils_module
    sys.modules["gym"] = gym_module
    sys.modules["gym.utils"] = gym_utils_module
    sys.modules["gym.utils.seeding"] = gym_seeding_module

if importlib.util.find_spec("ftfy") is None and "ftfy" not in sys.modules:
    ftfy_module = types.ModuleType("ftfy")
    ftfy_module.__spec__ = importlib.util.spec_from_loader("ftfy", loader=None)
    ftfy_module.fix_text = lambda text, *args, **kwargs: text
    sys.modules["ftfy"] = ftfy_module

try:
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    import_utils._apex_available = False
    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from pytorch_lightning import seed_everything

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition


SELECTOR_CLASSES = ["continue", "retry_demo", "recover"]
VJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
VJEPA_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SelectorMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReflectionCritic(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FutureCorrectionMLP(torch.nn.Module):
    def __init__(self, future_dim: int, reflection_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(future_dim + reflection_dim),
            torch.nn.Linear(future_dim + reflection_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, future_dim),
        )

    def forward(self, future: torch.Tensor, reflection: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(torch.cat([future, reflection], dim=-1)), dim=-1)


@dataclass
class MemoryReflectionDecision:
    trust_score: float
    mismatch_type: str
    correction_direction: str
    recoverability: str
    explanation: str
    raw_text: str
    hypothetical_failure: str = ""
    counterfactual_future: str = ""
    failure_factor: str = ""
    intervention: str = ""


def trust_label_to_score(value: Any) -> float:
    if isinstance(value, str):
        label = value.strip().lower()
        if label == "high":
            return 0.85
        if label == "medium":
            return 0.5
        if label == "low":
            return 0.2
    try:
        return float(value)
    except Exception:
        return 0.5


def trust_score_to_label(score: float) -> str:
    score = float(score)
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


RULE_MEMORY = {
    "stack_block": "Preserve object identity and alignment; avoid overestimating stack completion before stable contact.",
    "push_into_drawer": "Do not assume drawer progress or block displacement too early; strengthen sustained contact before predicting insertion.",
    "push_pink_block_right": "Reduce over-optimistic lateral displacement; require stronger side contact before predicting rightward motion.",
    "push_red_block_right": "Reduce over-optimistic lateral displacement; require stronger side contact before predicting rightward motion.",
    "push_blue_block_right": "Reduce over-optimistic lateral displacement; require stronger side contact before predicting rightward motion.",
    "lift_blue_block_drawer": "Do not assume successful grasp from cluttered drawer states; preserve gripper-object alignment until lift is realized.",
    "lift_pink_block_drawer": "Do not assume successful grasp from cluttered drawer states; preserve gripper-object alignment until lift is realized.",
}


class T5ReflectionEncoder:
    def __init__(self, t5_path: Path, device: torch.device) -> None:
        from transformers import T5EncoderModel, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(str(t5_path))
        self.encoder = T5EncoderModel.from_pretrained(str(t5_path)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled.float(), dim=-1)


class QwenLoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(int(rank), 1)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(x).to(self.lora_a.weight.dtype)))
        residual = residual.to(self.base.weight.dtype)
        base_out = self.base(x.to(self.base.weight.dtype))
        return base_out + residual * self.scaling


def replace_linear_with_qwen_lora(module: torch.nn.Module, rank: int, alpha: float, dropout: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, QwenLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += replace_linear_with_qwen_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return count


class ExternalQwenReflectionWorker:
    def __init__(
        self,
        python_bin: Path,
        qwen_path: Path,
        device: torch.device,
        lora_path: Optional[Path] = None,
    ) -> None:
        worker_script = Path(__file__).with_name("qwen_reflection_worker.py")
        cmd = [
            str(python_bin),
            str(worker_script),
            "--model-path",
            str(qwen_path),
            "--device",
            str(device),
        ]
        if lora_path is not None and lora_path.exists():
            cmd.extend(["--lora-path", str(lora_path)])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self.proc.stdout.readline().strip() if self.proc.stdout is not None else ""
        if not ready:
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
            raise RuntimeError(f"failed to start Qwen worker: {stderr[:800]}")
        payload = json.loads(ready)
        if payload.get("status") != "ready":
            raise RuntimeError(f"unexpected Qwen worker init response: {payload}")

    def generate_reflection(self, prompt: str, max_new_tokens: int = 96) -> str:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
            raise RuntimeError(f"Qwen worker returned empty output: {stderr[:800]}")
        payload = json.loads(line)
        if not payload.get("ok", False):
            raise RuntimeError(f"Qwen worker error: {payload.get('error', 'unknown')}")
        return str(payload.get("text", "")).strip()

    def close(self) -> None:
        if getattr(self, "proc", None) is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class QwenReflectionEncoder:
    def __init__(
        self,
        qwen_path: Path,
        device: torch.device,
        lora_path: Optional[Path] = None,
        python_bin: Optional[Path] = None,
    ) -> None:
        self.worker: Optional[ExternalQwenReflectionWorker] = None
        if python_bin is not None and python_bin.exists():
            self.worker = ExternalQwenReflectionWorker(
                python_bin=python_bin,
                qwen_path=qwen_path,
                device=device,
                lora_path=lora_path,
            )
            self.device = device
            self.tokenizer = None
            self.encoder = None
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.encoder = AutoModelForCausalLM.from_pretrained(
            str(qwen_path),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        if lora_path is not None and lora_path.exists():
            checkpoint = torch.load(lora_path, map_location="cpu")
            train_args = dict(checkpoint.get("args", {}))
            rank = int(train_args.get("lora_rank", 8))
            alpha = float(train_args.get("lora_alpha", 16.0))
            dropout = float(train_args.get("lora_dropout", 0.05))
            replace_linear_with_qwen_lora(self.encoder, rank=rank, alpha=alpha, dropout=dropout)
            lora_state = checkpoint.get("lora_state", {})
            missing, unexpected = self.encoder.load_state_dict(lora_state, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected Qwen LoRA keys: {unexpected[:8]}")
            missing = [key for key in missing if ".lora_a." not in key and ".lora_b." not in key]
            if missing:
                print(f"[WARN] Qwen LoRA load missing non-LoRA keys count={len(missing)}", flush=True)
        self.encoder = self.encoder.to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if self.worker is not None:
            raise RuntimeError("encode() is not supported when Qwen runs in external worker mode")
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.encoder(
            **tokens,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled.float(), dim=-1)

    @torch.no_grad()
    def generate_reflection(
        self,
        prompt: str,
        max_new_tokens: int = 96,
    ) -> str:
        if self.worker is not None:
            return self.worker.generate_reflection(prompt, max_new_tokens=max_new_tokens)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.encoder.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][tokens["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text or "progress_status: uncertain | failure_type: uncertain | prefer_next: continue | avoid_next: repeat failure | correction_direction: stabilize contact | memory_query: uncertain"

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


class VjepaFutureAdapter(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=-1)


class VjepaCrossAttentionFuture(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.query_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.demo_proj = torch.nn.Linear(input_dim * 3, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.MultiheadAttention(hidden_dim, heads, batch_first=True),
                        "norm1": torch.nn.LayerNorm(hidden_dim),
                        "ff": torch.nn.Sequential(
                            torch.nn.Linear(hidden_dim, hidden_dim * 4),
                            torch.nn.GELU(),
                            torch.nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                        "norm2": torch.nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.out = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, current: torch.Tensor, demo_current: torch.Tensor, demo_future: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(current).unsqueeze(1)
        demo_delta = demo_future - demo_current
        memory = self.demo_proj(torch.cat([demo_current, demo_future, demo_delta], dim=-1))
        for layer in self.layers:
            attended, _ = layer["attn"](query, memory, memory, need_weights=False)
            query = layer["norm1"](query + attended)
            query = layer["norm2"](query + layer["ff"](query))
        return torch.nn.functional.normalize(self.out(query.squeeze(1)), dim=-1)


STAGE_NAMES = ["approach", "grasp", "transport", "insert", "release"]


def infer_stage(task: str) -> int:
    task = str(task)
    if "lift" in task:
        return STAGE_NAMES.index("grasp")
    if "place_in_drawer" in task or "place_in_slider" in task or "push_into_drawer" in task:
        return STAGE_NAMES.index("insert")
    if "push" in task or "move_slider" in task:
        return STAGE_NAMES.index("transport")
    if "stack" in task or "unstack" in task or "rotate" in task:
        return STAGE_NAMES.index("release")
    return STAGE_NAMES.index("approach")


class VjepaMoeFuture(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        num_experts: int,
        future_aware_router: bool = False,
        num_tasks: int = 0,
        num_stages: int = 0,
        condition_router: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.future_aware_router = future_aware_router
        self.condition_router = condition_router
        self.query_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.demo_proj = torch.nn.Linear(input_dim * 3, hidden_dim)
        if condition_router:
            self.task_embed = torch.nn.Embedding(num_tasks, hidden_dim)
            self.stage_embed = torch.nn.Embedding(num_stages, hidden_dim)
        else:
            self.task_embed = None
            self.stage_embed = None
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.MultiheadAttention(hidden_dim, heads, batch_first=True),
                        "norm1": torch.nn.LayerNorm(hidden_dim),
                        "ff": torch.nn.Sequential(
                            torch.nn.Linear(hidden_dim, hidden_dim * 4),
                            torch.nn.GELU(),
                            torch.nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                        "norm2": torch.nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(layers)
            ]
        )
        if future_aware_router:
            router_dim = input_dim * (1 + num_experts * 2)
            if condition_router:
                router_dim += hidden_dim * 2
            self.router = torch.nn.Sequential(
                torch.nn.LayerNorm(router_dim),
                torch.nn.Linear(router_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, num_experts),
            )
        else:
            self.router = torch.nn.Sequential(torch.nn.LayerNorm(hidden_dim), torch.nn.Linear(hidden_dim, num_experts))
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        current: torch.Tensor,
        demo_current: torch.Tensor,
        demo_future: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        stage_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(current).unsqueeze(1)
        demo_delta = demo_future - demo_current
        memory = self.demo_proj(torch.cat([demo_current, demo_future, demo_delta], dim=-1))
        for layer in self.layers:
            attended, _ = layer["attn"](query, memory, memory, need_weights=False)
            query = layer["norm1"](query + attended)
            query = layer["norm2"](query + layer["ff"](query))
        hidden = query.squeeze(1)
        futures = torch.stack([torch.nn.functional.normalize(expert(hidden), dim=-1) for expert in self.experts], dim=1)
        if self.future_aware_router:
            current_expanded = current.unsqueeze(1).expand(-1, self.num_experts, -1)
            router_parts = [
                current,
                futures.reshape(futures.shape[0], -1),
                (futures - current_expanded).reshape(futures.shape[0], -1),
            ]
            if self.condition_router:
                if task_ids is None or stage_ids is None:
                    raise ValueError("task_ids and stage_ids are required for condition_router")
                router_parts.extend([self.task_embed(task_ids), self.stage_embed(stage_ids)])
            router_input = torch.cat(router_parts, dim=-1)
            logits = self.router(router_input)
        else:
            logits = self.router(hidden)
        return futures, logits


class UniversalMoeOracleRouter(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_embed = torch.nn.Embedding(num_experts, 16)
        self.score = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_dim + 16),
            torch.nn.Linear(feature_dim + 16, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_experts, _ = x.shape
        expert_ids = torch.arange(num_experts, device=x.device).unsqueeze(0).expand(batch, -1)
        expert_embed = self.expert_embed(expert_ids)
        return self.score(torch.cat([x, expert_embed], dim=-1)).squeeze(-1)


class FuturePortfolioSelector(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_experts: int,
        utility_score_weight: float = 1.0,
        viability_score_weight: float = 0.1,
        score_step_lambda: float = 0.05,
        execution_dim: int = 0,
        use_execution_residual_correction: bool = False,
        execution_correction_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.utility_score_weight = float(utility_score_weight)
        self.viability_score_weight = float(viability_score_weight)
        self.score_step_lambda = float(score_step_lambda)
        self.execution_dim = int(execution_dim)
        self.use_execution_residual_correction = bool(use_execution_residual_correction)
        self.execution_correction_scale = float(execution_correction_scale)
        self.expert_embed = torch.nn.Embedding(num_experts, 16)
        self.backbone = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_dim + 16),
            torch.nn.Linear(feature_dim + 16, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.viability = torch.nn.Linear(hidden_dim, 1)
        self.steps = torch.nn.Linear(hidden_dim, 1)
        self.utility = torch.nn.Linear(hidden_dim, 1)
        self.recovery = torch.nn.Linear(hidden_dim, 1)
        if self.execution_dim > 0:
            self.execution_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(self.execution_dim),
                torch.nn.Linear(self.execution_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.execution_readout = torch.nn.Linear(hidden_dim, hidden_dim)
            self.utility_correction = torch.nn.Linear(hidden_dim * 2, 1)
            self.recoverability_head = torch.nn.Linear(hidden_dim, 1)
        else:
            self.execution_proj = None
            self.execution_readout = None
            self.utility_correction = None
            self.recoverability_head = None

    def forward(self, x: torch.Tensor, execution: torch.Tensor | None = None) -> torch.Tensor:
        batch, num_experts, _ = x.shape
        expert_ids = torch.arange(num_experts, device=x.device).unsqueeze(0).expand(batch, -1)
        expert_embed = self.expert_embed(expert_ids)
        base_hidden = self.backbone(torch.cat([x, expert_embed], dim=-1))
        hidden = base_hidden
        utility = self.utility(base_hidden).squeeze(-1)
        if self.execution_proj is not None:
            if execution is None:
                execution = torch.zeros((batch, self.execution_dim), dtype=x.dtype, device=x.device)
            execution_repr = self.execution_proj(execution)
            hidden = hidden + self.execution_readout(execution_repr).unsqueeze(1)
            if self.utility_correction is not None and self.use_execution_residual_correction:
                exec_expand = execution_repr.unsqueeze(1).expand(-1, num_experts, -1)
                utility = utility + self.execution_correction_scale * self.utility_correction(
                    torch.cat([base_hidden, exec_expand], dim=-1)
                ).squeeze(-1)
        viability = torch.sigmoid(self.viability(hidden).squeeze(-1))
        steps_norm = torch.sigmoid(self.steps(hidden).squeeze(-1))
        return (
            self.utility_score_weight * utility
            + self.viability_score_weight * viability
            - self.score_step_lambda * steps_norm
        )


def count_success(results: Sequence[int], sequence_len: int = 5) -> List[float]:
    if len(results) == 0:
        return [0.0] * sequence_len
    return [float(np.mean(np.asarray(results) > i)) for i in range(sequence_len)]


def format_progress_success(results: Sequence[int], sequence_len: int = 5) -> str:
    rates = count_success(results, sequence_len)
    avg = float(np.mean(results)) if results else 0.0
    parts = [f"{i + 1}/{sequence_len} : {rate * 100:.1f}%" for i, rate in enumerate(rates)]
    parts.append(f"Average: {avg:.3f}")
    return " | ".join(parts)


def ensure_train_folder(output_dir: Path, checkpoint: Path) -> Path:
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


def load_model(cfg: Any, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model = hydra.utils.instantiate(cfg.model)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
    print(f"[CHECKPOINT] loaded={checkpoint}", flush=True)
    print(f"[CHECKPOINT] missing_keys={list(missing_keys)}", flush=True)
    print(f"[CHECKPOINT] unexpected_keys={list(unexpected_keys)}", flush=True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith("lam.text_encoder.") and name.endswith("layer_norm.bias"):
                print(f"[CHECKPOINT] zeroing T5 layer-norm bias for compatibility: {name}", flush=True)
                param.data.zero_()
            elif param.is_floating_point() and not torch.isfinite(param).all():
                print(f"[CHECKPOINT] replacing non-finite parameter with zeros: {name}", flush=True)
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point() and not torch.isfinite(buffer).all():
                print(f"[CHECKPOINT] replacing non-finite buffer with zeros: {name}", flush=True)
                buffer.data = torch.nan_to_num(buffer.data, nan=0.0, posinf=0.0, neginf=0.0)
    model.freeze()
    model.future_feature_mode = getattr(cfg, "future_feature_mode", "gfdm")
    model.future_frame_idx = getattr(cfg, "future_frame_idx", 5)
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
    if device.type == "cpu":
        model = model.float()
    model.process_device()
    model.eval()
    return model


def _apply_checkpoint_model_compat_overrides(cfg: Any, checkpoint: Path) -> None:
    """
    Align eval-time model shape with the checkpoint's original training config.
    Older DeFi checkpoints were trained with act_seq_len=10, while the current
    default eval config uses act_seq_len=50. Without this override, pos_emb and
    other sequence-shaped parameters mismatch at load time.
    """
    state = torch.load(str(checkpoint), map_location="cpu")
    train_args = state.get("args")
    if train_args is None:
        return

    if not isinstance(train_args, dict):
        train_args = OmegaConf.to_container(train_args, resolve=True)
    if not isinstance(train_args, dict):
        return

    model_args = train_args.get("model", {}) or {}
    with open_dict(cfg):
        if train_args.get("act_seq_len") is not None:
            cfg.act_seq_len = int(train_args["act_seq_len"])
        if model_args.get("action_seq_len") is not None:
            cfg.model.action_seq_len = int(model_args["action_seq_len"])
        if model_args.get("act_window_size") is not None:
            cfg.model.act_window_size = int(model_args["act_window_size"])
        if model_args.get("num_latents") is not None:
            cfg.model.num_latents = int(model_args["num_latents"])


def load_annotations(split_dir: Path) -> List[Dict[str, Any]]:
    data = np.load(split_dir / "lang_clip_resnet50" / "auto_lang_ann.npy", allow_pickle=True).item()
    rows = []
    for idx, (task, ann, interval) in enumerate(zip(data["language"]["task"], data["language"]["ann"], data["info"]["indx"])):
        rows.append({"index": idx, "task": str(task), "language": str(ann), "start_idx": int(interval[0]), "end_idx": int(interval[1])})
    return rows


def load_episode(split_dir: Path, episode_idx: int) -> Dict[str, np.ndarray]:
    with np.load(split_dir / f"episode_{episode_idx:07d}.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_demo_bank(calvin_dir: Path) -> Dict[str, Dict[str, Any]]:
    bank = {}
    split_dir = calvin_dir / "training"
    for row in load_annotations(split_dir):
        if row["task"] in bank:
            continue
        end_path = split_dir / f"episode_{row['end_idx']:07d}.npz"
        if end_path.exists():
            bank[row["task"]] = {"row": row, "end_ep": load_episode(split_dir, row["end_idx"])}
    return bank


def load_eval_sequences(
    num_sequences: Optional[int],
    sequences_path: Optional[Path],
    sequence_len: Optional[int],
) -> List[Tuple[Dict[str, Any], List[str]]]:
    if sequences_path is None:
        return get_sequences(num_sequences)
    with sequences_path.open("r") as f:
        rows = json.load(f)
    if sequence_len is not None:
        rows = [(initial_state, sequence[:sequence_len]) for initial_state, sequence in rows]
    if num_sequences is not None:
        rows = rows[:num_sequences]
    return rows


def build_demo_candidates(calvin_dir: Path, max_per_task: int) -> Dict[str, List[Dict[str, Any]]]:
    bank: Dict[str, List[Dict[str, Any]]] = {}
    split_dir = calvin_dir / "training"
    for row in load_annotations(split_dir):
        items = bank.setdefault(row["task"], [])
        if len(items) >= max_per_task:
            continue
        start_path = split_dir / f"episode_{row['start_idx']:07d}.npz"
        end_path = split_dir / f"episode_{row['end_idx']:07d}.npz"
        if start_path.exists() and end_path.exists():
            items.append(
                {
                    "row": row,
                    "start_ep": load_episode(split_dir, row["start_idx"]),
                    "end_ep": load_episode(split_dir, row["end_idx"]),
                }
            )
    return bank


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = cv2.resize(np.asarray(image), (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    tensor = tensor.div(255.0).sub(0.5).div(0.5)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


@torch.no_grad()
def episode_current_feature(model: torch.nn.Module, episode: Dict[str, np.ndarray], lang_text: str) -> torch.Tensor:
    static = image_to_tensor(episode["rgb_static"], model.device)
    gripper = image_to_tensor(episode["rgb_gripper"], model.device)
    input_rgb = torch.cat([static, gripper], dim=0)
    features = model.TVP_encoder(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    features = features[:, 0]
    features = features.reshape(features.shape[0], features.shape[1], -1).permute(0, 2, 1).float()
    return features[0].mean(dim=0).detach()


@torch.no_grad()
def demo_future_feature(model: torch.nn.Module, demo: Dict[str, Any], lang_text: str) -> torch.Tensor:
    static = image_to_tensor(demo["end_ep"]["rgb_static"], model.device)
    gripper = image_to_tensor(demo["end_ep"]["rgb_gripper"], model.device)
    input_rgb = torch.cat([static, gripper], dim=0)
    features = model.TVP_encoder(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    features = torch.einsum("bfchw->bfchw", features)
    features = features[:, 0]
    features = features.reshape(features.shape[0], features.shape[1], -1).permute(0, 2, 1).float()
    return features[0].detach()


@torch.no_grad()
def obs_current_feature(model: torch.nn.Module, obs: Dict[str, Any], lang_text: str) -> torch.Tensor:
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    features = model.TVP_encoder(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    features = features[:, 0]
    features = features.reshape(features.shape[0], features.shape[1], -1).permute(0, 2, 1).float()
    return features[0].mean(dim=0).detach()


@torch.no_grad()
def defi_future_feature(model: torch.nn.Module, obs: Dict[str, Any], lang_text: str) -> torch.Tensor:
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    perceptual = model.TVP_encoder(
        input_rgb,
        [lang_text, lang_text],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    perceptual = einops.rearrange(perceptual, "b f c h w -> b f c (h w)")
    perceptual = einops.rearrange(perceptual, "b f c l -> b f l c")
    perceptual = perceptual[:, : model.Former_num_time_embeds].to(torch.float32)
    batch = rgb_static.shape[0]
    perceptual, _ = torch.split(perceptual, [batch, batch], dim=0)
    frame_idx = min(max(int(getattr(model, "future_frame_idx", 1)), 0), perceptual.shape[1] - 1)
    return perceptual[:, frame_idx][0].detach()


def build_retrieval_vpp_bank(
    model: torch.nn.Module,
    calvin_dir: Path,
    max_per_task: int,
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    raw_bank = build_demo_candidates(calvin_dir, max_per_task)
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for task, demos in tqdm(raw_bank.items(), desc="build_nn_vpp_bank"):
        items = []
        for demo in demos:
            lang_text = str(demo["row"]["language"])
            current = episode_current_feature(model, demo["start_ep"], lang_text)
            future = demo_future_feature(model, demo, lang_text)
            items.append({"current": current.cpu(), "future": future.cpu()})
        if items:
            vpp_bank[task] = items
    return vpp_bank


def retrieve_vpp_future(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    lang_text: str,
    subtask: str,
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
) -> Optional[torch.Tensor]:
    candidates = vpp_bank.get(subtask, [])
    if not candidates:
        return None
    current = obs_current_feature(model, obs, lang_text).cpu()
    current_norm = torch.clamp(torch.norm(current), min=1e-12)
    best_idx = 0
    best_score = -float("inf")
    for idx, item in enumerate(candidates):
        ref = item["current"]
        score = float(torch.dot(current, ref) / (current_norm * torch.clamp(torch.norm(ref), min=1e-12)))
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]["future"].to(model.device)


def load_vjepa2_encoder(vjepa_root: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, vjepa_root.as_posix())
    encoder, _ = torch.hub.load(vjepa_root.as_posix(), "vjepa2_1_vit_base_384", source="local")
    return encoder.to(device).eval()


def vjepa_image_tensor_from_numpy(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = cv2.resize(np.asarray(image), (384, 384), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return ((tensor - VJEPA_MEAN) / VJEPA_STD).unsqueeze(2).to(device)


def vjepa_image_tensor_from_obs(obs_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = obs_tensor.detach().to(device)
    while tensor.ndim > 4:
        tensor = tensor[:, 0]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor.float(), size=(384, 384), mode="bilinear", align_corners=False)
    if float(tensor.max()) > 2.0:
        tensor = tensor / 255.0
    elif float(tensor.min()) < -0.1:
        tensor = tensor.mul(0.5).add(0.5).clamp(0.0, 1.0)
    return ((tensor - VJEPA_MEAN.to(device)) / VJEPA_STD.to(device)).unsqueeze(2)


@torch.no_grad()
def vjepa_encode_tensor(encoder: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    feat = encoder(tensor).mean(dim=1)
    return F.normalize(feat.float(), dim=-1).squeeze(0).detach()


def build_vjepa_vpp_bank(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    calvin_dir: Path,
    max_per_task: int,
    cache_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    if cache_path is not None and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    raw_bank = build_demo_candidates(calvin_dir, max_per_task)
    bank: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for task, demos in tqdm(raw_bank.items(), desc="build_vjepa_vpp_bank"):
        items = []
        for demo in demos:
            lang_text = str(demo["row"]["language"])
            current = vjepa_encode_tensor(
                vjepa_encoder,
                vjepa_image_tensor_from_numpy(demo["start_ep"]["rgb_static"], model.device),
            )
            vjepa_future = vjepa_encode_tensor(
                vjepa_encoder,
                vjepa_image_tensor_from_numpy(demo["end_ep"]["rgb_static"], model.device),
            )
            future = demo_future_feature(model, demo, lang_text)
            items.append({"current": current.cpu(), "vjepa_future": vjepa_future.cpu(), "future": future.cpu()})
        if items:
            bank[task] = items
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, cache_path)
    return bank


def retrieve_vjepa_vpp_future(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    obs: Dict[str, Any],
    subtask: str,
    bank: Dict[str, List[Dict[str, torch.Tensor]]],
) -> Optional[torch.Tensor]:
    candidates = bank.get(subtask, [])
    if not candidates:
        return None
    current = vjepa_encode_tensor(
        vjepa_encoder,
        vjepa_image_tensor_from_obs(obs["rgb_obs"]["rgb_static"], model.device),
    ).cpu()
    best_idx = 0
    best_score = -float("inf")
    for idx, item in enumerate(candidates):
        score = float(torch.dot(current, item["current"]))
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]["future"].to(model.device)


def load_vjepa_adapter(path: Path, device: torch.device) -> VjepaFutureAdapter:
    checkpoint = torch.load(path, map_location="cpu")
    adapter = VjepaFutureAdapter(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]))
    adapter.load_state_dict(checkpoint["model_state"])
    return adapter.to(device).eval()


def load_vjepa_cross_attention(path: Path, device: torch.device) -> Tuple[VjepaCrossAttentionFuture, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model = VjepaCrossAttentionFuture(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["heads"]),
        int(checkpoint["layers"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), int(checkpoint.get("k", 8))


def load_vjepa_moe(path: Path, device: torch.device) -> Tuple[VjepaMoeFuture, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model = VjepaMoeFuture(
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
    model.task_vocab = checkpoint.get("task_vocab", {})
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), int(checkpoint.get("k", 8))


def load_reflection_critic(path: Path, device: torch.device) -> ReflectionCritic:
    checkpoint = torch.load(path, map_location="cpu")
    model = ReflectionCritic(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def load_future_correction_mlp(path: Path, device: torch.device) -> FutureCorrectionMLP:
    checkpoint = torch.load(path, map_location="cpu")
    model = FutureCorrectionMLP(
        int(checkpoint["future_dim"]),
        int(checkpoint["reflection_dim"]),
        int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def load_universal_moe_oracle_router(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    if "utility_score_weight" in checkpoint:
        execution_dim = int(checkpoint.get("execution_dim", 0))
        if execution_dim <= 0 and "model_state" in checkpoint and "execution_proj.1.weight" in checkpoint["model_state"]:
            execution_dim = int(checkpoint["model_state"]["execution_proj.1.weight"].shape[1])
        model = FuturePortfolioSelector(
            int(checkpoint["feature_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["num_experts"]),
            float(checkpoint.get("utility_score_weight", 1.0)),
            float(checkpoint.get("viability_score_weight", 0.1)),
            float(checkpoint.get("score_step_lambda", 0.05)),
            execution_dim=execution_dim,
            use_execution_residual_correction=bool(checkpoint.get("use_execution_residual_correction", False)),
            execution_correction_scale=float(checkpoint.get("execution_correction_scale", 0.0)),
        )
    else:
        model = UniversalMoeOracleRouter(
            int(checkpoint["feature_dim"]),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["num_experts"]),
        )
    model.load_state_dict(checkpoint["model_state"])
    model.feature_layout = checkpoint.get(
        "feature_layout",
        [
            "obs_embedding",
            "expert_future",
            "expert_future_minus_obs",
            "router_logit",
            "router_score",
            "expert_contradiction",
            "expert_steps_norm",
        ],
    )
    return model.to(device).eval()


def load_portfolio_reflections(path: Path) -> Dict[Tuple[int, int], str]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["sequence_index"]), int(row["subtask_index"]))
            text = row.get("reflection_text")
            if not text:
                fields = [
                    "progress_status",
                    "failure_type",
                    "prefer_next",
                    "avoid_next",
                    "correction_direction",
                    "memory_query",
                ]
                text = " | ".join(f"{field}: {row.get(field, 'uncertain')}" for field in fields)
            rows[key] = str(text)
    return rows


def universal_moe_oracle_router_features(
    expert_items: Sequence[Dict[str, Any]],
    latent_tokens_by_expert: Dict[int, torch.Tensor],
    drift_memory: Sequence[float],
    cfg: Any,
    feature_layout: Optional[Sequence[str]] = None,
    reflection_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    feature_layout = feature_layout or [
        "obs_embedding",
        "expert_future",
        "expert_future_minus_obs",
        "router_logit",
        "router_score",
        "expert_contradiction",
        "expert_steps_norm",
    ]
    sorted_items = sorted(expert_items, key=lambda row: row["expert"])
    obs = sorted_items[0]["obs_vjepa"].to(torch.float32)
    router_logits = torch.tensor([float(item["router_logit"]) for item in sorted_items], dtype=torch.float32)
    router_scores = torch.softmax(router_logits, dim=0)
    rows = []
    for idx, item in enumerate(sorted_items):
        future = item["pred_future_vjepa"].detach().cpu().to(torch.float32)
        delta = future - obs
        _, contradiction = reflection_candidate_feature(item, latent_tokens_by_expert[int(item["expert"])], drift_memory, None)
        parts = []
        for name in feature_layout:
            if name == "obs_embedding":
                parts.append(obs)
            elif name == "expert_future":
                parts.append(future)
            elif name == "expert_future_minus_obs":
                parts.append(delta)
            elif name == "router_logit":
                parts.append(router_logits[idx : idx + 1])
            elif name == "router_score":
                parts.append(router_scores[idx : idx + 1])
            elif name == "expert_contradiction":
                parts.append(torch.tensor([float(contradiction)], dtype=torch.float32))
            elif name == "expert_steps_norm":
                parts.append(torch.tensor([0.0], dtype=torch.float32))
            elif name == "qwen_reflection_t5_embedding":
                if reflection_embedding is None:
                    raise ValueError("qwen_reflection_t5_embedding requires --portfolio-reflections")
                parts.append(reflection_embedding.detach().cpu().to(torch.float32))
            else:
                raise ValueError(f"unknown universal oracle router feature: {name}")
        row = torch.cat(parts, dim=0)
        rows.append(row)
    return torch.stack(rows, dim=0)


def retrieve_trained_vjepa_future(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    adapter: VjepaFutureAdapter,
    obs: Dict[str, Any],
    subtask: str,
    bank: Dict[str, List[Dict[str, torch.Tensor]]],
) -> Optional[torch.Tensor]:
    candidates = bank.get(subtask, [])
    if not candidates:
        return None
    current = vjepa_encode_tensor(
        vjepa_encoder,
        vjepa_image_tensor_from_obs(obs["rgb_obs"]["rgb_static"], model.device),
    ).to(model.device)
    with torch.no_grad():
        predicted_future = adapter(current.unsqueeze(0)).squeeze(0).cpu()
    best_idx = 0
    best_score = -float("inf")
    for idx, item in enumerate(candidates):
        score = float(torch.dot(predicted_future, item["vjepa_future"]))
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]["future"].to(model.device)


def retrieve_cross_attention_vjepa_future(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    predictor: VjepaCrossAttentionFuture,
    obs: Dict[str, Any],
    subtask: str,
    bank: Dict[str, List[Dict[str, torch.Tensor]]],
    k: int,
) -> Optional[torch.Tensor]:
    candidates = bank.get(subtask, [])
    if not candidates:
        return None
    current = vjepa_encode_tensor(
        vjepa_encoder,
        vjepa_image_tensor_from_obs(obs["rgb_obs"]["rgb_static"], model.device),
    ).to(model.device)
    scored = [(float(torch.dot(current.cpu(), item["current"])), idx) for idx, item in enumerate(candidates)]
    scored.sort(reverse=True)
    top_idx = [idx for _, idx in scored[:k]]
    if len(top_idx) < k:
        top_idx.extend([top_idx[-1]] * (k - len(top_idx)))
    demo_current = torch.stack([candidates[idx]["current"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    demo_future = torch.stack([candidates[idx]["vjepa_future"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    with torch.no_grad():
        predicted_future = predictor(current.unsqueeze(0), demo_current, demo_future).squeeze(0).cpu()
    best_idx = 0
    best_score = -float("inf")
    for idx, item in enumerate(candidates):
        score = float(torch.dot(predicted_future, item["vjepa_future"]))
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]["future"].to(model.device)


def retrieve_moe_vjepa_future(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    predictor: VjepaMoeFuture,
    obs: Dict[str, Any],
    subtask: str,
    bank: Dict[str, List[Dict[str, torch.Tensor]]],
    k: int,
) -> Optional[torch.Tensor]:
    candidates = bank.get(subtask, [])
    if not candidates:
        return None
    current = vjepa_encode_tensor(
        vjepa_encoder,
        vjepa_image_tensor_from_obs(obs["rgb_obs"]["rgb_static"], model.device),
    ).to(model.device)
    scored = [(float(torch.dot(current.cpu(), item["current"])), idx) for idx, item in enumerate(candidates)]
    scored.sort(reverse=True)
    top_idx = [idx for _, idx in scored[:k]]
    if len(top_idx) < k:
        top_idx.extend([top_idx[-1]] * (k - len(top_idx)))
    demo_current = torch.stack([candidates[idx]["current"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    demo_future = torch.stack([candidates[idx]["vjepa_future"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    with torch.no_grad():
        task_ids = None
        stage_ids = None
        if getattr(predictor, "condition_router", False):
            task_vocab = getattr(predictor, "task_vocab", {})
            task_ids = torch.tensor([int(task_vocab.get(subtask, 0))], dtype=torch.long, device=model.device)
            stage_ids = torch.tensor([infer_stage(subtask)], dtype=torch.long, device=model.device)
        futures, logits = predictor(current.unsqueeze(0), demo_current, demo_future, task_ids, stage_ids)
        predicted_future = futures[0, int(torch.argmax(logits, dim=-1).item())].cpu()
    best_idx = 0
    best_score = -float("inf")
    for idx, item in enumerate(candidates):
        score = float(torch.dot(predicted_future, item["vjepa_future"]))
        if score > best_score:
            best_score = score
            best_idx = idx
    return candidates[best_idx]["future"].to(model.device)


def task_caption(task: str) -> str:
    return str(task).replace("_", " ")


def flatten_vjepa_bank(bank: Dict[str, List[Dict[str, torch.Tensor]]]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    rows: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for task, items in bank.items():
        for item in items:
            rows.append((str(task), item))
    return rows


def nearest_bank_task(predicted_future: torch.Tensor, flat_bank: Sequence[Tuple[str, Dict[str, torch.Tensor]]]) -> str:
    if not flat_bank:
        return ""
    best_task = str(flat_bank[0][0])
    best_score = -float("inf")
    pred = predicted_future.detach().cpu()
    for task, item in flat_bank:
        score = float(torch.dot(pred, item["vjepa_future"]))
        if score > best_score:
            best_score = score
            best_task = str(task)
    return best_task


def reflection_text_for_future(task: str, hypothesis_task: str) -> str:
    if str(task) == str(hypothesis_task):
        return f"The future hypothesis {hypothesis_task} is consistent with task {task}."
    return (
        f"The future hypothesis {hypothesis_task} is inconsistent with task {task}; "
        f"correct it toward {task}. {task_caption(task)}"
    )


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


def random_reflection_text(task: str, hypothesis_task: str) -> str:
    payload = f"{task}|{hypothesis_task}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    idx = int.from_bytes(digest[:4], "little") % len(RANDOM_REFLECTION_TEMPLATES)
    return RANDOM_REFLECTION_TEMPLATES[idx]


def summarize_recent_memory(memory_rows: Sequence[Dict[str, Any]], keep_last: int) -> str:
    if not memory_rows:
        return "none"
    lines = []
    for row in list(memory_rows)[-keep_last:]:
        state = dict(row.get("state_semantics", {}))
        lines.append(
            f"task={row.get('task')} success={bool(row.get('success', False))} "
            f"state={state.get('object_state_summary', 'state_unknown')} "
            f"relation={state.get('relation_summary', 'relation_unknown')}"
        )
    return " ; ".join(lines)


def load_key_memory_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def retrieve_key_memory(
    key_memory_rows: Sequence[Dict[str, Any]],
    task: str,
    topk: int,
) -> List[Dict[str, Any]]:
    if not key_memory_rows:
        return []
    exact = [row for row in key_memory_rows if str(row.get("task")) == task]
    if len(exact) >= topk:
        return exact[:topk]
    fallback = [row for row in key_memory_rows if str(row.get("mismatch_type", "")) != "unknown"]
    merged = exact + fallback
    return merged[:topk]


def summarize_key_memory(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "none"
    lines = []
    for row in rows:
        state = dict(row.get("state_semantics", {}))
        completed = ",".join(state.get("completed_actions", [])[:2]) or "none"
        lines.append(
            f"task={row.get('task')} completed={completed} "
            f"state={state.get('object_state_summary', 'state_unknown')} "
            f"relation={state.get('relation_summary', 'relation_unknown')} "
            f"precond={state.get('precondition_status', 'unknown')}"
        )
    return " ; ".join(lines)


def summarize_key_diagnostics(rows: Sequence[Dict[str, Any]], topk: int = 2) -> str:
    if not rows:
        return "none"
    lines = []
    for row in list(rows)[:topk]:
        diag = dict(row.get("diagnostic_semantics", {}))
        lines.append(
            f"task={row.get('task')} mismatch={diag.get('mismatch_type', row.get('mismatch_type', 'unknown'))} "
            f"recoverability={diag.get('recoverability', row.get('recoverability', 'uncertain'))} "
            f"fix={diag.get('correction_direction', row.get('correction_direction', 'stabilize contact'))}"
        )
    return " ; ".join(lines)


def load_existing_sequence_logs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def rebuild_counts_from_logs(
    logs: Sequence[Dict[str, Any]],
) -> Tuple[List[int], Counter, Counter]:
    results: List[int] = []
    task_total = Counter()
    task_success = Counter()
    for row in logs:
        results.append(int(row.get("success_counter", 0)))
        for sub in row.get("subtasks", []):
            task = str(sub.get("task", ""))
            if not task:
                continue
            task_total[task] += 1
            task_success[task] += int(bool(sub.get("success", False)))
    return results, task_total, task_success


def append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def trace_rollout_enabled() -> bool:
    return os.environ.get("DEFI_TRACE_ROLLOUT", "0") == "1"


def trace_rollout(message: str) -> None:
    if trace_rollout_enabled():
        print(f"[TRACE] {message}", flush=True)


def infer_mismatch_type(task: str, success: bool, steps: int, ep_len: int) -> str:
    if success and steps < max(ep_len // 2, 1):
        return "none"
    if "drawer" in task and not success:
        return "drawer_progress_hallucinated"
    if task.startswith("push_") and not success:
        return "object_displacement_overestimated"
    if task.startswith("lift_") and not success:
        return "contact_not_realized"
    if task == "stack_block" and not success:
        return "wrong_object_identity"
    if not success:
        return "future_outcome_mismatch"
    return "slow_success"


def build_memory_reflection_prompt(
    task: str,
    next_task: Optional[str],
    recent_summary: str,
    key_summary: str,
    rule_text: str,
) -> str:
    next_text = next_task or "none"
    return (
        "You audit whether a robot imagined future is trustworthy.\n"
        "Return one compact JSON object only with keys: "
        '{"trust_score": number, "mismatch_type": string, "correction_direction": string, '
        '"recoverability": string, "explanation": string}.\n'
        "trust_score must be in [0,1]. Use short phrases.\n"
        f"task: {task}\n"
        f"next_task: {next_text}\n"
        f"recent_memory: {recent_summary}\n"
        f"key_memory: {key_summary}\n"
        f"repair_rule: {rule_text}\n"
    )


def summarize_important_key_memory(rows: Sequence[Dict[str, Any]], topk: int = 2) -> str:
    if not rows:
        return "none"
    important: List[Dict[str, Any]] = []
    for row in rows:
        mismatch = str(row.get("mismatch_type", "unknown"))
        if bool(row.get("repair_helped", False)) or mismatch not in {"unknown", "none"}:
            important.append(row)
    if not important:
        important = list(rows)
    lines = []
    for row in important[:topk]:
        lines.append(
            f"task={row.get('task')} mismatch={row.get('mismatch_type', 'unknown')} "
            f"fix={row.get('correction_direction', 'stabilize contact')}"
        )
    return " ; ".join(lines)


def build_postexec_reflection_prompt(
    task: str,
    next_task: Optional[str],
    success: bool,
    steps: int,
    mismatch_type: str,
    recent_summary: str,
    key_summary: str,
    important_summary: str,
    rule_text: str,
) -> str:
    next_text = next_task or "none"
    return (
        "You audit the executed robot step and write a compact repair hint for the next imagined future.\n"
        "Return one compact JSON object only with keys: "
        '{"trust_score": number, "mismatch_type": string, "correction_direction": string, '
        '"recoverability": string, "explanation": string}.\n'
        "trust_score must be in [0,1]. Use short phrases.\n"
        f"task: {task}\n"
        f"next_task: {next_text}\n"
        f"execution_success: {bool(success)}\n"
        f"execution_steps: {int(steps)}\n"
        f"observed_mismatch: {mismatch_type}\n"
        f"recent_memory: {recent_summary}\n"
        f"key_memory: {key_summary}\n"
        f"important_events: {important_summary}\n"
        f"repair_rule: {rule_text}\n"
    )


def build_augmented_future_instruction(base_text: str, prev_row: Dict[str, Any], important_summary: str) -> str:
    short_fix = str(prev_row.get("correction_direction", "stabilize contact")).replace(" ", "_")[:24]
    short_mismatch = str(prev_row.get("mismatch_type", "unknown")).replace(" ", "_")[:24]
    short_recover = str(prev_row.get("recoverability", "uncertain")).replace(" ", "_")[:16]
    short_events = important_summary.replace("task=", "").replace("mismatch=", "m=").replace("fix=", "f=")
    short_events = short_events.replace(" ; ", " | ")[:80]
    return (
        f"{base_text}. "
        f"Hint: trust={float(prev_row.get('trust_score', 0.5)):.2f}; "
        f"m={short_mismatch}; "
        f"fix={short_fix}; "
        f"r={short_recover}; "
        f"events={short_events}."
    )


def parse_memory_reflection_output(raw_text: str) -> MemoryReflectionDecision:
    clean = raw_text.strip()
    try:
        payload = json.loads(clean)
    except Exception:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match is None:
            payload = {}
        else:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = {}
    trust_value = payload.get("trust", payload.get("trust_score", 0.5))
    trust = trust_label_to_score(trust_value)
    trust = max(0.0, min(1.0, trust))
    return MemoryReflectionDecision(
        trust_score=trust,
        mismatch_type=str(payload.get("mismatch_type", "uncertain"))[:128],
        correction_direction=str(payload.get("correction_direction", "stabilize contact"))[:256],
        recoverability=str(payload.get("recoverability", "uncertain"))[:64],
        explanation=str(payload.get("explanation", "uncertain"))[:512],
        raw_text=clean[:1000],
        hypothetical_failure=str(payload.get("hypothetical_failure", payload.get("failure_hypothesis", "")))[:256],
        counterfactual_future=str(payload.get("counterfactual_future", payload.get("counterfactual_future_text", "")))[:256],
        failure_factor=str(payload.get("failure_factor", ""))[:128],
        intervention=str(payload.get("intervention", ""))[:256],
    )


def template_postexec_reflection_decision(
    task: str,
    success: bool,
    steps: int,
    ep_len: int,
    mismatch_type: str,
) -> MemoryReflectionDecision:
    if success and mismatch_type == "none":
        return MemoryReflectionDecision(
            trust_score=0.85,
            mismatch_type="none",
            correction_direction="preserve progress",
            recoverability="not_repairable",
            explanation=f"{task} succeeded cleanly; keep the future consistent with achieved progress.",
            raw_text='{"trust_score":0.85,"mismatch_type":"none","correction_direction":"preserve progress","recoverability":"not_repairable","explanation":"success"}',
            hypothetical_failure="none",
            counterfactual_future="preserve progress",
        )
    if mismatch_type == "drawer_progress_hallucinated":
        return MemoryReflectionDecision(
            trust_score=0.25,
            mismatch_type=mismatch_type,
            correction_direction="delay drawer progress and verify opening contact",
            recoverability="repairable",
            explanation="The imagined future advanced drawer state too early; require verified drawer motion first.",
            raw_text='{"trust_score":0.25,"mismatch_type":"drawer_progress_hallucinated","correction_direction":"delay drawer progress and verify opening contact","recoverability":"repairable","explanation":"template"}',
            hypothetical_failure="drawer progression is predicted too early",
            counterfactual_future="delay drawer progress until opening contact is verified",
        )
    if mismatch_type == "object_displacement_overestimated":
        return MemoryReflectionDecision(
            trust_score=0.2,
            mismatch_type=mismatch_type,
            correction_direction="reduce predicted object motion and strengthen lateral contact",
            recoverability="repairable",
            explanation="The future likely overestimates block displacement; require stronger contact before motion.",
            raw_text='{"trust_score":0.2,"mismatch_type":"object_displacement_overestimated","correction_direction":"reduce predicted object motion and strengthen lateral contact","recoverability":"repairable","explanation":"template"}',
            hypothetical_failure="object motion is overestimated",
            counterfactual_future="keep motion small and strengthen contact first",
        )
    if mismatch_type == "contact_not_realized":
        return MemoryReflectionDecision(
            trust_score=0.2,
            mismatch_type=mismatch_type,
            correction_direction="stabilize contact before lift",
            recoverability="repairable",
            explanation="The lift future is unreliable until contact is firmly established.",
            raw_text='{"trust_score":0.2,"mismatch_type":"contact_not_realized","correction_direction":"stabilize contact before lift","recoverability":"repairable","explanation":"template"}',
            hypothetical_failure="contact is not yet established",
            counterfactual_future="stabilize contact before attempting lift",
        )
    if mismatch_type == "wrong_object_identity":
        return MemoryReflectionDecision(
            trust_score=0.15,
            mismatch_type=mismatch_type,
            correction_direction="preserve object identity and align grasp before stacking",
            recoverability="repairable",
            explanation="The future likely confuses object relation; re-align before stacking.",
            raw_text='{"trust_score":0.15,"mismatch_type":"wrong_object_identity","correction_direction":"preserve object identity and align grasp before stacking","recoverability":"repairable","explanation":"template"}',
            hypothetical_failure="object identity may be confused",
            counterfactual_future="re-align grasp and preserve object identity",
        )
    if success:
        trust = 0.6 if steps > max(ep_len // 2, 1) else 0.75
        return MemoryReflectionDecision(
            trust_score=trust,
            mismatch_type=mismatch_type,
            correction_direction="be conservative and preserve verified progress",
            recoverability="uncertain",
            explanation="The step succeeded but was not fully clean; keep progress and avoid aggressive future changes.",
            raw_text='{"trust_score":0.6,"mismatch_type":"slow_success","correction_direction":"be conservative and preserve verified progress","recoverability":"uncertain","explanation":"template"}',
            hypothetical_failure="over-aggressive continuation may lose verified progress",
            counterfactual_future="preserve verified progress and avoid aggressive changes",
        )
    return MemoryReflectionDecision(
        trust_score=0.2,
        mismatch_type=mismatch_type,
        correction_direction="stabilize contact and reduce over-optimistic future progress",
        recoverability="repairable",
        explanation="The executed step failed; the next future should be more conservative and grounded in verified progress.",
        raw_text='{"trust_score":0.2,"mismatch_type":"future_outcome_mismatch","correction_direction":"stabilize contact and reduce over-optimistic future progress","recoverability":"repairable","explanation":"template"}',
        hypothetical_failure="future outcome looks too optimistic",
        counterfactual_future="stabilize contact and reduce over-optimistic progress",
    )


def apply_single_future_repair(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    goal: Dict[str, Any],
    future_feature: torch.Tensor,
    reflection_text: str,
    correction_mlp: FutureCorrectionMLP,
    reflection_encoder: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reflection_embedding = reflection_encoder.encode([reflection_text]).squeeze(0).to(model.device)
    corrected_future = correction_mlp(
        future_feature.unsqueeze(0).to(model.device),
        reflection_embedding.unsqueeze(0).to(model.device),
    ).squeeze(0)
    latent_tokens = latent_tokens_from_future_feature(model, obs, goal, corrected_future)
    return corrected_future.detach(), latent_tokens.detach()


@torch.no_grad()
def corrected_router_logits(
    predictor: VjepaMoeFuture,
    current: torch.Tensor,
    corrected_futures: torch.Tensor,
    subtask: str,
) -> torch.Tensor:
    if not getattr(predictor, "future_aware_router", False):
        raise ValueError("corrected future routing requires future_aware_router=True")
    current_b = current.unsqueeze(0)
    futures_b = corrected_futures.unsqueeze(0)
    expanded = current_b.unsqueeze(1).expand(-1, futures_b.shape[1], -1)
    router_parts = [
        current_b,
        futures_b.reshape(futures_b.shape[0], -1),
        (futures_b - expanded).reshape(futures_b.shape[0], -1),
    ]
    if getattr(predictor, "condition_router", False):
        task_vocab = getattr(predictor, "task_vocab", {})
        task_ids = torch.tensor([int(task_vocab.get(subtask, 0))], dtype=torch.long, device=current.device)
        stage_ids = torch.tensor([infer_stage(subtask)], dtype=torch.long, device=current.device)
        router_parts.extend([predictor.task_embed(task_ids), predictor.stage_embed(stage_ids)])
    return predictor.router(torch.cat(router_parts, dim=-1)).squeeze(0)


def retrieve_corrected_moe_vjepa_future(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    predictor: VjepaMoeFuture,
    correction_mlp: FutureCorrectionMLP,
    reflection_encoder: T5ReflectionEncoder,
    obs: Dict[str, Any],
    subtask: str,
    bank: Dict[str, List[Dict[str, torch.Tensor]]],
    k: int,
    reflection_text_mode: str = "task_hypothesis",
    return_trace: bool = False,
) -> Any:
    candidates = bank.get(subtask, [])
    if not candidates:
        return (None, None) if return_trace else None
    current = vjepa_encode_tensor(
        vjepa_encoder,
        vjepa_image_tensor_from_obs(obs["rgb_obs"]["rgb_static"], model.device),
    ).to(model.device)
    scored = [(float(torch.dot(current.cpu(), item["current"])), idx) for idx, item in enumerate(candidates)]
    scored.sort(reverse=True)
    top_idx = [idx for _, idx in scored[:k]]
    if len(top_idx) < k:
        top_idx.extend([top_idx[-1]] * (k - len(top_idx)))
    demo_current = torch.stack([candidates[idx]["current"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    demo_future = torch.stack([candidates[idx]["vjepa_future"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    with torch.no_grad():
        task_ids = None
        stage_ids = None
        if getattr(predictor, "condition_router", False):
            task_vocab = getattr(predictor, "task_vocab", {})
            task_ids = torch.tensor([int(task_vocab.get(subtask, 0))], dtype=torch.long, device=model.device)
            stage_ids = torch.tensor([infer_stage(subtask)], dtype=torch.long, device=model.device)
        futures, _ = predictor(current.unsqueeze(0), demo_current, demo_future, task_ids, stage_ids)
        predicted_futures = futures.squeeze(0)
        flat_bank = flatten_vjepa_bank(bank)
        texts = []
        hypothesis_tasks = []
        for predicted_future in predicted_futures:
            hypothesis_task = nearest_bank_task(predicted_future, flat_bank)
            hypothesis_tasks.append(hypothesis_task)
            if reflection_text_mode == "task_hypothesis":
                text = reflection_text_for_future(subtask, hypothesis_task)
            elif reflection_text_mode == "random":
                text = random_reflection_text(subtask, hypothesis_task)
            else:
                raise ValueError(f"unknown reflection_text_mode: {reflection_text_mode}")
            texts.append(text)
        reflections = reflection_encoder.encode(texts).to(model.device)
        corrected = correction_mlp(predicted_futures.to(model.device), reflections)
        logits = corrected_router_logits(predictor, current, corrected, subtask)
        chosen_expert = int(torch.argmax(logits).item())
        predicted_future = corrected[chosen_expert].detach().cpu()
    best_idx = 0
    best_score = -float("inf")
    for idx, item in enumerate(candidates):
        score = float(torch.dot(predicted_future, item["vjepa_future"]))
        if score > best_score:
            best_score = score
            best_idx = idx
    retrieved_future = candidates[best_idx]["future"].to(model.device)
    if not return_trace:
        return retrieved_future

    trace = {
        "subtask": subtask,
        "retrieval_topk_demo_indices": [int(idx) for idx in top_idx],
        "retrieval_topk_scores": [float(score) for score, _ in scored[:k]],
        "candidate_count": len(candidates),
        "reflection_text_mode": reflection_text_mode,
        "chosen_expert": chosen_expert,
        "router_logits": [float(x) for x in logits.detach().cpu().tolist()],
        "hypothesis_tasks": [str(task) for task in hypothesis_tasks],
        "reflection_texts": [str(text) for text in texts],
        "repair_delta_norms": [
            float(torch.norm((corrected[idx] - predicted_futures[idx]).detach(), p=2).item())
            for idx in range(corrected.shape[0])
        ],
        "selected_demo_index": int(best_idx),
        "selected_demo_score": float(best_score),
    }
    return retrieved_future, trace


def retrieve_moe_vjepa_expert_futures(
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    predictor: VjepaMoeFuture,
    obs: Dict[str, Any],
    subtask: str,
    bank: Dict[str, List[Dict[str, torch.Tensor]]],
    k: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    candidates = bank.get(subtask, [])
    if not candidates:
        return [], []
    current = vjepa_encode_tensor(
        vjepa_encoder,
        vjepa_image_tensor_from_obs(obs["rgb_obs"]["rgb_static"], model.device),
    ).to(model.device)
    scored = [(float(torch.dot(current.cpu(), item["current"])), idx) for idx, item in enumerate(candidates)]
    scored.sort(reverse=True)
    top_idx = [idx for _, idx in scored[:k]]
    if len(top_idx) < k:
        top_idx.extend([top_idx[-1]] * (k - len(top_idx)))
    demo_current = torch.stack([candidates[idx]["current"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    demo_future = torch.stack([candidates[idx]["vjepa_future"] for idx in top_idx], dim=0).unsqueeze(0).to(model.device)
    with torch.no_grad():
        task_ids = None
        stage_ids = None
        if getattr(predictor, "condition_router", False):
            task_vocab = getattr(predictor, "task_vocab", {})
            task_ids = torch.tensor([int(task_vocab.get(subtask, 0))], dtype=torch.long, device=model.device)
            stage_ids = torch.tensor([infer_stage(subtask)], dtype=torch.long, device=model.device)
        futures, logits = predictor(current.unsqueeze(0), demo_current, demo_future, task_ids, stage_ids)
        order = torch.argsort(logits.squeeze(0), descending=True).detach().cpu().tolist()
        predicted_futures = futures.squeeze(0).detach().cpu()
    expert_items = []
    for expert_idx, predicted_future in enumerate(predicted_futures):
        best_idx = 0
        best_score = -float("inf")
        for idx, item in enumerate(candidates):
            score = float(torch.dot(predicted_future, item["vjepa_future"]))
            if score > best_score:
                best_score = score
                best_idx = idx
        expert_items.append(
            {
                "expert": int(expert_idx),
                "router_rank": int(order.index(expert_idx)),
                "router_logit": float(logits.squeeze(0)[expert_idx].detach().cpu()),
                "future": candidates[best_idx]["future"].to(model.device),
                "obs_vjepa": current.detach().cpu(),
                "pred_future_vjepa": predicted_future.detach().cpu(),
                "demo_index": int(best_idx),
                "vjepa_score": float(best_score),
            }
        )
    return expert_items, [int(idx) for idx in order]


@torch.no_grad()
def latent_tokens_from_future_feature(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    goal: Dict[str, Any],
    future_feature: torch.Tensor,
) -> torch.Tensor:
    old_mode = getattr(model, "future_feature_mode", "gfdm")
    old_override = getattr(model, "override_future_feature", None)
    try:
        model.future_feature_mode = "override"
        model.override_future_feature = future_feature
        rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
        batch = rgb_static.shape[0]
        input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
        lang_text = goal["lang_text"]
        perceptual = model.TVP_encoder(
            input_rgb,
            [lang_text, lang_text],
            model.timestep,
            model.extract_layer_idx,
            all_layer=model.use_all_layer,
            step_time=1,
            max_length=model.max_length,
        )
        perceptual = einops.rearrange(perceptual, "b f c h w -> b f c (h w)")
        perceptual = einops.rearrange(perceptual, "b f c l -> b f l c")
        perceptual = perceptual[:, : model.Former_num_time_embeds].to(torch.float32)
        perceptual, _ = torch.split(perceptual, [batch, batch], dim=0)
        frame_idx = min(max(int(getattr(model, "future_frame_idx", 5)), 0), perceptual.shape[1] - 1)
        state_future = torch.stack([perceptual[:, 0], perceptual[:, frame_idx]], dim=1)
        state_future = model.goal_emb(state_future)
        time_pos_emb = model.time_pos_emb.unsqueeze(0).expand(state_future.size(0), -1, -1, -1)
        state_future = state_future + time_pos_emb
        univla_out = model.lam(state_future, lang_text)
        return univla_out["video_action_patches"].squeeze(1).detach()
    finally:
        model.future_feature_mode = old_mode
        model.override_future_feature = old_override


def slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    y = torch.tensor(values, dtype=torch.float32)
    x = torch.arange(len(values), dtype=torch.float32)
    x = x - x.mean()
    denom = torch.clamp(torch.sum(x * x), min=1e-12)
    return float(torch.sum(x * (y - y.mean())) / denom)


def latent_confidence_features(latent_tokens: torch.Tensor) -> List[float]:
    flat = latent_tokens.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(flat.norm() / max(1, flat.numel())),
        float(flat.abs().mean()),
        float(flat.std(unbiased=False)),
        float(flat.abs().max()),
    ]


def action_to_numpy(action: Any) -> np.ndarray:
    if torch.is_tensor(action):
        return action.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def array_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compact_obs(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    rgb_obs = obs.get("rgb_obs", obs)
    return {
        "rgb_static": array_to_numpy(rgb_obs["rgb_static"]).copy(),
        "rgb_gripper": array_to_numpy(rgb_obs["rgb_gripper"]).copy(),
        "robot_obs": array_to_numpy(obs["robot_obs"]).copy(),
        "scene_obs": array_to_numpy(obs["scene_obs"]).copy(),
    }


def static_frame_from_raw(raw: Dict[str, np.ndarray]) -> np.ndarray:
    frame = np.asarray(compact_obs(raw)["rgb_static"])
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0 and frame.min() >= 0.0:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


def write_mp4(frames: Sequence[np.ndarray], path: Path, fps: int = 10) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    height, width = int(first.shape[0]), int(first.shape[1])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    try:
        for frame in frames:
            arr = frame
            if arr.shape[:2] != (height, width):
                arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def action_execution_features(actions: Sequence[Any], max_steps: int = 5) -> List[float]:
    rows = []
    for action in list(actions)[:max_steps]:
        arr = action_to_numpy(action)
        if arr.size:
            rows.append(arr)
    if not rows:
        return [0.0] * 24
    mat = np.stack(rows, axis=0)
    flat = mat.reshape(-1)
    delta = mat[-1] - mat[0] if len(mat) > 1 else np.zeros_like(mat[0])
    stats = [
        float(np.mean(np.abs(flat))),
        float(np.std(flat)),
        float(np.max(np.abs(flat))),
        float(np.mean(np.linalg.norm(mat[:, :3], axis=1))) if mat.shape[1] >= 3 else 0.0,
        float(np.mean(np.abs(mat[:, -1]))) if mat.shape[1] >= 1 else 0.0,
        float(np.linalg.norm(delta)),
        float(len(rows)) / float(max_steps),
        float(np.mean(np.abs(np.diff(mat, axis=0)))) if len(mat) > 1 else 0.0,
    ]
    head = mat[:max_steps, : min(3, mat.shape[1])].reshape(-1).tolist()
    head = (head + [0.0] * 15)[:15]
    return stats + [float(x) for x in head] + [0.0]


def reflection_candidate_feature(
    item: Dict[str, Any],
    latent_tokens: torch.Tensor,
    drift_memory: Sequence[float],
    execution_features: Optional[Sequence[float]] = None,
) -> Tuple[List[float], float]:
    obs_vjepa = item["obs_vjepa"].to(torch.float32).reshape(-1)
    future_vjepa = item["pred_future_vjepa"].to(torch.float32).reshape(-1)
    delta = future_vjepa - obs_vjepa
    cosine = float(torch.dot(obs_vjepa, future_vjepa) / torch.clamp(torch.norm(obs_vjepa) * torch.norm(future_vjepa), min=1e-12))
    contradiction = float(1.0 - cosine)
    memory = list(drift_memory[-4:])
    memory = ([0.0] * (4 - len(memory))) + memory
    drift = slope(memory + [contradiction])
    scalars = [
        float(item["router_logit"]),
        float(item["router_rank"]),
        float(item["vjepa_score"]),
        contradiction,
        drift,
        *memory,
        *latent_confidence_features(latent_tokens),
    ]
    if execution_features is not None:
        scalars.extend(float(x) for x in execution_features)
    feature = torch.cat([obs_vjepa, future_vjepa, delta, torch.tensor(scalars, dtype=torch.float32)])
    return feature.tolist(), contradiction


@torch.no_grad()
def selector_observation_feature(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    goal: Dict[str, Any],
    error_history: Sequence[float],
) -> Tuple[List[float], Dict[str, float]]:
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    language = goal["lang_text"]
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    perceptual = model.TVP_encoder(
        input_rgb,
        [language, language],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    perceptual = perceptual.permute(0, 1, 3, 4, 2).reshape(perceptual.shape[0], perceptual.shape[1], -1, perceptual.shape[2])
    perceptual = perceptual[:, : model.Former_num_time_embeds].to(torch.float32)
    static_feature, gripper_feature = torch.split(perceptual, [rgb_static.shape[0], rgb_static.shape[0]], dim=0)
    frame_idx = min(max(int(model.future_frame_idx), 0), static_feature.shape[1] - 1)
    current = static_feature[:, 0]
    future = static_feature[:, frame_idx]
    if getattr(model, "use_gripper", False):
        current = torch.cat([current, gripper_feature[:, 0]], dim=1)
        future = torch.cat([future, gripper_feature[:, frame_idx]], dim=1)

    current_vec = current.mean(dim=1).reshape(-1)
    future_vec = future.mean(dim=1).reshape(-1)
    delta_vec = future_vec - current_vec
    current_flat = current.reshape(-1)
    future_flat = future.reshape(-1)
    denom = torch.norm(current_flat) * torch.norm(future_flat)
    cosine_error = 0.0 if float(denom) <= 1e-12 else float(1.0 - torch.dot(current_flat, future_flat) / denom)
    l2_error = float(torch.norm(future_flat - current_flat) / max(1, current_flat.numel()))
    memory = list(error_history[-4:])
    memory = ([0.0] * (4 - len(memory))) + memory
    drift = slope(memory + [cosine_error])
    contradiction_score = float(min(1.0, max(0.0, cosine_error / 0.6)))
    scalars = {
        "error_score": cosine_error,
        "l2_error": l2_error,
        "contradiction_score": contradiction_score,
        "drift_score": drift,
        "error_memory_0": memory[0],
        "error_memory_1": memory[1],
        "error_memory_2": memory[2],
        "error_memory_3": memory[3],
    }
    feature = torch.cat(
        [
            current_vec.detach().cpu(),
            future_vec.detach().cpu(),
            delta_vec.detach().cpu(),
            torch.tensor([scalars[key] for key in sorted(scalars)], dtype=torch.float32),
        ]
    )
    return feature.tolist(), scalars


@torch.no_grad()
def selector_feature_parts(
    model: torch.nn.Module,
    obs: Dict[str, Any],
    goal: Dict[str, Any],
    error_history: Sequence[float],
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    rgb_static = obs["rgb_obs"]["rgb_static"].to(model.device)
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"].to(model.device)
    language = goal["lang_text"]
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
    perceptual = model.TVP_encoder(
        input_rgb,
        [language, language],
        model.timestep,
        model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1,
        max_length=model.max_length,
    )
    perceptual = perceptual.permute(0, 1, 3, 4, 2).reshape(perceptual.shape[0], perceptual.shape[1], -1, perceptual.shape[2])
    perceptual = perceptual[:, : model.Former_num_time_embeds].to(torch.float32)
    static_feature, _ = torch.split(perceptual, [rgb_static.shape[0], rgb_static.shape[0]], dim=0)
    frame_idx = min(max(int(model.future_frame_idx), 0), static_feature.shape[1] - 1)
    current = static_feature[:, 0]
    continue_future = static_feature[:, frame_idx]

    current_vec = current.mean(dim=1).reshape(-1)
    continue_vec = continue_future.mean(dim=1).reshape(-1)
    current_flat = current.reshape(-1)
    future_flat = continue_future.reshape(-1)
    denom = torch.norm(current_flat) * torch.norm(future_flat)
    cosine_error = 0.0 if float(denom) <= 1e-12 else float(1.0 - torch.dot(current_flat, future_flat) / denom)
    l2_error = float(torch.norm(future_flat - current_flat) / max(1, current_flat.numel()))
    memory = list(error_history[-4:])
    memory = ([0.0] * (4 - len(memory))) + memory
    drift = slope(memory + [cosine_error])
    contradiction_score = float(min(1.0, max(0.0, cosine_error / 0.6)))
    scalars = {
        "error_score": cosine_error,
        "l2_error": l2_error,
        "contradiction_score": contradiction_score,
        "drift_score": drift,
        "error_memory_0": memory[0],
        "error_memory_1": memory[1],
        "error_memory_2": memory[2],
        "error_memory_3": memory[3],
    }
    return {
        "state": current_vec.detach().cpu().tolist(),
        "continue_future": continue_vec.detach().cpu().tolist(),
        "recover_future": current_vec.detach().cpu().tolist(),
    }, scalars


def pooled_future_feature(feature: torch.Tensor) -> List[float]:
    if feature.ndim == 2:
        return feature.to(torch.float32).mean(dim=0).detach().cpu().tolist()
    return feature.reshape(-1, feature.shape[-1]).to(torch.float32).mean(dim=0).detach().cpu().tolist()


def branch_value(branch: Dict[str, Any], ep_len: int) -> float:
    if not branch.get("success", False):
        return 0.0
    return float(1.0 + (ep_len - float(branch["steps"])) / ep_len)


def raw_env_obs(env: Any) -> Dict[str, np.ndarray]:
    return env.env.get_obs()


def goal_to_device(goal: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in goal.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = goal_to_device(value, device)
        else:
            moved[key] = value
    return moved


def reset_raw(env: Any, raw: Dict[str, np.ndarray]) -> None:
    env.reset(robot_obs=raw["robot_obs"], scene_obs=raw["scene_obs"])


def rollout_once(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    start_info: Dict[str, Any],
    mode: str,
    override_feature: Optional[torch.Tensor] = None,
    override_latent_motion_tokens_up: Optional[torch.Tensor] = None,
) -> Tuple[bool, int, Dict[str, np.ndarray]]:
    model.reset()
    model.future_feature_mode = mode
    model.override_future_feature = override_feature
    model.override_latent_motion_tokens_up = override_latent_motion_tokens_up
    obs = env.get_obs()
    final_raw = raw_env_obs(env)
    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        try:
            action_dbg = action.detach().cpu().numpy()
        except Exception:
            action_dbg = np.asarray(action)
        if getattr(cfg, "debug_action_before_env_step", False) and step < getattr(cfg, "debug_action_steps", 5):
            print(f"\n[DEBUG ACTION] step={step} value={action_dbg}", flush=True)
            print(f"[DEBUG ACTION] step={step} shape={action_dbg.shape}", flush=True)
            print(f"[DEBUG ACTION] step={step} finite={np.isfinite(action_dbg).all()}", flush=True)
            print(
                f"[DEBUG ACTION] step={step} minmax={np.nanmin(action_dbg)} {np.nanmax(action_dbg)}",
                flush=True,
            )
        if not np.isfinite(action_dbg).all():
            raise FloatingPointError(
                f"non-finite action before env.step: mode={mode} subtask={subtask} "
                f"step={step} shape={action_dbg.shape} action={action_dbg}"
            )
        obs, _, _, current_info = env.step(action)
        final_raw = raw_env_obs(env)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, step + 1, final_raw
    return False, cfg.ep_len, final_raw


def rollout_once_with_actions(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    start_info: Dict[str, Any],
    mode: str,
    override_feature: Optional[torch.Tensor] = None,
    override_latent_motion_tokens_up: Optional[torch.Tensor] = None,
    action_trace_steps: int = 5,
    record_frames: bool = False,
) -> Tuple[bool, int, Dict[str, np.ndarray], List[np.ndarray], List[np.ndarray]]:
    model.reset()
    model.future_feature_mode = mode
    model.override_future_feature = override_feature
    model.override_latent_motion_tokens_up = override_latent_motion_tokens_up
    obs = env.get_obs()
    final_raw = raw_env_obs(env)
    actions: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    if record_frames:
        frames.append(static_frame_from_raw(final_raw))
    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        action_np = action_to_numpy(action)
        if len(actions) < action_trace_steps:
            actions.append(action_np.copy())
        obs, _, _, current_info = env.step(action)
        final_raw = raw_env_obs(env)
        if record_frames:
            frames.append(static_frame_from_raw(final_raw))
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, step + 1, final_raw, actions, frames
    return False, cfg.ep_len, final_raw, actions, frames


def rollout_once_with_policy_trace(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    subtask: str,
    goal: Dict[str, Any],
    start_info: Dict[str, Any],
    mode: str,
    override_feature: Optional[torch.Tensor] = None,
    override_latent_motion_tokens_up: Optional[torch.Tensor] = None,
) -> Tuple[bool, int, Dict[str, np.ndarray], List[Dict[str, np.ndarray]], List[np.ndarray]]:
    model.reset()
    model.future_feature_mode = mode
    model.override_future_feature = override_feature
    model.override_latent_motion_tokens_up = override_latent_motion_tokens_up
    obs = env.get_obs()
    final_raw = raw_env_obs(env)
    obs_trace: List[Dict[str, np.ndarray]] = []
    actions: List[np.ndarray] = []
    for step in range(cfg.ep_len):
        obs_trace.append(compact_obs(final_raw))
        action = model.step(obs, goal)
        action_np = action_to_numpy(action)
        actions.append(action_np.copy())
        obs, _, _, current_info = env.step(action)
        final_raw = raw_env_obs(env)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True, step + 1, final_raw, obs_trace, actions
    return False, cfg.ep_len, final_raw, obs_trace, actions


def short_rollout_actions(
    env: Any,
    model: torch.nn.Module,
    cfg: Any,
    goal: Dict[str, Any],
    raw_state: Dict[str, np.ndarray],
    latent_tokens: torch.Tensor,
    steps: int,
) -> List[np.ndarray]:
    reset_raw(env, raw_state)
    model.reset()
    model.future_feature_mode = "current"
    model.override_future_feature = None
    model.override_latent_motion_tokens_up = latent_tokens
    obs = env.get_obs()
    actions: List[np.ndarray] = []
    for _ in range(max(0, steps)):
        action = model.step(obs, goal)
        action_np = action_to_numpy(action)
        actions.append(action_np.copy())
        obs, _, _, _ = env.step(action)
    reset_raw(env, raw_state)
    return actions


def evaluate_defi(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    output_dir: Path,
    record_video_dir: Optional[Path] = None,
    record_sequence_indices: Optional[Sequence[int]] = None,
    record_video_fps: int = 10,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    record_sequence_indices = set(record_sequence_indices or [])
    if record_video_dir is not None:
        record_video_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(sequences, desc="defi")
    for seq_idx, (initial_state, eval_sequence) in enumerate(progress):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        sequence_frames: List[np.ndarray] = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            if record_video_dir is not None and seq_idx in record_sequence_indices:
                success, steps, _, _, frames = rollout_once_with_actions(
                    env,
                    model,
                    task_oracle,
                    cfg,
                    subtask,
                    goal,
                    start_info,
                    "gfdm",
                    None,
                    None,
                    0,
                    record_frames=True,
                )
                sequence_frames.extend(frames)
            else:
                success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append({"subtask_index": sub_idx, "task": subtask, "success": success, "steps": steps})
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
        progress.set_description(format_progress_success(results, sequence_len=len(eval_sequence)))
        if record_video_dir is not None and seq_idx in record_sequence_indices and sequence_frames:
            write_mp4(sequence_frames, record_video_dir / f"sequence_{seq_idx:04d}.mp4", fps=record_video_fps)
    return write_variant(output_dir, "defi", results, logs, task_total, task_success)


def evaluate_prototype_vpp(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    demo_bank: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    feature_cache: Dict[Tuple[str, str], torch.Tensor] = {}
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="prototype_vpp")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            mode = "gfdm"
            override = None
            if subtask in demo_bank:
                key = (subtask, lang_text)
                if key not in feature_cache:
                    feature_cache[key] = demo_future_feature(model, demo_bank[subtask], lang_text)
                mode = "override"
                override = feature_cache[key]
            success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, mode, override)
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": success,
                    "steps": steps,
                    "future_source": "prototype_demo" if override is not None else "gfdm_fallback",
                }
            )
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
    return write_variant(output_dir, "prototype_vpp", results, logs, task_total, task_success)


def evaluate_defi_memory_repair(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    output_dir: Path,
    reflection_generator: QwenReflectionEncoder,
    key_memory_rows: Sequence[Dict[str, Any]],
    key_topk: int,
    recent_k: int,
    trust_threshold: float,
    future_correction_mlp: Optional[FutureCorrectionMLP] = None,
    reflection_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    memory_rows_all: List[Dict[str, Any]] = []
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="defi_memory_repair")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        recent_memory: List[Dict[str, Any]] = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            decision_obs = env.get_obs()
            original_future = defi_future_feature(model, decision_obs, lang_text)
            next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
            recent_summary = summarize_recent_memory(recent_memory, recent_k)
            key_rows = retrieve_key_memory(key_memory_rows, subtask, key_topk)
            key_summary = summarize_key_memory(key_rows)
            rule_text = RULE_MEMORY.get(subtask, "Prefer conservative future calibration and preserve verified progress.")
            prompt = build_memory_reflection_prompt(subtask, next_task, recent_summary, key_summary, rule_text)
            decision = parse_memory_reflection_output(
                reflection_generator.generate_reflection(prompt, max_new_tokens=192)
            )
            use_repair = (
                future_correction_mlp is not None
                and reflection_encoder is not None
                and decision.trust_score < trust_threshold
            )
            if use_repair:
                repaired_future, repaired_tokens = apply_single_future_repair(
                    model,
                    decision_obs,
                    goal,
                    original_future,
                    (
                        f"mismatch_type: {decision.mismatch_type} | "
                        f"correction_direction: {decision.correction_direction} | "
                        f"recoverability: {decision.recoverability} | "
                        f"explanation: {decision.explanation}"
                    ),
                    future_correction_mlp,
                    reflection_encoder,
                )
                success, steps, _ = rollout_once(
                    env,
                    model,
                    task_oracle,
                    cfg,
                    subtask,
                    goal,
                    start_info,
                    "current",
                    None,
                    repaired_tokens,
                )
                used_future = "repaired"
            else:
                repaired_future = None
                success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")
                used_future = "original"
            mismatch_type = infer_mismatch_type(subtask, success, steps, cfg.ep_len)
            row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "success": bool(success),
                "steps": int(steps),
                "trust_score": float(decision.trust_score),
                "mismatch_type": mismatch_type if mismatch_type != "none" else decision.mismatch_type,
                "correction_direction": decision.correction_direction,
                "recoverability": decision.recoverability,
                "reflection_text": decision.raw_text,
                "used_future": used_future,
                "repair_helped": bool(use_repair and success),
                "rule_text": rule_text,
                "recent_summary": recent_summary,
                "key_summary": key_summary,
            }
            recent_memory.append(row)
            recent_memory = recent_memory[-recent_k:]
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(row)
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
    summary = write_variant(output_dir, "defi_memory_repair", results, logs, task_total, task_success)
    variant_dir = output_dir / "defi_memory_repair"
    with (variant_dir / "memory_rollout_rows.jsonl").open("w") as handle:
        for row in memory_rows_all:
            handle.write(json.dumps(row) + "\n")
    summary["memory_rows"] = len(memory_rows_all)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_defi_postexec_language_conditioned(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    output_dir: Path,
    reflection_generator: Optional[QwenReflectionEncoder],
    key_memory_rows: Sequence[Dict[str, Any]],
    key_topk: int,
    recent_k: int,
) -> Dict[str, Any]:
    variant = "defi_postexec_language_conditioned"
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = variant_dir / "sequences.jsonl"
    memory_rows_path = variant_dir / "memory_rollout_rows.jsonl"
    logs = load_existing_sequence_logs(sequences_path)
    results, task_total, task_success = rebuild_counts_from_logs(logs)
    completed = {int(row.get("sequence_index", -1)) for row in logs}
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant)):
        if seq_idx in completed:
            continue
        trace_rollout(f"{variant} seq={seq_idx} begin")
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        recent_memory: List[Dict[str, Any]] = []
        prev_row: Optional[Dict[str, Any]] = None
        for sub_idx, subtask in enumerate(eval_sequence):
            base_lang_text = cfg.annotations[subtask][0]
            key_rows_for_lang = retrieve_key_memory(key_memory_rows, subtask, key_topk)
            important_summary_for_lang = summarize_important_key_memory(key_rows_for_lang)
            lang_text = (
                base_lang_text
                if prev_row is None
                else build_augmented_future_instruction(base_lang_text, prev_row, important_summary_for_lang)
            )
            goal = lang_embeddings.get_lang_goal(base_lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before rollout_once")
            success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after rollout_once success={bool(success)} steps={int(steps)}")
            mismatch_type = infer_mismatch_type(subtask, success, steps, cfg.ep_len)
            next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
            recent_summary = summarize_recent_memory(recent_memory, recent_k)
            key_summary = summarize_key_memory(key_rows_for_lang)
            important_summary = summarize_important_key_memory(key_rows_for_lang)
            rule_text = RULE_MEMORY.get(subtask, "Prefer conservative future calibration and preserve verified progress.")
            if reflection_generator is None:
                decision = template_postexec_reflection_decision(
                    subtask,
                    bool(success),
                    int(steps),
                    int(cfg.ep_len),
                    mismatch_type,
                )
            else:
                prompt = build_postexec_reflection_prompt(
                    subtask,
                    next_task,
                    bool(success),
                    int(steps),
                    mismatch_type,
                    recent_summary,
                    key_summary,
                    important_summary,
                    rule_text,
                )
                decision = parse_memory_reflection_output(
                    reflection_generator.generate_reflection(prompt, max_new_tokens=192)
                )
            row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "success": bool(success),
                "steps": int(steps),
                "trust_score": float(decision.trust_score),
                "mismatch_type": mismatch_type if mismatch_type != "none" else decision.mismatch_type,
                "correction_direction": decision.correction_direction,
                "recoverability": decision.recoverability,
                "reflection_text": decision.raw_text,
                "used_future": "original",
                "language_augmented": prev_row is not None,
                "lang_text": lang_text,
                "rule_text": rule_text,
                "recent_summary": recent_summary,
                "key_summary": key_summary,
                "important_summary": important_summary,
            }
            recent_memory.append(row)
            recent_memory = recent_memory[-recent_k:]
            prev_row = row
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(row)
            if success:
                success_counter += 1
            else:
                break
        sequence_row = {"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks}
        results.append(success_counter)
        logs.append(sequence_row)
        append_jsonl_row(sequences_path, sequence_row)
        for row in subtasks:
            append_jsonl_row(memory_rows_path, row)
        summary = write_variant(output_dir, variant, results, logs, task_total, task_success)
        summary["memory_rows"] = sum(len(item.get("subtasks", [])) for item in logs)
        (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary = write_variant(output_dir, variant, results, logs, task_total, task_success)
    summary["memory_rows"] = sum(len(item.get("subtasks", [])) for item in logs)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_defi_memory_reflect_only(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    output_dir: Path,
    reflection_generator: QwenReflectionEncoder,
    key_memory_rows: Sequence[Dict[str, Any]],
    key_topk: int,
    recent_k: int,
) -> Dict[str, Any]:
    variant = "defi_memory_reflect_only"
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = variant_dir / "sequences.jsonl"
    memory_rows_path = variant_dir / "memory_rollout_rows.jsonl"
    logs = load_existing_sequence_logs(sequences_path)
    results, task_total, task_success = rebuild_counts_from_logs(logs)
    completed = {int(row.get("sequence_index", -1)) for row in logs}
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant)):
        if seq_idx in completed:
            continue
        trace_rollout(f"{variant} seq={seq_idx} begin")
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        trace_rollout(f"{variant} seq={seq_idx} before env.reset")
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        trace_rollout(f"{variant} seq={seq_idx} after env.reset")
        success_counter = 0
        subtasks = []
        recent_memory: List[Dict[str, Any]] = []
        for sub_idx, subtask in enumerate(eval_sequence):
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before get_lang_goal")
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after get_lang_goal")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before env.get_info")
            start_info = env.get_info()
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after env.get_info")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before env.get_obs")
            _ = env.get_obs()
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after env.get_obs")
            next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
            recent_summary = summarize_recent_memory(recent_memory, recent_k)
            key_rows = retrieve_key_memory(key_memory_rows, subtask, key_topk)
            key_summary = summarize_key_memory(key_rows)
            rule_text = RULE_MEMORY.get(subtask, "Prefer conservative future calibration and preserve verified progress.")
            prompt = build_memory_reflection_prompt(subtask, next_task, recent_summary, key_summary, rule_text)
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before reflection")
            decision = parse_memory_reflection_output(
                reflection_generator.generate_reflection(prompt, max_new_tokens=192)
            )
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after reflection trust={decision.trust_score:.3f}")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before rollout_once")
            success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "gfdm")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after rollout_once success={bool(success)} steps={int(steps)}")
            mismatch_type = infer_mismatch_type(subtask, success, steps, cfg.ep_len)
            row = {
                "sequence_index": seq_idx,
                "subtask_index": sub_idx,
                "task": subtask,
                "success": bool(success),
                "steps": int(steps),
                "trust_score": float(decision.trust_score),
                "mismatch_type": mismatch_type if mismatch_type != "none" else decision.mismatch_type,
                "correction_direction": decision.correction_direction,
                "recoverability": decision.recoverability,
                "reflection_text": decision.raw_text,
                "used_future": "original",
                "repair_helped": False,
                "rule_text": rule_text,
                "recent_summary": recent_summary,
                "key_summary": key_summary,
            }
            recent_memory.append(row)
            recent_memory = recent_memory[-recent_k:]
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(row)
            if success:
                success_counter += 1
            else:
                break
        sequence_row = {"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks}
        results.append(success_counter)
        logs.append(sequence_row)
        append_jsonl_row(sequences_path, sequence_row)
        for row in subtasks:
            append_jsonl_row(memory_rows_path, row)
        trace_rollout(f"{variant} seq={seq_idx} written subtasks={len(subtasks)} success_counter={success_counter}")
        summary = write_variant(output_dir, variant, results, logs, task_total, task_success)
        summary["memory_rows"] = sum(len(item.get("subtasks", [])) for item in logs)
        (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary = write_variant(output_dir, variant, results, logs, task_total, task_success)
    summary["memory_rows"] = sum(len(item.get("subtasks", [])) for item in logs)
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_nn_vpp(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    source_counter = Counter()
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="nn_vpp")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            decision_obs = env.get_obs()
            override = retrieve_vpp_future(model, decision_obs, lang_text, subtask, vpp_bank)
            mode = "override" if override is not None else "gfdm"
            source = "nn_vpp" if override is not None else "gfdm_fallback"
            source_counter[source] += 1
            success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, mode, override)
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": success,
                    "steps": steps,
                    "future_source": source,
                }
            )
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
    summary = write_variant(output_dir, "nn_vpp", results, logs, task_total, task_success)
    summary["future_source_counts"] = dict(source_counter)
    (output_dir / "nn_vpp" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_vjepa_nn_vpp(
    env: Any,
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    source_counter = Counter()
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="vjepa_nn_vpp")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            decision_obs = env.get_obs()
            override = retrieve_vjepa_vpp_future(model, vjepa_encoder, decision_obs, subtask, vpp_bank)
            mode = "override" if override is not None else "gfdm"
            source = "vjepa_nn_vpp" if override is not None else "gfdm_fallback"
            source_counter[source] += 1
            success, steps, _ = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, mode, override)
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": success,
                    "steps": steps,
                    "future_source": source,
                }
            )
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
    summary = write_variant(output_dir, "vjepa_nn_vpp", results, logs, task_total, task_success)
    summary["future_source_counts"] = dict(source_counter)
    (output_dir / "vjepa_nn_vpp" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_vjepa_gidm_tokens(
    env: Any,
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
    visual_mode: str = "gfdm",
    variant_name: str = "vjepa_gidm_tokens",
    vjepa_adapter: Optional[VjepaFutureAdapter] = None,
    vjepa_cross_attention: Optional[VjepaCrossAttentionFuture] = None,
    vjepa_cross_attention_k: int = 8,
    vjepa_moe: Optional[VjepaMoeFuture] = None,
    vjepa_moe_k: int = 8,
    future_correction_mlp: Optional[FutureCorrectionMLP] = None,
    reflection_encoder: Optional[Any] = None,
    reflection_text_mode: str = "task_hypothesis",
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    source_counter = Counter()
    reflection_traces = []
    reflection_memory_rows = []
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant_name)):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            decision_obs = env.get_obs()
            trace = None
            if vjepa_moe is not None and future_correction_mlp is not None and reflection_encoder is not None:
                future_feature, trace = retrieve_corrected_moe_vjepa_future(
                    model,
                    vjepa_encoder,
                    vjepa_moe,
                    future_correction_mlp,
                    reflection_encoder,
                    decision_obs,
                    subtask,
                    vpp_bank,
                    vjepa_moe_k,
                    reflection_text_mode=reflection_text_mode,
                    return_trace=True,
                )
            elif vjepa_moe is not None:
                future_feature = retrieve_moe_vjepa_future(
                    model,
                    vjepa_encoder,
                    vjepa_moe,
                    decision_obs,
                    subtask,
                    vpp_bank,
                    vjepa_moe_k,
                )
            elif vjepa_cross_attention is not None:
                future_feature = retrieve_cross_attention_vjepa_future(
                    model,
                    vjepa_encoder,
                    vjepa_cross_attention,
                    decision_obs,
                    subtask,
                    vpp_bank,
                    vjepa_cross_attention_k,
                )
            elif vjepa_adapter is None:
                future_feature = retrieve_vjepa_vpp_future(model, vjepa_encoder, decision_obs, subtask, vpp_bank)
            else:
                future_feature = retrieve_trained_vjepa_future(model, vjepa_encoder, vjepa_adapter, decision_obs, subtask, vpp_bank)
            latent_tokens = None
            source = "gfdm_fallback"
            if future_feature is not None:
                latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, future_feature)
                source = "vjepa_gidm_tokens"
            source_counter[source] += 1
            success, steps, _ = rollout_once(
                env,
                model,
                task_oracle,
                cfg,
                subtask,
                goal,
                start_info,
                visual_mode,
                None,
                latent_tokens,
            )
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            if trace is not None:
                trace_row = {
                    "sequence_index": int(seq_idx),
                    "subtask_index": int(sub_idx),
                    "task": subtask,
                    "lang_text": lang_text,
                    "success": bool(success),
                    "steps": int(steps),
                    "token_source": source,
                    **trace,
                }
                reflection_traces.append(trace_row)
                reflection_memory_rows.append(
                    {
                        "sequence_index": int(seq_idx),
                        "subtask_index": int(sub_idx),
                        "task": subtask,
                        "context": {
                            "lang_text": lang_text,
                            "eval_sequence": list(eval_sequence),
                        },
                        "imagined_future_source": source,
                        "outcome": "success" if success else "failure",
                        "future_outcome_mismatch": "unknown" if success else "needs_reflection",
                        "reflection_texts": trace.get("reflection_texts", []),
                        "hypothesis_tasks": trace.get("hypothesis_tasks", []),
                        "repair_delta_norms": trace.get("repair_delta_norms", []),
                        "selected_demo_index": trace.get("selected_demo_index"),
                        "selected_demo_score": trace.get("selected_demo_score"),
                        "repair_success": bool(success),
                    }
                )
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": success,
                    "steps": steps,
                    "token_source": source,
                }
            )
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
    summary = write_variant(output_dir, variant_name, results, logs, task_total, task_success)
    summary["token_source_counts"] = dict(source_counter)
    summary["visual_future_mode"] = visual_mode
    if reflection_traces:
        variant_dir = output_dir / variant_name
        with (variant_dir / "reflection_traces.jsonl").open("w") as handle:
            for row in reflection_traces:
                handle.write(json.dumps(row) + "\n")
        with (variant_dir / "reflection_memory.jsonl").open("w") as handle:
            for row in reflection_memory_rows:
                handle.write(json.dumps(row) + "\n")
        summary["reflection_trace_count"] = len(reflection_traces)
        summary["reflection_memory_count"] = len(reflection_memory_rows)
    (output_dir / variant_name / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_gt_oracle_tokens(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
    num_candidates: int,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    choice_counter = Counter()
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc="gt_oracle_tokens_nogfdm")):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            branch_start_raw = raw_env_obs(env)
            decision_obs = env.get_obs()
            candidates = vpp_bank.get(subtask, [])[:num_candidates]
            branch_results = []
            if not candidates:
                success, steps, final_raw = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "current")
                branch_results.append({"idx": -1, "success": success, "steps": steps, "final_raw": final_raw})
            else:
                for idx, item in enumerate(candidates):
                    reset_raw(env, branch_start_raw)
                    latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, item["future"].to(model.device))
                    success, steps, final_raw = rollout_once(
                        env,
                        model,
                        task_oracle,
                        cfg,
                        subtask,
                        goal,
                        start_info,
                        "current",
                        None,
                        latent_tokens,
                    )
                    branch_results.append({"idx": idx, "success": success, "steps": steps, "final_raw": final_raw})
            successful = [item for item in branch_results if item["success"]]
            chosen = min(successful, key=lambda item: item["steps"]) if successful else branch_results[0]
            reset_raw(env, chosen["final_raw"])
            choice_counter[str(chosen["idx"])] += 1
            task_total[subtask] += 1
            task_success[subtask] += int(chosen["success"])
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": bool(chosen["success"]),
                    "steps": int(chosen["steps"]),
                    "chosen_candidate": int(chosen["idx"]),
                    "num_candidates": len(candidates),
                }
            )
            if chosen["success"]:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
    summary = write_variant(output_dir, "gt_oracle_tokens_nogfdm", results, logs, task_total, task_success)
    summary["choice_counts"] = dict(choice_counter)
    summary["visual_future_mode"] = "current"
    summary["oracle_candidates_per_task"] = num_candidates
    (output_dir / "gt_oracle_tokens_nogfdm" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_moe_expert_oracle_tokens(
    env: Any,
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    vjepa_moe: VjepaMoeFuture,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
    moe_k: int,
    variant_name: str,
    top_router_k: Optional[int] = None,
    reflection_critic: Optional[ReflectionCritic] = None,
    collect_reflection_dataset: bool = False,
    selective_gate: bool = False,
    switch_high: float = 0.8,
    safe_low: float = 0.4,
    risk_margin: float = 0.25,
    value_gate_margin: Optional[float] = None,
    execution_signal_steps: int = 0,
    universal_oracle_router: Optional[UniversalMoeOracleRouter] = None,
    portfolio_reflections: Optional[Dict[Tuple[int, int], str]] = None,
    portfolio_reflection_encoder: Optional[Any] = None,
    online_postexec_reflection_generator: Optional[QwenReflectionEncoder] = None,
    online_postexec_reflection_max_new_tokens: int = 96,
    record_video_dir: Optional[Path] = None,
    record_sequence_indices: Optional[Sequence[int]] = None,
    record_video_fps: int = 10,
) -> Dict[str, Any]:
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    choice_counter = Counter()
    reflection_features = []
    reflection_values = []
    reflection_meta = []
    online_reflection_rows = []
    record_sequence_indices = set(record_sequence_indices or [])
    if record_video_dir is not None:
        record_video_dir.mkdir(parents=True, exist_ok=True)
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant_name)):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        success_counter = 0
        subtasks = []
        drift_memory: List[float] = []
        latest_online_reflection_text: Optional[str] = None
        sequence_frames: List[np.ndarray] = []
        for sub_idx, subtask in enumerate(eval_sequence):
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            start_info = env.get_info()
            branch_start_raw = raw_env_obs(env)
            decision_obs = env.get_obs()
            expert_items, router_order = retrieve_moe_vjepa_expert_futures(
                model,
                vjepa_encoder,
                vjepa_moe,
                decision_obs,
                subtask,
                vpp_bank,
                moe_k,
            )
            if top_router_k is not None:
                allowed = set(router_order[:top_router_k])
                expert_items = [item for item in expert_items if item["expert"] in allowed]
            branch_results = []
            if not expert_items:
                success, steps, final_raw = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "current")
                branch_results.append({"expert": -1, "success": success, "steps": steps, "final_raw": final_raw})
            elif universal_oracle_router is not None:
                latent_tokens_by_expert = {
                    int(item["expert"]): latent_tokens_from_future_feature(model, decision_obs, goal, item["future"])
                    for item in expert_items
                }
                reflection_embedding = None
                if latest_online_reflection_text is not None and portfolio_reflection_encoder is not None:
                    reflection_embedding = portfolio_reflection_encoder.encode([latest_online_reflection_text]).squeeze(0)
                elif portfolio_reflections is not None and portfolio_reflection_encoder is not None:
                    reflection_embedding = portfolio_reflection_encoder.encode(
                        [
                            portfolio_reflections.get(
                                (seq_idx, sub_idx),
                                (
                                    f"progress_status: uncertain | failure_type: uncertain | "
                                    f"prefer_next: choose viable future for {subtask} | "
                                    "avoid_next: avoid failed futures | correction_direction: use portfolio utility | "
                                    f"memory_query: {subtask}"
                                ),
                            )
                        ]
                    ).squeeze(0)
                features = universal_moe_oracle_router_features(
                    expert_items,
                    latent_tokens_by_expert,
                    drift_memory,
                    cfg,
                    getattr(universal_oracle_router, "feature_layout", None),
                    reflection_embedding,
                )
                with torch.no_grad():
                    scores = universal_oracle_router(features.unsqueeze(0).to(model.device)).squeeze(0)
                expert_to_score = {
                    int(item["expert"]): float(scores[idx].detach().cpu())
                    for idx, item in enumerate(sorted(expert_items, key=lambda row: row["expert"]))
                }
                item = max(expert_items, key=lambda row: expert_to_score[int(row["expert"])])
                latent_tokens = latent_tokens_by_expert[int(item["expert"])]
                if record_video_dir is not None and seq_idx in record_sequence_indices:
                    success, steps, final_raw, _, frames = rollout_once_with_actions(
                        env,
                        model,
                        task_oracle,
                        cfg,
                        subtask,
                        goal,
                        start_info,
                        "current",
                        None,
                        latent_tokens,
                        0,
                        record_frames=True,
                    )
                else:
                    success, steps, final_raw = rollout_once(
                        env,
                        model,
                        task_oracle,
                        cfg,
                        subtask,
                        goal,
                        start_info,
                        "current",
                        None,
                        latent_tokens,
                    )
                    frames = []
                _, contradiction = reflection_candidate_feature(item, latent_tokens, drift_memory, None)
                branch_results.append(
                    {
                        "expert": item["expert"],
                        "router_rank": item["router_rank"],
                        "router_logit": item["router_logit"],
                        "demo_index": item["demo_index"],
                        "vjepa_score": item["vjepa_score"],
                        "learned_score": expert_to_score[int(item["expert"])],
                        "contradiction": contradiction,
                        "success": success,
                        "steps": steps,
                        "final_raw": final_raw,
                        "frames": frames,
                    }
                )
                drift_memory.append(contradiction)
            elif reflection_critic is not None:
                scored_items = []
                for item in expert_items:
                    latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, item["future"])
                    exec_features = None
                    if execution_signal_steps > 0:
                        actions = short_rollout_actions(env, model, cfg, goal, branch_start_raw, latent_tokens, execution_signal_steps)
                        exec_features = action_execution_features(actions, execution_signal_steps)
                    feature, contradiction = reflection_candidate_feature(item, latent_tokens, drift_memory, exec_features)
                    with torch.no_grad():
                        score = float(reflection_critic(torch.tensor(feature, dtype=torch.float32, device=model.device).unsqueeze(0)).item())
                    scored_items.append((score, item, latent_tokens, feature, contradiction))
                if selective_gate:
                    top1 = min(scored_items, key=lambda row: row[1]["router_rank"])
                    best_safe = min(scored_items, key=lambda row: row[0])
                    top1_risk = 1.0 / (1.0 + np.exp(-top1[0]))
                    best_risk = 1.0 / (1.0 + np.exp(-best_safe[0]))
                    if top1_risk > switch_high and best_risk < safe_low and (top1_risk - best_risk) > risk_margin:
                        score, item, latent_tokens, feature, contradiction = best_safe
                    else:
                        score, item, latent_tokens, feature, contradiction = top1
                else:
                    if value_gate_margin is None:
                        score, item, latent_tokens, feature, contradiction = max(scored_items, key=lambda row: row[0])
                    else:
                        top1 = min(scored_items, key=lambda row: row[1]["router_rank"])
                        best_value = max(scored_items, key=lambda row: row[0])
                        if best_value[0] > top1[0] + value_gate_margin:
                            score, item, latent_tokens, feature, contradiction = best_value
                        else:
                            score, item, latent_tokens, feature, contradiction = top1
                    top1_risk = None
                    best_risk = None
                success, steps, final_raw = rollout_once(
                    env,
                    model,
                    task_oracle,
                    cfg,
                    subtask,
                    goal,
                    start_info,
                    "current",
                    None,
                    latent_tokens,
                )
                branch_results.append(
                    {
                        "expert": item["expert"],
                        "router_rank": item["router_rank"],
                        "router_logit": item["router_logit"],
                        "demo_index": item["demo_index"],
                        "vjepa_score": item["vjepa_score"],
                        "reflection_score": score,
                        "top1_risk": top1_risk,
                        "best_risk": best_risk,
                        "contradiction": contradiction,
                        "success": success,
                        "steps": steps,
                        "final_raw": final_raw,
                    }
                )
                drift_memory.append(contradiction)
            else:
                for item in expert_items:
                    reset_raw(env, branch_start_raw)
                    latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, item["future"])
                    if execution_signal_steps > 0:
                        success, steps, final_raw, actions, _ = rollout_once_with_actions(
                            env,
                            model,
                            task_oracle,
                            cfg,
                            subtask,
                            goal,
                            start_info,
                            "current",
                            None,
                            latent_tokens,
                            execution_signal_steps,
                        )
                        exec_features = action_execution_features(actions, execution_signal_steps)
                    else:
                        feature_exec = None
                        success, steps, final_raw = rollout_once(
                            env,
                            model,
                            task_oracle,
                            cfg,
                            subtask,
                            goal,
                            start_info,
                            "current",
                            None,
                            latent_tokens,
                        )
                        exec_features = feature_exec
                    feature, contradiction = reflection_candidate_feature(item, latent_tokens, drift_memory, exec_features)
                    branch_results.append(
                        {
                            "expert": item["expert"],
                            "router_rank": item["router_rank"],
                            "router_logit": item["router_logit"],
                            "demo_index": item["demo_index"],
                            "vjepa_score": item["vjepa_score"],
                            "contradiction": contradiction,
                            "success": success,
                            "steps": steps,
                            "final_raw": final_raw,
                        }
                    )
                    if collect_reflection_dataset:
                        reflection_features.append(feature)
                        reflection_values.append(branch_value({"success": success, "steps": steps}, cfg.ep_len))
                        reflection_meta.append(
                            {
                                "sequence_index": seq_idx,
                                "subtask_index": sub_idx,
                                "task": subtask,
                                "expert": int(item["expert"]),
                                "router_rank": int(item["router_rank"]),
                                "success": bool(success),
                                "steps": int(steps),
                            }
                        )
            successful = [item for item in branch_results if item["success"]]
            chosen = min(successful, key=lambda item: item["steps"]) if successful else branch_results[0]
            reset_raw(env, chosen["final_raw"])
            success = bool(chosen["success"])
            if "contradiction" in chosen:
                drift_memory.append(float(chosen["contradiction"]))
            if online_postexec_reflection_generator is not None:
                next_task = eval_sequence[sub_idx + 1] if sub_idx + 1 < len(eval_sequence) else None
                prompt = build_postexec_qwen_prompt(subtask, success, int(chosen["steps"]), chosen, next_task)
                reflection_text = online_postexec_reflection_generator.generate_reflection(
                    prompt,
                    max_new_tokens=online_postexec_reflection_max_new_tokens,
                )
                latest_online_reflection_text = reflection_text
                online_reflection_rows.append(
                    {
                        "sequence_index": int(seq_idx),
                        "subtask_index": int(sub_idx),
                        "task": subtask,
                        "next_task": next_task,
                        "success": success,
                        "steps": int(chosen["steps"]),
                        "chosen_expert": int(chosen["expert"]),
                        "prompt": prompt,
                        "reflection_text": reflection_text,
                    }
                )
            choice_counter[str(chosen["expert"])] += 1
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": success,
                    "steps": int(chosen["steps"]),
                    "chosen_expert": int(chosen["expert"]),
                    "router_order": router_order,
                    "branches": [{k: v for k, v in item.items() if k not in {"final_raw", "frames"}} for item in branch_results],
                }
            )
            if record_video_dir is not None and seq_idx in record_sequence_indices:
                sequence_frames.extend(chosen.get("frames", []))
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
        if record_video_dir is not None and seq_idx in record_sequence_indices and sequence_frames:
            write_mp4(sequence_frames, record_video_dir / f"sequence_{seq_idx:04d}.mp4", fps=record_video_fps)
    summary = write_variant(output_dir, variant_name, results, logs, task_total, task_success)
    summary["choice_counts"] = dict(choice_counter)
    summary["visual_future_mode"] = "current"
    summary["moe_k"] = int(moe_k)
    summary["top_router_k"] = top_router_k
    if online_reflection_rows:
        with (output_dir / variant_name / "online_postexec_reflections.jsonl").open("w") as handle:
            for row in online_reflection_rows:
                handle.write(json.dumps(row) + "\n")
        summary["online_postexec_reflections"] = len(online_reflection_rows)
    (output_dir / variant_name / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if collect_reflection_dataset and reflection_features:
        np.savez_compressed(
            output_dir / variant_name / "reflection_dataset.npz",
            x=np.asarray(reflection_features, dtype=np.float32),
            y=np.asarray(reflection_values, dtype=np.float32),
            meta=np.asarray([json.dumps(row) for row in reflection_meta]),
        )
        summary["reflection_examples"] = len(reflection_features)
        (output_dir / variant_name / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_moe_universal_self_dataset(
    env: Any,
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    vjepa_moe: VjepaMoeFuture,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
    moe_k: int,
    execution_signal_steps: int = 5,
    record_video_dir: Optional[Path] = None,
    record_sequence_indices: Optional[Sequence[int]] = None,
    record_video_fps: int = 10,
) -> Dict[str, Any]:
    variant_name = "collect_moe_universal_self_dataset"
    dataset_dir = output_dir / variant_name / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / variant_name / "index.jsonl"
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    choice_counter = Counter()
    examples = 0
    record_sequence_indices = set(record_sequence_indices or [])
    if record_video_dir is not None:
        record_video_dir.mkdir(parents=True, exist_ok=True)

    with index_path.open("w") as index_handle:
        for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant_name)):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success_counter = 0
            subtasks = []
            drift_memory: List[float] = []
            sequence_frames: List[np.ndarray] = []
            traj_dir = dataset_dir / f"traj_{seq_idx:04d}"
            traj_dir.mkdir(parents=True, exist_ok=True)

            for sub_idx, subtask in enumerate(eval_sequence):
                lang_text = cfg.annotations[subtask][0]
                goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
                goal["lang_text"] = lang_text
                start_info = env.get_info()
                branch_start_raw = raw_env_obs(env)
                decision_obs = env.get_obs()
                expert_items, router_order = retrieve_moe_vjepa_expert_futures(
                    model,
                    vjepa_encoder,
                    vjepa_moe,
                    decision_obs,
                    subtask,
                    vpp_bank,
                    moe_k,
                )
                if not expert_items:
                    success, steps, final_raw = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, "current")
                    reset_raw(env, final_raw)
                    task_total[subtask] += 1
                    task_success[subtask] += int(success)
                    subtasks.append(
                        {
                            "subtask_index": sub_idx,
                            "task": subtask,
                            "success": bool(success),
                            "steps": int(steps),
                            "chosen_expert": -1,
                            "router_order": [],
                            "branches": [],
                            "dataset_file": None,
                        }
                    )
                    if success:
                        success_counter += 1
                        continue
                    break

                branch_results = []
                future_rows = []
                value_rows = []
                success_rows = []
                steps_rows = []
                contradiction_rows = []
                token_rows = []
                exec_rows = []
                action_rows = []

                for item in expert_items:
                    reset_raw(env, branch_start_raw)
                    latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, item["future"])
                    success, steps, final_raw, actions, frames = rollout_once_with_actions(
                        env,
                        model,
                        task_oracle,
                        cfg,
                        subtask,
                        goal,
                        start_info,
                        "current",
                        None,
                        latent_tokens,
                        execution_signal_steps,
                        record_frames=bool(record_video_dir is not None and seq_idx in record_sequence_indices),
                    )
                    exec_features = action_execution_features(actions, execution_signal_steps)
                    _, contradiction = reflection_candidate_feature(item, latent_tokens, drift_memory, exec_features)
                    branch = {
                        "expert": int(item["expert"]),
                        "router_rank": int(item["router_rank"]),
                        "router_logit": float(item["router_logit"]),
                        "demo_index": int(item["demo_index"]),
                        "vjepa_score": float(item["vjepa_score"]),
                        "contradiction": float(contradiction),
                        "success": bool(success),
                        "steps": int(steps),
                        "final_raw": final_raw,
                        "frames": frames,
                    }
                    branch_results.append(branch)
                    future_rows.append(item["pred_future_vjepa"].detach().cpu().numpy().astype(np.float32))
                    value_rows.append(branch_value(branch, cfg.ep_len))
                    success_rows.append(float(success))
                    steps_rows.append(float(steps))
                    contradiction_rows.append(float(contradiction))
                    token_rows.append(latent_tokens.detach().cpu().reshape(-1).numpy().astype(np.float32))
                    exec_rows.append(np.asarray(exec_features, dtype=np.float32))
                    padded_actions = np.zeros((execution_signal_steps, 7), dtype=np.float32)
                    for action_idx, action in enumerate(actions[:execution_signal_steps]):
                        arr = np.asarray(action, dtype=np.float32).reshape(-1)
                        padded_actions[action_idx, : min(7, arr.size)] = arr[:7]
                    action_rows.append(padded_actions)

                chosen = min(branch_results, key=lambda item: item["router_rank"])
                reset_raw(env, chosen["final_raw"])
                success = bool(chosen["success"])
                if record_video_dir is not None and seq_idx in record_sequence_indices:
                    sequence_frames.extend(chosen.get("frames", []))
                post_task_obs = env.get_obs()
                post_task_embedding = vjepa_encode_tensor(
                    vjepa_encoder,
                    vjepa_image_tensor_from_obs(post_task_obs["rgb_obs"]["rgb_static"], model.device),
                ).detach().cpu().numpy().astype(np.float32)
                obs_embedding = expert_items[0]["obs_vjepa"].detach().cpu().numpy().astype(np.float32)
                task_delta_embedding = (post_task_embedding - obs_embedding).astype(np.float32)
                if "contradiction" in chosen:
                    drift_memory.append(float(chosen["contradiction"]))
                choice_counter[str(chosen["expert"])] += 1
                task_total[subtask] += 1
                task_success[subtask] += int(success)

                order_by_expert = np.argsort([item["expert"] for item in expert_items])
                ordered_experts = np.asarray(
                    [item["expert"] for item in sorted(expert_items, key=lambda row: row["expert"])],
                    dtype=np.int32,
                )
                ordered_values = np.asarray(value_rows, dtype=np.float32)[order_by_expert]
                oracle_idx = int(np.argmax(ordered_values))
                oracle_expert = int(ordered_experts[oracle_idx])
                oracle_value = float(ordered_values[oracle_idx])
                router_logits = np.asarray(
                    [item["router_logit"] for item in sorted(expert_items, key=lambda row: row["expert"])],
                    dtype=np.float32,
                )
                router_scores = np.exp(router_logits - np.max(router_logits))
                router_scores = router_scores / np.clip(np.sum(router_scores), 1e-12, None)
                step_file = traj_dir / f"step_{sub_idx:04d}.npz"
                np.savez_compressed(
                    step_file,
                    t=np.asarray([sub_idx], dtype=np.int32),
                    obs_embedding=obs_embedding,
                    post_task_obs_embedding=post_task_embedding,
                    next_task_obs_embedding=post_task_embedding,
                    task_delta_embedding=task_delta_embedding,
                    expert_futures=np.stack(future_rows, axis=0)[order_by_expert],
                    router_logits=router_logits,
                    router_scores=router_scores.astype(np.float32),
                    gidm_tokens=np.stack(token_rows, axis=0)[order_by_expert],
                    expert_values=ordered_values,
                    expert_success=np.asarray(success_rows, dtype=np.float32)[order_by_expert],
                    expert_steps=np.asarray(steps_rows, dtype=np.float32)[order_by_expert],
                    expert_contradiction=np.asarray(contradiction_rows, dtype=np.float32)[order_by_expert],
                    expert_execution_features=np.stack(exec_rows, axis=0)[order_by_expert],
                    expert_short_actions=np.stack(action_rows, axis=0)[order_by_expert],
                    router_order=np.asarray(router_order, dtype=np.int32),
                    chosen_expert=np.asarray([int(chosen["expert"])], dtype=np.int32),
                    oracle_expert=np.asarray([oracle_expert], dtype=np.int32),
                    oracle_value=np.asarray([oracle_value], dtype=np.float32),
                    rollout_success=np.asarray([float(success)], dtype=np.float32),
                    rollout_steps=np.asarray([float(chosen["steps"])], dtype=np.float32),
                )
                index_row = {
                    "trajectory_id": f"traj_{seq_idx:04d}",
                    "sequence_index": seq_idx,
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "path": str(step_file.relative_to(output_dir / variant_name)),
                    "chosen_expert": int(chosen["expert"]),
                    "oracle_expert": oracle_expert,
                    "oracle_value": oracle_value,
                    "router_order": router_order,
                    "success": success,
                    "steps": int(chosen["steps"]),
                    "branches": [{k: v for k, v in item.items() if k not in {"final_raw", "frames"}} for item in branch_results],
                }
                index_handle.write(json.dumps(index_row) + "\n")
                index_handle.flush()
                examples += 1

                subtasks.append(
                    {
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": success,
                        "steps": int(chosen["steps"]),
                        "chosen_expert": int(chosen["expert"]),
                        "router_order": router_order,
                        "branches": index_row["branches"],
                        "dataset_file": index_row["path"],
                    }
                )
                if success:
                    success_counter += 1
                else:
                    break
            results.append(success_counter)
            logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})
            if record_video_dir is not None and seq_idx in record_sequence_indices and sequence_frames:
                write_mp4(sequence_frames, record_video_dir / f"sequence_{seq_idx:04d}.mp4", fps=record_video_fps)

    summary = write_variant(output_dir, variant_name, results, logs, task_total, task_success)
    summary["choice_counts"] = dict(choice_counter)
    summary["visual_future_mode"] = "current"
    summary["moe_k"] = int(moe_k)
    summary["execution_signal_steps"] = int(execution_signal_steps)
    summary["examples"] = int(examples)
    summary["dataset_dir"] = str(dataset_dir)
    summary["index_path"] = str(index_path)
    summary["selection_note"] = (
        "Environment progression uses router top-1 expert only. Other experts are rolled out "
        "from the same self-rollout state to label expert value/success/contradiction."
    )
    (output_dir / variant_name / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_success_expert_policy_dataset(
    env: Any,
    model: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    vjepa_moe: VjepaMoeFuture,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    vpp_bank: Dict[str, List[Dict[str, torch.Tensor]]],
    output_dir: Path,
    moe_k: int,
) -> Dict[str, Any]:
    variant_name = "collect_success_expert_policy_dataset"
    variant_dir = output_dir / variant_name
    dataset_dir = variant_dir / "policy_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_path = variant_dir / "index.jsonl"
    results = []
    logs = []
    task_total = Counter()
    task_success = Counter()
    choice_counter = Counter()
    saved_examples = 0
    total_decisions = 0
    total_expert_trials = 0
    successful_expert_trials = 0
    all_experts_failed = 0
    successful_experts_per_decision: List[int] = []
    chosen_success_steps: List[int] = []

    with index_path.open("w") as index_handle:
        for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant_name)):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success_counter = 0
            subtasks = []
            traj_dir = dataset_dir / f"traj_{seq_idx:04d}"
            traj_dir.mkdir(parents=True, exist_ok=True)

            for sub_idx, subtask in enumerate(eval_sequence):
                lang_text = cfg.annotations[subtask][0]
                goal = goal_to_device(lang_embeddings.get_lang_goal(lang_text), model.device)
                goal["lang_text"] = lang_text
                start_info = env.get_info()
                branch_start_raw = raw_env_obs(env)
                decision_obs = env.get_obs()
                expert_items, router_order = retrieve_moe_vjepa_expert_futures(
                    model, vjepa_encoder, vjepa_moe, decision_obs, subtask, vpp_bank, moe_k
                )
                branch_results = []
                for item in expert_items:
                    reset_raw(env, branch_start_raw)
                    latent_tokens = latent_tokens_from_future_feature(model, decision_obs, goal, item["future"])
                    success, steps, final_raw, obs_trace, actions = rollout_once_with_policy_trace(
                        env, model, task_oracle, cfg, subtask, goal, start_info, "current", None, latent_tokens
                    )
                    branch_results.append(
                        {
                            "expert": int(item["expert"]),
                            "router_rank": int(item["router_rank"]),
                            "router_logit": float(item["router_logit"]),
                            "demo_index": int(item["demo_index"]),
                            "vjepa_score": float(item["vjepa_score"]),
                            "success": bool(success),
                            "steps": int(steps),
                            "final_raw": final_raw,
                            "latent_tokens": latent_tokens.detach().cpu().reshape(-1).numpy().astype(np.float32),
                            "obs_trace": obs_trace,
                            "actions": actions,
                        }
                    )

                successful = [item for item in branch_results if item["success"]]
                total_decisions += 1
                total_expert_trials += len(branch_results)
                successful_expert_trials += len(successful)
                successful_experts_per_decision.append(len(successful))
                all_experts_failed += int(len(successful) == 0)
                chosen = min(successful, key=lambda item: item["steps"]) if successful else (
                    min(branch_results, key=lambda item: item["router_rank"]) if branch_results else None
                )
                success = bool(chosen is not None and chosen["success"])
                dataset_file = None
                if chosen is not None:
                    reset_raw(env, chosen["final_raw"])
                if success and chosen is not None:
                    actions = np.stack(chosen["actions"], axis=0).astype(np.float32)
                    obs_trace = chosen["obs_trace"]
                    step_file = traj_dir / f"step_{sub_idx:04d}.npz"
                    np.savez_compressed(
                        step_file,
                        rgb_static=np.stack([row["rgb_static"] for row in obs_trace], axis=0),
                        rgb_gripper=np.stack([row["rgb_gripper"] for row in obs_trace], axis=0),
                        robot_obs=np.stack([row["robot_obs"] for row in obs_trace], axis=0).astype(np.float32),
                        scene_obs=np.stack([row["scene_obs"] for row in obs_trace], axis=0).astype(np.float32),
                        actions=actions,
                        latent_tokens=chosen["latent_tokens"],
                        t=np.asarray([sub_idx], dtype=np.int32),
                        sequence_index=np.asarray([seq_idx], dtype=np.int32),
                        expert=np.asarray([int(chosen["expert"])], dtype=np.int32),
                        router_rank=np.asarray([int(chosen["router_rank"])], dtype=np.int32),
                        success=np.asarray([1.0], dtype=np.float32),
                        steps=np.asarray([int(chosen["steps"])], dtype=np.int32),
                    )
                    dataset_file = str(step_file.relative_to(variant_dir))
                    index_handle.write(
                        json.dumps(
                            {
                                "trajectory_id": f"traj_{seq_idx:04d}",
                                "sequence_index": seq_idx,
                                "subtask_index": sub_idx,
                                "task": subtask,
                                "lang_text": lang_text,
                                "path": dataset_file,
                                "expert": int(chosen["expert"]),
                                "router_rank": int(chosen["router_rank"]),
                                "steps": int(chosen["steps"]),
                                "num_actions": int(actions.shape[0]),
                                "source": "successful_moe_expert_rollout",
                            }
                        )
                        + "\n"
                    )
                    index_handle.flush()
                    saved_examples += 1
                    chosen_success_steps.append(int(chosen["steps"]))

                if chosen is not None:
                    choice_counter[str(chosen["expert"])] += 1
                task_total[subtask] += 1
                task_success[subtask] += int(success)
                subtasks.append(
                    {
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "success": success,
                        "steps": int(chosen["steps"]) if chosen is not None else int(cfg.ep_len),
                        "chosen_expert": int(chosen["expert"]) if chosen is not None else -1,
                        "router_order": router_order,
                        "dataset_file": dataset_file,
                        "branches": [
                            {k: v for k, v in item.items() if k not in {"final_raw", "latent_tokens", "obs_trace", "actions"}}
                            for item in branch_results
                        ],
                    }
                )
                if success:
                    success_counter += 1
                else:
                    break
            results.append(success_counter)
            logs.append({"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks})

    summary = write_variant(output_dir, variant_name, results, logs, task_total, task_success)
    summary["saved_examples"] = int(saved_examples)
    summary["total_decisions"] = int(total_decisions)
    summary["total_expert_trials"] = int(total_expert_trials)
    summary["successful_expert_trials"] = int(successful_expert_trials)
    summary["expert_candidate_success_rate"] = float(successful_expert_trials / total_expert_trials) if total_expert_trials else 0.0
    summary["all_experts_failed"] = int(all_experts_failed)
    summary["all_experts_failed_rate"] = float(all_experts_failed / total_decisions) if total_decisions else 0.0
    summary["saved_example_rate"] = float(saved_examples / total_decisions) if total_decisions else 0.0
    summary["mean_successful_experts_per_decision"] = (
        float(np.mean(successful_experts_per_decision)) if successful_experts_per_decision else 0.0
    )
    summary["mean_chosen_success_steps"] = float(np.mean(chosen_success_steps)) if chosen_success_steps else 0.0
    summary["choice_counts"] = dict(choice_counter)
    summary["dataset_dir"] = str(dataset_dir)
    summary["index_path"] = str(index_path)
    summary["dataset_schema"] = {
        "rgb_static": "T x ... raw env rgb_static before each executed action",
        "rgb_gripper": "T x ... raw env rgb_gripper before each executed action",
        "robot_obs": "T x robot state",
        "scene_obs": "T x scene state",
        "actions": "T x 7 executed action sequence",
        "latent_tokens": "flattened successful expert latent_motion_tokens_up",
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def heuristic_choice(branch_results: Sequence[Dict[str, Any]], subtask: str) -> Dict[str, Any]:
    by_name = {item["name"]: item for item in branch_results}
    contact_keywords = ("block", "place", "stack", "rotate", "push", "lift")
    if any(keyword in subtask for keyword in contact_keywords) and "retry_demo" in by_name:
        return by_name["retry_demo"]
    return by_name.get("continue", branch_results[0])


def select_hypothesis(
    selector: str,
    branch_results: Sequence[Dict[str, Any]],
    subtask: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if selector == "oracle":
        successful = [item for item in branch_results if item["success"]]
        return min(successful, key=lambda item: item["steps"]) if successful else branch_results[0]
    if selector == "random":
        return branch_results[int(rng.integers(0, len(branch_results)))]
    if selector == "heuristic":
        return heuristic_choice(branch_results, subtask)
    raise ValueError(f"Unknown selector: {selector}")


def load_selector_model(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model = SelectorMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=len(checkpoint["classes"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "classes": list(checkpoint["classes"]),
        "mean": torch.as_tensor(checkpoint["mean"], dtype=torch.float32),
        "std": torch.as_tensor(checkpoint["std"], dtype=torch.float32),
    }


def load_value_model(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model = ValueMLP(input_dim=int(checkpoint["input_dim"]), hidden_dim=int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "mean": torch.as_tensor(checkpoint["mean"], dtype=torch.float32),
        "std": torch.as_tensor(checkpoint["std"], dtype=torch.float32),
        "scalars": list(checkpoint.get("scalars", [])),
    }


def select_mlp_hypothesis(
    branch_results: Sequence[Dict[str, Any]],
    selector_feature: Sequence[float],
    selector_model: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by_name = {item["name"]: item for item in branch_results}
    feature = torch.as_tensor(selector_feature, dtype=torch.float32)
    feature = (feature - selector_model["mean"]) / selector_model["std"]
    with torch.no_grad():
        logits = selector_model["model"](feature.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(logits, dim=0)
    ranked = torch.argsort(probs, descending=True).tolist()
    predicted = selector_model["classes"][ranked[0]]
    chosen_name = None
    for idx in ranked:
        candidate = selector_model["classes"][idx]
        if candidate in by_name:
            chosen_name = candidate
            break
    if chosen_name is None:
        chosen_name = "continue"
    return by_name.get(chosen_name, branch_results[0]), {
        "predicted": predicted,
        "chosen": chosen_name,
        "probs": {selector_model["classes"][idx]: float(probs[idx]) for idx in range(len(selector_model["classes"]))},
    }


def value_pair_feature(
    state_feature: Sequence[float],
    future_feature: Sequence[float],
    scalars: Dict[str, float],
    scalar_names: Sequence[str],
) -> torch.Tensor:
    state = torch.as_tensor(state_feature, dtype=torch.float32)
    future = torch.as_tensor(future_feature, dtype=torch.float32)
    scalar_tensor = torch.tensor([float(scalars[name]) for name in scalar_names], dtype=torch.float32)
    return torch.cat([state, future, future - state, scalar_tensor], dim=0)


def select_value_hypothesis(
    branch_results: Sequence[Dict[str, Any]],
    state_feature: Sequence[float],
    hypothesis_features: Dict[str, Sequence[float]],
    scalars: Dict[str, float],
    value_model: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by_name = {item["name"]: item for item in branch_results}
    values = {}
    for name in by_name:
        feature = value_pair_feature(state_feature, hypothesis_features[name], scalars, value_model["scalars"])
        feature = (feature - value_model["mean"]) / value_model["std"]
        with torch.no_grad():
            values[name] = float(value_model["model"](feature.unsqueeze(0)).item())
    chosen_name = max(values, key=values.get)
    return by_name[chosen_name], {"predicted_values": values, "chosen": chosen_name}


def evaluate_hypothesis_selector(
    env: Any,
    model: torch.nn.Module,
    task_oracle: Any,
    cfg: Any,
    lang_embeddings: Any,
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    demo_bank: Dict[str, Dict[str, Any]],
    output_dir: Path,
    variant: str,
    selector: str,
    seed: int = 0,
    collect_examples: bool = False,
    selector_model: Optional[Dict[str, Any]] = None,
    value_model: Optional[Dict[str, Any]] = None,
    collect_hypothesis_examples: bool = False,
) -> Dict[str, Any]:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = variant_dir / "sequences.jsonl"
    selector_examples_path = variant_dir / "selector_examples.jsonl"
    logs = load_existing_sequence_logs(sequences_path)
    results, task_total, task_success = rebuild_counts_from_logs(logs)
    choice_counter = Counter()
    feature_cache: Dict[Tuple[str, str], torch.Tensor] = {}
    completed = {int(row.get("sequence_index", -1)) for row in logs}
    for row in logs:
        for sub in row.get("subtasks", []):
            chosen = sub.get("chosen")
            if chosen:
                choice_counter[str(chosen)] += 1
    rng = np.random.default_rng(seed)
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm(sequences, desc=variant)):
        if seq_idx in completed:
            continue
        trace_rollout(f"{variant} seq={seq_idx} begin")
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        trace_rollout(f"{variant} seq={seq_idx} before env.reset")
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        trace_rollout(f"{variant} seq={seq_idx} after env.reset")
        success_counter = 0
        subtasks = []
        selector_error_history: List[float] = []
        selector_examples_seq = []
        for sub_idx, subtask in enumerate(eval_sequence):
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before get_lang_goal")
            lang_text = cfg.annotations[subtask][0]
            goal = lang_embeddings.get_lang_goal(lang_text)
            goal["lang_text"] = lang_text
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after get_lang_goal")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before raw_env_obs")
            branch_start_raw = raw_env_obs(env)
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after raw_env_obs")
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} before env.get_obs")
            decision_obs = env.get_obs()
            trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} after env.get_obs")
            selector_feature = None
            selector_scalars = None
            feature_parts = None
            if collect_examples or selector == "mlp":
                selector_feature, selector_scalars = selector_observation_feature(model, decision_obs, goal, selector_error_history)
            if collect_hypothesis_examples or selector == "value":
                feature_parts, selector_scalars = selector_feature_parts(model, decision_obs, goal, selector_error_history)
            start_info = env.get_info()

            hypotheses = [("continue", "gfdm", None), ("recover", "current", None)]
            if subtask in demo_bank:
                key = (subtask, lang_text)
                if key not in feature_cache:
                    feature_cache[key] = demo_future_feature(model, demo_bank[subtask], lang_text)
                hypotheses.append(("retry_demo", "override", feature_cache[key]))

            hypothesis_features = None
            if feature_parts is not None:
                hypothesis_features = {
                    "continue": feature_parts["continue_future"],
                    "recover": feature_parts["recover_future"],
                }
                if subtask in demo_bank:
                    hypothesis_features["retry_demo"] = pooled_future_feature(feature_cache[(subtask, lang_text)])

            branch_results = []
            for name, mode, feature in hypotheses:
                trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} branch={name} before reset_raw")
                reset_raw(env, branch_start_raw)
                trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} branch={name} before rollout_once mode={mode}")
                success, steps, final_raw = rollout_once(env, model, task_oracle, cfg, subtask, goal, start_info, mode, feature)
                trace_rollout(f"{variant} seq={seq_idx} sub={sub_idx} task={subtask} branch={name} after rollout_once success={bool(success)} steps={int(steps)}")
                branch_results.append({"name": name, "mode": mode, "success": success, "steps": steps, "final_raw": final_raw})

            mlp_info = None
            value_info = None
            if selector == "mlp":
                if selector_model is None or selector_feature is None:
                    raise ValueError("selector='mlp' requires --selector-model and selector features")
                chosen, mlp_info = select_mlp_hypothesis(branch_results, selector_feature, selector_model)
            elif selector == "value":
                if value_model is None or feature_parts is None or hypothesis_features is None or selector_scalars is None:
                    raise ValueError("selector='value' requires --value-model and hypothesis features")
                chosen, value_info = select_value_hypothesis(
                    branch_results,
                    feature_parts["state"],
                    hypothesis_features,
                    selector_scalars,
                    value_model,
                )
            else:
                chosen = select_hypothesis(selector, branch_results, subtask, rng)
            choice_counter[chosen["name"]] += 1
            if collect_examples and selector_feature is not None and selector_scalars is not None:
                selector_examples_seq.append(
                    {
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "lang_text": lang_text,
                        "label": chosen["name"],
                        "feature": selector_feature,
                        "scalars": selector_scalars,
                        "branches": [{k: v for k, v in item.items() if k != "final_raw"} for item in branch_results],
                    }
                )
                selector_error_history.append(float(selector_scalars["error_score"]))
            if collect_hypothesis_examples and feature_parts is not None and hypothesis_features is not None and selector_scalars is not None:
                branch_by_name = {item["name"]: item for item in branch_results}
                selector_examples_seq.append(
                    {
                        "sequence_index": seq_idx,
                        "subtask_index": sub_idx,
                        "task": subtask,
                        "lang_text": lang_text,
                        "label": chosen["name"],
                        "state_feature": feature_parts["state"],
                        "hypothesis_features": hypothesis_features,
                        "hypothesis_values": {
                            name: branch_value(branch_by_name[name], cfg.ep_len)
                            for name in hypothesis_features
                            if name in branch_by_name
                        },
                        "scalars": selector_scalars,
                        "branches": [{k: v for k, v in item.items() if k != "final_raw"} for item in branch_results],
                    }
                )
                selector_error_history.append(float(selector_scalars["error_score"]))
            elif selector in {"mlp", "value"} and selector_scalars is not None:
                selector_error_history.append(float(selector_scalars["error_score"]))
            reset_raw(env, chosen["final_raw"])
            success = bool(chosen["success"])
            task_total[subtask] += 1
            task_success[subtask] += int(success)
            subtasks.append(
                {
                    "subtask_index": sub_idx,
                    "task": subtask,
                    "success": success,
                    "chosen": chosen["name"],
                    "mlp": mlp_info,
                    "value": value_info,
                    "branches": [{k: v for k, v in item.items() if k != "final_raw"} for item in branch_results],
                }
            )
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        sequence_row = {"sequence_index": seq_idx, "eval_sequence": list(eval_sequence), "success_counter": success_counter, "subtasks": subtasks}
        logs.append(sequence_row)
        append_jsonl_row(sequences_path, sequence_row)
        if collect_examples or collect_hypothesis_examples:
            for row in selector_examples_seq:
                append_jsonl_row(selector_examples_path, row)
        trace_rollout(f"{variant} seq={seq_idx} written subtasks={len(subtasks)} success_counter={success_counter}")
        summary = write_variant(output_dir, variant, results, logs, task_total, task_success)
        summary["selector"] = selector
        summary["choice_counts"] = dict(choice_counter.most_common())
        if collect_examples or collect_hypothesis_examples:
            summary["selector_examples"] = sum(1 for _ in iter_jsonl(selector_examples_path)) if selector_examples_path.exists() else 0
        (output_dir / variant / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary = write_variant(output_dir, variant, results, logs, task_total, task_success)
    summary["selector"] = selector
    summary["choice_counts"] = dict(choice_counter.most_common())
    (output_dir / variant / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if collect_examples or collect_hypothesis_examples:
        summary["selector_examples"] = sum(1 for _ in iter_jsonl(selector_examples_path)) if selector_examples_path.exists() else 0
        (output_dir / variant / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_variant(output_dir: Path, name: str, results: List[int], logs: List[Dict[str, Any]], task_total: Counter, task_success: Counter) -> Dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    sequence_len = max((len(row.get("eval_sequence", [])) for row in logs), default=5)
    summary = {
        "variant": name,
        "num_sequences": len(results),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": {str(i + 1): rate for i, rate in enumerate(count_success(results, sequence_len))},
        "results": results,
        "task_info": {
            task: {"success": int(task_success[task]), "total": int(task_total[task])}
            for task in sorted(task_total)
        },
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (variant_dir / "sequences.jsonl").open("w") as handle:
        for row in logs:
            handle.write(json.dumps(row) + "\n")
    return summary


def filter_sequences_by_tasks(
    sequences: Sequence[Tuple[Any, Sequence[str]]],
    focus_tasks: Sequence[str],
) -> List[Tuple[Any, Sequence[str]]]:
    allowed = {task.strip() for task in focus_tasks if task.strip()}
    if not allowed:
        return list(sequences)
    return [(initial_state, seq) for initial_state, seq in sequences if any(task in allowed for task in seq)]


def build_postexec_qwen_prompt(
    subtask: str,
    success: bool,
    steps: int,
    chosen_branch: Dict[str, Any],
    next_task: Optional[str],
) -> str:
    outcome = "success" if success else "failure"
    contradiction = chosen_branch.get("contradiction", "unknown")
    router_rank = chosen_branch.get("router_rank", "unknown")
    router_logit = chosen_branch.get("router_logit", "unknown")
    next_task_text = next_task or "none"
    return (
        "You are writing a compact post-execution reflection for robot control.\n"
        "Return exactly one line using this schema:\n"
        "progress_status: <status> | failure_type: <type> | prefer_next: <advice> | "
        "avoid_next: <warning> | correction_direction: <direction> | memory_query: <query>\n\n"
        f"task: {subtask}\n"
        f"outcome: {outcome}\n"
        f"steps: {steps}\n"
        f"router_rank: {router_rank}\n"
        f"router_logit: {router_logit}\n"
        f"contradiction: {contradiction}\n"
        f"next_task: {next_task_text}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout upper bound for Continue/Retry/Recover future hypotheses.")
    parser.add_argument("--video-model-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--t5-model-path", default="")
    parser.add_argument("--language-goal-path", default="")
    parser.add_argument("--calvin-abc-dir", required=True)
    parser.add_argument("--demo-calvin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--eval-sequences-path", type=Path, default=None)
    parser.add_argument("--eval-sequence-len", type=int, default=None)
    parser.add_argument("--focus-tasks", type=str, default="")
    parser.add_argument("--ep-len", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug-action-before-env-step", action="store_true")
    parser.add_argument("--debug-action-steps", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["defi", "random_hypothesis", "heuristic_selector", "oracle_selector"],
        choices=[
            "defi",
            "prototype_vpp",
            "nn_vpp",
            "vjepa_nn_vpp",
            "vjepa_gidm_tokens",
            "vjepa_gidm_tokens_nogfdm",
            "trained_vjepa_gidm_tokens_nogfdm",
            "cross_attention_vjepa_gidm_tokens_nogfdm",
            "moe_vjepa_gidm_tokens_nogfdm",
            "moe_corrected_vjepa_gidm_tokens_nogfdm",
            "moe_qwen_corrected_vjepa_gidm_tokens_nogfdm",
            "moe_oracle_expert_tokens_nogfdm",
            "moe_router_top2_oracle_tokens_nogfdm",
            "moe_universal_oracle_router_tokens_nogfdm",
            "collect_moe_reflection_top2_dataset",
            "collect_moe_reflection_all_dataset",
            "collect_moe_execution_value_dataset",
            "collect_moe_universal_self_dataset",
            "collect_success_expert_policy_dataset",
            "moe_reflection_top2_tokens_nogfdm",
            "moe_value_all_tokens_nogfdm",
            "moe_value_gate_all_tokens_nogfdm",
            "moe_execution_value_gate_tokens_nogfdm",
            "moe_selective_reflection_top2_tokens_nogfdm",
            "gt_oracle_tokens_nogfdm",
            "defi_memory_repair",
            "defi_memory_reflect_only",
            "defi_postexec_language_conditioned",
            "random_hypothesis",
            "heuristic_selector",
            "mlp_selector",
            "value_selector",
            "oracle_selector",
        ],
    )
    parser.add_argument("--collect-selector-examples", action="store_true")
    parser.add_argument("--collect-hypothesis-examples", action="store_true")
    parser.add_argument("--selector-model", type=Path, default=None)
    parser.add_argument("--value-model", type=Path, default=None)
    parser.add_argument("--vpp-bank-per-task", type=int, default=5)
    parser.add_argument("--vjepa-root", type=Path, default=Path("/mnt/data/manipulation/vjepa2"))
    parser.add_argument("--vjepa-adapter", type=Path, default=None)
    parser.add_argument("--vjepa-cross-attention", type=Path, default=None)
    parser.add_argument("--vjepa-moe", type=Path, default=None)
    parser.add_argument("--vjepa-bank-cache", type=Path, default=None)
    parser.add_argument("--future-correction-mlp", type=Path, default=None)
    parser.add_argument("--reflection-t5-path", type=Path, default=Path("ckpts/t5_base"))
    parser.add_argument("--reflection-qwen-path", type=Path, default=None)
    parser.add_argument("--reflection-encoder-type", choices=["t5", "qwen"], default="t5")
    parser.add_argument("--reflection-text-mode", choices=["task_hypothesis", "random"], default="task_hypothesis")
    parser.add_argument("--reflection-critic", type=Path, default=None)
    parser.add_argument("--portfolio-reflections", type=Path, default=None)
    parser.add_argument("--portfolio-reflection-t5-path", type=Path, default=Path("ckpts/t5_base"))
    parser.add_argument("--online-postexec-qwen-reflection", action="store_true")
    parser.add_argument("--online-postexec-reflection-max-new-tokens", type=int, default=96)
    parser.add_argument("--defi-memory-key-path", type=Path, default=None)
    parser.add_argument("--memory-recent-k", type=int, default=4)
    parser.add_argument("--memory-key-topk", type=int, default=3)
    parser.add_argument("--memory-trust-threshold", type=float, default=0.5)
    parser.add_argument("--universal-moe-oracle-router", type=Path, default=None)
    parser.add_argument("--selective-switch-high", type=float, default=0.8)
    parser.add_argument("--selective-safe-low", type=float, default=0.4)
    parser.add_argument("--selective-risk-margin", type=float, default=0.25)
    parser.add_argument("--value-gate-margin", type=float, default=0.1)
    parser.add_argument("--execution-signal-steps", type=int, default=5)
    parser.add_argument("--oracle-token-candidates", type=int, default=5)
    parser.add_argument("--record-video-dir", type=Path, default=None)
    parser.add_argument("--record-sequence-indices", type=int, nargs="*", default=[])
    parser.add_argument("--record-video-fps", type=int, default=10)
    args = parser.parse_args()

    seed_everything(0, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with initialize(config_path="../policy_conf", job_name="oracle_hypothesis_rollout"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    _apply_checkpoint_model_compat_overrides(cfg, args.checkpoint)
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    if args.t5_model_path:
        cfg.model.t5_model_path = args.t5_model_path
    if args.language_goal_path:
        cfg.model.language_goal_path = args.language_goal_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.train_folder = str(ensure_train_folder(args.output_dir, args.checkpoint))
    cfg.ep_len = args.ep_len
    cfg.num_sequences = args.num_sequences
    cfg.log_wandb = False
    with open_dict(cfg):
        cfg.future_feature_mode = "gfdm"
        cfg.future_frame_idx = 5

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder,
        cfg.root_data_dir,
        args.checkpoint,
        env=None,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=device.index if device.type == "cuda" and device.index is not None else "cpu",
        cfg=cfg,
    )
    model = load_model(cfg, args.checkpoint, device)
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    if args.sequence_offset < 0:
        raise ValueError("--sequence-offset must be non-negative")
    focus_tasks = [task.strip() for task in args.focus_tasks.split(",") if task.strip()]
    requested_sequences = None if (focus_tasks and args.eval_sequences_path is not None) else (args.num_sequences + args.sequence_offset)
    generated_sequences = load_eval_sequences(
        requested_sequences,
        args.eval_sequences_path,
        args.eval_sequence_len,
    )
    if focus_tasks:
        generated_sequences = filter_sequences_by_tasks(generated_sequences, focus_tasks)
    sequences = generated_sequences[args.sequence_offset : args.sequence_offset + args.num_sequences]
    demo_bank = build_demo_bank(args.demo_calvin_dir)
    metadata = {
        "num_sequences": args.num_sequences,
        "sequence_offset": args.sequence_offset,
        "eval_sequences_path": str(args.eval_sequences_path) if args.eval_sequences_path is not None else None,
        "eval_sequence_len": args.eval_sequence_len,
        "focus_tasks": focus_tasks,
        "num_sequences_after_filter": len(sequences),
        "ep_len": args.ep_len,
        "hypotheses": ["continue:gfdm", "recover:current", "retry_demo:override first same-task training GT future feature"],
        "selectors": {
            "prototype_vpp": "replace GFDM future with first same-task training demo GT future feature for every subtask",
            "nn_vpp": "replace GFDM future with nearest same-task training demo GT future feature, matched by current TVP feature",
            "vjepa_nn_vpp": "replace GFDM future with nearest same-task training demo GT future feature, matched by V-JEPA2 current image feature",
            "vjepa_gidm_tokens": "keep GFDM visual future, but override latent_motion_tokens_up with LAM/GIDM tokens from V-JEPA2 nearest demo future",
            "vjepa_gidm_tokens_nogfdm": "no GFDM future: use current visual feature and override latent_motion_tokens_up with LAM/GIDM tokens from V-JEPA2 nearest demo future",
            "trained_vjepa_gidm_tokens_nogfdm": "no GFDM future: train V-JEPA2 current->future adapter, retrieve demo future by predicted future, then override GIDM tokens",
            "cross_attention_vjepa_gidm_tokens_nogfdm": "no GFDM future: V-JEPA2 retrieve K demos, cross-attend to predict continuous future, retrieve nearest demo future, then override GIDM tokens",
            "moe_vjepa_gidm_tokens_nogfdm": "no GFDM future: V-JEPA2 retrieve K demos, MoE predicts multiple futures, router selects one, retrieve nearest demo future, then override GIDM tokens",
            "moe_corrected_vjepa_gidm_tokens_nogfdm": "no GFDM future: correct each MoE V-JEPA future using reflection-conditioned MLP, frozen router selects top1, retrieve nearest demo future, then override GIDM tokens",
            "moe_qwen_corrected_vjepa_gidm_tokens_nogfdm": "no GFDM future: correct each MoE V-JEPA future using Qwen-conditioned correction, frozen router selects top1, retrieve nearest demo future, then override GIDM tokens",
            "moe_oracle_expert_tokens_nogfdm": "no GFDM future: try all MoE experts and cheat by choosing successful fastest expert",
            "moe_router_top2_oracle_tokens_nogfdm": "no GFDM future: try router top-2 MoE experts and cheat by choosing successful fastest expert",
            "collect_moe_reflection_top2_dataset": "collect top-2 MoE branch features and rollout values for Reflection Critic training",
            "collect_moe_reflection_all_dataset": "collect all MoE expert branch features and rollout values for value-router training",
            "collect_moe_execution_value_dataset": "collect all MoE expert branch values with GIDM/action execution features",
            "collect_moe_universal_self_dataset": "collect compact Universal MoE Dataset from router-top1 self-rollout states; all experts are branch-rolled from the same state for value/success/contradiction labels",
            "collect_success_expert_policy_dataset": "collect DeFI-style obs/action traces from the successful fastest MoE expert for policy/refiner fine-tuning",
            "moe_reflection_top2_tokens_nogfdm": "no GFDM future: Reflection Critic reranks router top-2 MoE experts",
            "moe_value_all_tokens_nogfdm": "no GFDM future: value critic scores all MoE experts and chooses argmax",
            "moe_value_gate_all_tokens_nogfdm": "no GFDM future: keep router top1 unless value critic strongly prefers another expert",
            "moe_execution_value_gate_tokens_nogfdm": "no GFDM future: score experts with value critic using short GIDM action rollout features",
            "moe_selective_reflection_top2_tokens_nogfdm": "no GFDM future: keep router top1 unless risk detector says top1 is dangerous and top2 is safe",
            "gt_oracle_tokens_nogfdm": "no GFDM future: cheat by trying K demo-future GIDM token candidates and choosing the successful fastest rollout",
            "defi_memory_repair": "Frozen DeFI single future with recent/key/rule memory, Qwen trust reflection, and optional selective future repair",
            "defi_memory_reflect_only": "Frozen DeFI single future with recent/key/rule memory and Qwen trust reflection logging only; no future repair",
            "defi_postexec_language_conditioned": "Frozen DeFI single future; generate post-exec reflection and inject previous-step repair text plus important memory events into the next GFDM language instruction",
            "random_hypothesis": "uniform random over available hypotheses, seed=0",
            "heuristic_selector": "retry_demo for block/place/stack/rotate/push/lift tasks when available, otherwise continue",
            "mlp_selector": "small MLP over current obs/current future/error memory/contradiction proxy/drift score",
            "value_selector": "predict V(state, hypothesis_future) for each hypothesis and choose argmax",
            "oracle_selector": "choose a successful branch with the fewest rollout steps; fallback to continue",
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    summaries = []
    if "defi" in args.variants:
        summaries.append(
            evaluate_defi(
                env,
                model,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                args.output_dir,
                args.record_video_dir,
                args.record_sequence_indices,
                args.record_video_fps,
            )
        )
    if "defi_memory_repair" in args.variants:
        if args.reflection_qwen_path is None:
            raise ValueError("--reflection-qwen-path is required for defi_memory_repair")
        memory_reflection_generator = QwenReflectionEncoder(args.reflection_qwen_path, device)
        key_memory_rows = load_key_memory_rows(args.defi_memory_key_path)
        future_correction_mlp = (
            load_future_correction_mlp(args.future_correction_mlp, device)
            if args.future_correction_mlp is not None
            else None
        )
        memory_reflection_encoder = None
        if future_correction_mlp is not None:
            if args.reflection_encoder_type == "qwen":
                memory_reflection_encoder = QwenReflectionEncoder(args.reflection_qwen_path, device)
            else:
                memory_reflection_encoder = T5ReflectionEncoder(args.reflection_t5_path, device)
        summaries.append(
            evaluate_defi_memory_repair(
                env,
                model,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                args.output_dir,
                memory_reflection_generator,
                key_memory_rows,
                args.memory_key_topk,
                args.memory_recent_k,
                args.memory_trust_threshold,
                future_correction_mlp,
                memory_reflection_encoder,
            )
        )
    if "defi_memory_reflect_only" in args.variants:
        if args.reflection_qwen_path is None:
            raise ValueError("--reflection-qwen-path is required for defi_memory_reflect_only")
        memory_reflection_generator = QwenReflectionEncoder(args.reflection_qwen_path, device)
        key_memory_rows = load_key_memory_rows(args.defi_memory_key_path)
        summaries.append(
            evaluate_defi_memory_reflect_only(
                env,
                model,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                args.output_dir,
                memory_reflection_generator,
                key_memory_rows,
                args.memory_key_topk,
                args.memory_recent_k,
            )
        )
    if "defi_postexec_language_conditioned" in args.variants:
        memory_reflection_generator = (
            QwenReflectionEncoder(args.reflection_qwen_path, device)
            if args.reflection_qwen_path is not None
            else None
        )
        key_memory_rows = load_key_memory_rows(args.defi_memory_key_path)
        summaries.append(
            evaluate_defi_postexec_language_conditioned(
                env,
                model,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                args.output_dir,
                memory_reflection_generator,
                key_memory_rows,
                args.memory_key_topk,
                args.memory_recent_k,
            )
        )
    if "prototype_vpp" in args.variants:
        summaries.append(
            evaluate_prototype_vpp(
                env, model, task_oracle, cfg, lang_embeddings, sequences, demo_bank, args.output_dir,
            )
        )
    if "nn_vpp" in args.variants:
        vpp_bank = build_retrieval_vpp_bank(model, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_nn_vpp(
                env, model, task_oracle, cfg, lang_embeddings, sequences, vpp_bank, args.output_dir,
            )
        )
    if "vjepa_nn_vpp" in args.variants:
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_bank = build_vjepa_vpp_bank(
            model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task, cache_path=args.vjepa_bank_cache
        )
        summaries.append(
            evaluate_vjepa_nn_vpp(
                env, model, vjepa_encoder, task_oracle, cfg, lang_embeddings, sequences, vjepa_bank, args.output_dir,
            )
        )
    if "vjepa_gidm_tokens" in args.variants:
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_bank = build_vjepa_vpp_bank(
            model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task, cache_path=args.vjepa_bank_cache
        )
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env, model, vjepa_encoder, task_oracle, cfg, lang_embeddings, sequences, vjepa_bank, args.output_dir,
            )
        )
    if "vjepa_gidm_tokens_nogfdm" in args.variants:
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env,
                model,
                vjepa_encoder,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                visual_mode="current",
                variant_name="vjepa_gidm_tokens_nogfdm",
            )
        )
    if "trained_vjepa_gidm_tokens_nogfdm" in args.variants:
        if args.vjepa_adapter is None:
            raise ValueError("--vjepa-adapter is required for trained_vjepa_gidm_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_adapter = load_vjepa_adapter(args.vjepa_adapter, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env,
                model,
                vjepa_encoder,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                visual_mode="current",
                variant_name="trained_vjepa_gidm_tokens_nogfdm",
                vjepa_adapter=vjepa_adapter,
            )
        )
    if "cross_attention_vjepa_gidm_tokens_nogfdm" in args.variants:
        if args.vjepa_cross_attention is None:
            raise ValueError("--vjepa-cross-attention is required for cross_attention_vjepa_gidm_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_cross_attention, vjepa_cross_attention_k = load_vjepa_cross_attention(args.vjepa_cross_attention, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env,
                model,
                vjepa_encoder,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                visual_mode="current",
                variant_name="cross_attention_vjepa_gidm_tokens_nogfdm",
                vjepa_cross_attention=vjepa_cross_attention,
                vjepa_cross_attention_k=vjepa_cross_attention_k,
            )
        )
    if "moe_vjepa_gidm_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for moe_vjepa_gidm_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env,
                model,
                vjepa_encoder,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                visual_mode="current",
                variant_name="moe_vjepa_gidm_tokens_nogfdm",
                vjepa_moe=vjepa_moe,
                vjepa_moe_k=vjepa_moe_k,
            )
        )
    if "moe_corrected_vjepa_gidm_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.future_correction_mlp is None:
            raise ValueError("--vjepa-moe and --future-correction-mlp are required for moe_corrected_vjepa_gidm_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        future_correction_mlp = load_future_correction_mlp(args.future_correction_mlp, device)
        reflection_encoder = T5ReflectionEncoder(args.reflection_t5_path, device)
        vjepa_bank = build_vjepa_vpp_bank(
            model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task, cache_path=args.vjepa_bank_cache
        )
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env,
                model,
                vjepa_encoder,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                visual_mode="current",
                variant_name="moe_corrected_vjepa_gidm_tokens_nogfdm",
                vjepa_moe=vjepa_moe,
                vjepa_moe_k=vjepa_moe_k,
                future_correction_mlp=future_correction_mlp,
                reflection_encoder=reflection_encoder,
                reflection_text_mode=args.reflection_text_mode,
            )
        )
    if "moe_qwen_corrected_vjepa_gidm_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.future_correction_mlp is None:
            raise ValueError("--vjepa-moe and --future-correction-mlp are required for moe_qwen_corrected_vjepa_gidm_tokens_nogfdm")
        if args.reflection_qwen_path is None:
            raise ValueError("--reflection-qwen-path is required for moe_qwen_corrected_vjepa_gidm_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        future_correction_mlp = load_future_correction_mlp(args.future_correction_mlp, device)
        reflection_encoder = QwenReflectionEncoder(args.reflection_qwen_path, device)
        vjepa_bank = build_vjepa_vpp_bank(
            model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task, cache_path=args.vjepa_bank_cache
        )
        summaries.append(
            evaluate_vjepa_gidm_tokens(
                env,
                model,
                vjepa_encoder,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                visual_mode="current",
                variant_name="moe_qwen_corrected_vjepa_gidm_tokens_nogfdm",
                vjepa_moe=vjepa_moe,
                vjepa_moe_k=vjepa_moe_k,
                future_correction_mlp=future_correction_mlp,
                reflection_encoder=reflection_encoder,
                reflection_text_mode=args.reflection_text_mode,
            )
        )
    if "moe_oracle_expert_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for moe_oracle_expert_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_oracle_expert_tokens_nogfdm",
                None,
            )
        )
    if "moe_router_top2_oracle_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for moe_router_top2_oracle_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_router_top2_oracle_tokens_nogfdm",
                2,
            )
        )
    if "moe_universal_oracle_router_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.universal_moe_oracle_router is None:
            raise ValueError("--vjepa-moe and --universal-moe-oracle-router are required for moe_universal_oracle_router_tokens_nogfdm")
        if args.online_postexec_qwen_reflection and args.reflection_qwen_path is None:
            raise ValueError("--reflection-qwen-path is required when --online-postexec-qwen-reflection is enabled")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        universal_oracle_router = load_universal_moe_oracle_router(args.universal_moe_oracle_router, device)
        portfolio_reflections = load_portfolio_reflections(args.portfolio_reflections) if args.portfolio_reflections is not None else None
        portfolio_reflection_encoder = (
            T5ReflectionEncoder(args.portfolio_reflection_t5_path, device)
            if (args.portfolio_reflections is not None or args.online_postexec_qwen_reflection)
            else None
        )
        online_postexec_reflection_generator = (
            QwenReflectionEncoder(args.reflection_qwen_path, device)
            if args.online_postexec_qwen_reflection
            else None
        )
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_universal_oracle_router_tokens_nogfdm",
                None,
                None,
                False,
                False,
                args.selective_switch_high,
                args.selective_safe_low,
                args.selective_risk_margin,
                None,
                0,
                universal_oracle_router,
                portfolio_reflections,
                portfolio_reflection_encoder,
                online_postexec_reflection_generator,
                args.online_postexec_reflection_max_new_tokens,
                args.record_video_dir,
                args.record_sequence_indices,
                args.record_video_fps,
            )
        )
    if "collect_moe_reflection_top2_dataset" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for collect_moe_reflection_top2_dataset")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "collect_moe_reflection_top2_dataset",
                2,
                None,
                True,
            )
        )
    if "collect_moe_reflection_all_dataset" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for collect_moe_reflection_all_dataset")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "collect_moe_reflection_all_dataset",
                None,
                None,
                True,
            )
        )
    if "collect_moe_execution_value_dataset" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for collect_moe_execution_value_dataset")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "collect_moe_execution_value_dataset",
                None,
                None,
                True,
                False,
                execution_signal_steps=args.execution_signal_steps,
            )
        )
    if "collect_moe_universal_self_dataset" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for collect_moe_universal_self_dataset")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_universal_self_dataset(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                args.execution_signal_steps,
                args.record_video_dir,
                args.record_sequence_indices,
                args.record_video_fps,
            )
        )
    if "collect_success_expert_policy_dataset" in args.variants:
        if args.vjepa_moe is None:
            raise ValueError("--vjepa-moe is required for collect_success_expert_policy_dataset")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_success_expert_policy_dataset(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
            )
        )
    if "moe_reflection_top2_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.reflection_critic is None:
            raise ValueError("--vjepa-moe and --reflection-critic are required for moe_reflection_top2_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        reflection_critic = load_reflection_critic(args.reflection_critic, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_reflection_top2_tokens_nogfdm",
                2,
                reflection_critic,
                False,
            )
        )
    if "moe_value_all_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.reflection_critic is None:
            raise ValueError("--vjepa-moe and --reflection-critic are required for moe_value_all_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        reflection_critic = load_reflection_critic(args.reflection_critic, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_value_all_tokens_nogfdm",
                None,
                reflection_critic,
                False,
            )
        )
    if "moe_value_gate_all_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.reflection_critic is None:
            raise ValueError("--vjepa-moe and --reflection-critic are required for moe_value_gate_all_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        reflection_critic = load_reflection_critic(args.reflection_critic, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_value_gate_all_tokens_nogfdm",
                None,
                reflection_critic,
                False,
                False,
                value_gate_margin=args.value_gate_margin,
            )
        )
    if "moe_execution_value_gate_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.reflection_critic is None:
            raise ValueError("--vjepa-moe and --reflection-critic are required for moe_execution_value_gate_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        reflection_critic = load_reflection_critic(args.reflection_critic, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_execution_value_gate_tokens_nogfdm",
                None,
                reflection_critic,
                False,
                False,
                value_gate_margin=args.value_gate_margin,
                execution_signal_steps=args.execution_signal_steps,
            )
        )
    if "moe_selective_reflection_top2_tokens_nogfdm" in args.variants:
        if args.vjepa_moe is None or args.reflection_critic is None:
            raise ValueError("--vjepa-moe and --reflection-critic are required for moe_selective_reflection_top2_tokens_nogfdm")
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_moe, vjepa_moe_k = load_vjepa_moe(args.vjepa_moe, device)
        reflection_critic = load_reflection_critic(args.reflection_critic, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_moe_expert_oracle_tokens(
                env,
                model,
                vjepa_encoder,
                vjepa_moe,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                vjepa_moe_k,
                "moe_selective_reflection_top2_tokens_nogfdm",
                2,
                reflection_critic,
                False,
                True,
                args.selective_switch_high,
                args.selective_safe_low,
                args.selective_risk_margin,
            )
        )
    if "gt_oracle_tokens_nogfdm" in args.variants:
        vjepa_encoder = load_vjepa2_encoder(args.vjepa_root, device)
        vjepa_bank = build_vjepa_vpp_bank(model, vjepa_encoder, args.demo_calvin_dir, args.vpp_bank_per_task)
        summaries.append(
            evaluate_gt_oracle_tokens(
                env,
                model,
                task_oracle,
                cfg,
                lang_embeddings,
                sequences,
                vjepa_bank,
                args.output_dir,
                args.oracle_token_candidates,
            )
        )
    if "random_hypothesis" in args.variants:
        summaries.append(
            evaluate_hypothesis_selector(
                env, model, task_oracle, cfg, lang_embeddings, sequences, demo_bank, args.output_dir,
                variant="random_hypothesis", selector="random", seed=0,
            )
        )
    if "heuristic_selector" in args.variants:
        summaries.append(
            evaluate_hypothesis_selector(
                env, model, task_oracle, cfg, lang_embeddings, sequences, demo_bank, args.output_dir,
                variant="heuristic_selector", selector="heuristic", seed=0,
            )
        )
    if "mlp_selector" in args.variants:
        if args.selector_model is None:
            raise ValueError("--selector-model is required for mlp_selector")
        selector_model = load_selector_model(args.selector_model)
        summaries.append(
            evaluate_hypothesis_selector(
                env, model, task_oracle, cfg, lang_embeddings, sequences, demo_bank, args.output_dir,
                variant="mlp_selector", selector="mlp", seed=0, selector_model=selector_model,
            )
        )
    if "value_selector" in args.variants:
        if args.value_model is None:
            raise ValueError("--value-model is required for value_selector")
        value_model = load_value_model(args.value_model)
        summaries.append(
            evaluate_hypothesis_selector(
                env, model, task_oracle, cfg, lang_embeddings, sequences, demo_bank, args.output_dir,
                variant="value_selector", selector="value", seed=0, value_model=value_model,
            )
        )
    if "oracle_selector" in args.variants:
        summaries.append(
            evaluate_hypothesis_selector(
                env, model, task_oracle, cfg, lang_embeddings, sequences, demo_bank, args.output_dir,
                variant="oracle_selector", selector="oracle", seed=0,
                collect_examples=args.collect_selector_examples,
                collect_hypothesis_examples=args.collect_hypothesis_examples,
            )
        )
    (args.output_dir / "comparison.json").write_text(json.dumps({item["variant"]: item for item in summaries}, indent=2) + "\n")
    print(json.dumps({item["variant"]: item for item in summaries}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PL_TORCH_DISTRIBUTED_BACKEND", "gloo")
    main()
