import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DEFI_ROOT = Path("/mnt/workspace/manipulation/DeFi/Step3_DeFI")
for path in [ROOT, DEFI_ROOT]:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _to_float_vector(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], (list, np.ndarray)):
            value = value[0]
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _match_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[0] > target_dim:
        return arr[:target_dim]
    if arr.shape[0] < target_dim:
        repeat = (target_dim + arr.shape[0] - 1) // arr.shape[0]
        return np.tile(arr, repeat)[:target_dim]
    return arr


class BridgeLatentHeads(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.state_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.future_head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.motion_head = nn.Sequential(nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))


def _load_lam(t5_model_path: str, ckpt_path: str, device: torch.device):
    if not ckpt_path:
        return None
    from policy_models.m_former_univla.latent_motion_tokenizer_univla import UncontrolledDINOLatentActionModel

    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    filtered = {}
    for key, value in state_dict.items():
        if not key.startswith("lam."):
            continue
        key = key[len("lam.") :]
        if key.startswith("dino_encoder."):
            continue
        filtered[key] = value
    model = UncontrolledDINOLatentActionModel(t5_model_path=t5_model_path)
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def _run_lam(lam, state_latent: torch.Tensor, future_latent: torch.Tensor, lang_text: str):
    if lam is None:
        return None
    state_latent = state_latent[:, :768]
    future_latent = future_latent[:, :768]
    current_tokens = state_latent.unsqueeze(1).repeat(1, 256, 1)
    future_tokens = future_latent.unsqueeze(1).repeat(1, 256, 1)
    out = lam(torch.stack([current_tokens, future_tokens], dim=1), [lang_text])
    patches = out.get("video_action_patches", None)
    if patches is None:
        return None
    return patches.reshape(patches.shape[0], -1, patches.shape[-1]).mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--latent-head-ckpt", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--standalone-gidm-ckpt", default="")
    parser.add_argument("--standalone-gidm-t5-path", default="/mnt/workspace/manipulation/DeFi/ckpts/t5_base")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    payload = torch.load(args.latent_head_ckpt, map_location="cpu")
    meta = payload["meta"]
    model = BridgeLatentHeads(int(meta["input_dim"]), int(meta["latent_dim"]), int(meta["hidden_dim"]))
    model.load_state_dict(payload["bridge_latent_heads"], strict=False)
    model.to(device).eval()
    lam = _load_lam(args.standalone_gidm_t5_path, args.standalone_gidm_ckpt, device)

    items = []
    with Path(args.rows_jsonl).open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            row = json.loads(line)
            feature_path = row.get("feature_path", "")
            if not feature_path:
                continue
            npz_path = Path(feature_path)
            if not npz_path.is_file():
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            current = _to_float_vector(data.get("obs_current_feature"))
            if current is None:
                continue
            current = _match_dim(current, int(meta["input_dim"]))
            current_t = torch.from_numpy(current).to(device=device, dtype=torch.float32).unsqueeze(0)
            state_latent = torch.tanh(model.state_proj(current_t))
            future_latent = torch.tanh(model.future_head(current_t))
            lang_goal = str(row.get("lang_goal", row.get("task", "")))
            motion_latent = _run_lam(lam, state_latent, future_latent, lang_goal)
            if motion_latent is None:
                motion_latent = torch.tanh(model.motion_head(torch.cat([state_latent, future_latent], dim=-1)))
            items.append(
                {
                    "task": str(row.get("task", "")),
                    "lang_goal": lang_goal,
                    "success": 1.0 if bool(row.get("success", False)) else 0.0,
                    "episode": int(row.get("episode", row_idx)),
                    "attempt_idx": int(row.get("attempt_idx", 0)),
                    "state_latent": state_latent.squeeze(0).detach().cpu().numpy().astype(np.float32),
                    "future_latent": future_latent.squeeze(0).detach().cpu().numpy().astype(np.float32),
                    "motion_latent": motion_latent.squeeze(0).detach().cpu().numpy().astype(np.float32),
                }
            )
            if args.max_rows > 0 and len(items) >= args.max_rows:
                break

    if not items:
        raise RuntimeError("no usable rows found")

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        task=np.asarray([x["task"] for x in items], dtype=object),
        lang_goal=np.asarray([x["lang_goal"] for x in items], dtype=object),
        success=np.asarray([x["success"] for x in items], dtype=np.float32),
        episode=np.asarray([x["episode"] for x in items], dtype=np.int32),
        attempt_idx=np.asarray([x["attempt_idx"] for x in items], dtype=np.int32),
        state_latent=np.stack([x["state_latent"] for x in items], axis=0).astype(np.float32),
        future_latent=np.stack([x["future_latent"] for x in items], axis=0).astype(np.float32),
        motion_latent=np.stack([x["motion_latent"] for x in items], axis=0).astype(np.float32),
    )
    print(json.dumps({"output_npz": str(out), "num_items": len(items)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
