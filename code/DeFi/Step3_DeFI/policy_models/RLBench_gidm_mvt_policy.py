from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from policy_models.m_former_univla.latent_motion_tokenizer_univla import UncontrolledDINOLatentActionModel
from policy_models.module.bridge_full_mvt import BridgeFullMVT
from policy_models.module.clip_lang_encoder import LangClip
from policy_models.module.gfdm_future_predictor import GFDMFuturePredictor
from policy_models.module.rlbench_bridge_head import RLBenchBridgeHead
from policy_models.module.rendered_view_token_adapter import RenderedViewTokenAdapter
from policy_models.module.residual_action_head import MotionAuxHead
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


def _ensure_bridgevla_imports():
    root = Path(__file__).resolve().parents[3]
    finetune_root = root / "BridgeVLA" / "finetune"
    point_renderer_root = finetune_root / "bridgevla" / "libs" / "point-renderer"
    for path in (finetune_root, point_renderer_root):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_ensure_bridgevla_imports()
from bridgevla.mvt.config import get_cfg_defaults  # noqa: E402
import bridgevla.mvt.utils as bridge_mvt_utils  # noqa: E402


class RLBenchGIDMMVTPolicy(nn.Module):
    def __init__(
        self,
        optimizer,
        lr_scheduler,
        latent_dim: int = 384,
        use_lr_scheduler: bool = True,
        action_dim: int = 9,
        t5_model_path: str = "/ckpt/t5_base",
        language_goal_path: str = "/ckpt/ViT-B-32.pt",
        waypoint_hidden_dim: int = 512,
        local_delta_max: float = 0.04,
        lambda_action: float = 1.0,
        lambda_future: float = 0.5,
        lambda_motion: float = 0.2,
        lambda_gripper: float = 0.2,
        lambda_collision: float = 0.1,
        future_mode: str = "pred",
        freeze_action_head: bool = False,
        bridge_ckpt_path: str = "",
        freeze_bridge_mvt: bool = False,
        freeze_bridge_encoder: bool = False,
        train_motion_only: bool = False,
        n_trans_bins: int = 100,
        n_rot_bins: int = 72,
        scene_bounds: list[float] | None = None,
        mvt_cfg_overrides: dict | None = None,
        **unused_kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_lr_scheduler = use_lr_scheduler
        self.action_dim = action_dim
        self.scene_bounds = scene_bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self.local_delta_max = float(local_delta_max)
        self.lambda_action = float(lambda_action)
        self.lambda_future = float(lambda_future)
        self.lambda_motion = float(lambda_motion)
        self.lambda_gripper = float(lambda_gripper)
        self.lambda_collision = float(lambda_collision)
        self.future_mode = future_mode
        self.freeze_action_head = freeze_action_head
        self.bridge_ckpt_path = bridge_ckpt_path
        self.freeze_bridge_mvt = freeze_bridge_mvt
        self.freeze_bridge_encoder = freeze_bridge_encoder
        self.train_motion_only = train_motion_only
        self.n_trans_bins = int(n_trans_bins)
        self.n_rot_bins = int(n_rot_bins)
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.action_head_type = "rlbench_bridge"
        self.rollout_step_counter = 0
        self.pred_action_seq = None

        self.trans_min = torch.tensor(self.scene_bounds[:3], dtype=torch.float32)
        self.trans_max = torch.tensor(self.scene_bounds[3:], dtype=torch.float32)

        self.language_goal = LangClip(model_name=language_goal_path)
        self.render_token_adapter = RenderedViewTokenAdapter(
            image_size=224, patch_size=14, gidm_dim=768, head_dim=latent_dim
        )
        self.gfdm = GFDMFuturePredictor(token_dim=latent_dim, goal_dim=512, hidden_dim=waypoint_hidden_dim)
        self.future_to_gidm = nn.Linear(latent_dim, 768)
        self.lam = UncontrolledDINOLatentActionModel(t5_model_path=t5_model_path)
        self.gfdm_bridge_proj = nn.Linear(latent_dim, latent_dim)
        self.gidm_bridge_proj = nn.Linear(768, latent_dim)
        self.native_head = RLBenchBridgeHead(
            obs_dim=latent_dim,
            goal_dim=512,
            hidden_dim=waypoint_hidden_dim,
            action_seq_len=1,
            n_trans_bins=self.n_trans_bins,
            n_rot_bins=self.n_rot_bins,
            motion_dim=None,
        )
        self.motion_aux_head = MotionAuxHead(motion_dim=768, hidden_dim=waypoint_hidden_dim)

        cfg = get_cfg_defaults()
        if mvt_cfg_overrides:
            override_dict = OmegaConf.to_container(OmegaConf.create(mvt_cfg_overrides), resolve=True)
            for k, v in override_dict.items():
                setattr(cfg, k, v)
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mvt = BridgeFullMVT(renderer_device=device_str, **cfg)
        self._maybe_load_bridge_mvt()
        self._printed_label_stats = False

    def _maybe_load_bridge_mvt(self) -> None:
        if not self.bridge_ckpt_path:
            return
        checkpoint = torch.load(self.bridge_ckpt_path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        mvt_state = {k: v for k, v in state.items() if k.startswith("mvt1.")}
        load_result = self.mvt.load_state_dict(mvt_state, strict=False)
        print(f"[bridge-init] loaded MVT backbone from {self.bridge_ckpt_path}")
        if getattr(load_result, "missing_keys", None):
            print(f"[bridge-init] missing keys: {len(load_result.missing_keys)}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"[bridge-init] unexpected keys: {len(load_result.unexpected_keys)}")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def log(self, *args, **kwargs):
        return None

    def process_device(self):
        self.language_goal = self.language_goal.to(self.device)
        self.render_token_adapter = self.render_token_adapter.to(self.device)
        self.gfdm = self.gfdm.to(self.device)
        self.future_to_gidm = self.future_to_gidm.to(self.device)
        self.lam = self.lam.to(self.device)
        self.gfdm_bridge_proj = self.gfdm_bridge_proj.to(self.device)
        self.gidm_bridge_proj = self.gidm_bridge_proj.to(self.device)
        self.native_head = self.native_head.to(self.device)
        self.motion_aux_head = self.motion_aux_head.to(self.device)
        self.mvt = self.mvt.to(self.device)
        if self.freeze_bridge_mvt or self.freeze_bridge_encoder or self.train_motion_only:
            self.mvt.eval()
            self.render_token_adapter.eval()
            for p in self.mvt.parameters():
                p.requires_grad = False
            for p in self.render_token_adapter.parameters():
                p.requires_grad = False
        if self.freeze_action_head:
            for module in (self.native_head,):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        if self.train_motion_only:
            self.native_head.eval()
            for p in self.native_head.parameters():
                p.requires_grad = False

    def on_train_start(self) -> None:
        return None

    def configure_optimizers(self):
        optim_groups = []
        for module in (self.gfdm, self.future_to_gidm, self.lam, self.motion_aux_head, self.gfdm_bridge_proj, self.gidm_bridge_proj):
            optim_groups.append(
                {"params": [p for p in module.parameters() if p.requires_grad], "weight_decay": self.optimizer_config.transformer_weight_decay}
            )
        native_params = [p for p in self.native_head.parameters() if p.requires_grad]
        if native_params:
            optim_groups.append(
                {
                    "params": native_params,
                    "weight_decay": self.optimizer_config.transformer_weight_decay,
                    "lr": self.optimizer_config.learning_rate * 0.1,
                }
            )
        adapter_params = [p for p in self.render_token_adapter.parameters() if p.requires_grad]
        if adapter_params:
            optim_groups.append(
                {"params": adapter_params, "weight_decay": self.optimizer_config.transformer_weight_decay}
            )
        optimizer = torch.optim.AdamW(
            [group for group in optim_groups if group["params"]],
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.betas,
        )
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": None, "interval": "step", "frequency": 1}}

    def _get_pc_img_feat(self, point_cloud_obs: Dict[str, torch.Tensor], rgb_obs: Dict[str, torch.Tensor]):
        key_pairs = [
            ("pcd_front", "rgb_static"),
            ("pcd_left_shoulder", "rgb_left_shoulder"),
            ("pcd_right_shoulder", "rgb_right_shoulder"),
            ("pcd_wrist", "rgb_gripper"),
        ]
        pcs, feats = [], []
        for pcd_key, rgb_key in key_pairs:
            if pcd_key not in point_cloud_obs or rgb_key not in rgb_obs:
                continue
            pcd = point_cloud_obs[pcd_key]
            rgb = rgb_obs[rgb_key]
            if pcd.ndim == 5:
                pcd = pcd[:, 0]
            if rgb.ndim == 5:
                rgb = rgb[:, 0]
            if pcd.shape[-2:] != rgb.shape[-2:]:
                pcd = F.interpolate(pcd, size=rgb.shape[-2:], mode="nearest")
            pcd_flat = pcd.permute(0, 2, 3, 1).reshape(pcd.shape[0], -1, 3)
            rgb_flat = ((rgb.permute(0, 2, 3, 1) + 1.0) * 0.5).reshape(rgb.shape[0], -1, rgb.shape[1]).clamp(0.0, 1.0)
            n_pts = min(pcd_flat.shape[1], rgb_flat.shape[1])
            pcs.append(pcd_flat[:, :n_pts])
            feats.append(rgb_flat[:, :n_pts])
        return torch.cat(pcs, dim=1).float(), torch.cat(feats, dim=1).float()

    def _move_pc_in_bound(self, pc: torch.Tensor, img_feat: torch.Tensor):
        x_min, y_min, z_min, x_max, y_max, z_max = self.scene_bounds
        inv = (
            (pc[:, :, 0] < x_min)
            | (pc[:, :, 0] > x_max)
            | (pc[:, :, 1] < y_min)
            | (pc[:, :, 1] > y_max)
            | (pc[:, :, 2] < z_min)
            | (pc[:, :, 2] > z_max)
            | torch.isnan(pc[:, :, 0])
            | torch.isnan(pc[:, :, 1])
            | torch.isnan(pc[:, :, 2])
        )
        return [pc[i, ~m] for i, m in enumerate(inv)], [img_feat[i, ~m] for i, m in enumerate(inv)]

    def _render_token_dict(self, rgb_obs: Dict[str, torch.Tensor], point_cloud_obs: Dict[str, torch.Tensor], lang_text):
        pc, img_feat = self._get_pc_img_feat(point_cloud_obs, rgb_obs)
        pc, img_feat = self._move_pc_in_bound(pc, img_feat)
        rendered = self.mvt.render(pc=pc, img_feat=img_feat, img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None)
        rendered_rgb = rendered[:, :, 3:6]
        token_out = self.render_token_adapter(rendered_rgb)
        latent_goal = self.language_goal(lang_text)
        return token_out, latent_goal

    def _future_target_tokens(self, dataset_batch, lang_text):
        token_out, _ = self._render_token_dict(
            dataset_batch["future_rgb_obs"], dataset_batch["future_point_cloud_obs"], lang_text
        )
        return token_out["head_tokens"].detach()

    def _build_motion_latent(self, current_tokens: torch.Tensor, future_tokens: torch.Tensor, lang_text):
        proj_weight = self.future_to_gidm.weight
        proj_dtype = proj_weight.dtype
        proj_device = proj_weight.device
        current_gidm = self.future_to_gidm(current_tokens.to(device=proj_device, dtype=proj_dtype))
        future_gidm = self.future_to_gidm(future_tokens.to(device=proj_device, dtype=proj_dtype))
        current_gidm = torch.nan_to_num(current_gidm, nan=0.0, posinf=0.0, neginf=0.0)
        future_gidm = torch.nan_to_num(future_gidm, nan=0.0, posinf=0.0, neginf=0.0)
        gidm_in = torch.stack([current_gidm, future_gidm], dim=1)
        gidm_in = torch.nan_to_num(gidm_in, nan=0.0, posinf=0.0, neginf=0.0)
        lam_out = self.lam(gidm_in, lang_text)
        video_action_patches = lam_out.get("video_action_patches", None)
        if video_action_patches is None:
            fallback_motion = torch.tanh(future_gidm - current_gidm)
            return lam_out, fallback_motion
        video_action_patches = torch.nan_to_num(video_action_patches, nan=0.0, posinf=0.0, neginf=0.0)
        motion_latent = video_action_patches.reshape(gidm_in.shape[0], -1, 768).mean(dim=1)
        motion_latent = torch.nan_to_num(motion_latent, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(motion_latent).all():
            motion_latent = torch.tanh(future_gidm - current_gidm)
            motion_latent = torch.nan_to_num(motion_latent, nan=0.0, posinf=0.0, neginf=0.0)
        lam_out["video_action_patches"] = video_action_patches
        return lam_out, motion_latent

    def _get_current_pos(self, robot_obs: torch.Tensor) -> torch.Tensor:
        if robot_obs.ndim == 3:
            return robot_obs[:, -1, :3].float()
        return robot_obs[:, :3].float()

    def _get_current_quat(self, robot_obs: torch.Tensor) -> torch.Tensor:
        if robot_obs.ndim == 3:
            euler = robot_obs[:, -1, 3:6].detach().cpu().numpy()
        else:
            euler = robot_obs[:, 3:6].detach().cpu().numpy()
        quat_np = R.from_euler("xyz", euler, degrees=False).as_quat().astype("float32")
        quat = torch.from_numpy(quat_np).to(robot_obs.device)
        return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _clip_target_delta(self, delta_xyz: torch.Tensor) -> torch.Tensor:
        return delta_xyz.clamp(-self.local_delta_max, self.local_delta_max)

    def _decode_translation(self, trans_idx: torch.Tensor) -> torch.Tensor:
        mins = torch.tensor([-0.3, -0.5, 0.6], dtype=torch.float32, device=trans_idx.device)
        maxs = torch.tensor([0.7, 0.5, 1.6], dtype=torch.float32, device=trans_idx.device)
        scaled = trans_idx.float() / max(self.n_trans_bins - 1, 1)
        return mins + scaled * (maxs - mins)

    def _decode_quaternion(self, rot_idx: torch.Tensor) -> torch.Tensor:
        resolution = 360.0 / float(self.n_rot_bins)
        euler_deg = rot_idx.float() * resolution
        quat_np = R.from_euler("xyz", euler_deg.detach().cpu().numpy(), degrees=True).as_quat().astype("float32")
        quat = torch.from_numpy(quat_np).to(rot_idx.device)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sign = torch.where(quat[..., -1:] < 0, -torch.ones_like(quat[..., -1:]), torch.ones_like(quat[..., -1:]))
        return quat * sign

    def _build_target_residual_heatmap(self, dataset_batch: Dict[str, Dict], target_pos: torch.Tensor) -> torch.Tensor:
        rgb_obs = dataset_batch["rgb_obs"]
        point_cloud_obs = dataset_batch["point_cloud_obs"]
        pc, _ = self._get_pc_img_feat(point_cloud_obs, rgb_obs)
        pc, _ = self._move_pc_in_bound(pc, torch.zeros_like(pc))
        bs = target_pos.shape[0]
        img_size = self.mvt.mvt1.img_size
        num_img = self.mvt.mvt1.num_img
        yy, xx = torch.meshgrid(
            torch.arange(img_size, device=target_pos.device, dtype=torch.float32),
            torch.arange(img_size, device=target_pos.device, dtype=torch.float32),
            indexing="ij",
        )
        heatmaps = []
        for i in range(bs):
            pc_i = pc[i]
            tgt_i = target_pos[i]
            tgt_cube, _ = bridge_mvt_utils.place_pc_in_cube(
                pc_i,
                app_pc=tgt_i.unsqueeze(0),
                with_mean_or_bounds=True,
                scene_bounds=None,
            )
            pt_img = self.mvt.get_pt_loc_on_img(tgt_cube.unsqueeze(1), True, None)
            pt_img = pt_img[:, 0]  # [1, num_img, 2]
            cam_maps = []
            for v in range(num_img):
                x = pt_img[0, v, 0]
                y = pt_img[0, v, 1]
                valid = (x >= 0) & (x < img_size) & (y >= 0) & (y < img_size)
                dist2 = (xx - x) ** 2 + (yy - y) ** 2
                hm = torch.exp(-dist2 / (2.0 * 4.0 * 4.0))
                cam_maps.append(hm * valid.float())
            heatmaps.append(torch.stack(cam_maps, dim=0))
        return torch.stack(heatmaps, dim=0)

    def _future_loss(self, pred_future: torch.Tensor, target_future: torch.Tensor) -> torch.Tensor:
        loss_s1 = F.smooth_l1_loss(pred_future, target_future.detach())
        loss_cos = 1.0 - F.cosine_similarity(
            pred_future.reshape(pred_future.shape[0], -1),
            target_future.detach().reshape(target_future.shape[0], -1),
            dim=-1,
        ).mean()
        return loss_s1 + 0.5 * loss_cos

    def _float32_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        with self._float32_context():
            lang_text = dataset_batch["lang_text"]
            token_out, latent_goal = self._render_token_dict(
                dataset_batch["rgb_obs"], dataset_batch["point_cloud_obs"], lang_text
            )
            current_tokens = token_out["head_tokens"]
            target_future = self._future_target_tokens(dataset_batch, lang_text)
            pred_future = self.gfdm(current_tokens, latent_goal)
            future_tokens = {
                "pred": pred_future,
                "gt": target_future,
                "none": torch.zeros_like(current_tokens),
            }[self.future_mode]

            lam_out, motion_latent = self._build_motion_latent(current_tokens, future_tokens, lang_text)
            bridge_feature = {"state_images_for_videoformer": current_tokens}
            native_pred = self.native_head(bridge_feature, latent_goal)
            motion_pred = self.motion_aux_head(motion_latent)

            if "trans_action_indicies" in dataset_batch:
                trans_idx = dataset_batch["trans_action_indicies"].to(self.device).long()[:, :1]
                rot_grip = dataset_batch["rot_grip_action_indicies"].to(self.device).long()[:, :1]
                rot_idx = rot_grip[..., :3]
                grip_idx = rot_grip[..., 3]
                coll_idx = dataset_batch["ignore_collisions"].to(self.device).long()[:, :1, 0]
            else:
                target = dataset_batch["actions"].to(self.device).long()
                target = target[:, :1]
                trans_idx = target[..., :3]
                rot_idx = target[..., 3:6]
                grip_idx = target[..., 6]
                coll_idx = target[..., 7]

            if not self._printed_label_stats:
                unique, counts = torch.unique(grip_idx, return_counts=True)
                stats = {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}
                print(f"[train-debug] gripper_label_counts={stats}")
                self._printed_label_stats = True

            if self.train_motion_only:
                loss_action = torch.zeros((), device=self.device, dtype=current_tokens.dtype)
            else:
                loss_action = F.cross_entropy(native_pred["trans_logits"].reshape(-1, self.n_trans_bins), trans_idx.reshape(-1))
                loss_action = loss_action + F.cross_entropy(native_pred["rot_logits"].reshape(-1, self.n_rot_bins), rot_idx.reshape(-1))
                loss_action = loss_action + self.lambda_gripper * F.cross_entropy(native_pred["gripper_logits"].reshape(-1, 2), grip_idx.reshape(-1))
                loss_action = loss_action + self.lambda_collision * F.cross_entropy(native_pred["collision_logits"].reshape(-1, 2), coll_idx.reshape(-1))

            loss_future = self._future_loss(pred_future, target_future)
            target_delta = self._clip_target_delta(dataset_batch["target_delta_xyz"].to(self.device).float())
            target_gripper = dataset_batch["target_gripper"].to(self.device).long().reshape(-1)
            loss_motion = F.smooth_l1_loss(motion_pred["delta_xyz"], target_delta)
            loss_motion = loss_motion + self.lambda_gripper * F.cross_entropy(motion_pred["gripper_logits"], target_gripper)

            q_loss = ((lam_out["emb"].detach() - lam_out["z"]) ** 2).mean()
            commit_loss = ((lam_out["emb"] - lam_out["z"].detach()) ** 2).mean()

            total = (
                self.lambda_action * loss_action
                + self.lambda_future * loss_future
                + self.lambda_motion * loss_motion
                + q_loss
                + 0.25 * commit_loss
            )
            return total

    def forward(self, batch):
        return self.training_step(batch)

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None

    def eval_forward(self, obs, goal):
        lang_text = goal["lang_text"]
        if isinstance(lang_text, str):
            lang_text = [lang_text]
        with self._float32_context():
            token_out, latent_goal = self._render_token_dict(obs["rgb_obs"], obs["point_cloud_obs"], lang_text)
            current_tokens = token_out["head_tokens"]
            pred_future = self.gfdm(current_tokens, latent_goal)
            lam_out, motion_latent = self._build_motion_latent(current_tokens, pred_future, lang_text)
            bridge_feature = {"state_images_for_videoformer": current_tokens}
            native_pred = self.native_head(bridge_feature, latent_goal)

            trans_idx = native_pred["trans_logits"].argmax(dim=-1)
            rot_idx = native_pred["rot_logits"].argmax(dim=-1)
            grip_idx = native_pred["gripper_logits"].argmax(dim=-1).float()
            coll_idx = native_pred["collision_logits"].argmax(dim=-1).float()

            pos = self._decode_translation(trans_idx)
            quat = self._decode_quaternion(rot_idx.reshape(-1, 3)).reshape(rot_idx.shape[0], rot_idx.shape[1], 4)
            gripper = torch.where(grip_idx > 0.5, torch.ones_like(grip_idx), -torch.ones_like(grip_idx)).unsqueeze(-1)
            collision = coll_idx.unsqueeze(-1)
            return torch.cat([pos, quat, gripper, collision], dim=-1)

    def step(self, obs, goal):
        self.pred_action_seq = self.eval_forward(obs, goal)
        current_action = self.pred_action_seq[0]
        if len(current_action.shape) == 1:
            current_action = current_action.unsqueeze(0)
        self.rollout_step_counter += 1
        return current_action
