from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from policy_models.module.bridge_full_mvt import BridgeFullMVT
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


def _ensure_bridgevla_imports():
    root = Path(__file__).resolve().parents[3]
    finetune_root = root / "BridgeVLA" / "finetune"
    point_renderer_root = finetune_root / "bridgevla" / "libs" / "point-renderer"
    for path in (finetune_root, point_renderer_root):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_ensure_bridgevla_imports()
import bridgevla.mvt.utils as bridge_mvt_utils  # noqa: E402
from bridgevla.mvt.config import get_cfg_defaults  # noqa: E402


class RLBenchFullMVTPolicy(nn.Module):
    def __init__(
        self,
        optimizer,
        lr_scheduler,
        use_lr_scheduler: bool = True,
        act_window_size: int = 1,
        action_dim: int = 9,
        num_rotation_classes: int = 72,
        gt_hm_sigma: float = 1.5,
        img_aug: float = 0.0,
        add_rgc_loss: bool = True,
        place_with_mean: bool = True,
        move_pc_in_bound: bool = True,
        scene_bounds: list[float] | None = None,
        mvt_cfg_overrides: dict | None = None,
        **unused_kwargs,
    ):
        super().__init__()
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.use_lr_scheduler = use_lr_scheduler
        self.act_window_size = act_window_size
        self.action_dim = action_dim
        self.action_head_type = "rlbench_bridge"
        self.num_rotation_classes = num_rotation_classes
        self.rotation_resolution = 360.0 / num_rotation_classes
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.place_with_mean = place_with_mean
        self.move_pc_in_bound = move_pc_in_bound
        self.scene_bounds = scene_bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.num_all_rot = self.num_rotation_classes * 3
        self.pred_action_seq = None
        self.rollout_step_counter = 0

        cfg = get_cfg_defaults()
        if mvt_cfg_overrides:
            override_dict = OmegaConf.to_container(OmegaConf.create(mvt_cfg_overrides), resolve=True)
            for key, value in override_dict.items():
                setattr(cfg, key, value)
        cfg.num_rot = num_rotation_classes
        cfg.feat_dim = (num_rotation_classes * 3) + 2 + 2

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mvt = BridgeFullMVT(renderer_device=device_str, **cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def log(self, *args, **kwargs):
        return None

    def process_device(self):
        self.mvt = self.mvt.to(self.device)

    def on_train_start(self) -> None:
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.optimizer_config.learning_rate,
            weight_decay=self.optimizer_config.transformer_weight_decay,
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
        pcs = []
        feats = []
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
                pcd = torch.nn.functional.interpolate(pcd, size=rgb.shape[-2:], mode="nearest")
            pcd_flat = pcd.permute(0, 2, 3, 1).reshape(pcd.shape[0], -1, 3)
            rgb_flat = ((rgb.permute(0, 2, 3, 1) + 1.0) * 0.5).reshape(rgb.shape[0], -1, rgb.shape[1]).clamp(0.0, 1.0)
            if pcd_flat.shape[1] != rgb_flat.shape[1]:
                n_pts = min(pcd_flat.shape[1], rgb_flat.shape[1])
                pcd_flat = pcd_flat[:, :n_pts]
                rgb_flat = rgb_flat[:, :n_pts]
            pcs.append(pcd_flat)
            feats.append(rgb_flat)
        pc = torch.cat(pcs, dim=1)
        img_feat = torch.cat(feats, dim=1)
        return pc, img_feat

    def _move_pc_in_bound(self, pc: torch.Tensor, img_feat: torch.Tensor):
        if not self.move_pc_in_bound:
            return [pc[i] for i in range(pc.shape[0])], [img_feat[i] for i in range(img_feat.shape[0])]
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
        pc_list = [pc[i, ~mask] for i, mask in enumerate(inv)]
        feat_list = [img_feat[i, ~mask] for i, mask in enumerate(inv)]
        return pc_list, feat_list

    def _wrap_language_goal(self, lang_text):
        if isinstance(lang_text, str):
            return [[lang_text]]
        return [[x] for x in lang_text]

    def _normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        quat = quat.astype(np.float32)
        quat = quat / np.linalg.norm(quat).clip(min=1e-6)
        if quat[-1] < 0:
            quat = -quat
        return quat

    def _quat_to_disc_euler(self, quat_xyzw: np.ndarray) -> np.ndarray:
        quat = self._normalize_quaternion(quat_xyzw)
        euler = R.from_quat(quat).as_euler("xyz", degrees=True)
        euler = np.mod(np.round(euler), 360.0)
        disc = np.floor(euler / self.rotation_resolution + 0.5).astype(np.int64) % self.num_rotation_classes
        return disc

    def _get_one_hot_expert_actions(self, action_rot, action_grip, action_ignore_collisions, device):
        bs = action_rot.shape[0]
        rot_x = torch.zeros((bs, self.num_rotation_classes), dtype=torch.int64, device=device)
        rot_y = torch.zeros((bs, self.num_rotation_classes), dtype=torch.int64, device=device)
        rot_z = torch.zeros((bs, self.num_rotation_classes), dtype=torch.int64, device=device)
        grip = torch.zeros((bs, 2), dtype=torch.int64, device=device)
        coll = torch.zeros((bs, 2), dtype=torch.int64, device=device)
        action_rot_np = action_rot.detach().cpu().numpy()
        for b in range(bs):
            disc = self._quat_to_disc_euler(action_rot_np[b])
            rot_x[b, disc[0]] = 1
            rot_y[b, disc[1]] = 1
            rot_z[b, disc[2]] = 1
            grip[b, int(action_grip[b].item())] = 1
            coll[b, int(action_ignore_collisions[b, 0].item())] = 1
        return rot_x, rot_y, rot_z, grip, coll

    def _preprocess_batch(self, dataset_batch: Dict[str, Dict]):
        point_cloud_obs = dataset_batch["point_cloud_obs"]
        rgb_obs = dataset_batch["rgb_obs"]
        pc, img_feat = self._get_pc_img_feat(point_cloud_obs, rgb_obs)
        pc, img_feat = self._move_pc_in_bound(pc, img_feat)
        action_gripper_pose = dataset_batch["gripper_pose"][:, -1].to(self.device).float()
        action_ignore_collisions = dataset_batch["ignore_collisions"][:, -1].to(self.device).int()
        action_rot_grip = dataset_batch["rot_grip_action_indicies"][:, -1].to(self.device).int()
        action_rot = action_gripper_pose[:, 3:7]
        action_grip = action_rot_grip[:, -1]
        wpt = [x[:3] for x in action_gripper_pose]
        wpt_local = []
        rev_trans = []
        for _pc, _wpt in zip(pc, wpt):
            a, b = bridge_mvt_utils.place_pc_in_cube(
                _pc,
                _wpt,
                with_mean_or_bounds=self.place_with_mean,
                scene_bounds=None if self.place_with_mean else self.scene_bounds,
            )
            wpt_local.append(a.unsqueeze(0))
            rev_trans.append(b)
        wpt_local = torch.cat(wpt_local, dim=0)
        pc_local = [
            bridge_mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self.place_with_mean,
                scene_bounds=None if self.place_with_mean else self.scene_bounds,
            )[0]
            for _pc in pc
        ]
        return pc_local, img_feat, wpt_local, rev_trans, action_rot, action_grip, action_ignore_collisions, action_gripper_pose

    def _get_q(self, out, dims):
        bs, nc, h, w = dims
        q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
        rot_q = out["feat"].view(bs, -1)[:, 0:self.num_all_rot]
        grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot:self.num_all_rot + 2]
        collision_q = out["feat"].view(bs, -1)[:, self.num_all_rot + 2:self.num_all_rot + 4]
        return q_trans, rot_q, grip_q, collision_q

    @torch.no_grad()
    def _get_action_trans(self, wpt_local, out, dims):
        bs, nc, h, w = dims
        wpt_img = self.mvt.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=None,
            out=None,
        ).squeeze(1)
        action_trans = bridge_mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2), (h, w), sigma=self.gt_hm_sigma, thres_sigma_times=3
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()
        return action_trans

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        pc_local, img_feat, wpt_local, _, action_rot, action_grip, action_ignore_collisions, _ = self._preprocess_batch(dataset_batch)
        language_goal = self._wrap_language_goal(dataset_batch["lang_text"])
        out = self.mvt(
            pc=pc_local,
            img_feat=img_feat,
            img_aug=self.img_aug if self.training else 0,
            wpt_local=wpt_local,
            rot_x_y=None,
            language_goal=language_goal,
        )
        bs = len(pc_local)
        nc = self.mvt.num_img
        h = w = self.mvt.img_size
        q_trans, rot_q, grip_q, collision_q = self._get_q(out, dims=(bs, nc, h, w))
        action_trans = self._get_action_trans(wpt_local, out, dims=(bs, nc, h, w))
        action_rot_x_one_hot, action_rot_y_one_hot, action_rot_z_one_hot, action_grip_one_hot, action_collision_one_hot = (
            self._get_one_hot_expert_actions(action_rot, action_grip, action_ignore_collisions, device=self.device)
        )
        trans_loss = self.cross_entropy(q_trans, action_trans).mean()
        rot_loss_x = self.cross_entropy(
            rot_q[:, 0 * self.num_rotation_classes:1 * self.num_rotation_classes],
            action_rot_x_one_hot.argmax(-1),
        ).mean()
        rot_loss_y = self.cross_entropy(
            rot_q[:, 1 * self.num_rotation_classes:2 * self.num_rotation_classes],
            action_rot_y_one_hot.argmax(-1),
        ).mean()
        rot_loss_z = self.cross_entropy(
            rot_q[:, 2 * self.num_rotation_classes:3 * self.num_rotation_classes],
            action_rot_z_one_hot.argmax(-1),
        ).mean()
        grip_loss = self.cross_entropy(grip_q, action_grip_one_hot.argmax(-1)).mean()
        collision_loss = self.cross_entropy(collision_q, action_collision_one_hot.argmax(-1)).mean()
        return trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss

    def forward(self, batch):
        return self.training_step(batch)

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None

    def _disc_to_quat(self, disc_xyz: np.ndarray) -> np.ndarray:
        euler_deg = disc_xyz.astype(np.float32) * self.rotation_resolution
        quat = R.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
        quat = quat / np.linalg.norm(quat).clip(min=1e-6)
        if quat[-1] < 0:
            quat = -quat
        return quat

    def eval_forward(self, obs, goal):
        point_cloud_obs = {k: v.to(self.device) for k, v in obs["point_cloud_obs"].items()}
        rgb_obs = {k: v.to(self.device) for k, v in obs["rgb_obs"].items()}
        pc, img_feat = self._get_pc_img_feat(point_cloud_obs, rgb_obs)
        pc, img_feat = self._move_pc_in_bound(pc, img_feat)
        rev_trans = []
        pc_local = []
        for _pc in pc:
            a, b = bridge_mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self.place_with_mean,
                scene_bounds=None if self.place_with_mean else self.scene_bounds,
            )
            pc_local.append(a)
            rev_trans.append(b)
        language_goal = self._wrap_language_goal(goal["lang_text"] if isinstance(goal["lang_text"], list) else [goal["lang_text"]])
        out = self.mvt(pc=pc_local, img_feat=img_feat, img_aug=0, language_goal=language_goal)
        bs = len(pc_local)
        nc = self.mvt.num_img
        h = w = self.mvt.img_size
        _, rot_q, grip_q, collision_q = self._get_q(out, dims=(bs, nc, h, w))
        pred_local = self.mvt.get_wpt(out, mvt1_or_mvt2=True, dyn_cam_info=None).detach().cpu()
        pred_world = torch.stack([rev(pred_local[i:i+1]).squeeze(0).to(self.device) for i, rev in enumerate(rev_trans)], dim=0)

        rot_q = rot_q.view(bs, 3, self.num_rotation_classes)
        rot_idx = rot_q.argmax(dim=-1).detach().cpu().numpy()
        quat = np.stack([self._disc_to_quat(rot_idx[i]) for i in range(bs)], axis=0)
        quat = torch.from_numpy(quat).to(self.device)
        gripper = torch.where(grip_q.argmax(dim=-1) > 0, torch.ones(bs, device=self.device), -torch.ones(bs, device=self.device)).unsqueeze(-1)
        collision = collision_q.argmax(dim=-1).float().unsqueeze(-1)
        action = torch.cat([pred_world, quat, gripper, collision], dim=-1)
        return action.unsqueeze(1)

    def step(self, obs, goal):
        self.pred_action_seq = self.eval_forward(obs, goal)
        current_action = self.pred_action_seq[0, 0]
        if len(current_action.shape) == 2:
            current_action = current_action.view(current_action.shape[0], 1, current_action.shape[1])
        self.rollout_step_counter += 1
        return current_action
