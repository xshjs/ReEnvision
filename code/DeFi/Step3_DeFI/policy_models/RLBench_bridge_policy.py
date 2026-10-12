import numpy as np
from typing import Dict

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from policy_models.RLBench_native_policy import RLBenchNativePolicy
from policy_models.module.rlbench_bridge_head import RLBenchBridgeHead
from policy_models.module.rlbench_mvt import RLBenchMVT
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


class RLBenchBridgePolicy(RLBenchNativePolicy):
    """Bridge-style RLBench discrete policy with DeFi visual backbone."""

    def __init__(
        self,
        *args,
        waypoint_hidden_dim: int = 512,
        n_trans_bins: int = 100,
        n_rot_bins: int = 72,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        gripper_loss_weight: float = 0.2,
        collision_loss_weight: float = 1.0,
        scene_bounds: list[float] | None = None,
        voxel_size: int = 100,
        mvt_token_dim: int = 256,
        mvt_max_tokens: int = 256,
        mvt_transformer_depth: int = 4,
        **kwargs,
    ):
        kwargs["action_dim"] = 9
        kwargs["action_head_type"] = "rlbench_bridge"
        super().__init__(*args, **kwargs)
        self.n_trans_bins = n_trans_bins
        self.n_rot_bins = n_rot_bins
        self.trans_loss_weight = trans_loss_weight
        self.rot_loss_weight = rot_loss_weight
        self.gripper_loss_weight = gripper_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.scene_bounds = scene_bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self.bridge_head = RLBenchBridgeHead(
            obs_dim=self.latent_dim,
            goal_dim=512,
            hidden_dim=waypoint_hidden_dim,
            action_seq_len=self.act_window_size,
            n_trans_bins=n_trans_bins,
            n_rot_bins=n_rot_bins,
        )
        self.mvt = RLBenchMVT(
            scene_bounds=self.scene_bounds,
            voxel_size=voxel_size,
            out_dim=waypoint_hidden_dim,
            token_dim=mvt_token_dim,
            max_tokens=mvt_max_tokens,
            depth=mvt_transformer_depth,
        )

    def process_device(self):
        super().process_device()
        self.bridge_head = self.bridge_head.to(self.device)
        self.mvt = self.mvt.to(self.device)

    def on_train_start(self) -> None:
        super().on_train_start()
        dtype = next(self.parameters()).dtype
        self.bridge_head.to(dtype=dtype)
        self.mvt.to(dtype=dtype)

    def configure_optimizers(self):
        optim_groups = [
            {
                "params": self.Video_Former.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
            {
                "params": self.goal_emb.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
            {
                "params": [self.time_pos_emb],
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
            {
                "params": self.bridge_head.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
            {
                "params": self.mvt.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.betas,
        )
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": None, "interval": "step", "frequency": 1}}

    def extract_predictive_feature(self, dataset_batch):
        predictive_feature, latent_goal = super().extract_predictive_feature(dataset_batch)
        point_cloud_obs = dataset_batch.get("point_cloud_obs", {})
        if point_cloud_obs:
            voxel_out = self.mvt(point_cloud_obs, dataset_batch["rgb_obs"])
            predictive_feature["voxel_features"] = voxel_out["global_feature"]
            predictive_feature["voxel_tokens"] = voxel_out["token_features"]
            predictive_feature["voxel_token_padding_mask"] = voxel_out["token_padding_mask"]
        return predictive_feature, latent_goal

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)
        pred = self.bridge_head(predictive_feature, latent_goal)

        if "trans_action_indicies" in dataset_batch:
            trans_idx = dataset_batch["trans_action_indicies"].to(self.device).long()[:, : self.act_window_size]
            rot_grip = dataset_batch["rot_grip_action_indicies"].to(self.device).long()[:, : self.act_window_size]
            rot_idx = rot_grip[..., :3]
            grip_idx = rot_grip[..., 3]
            coll_idx = dataset_batch["ignore_collisions"].to(self.device).long()[:, : self.act_window_size, 0]
        else:
            target = dataset_batch["actions"].to(self.device).long()
            target = target[:, : self.act_window_size]
            trans_idx = target[..., :3]
            rot_idx = target[..., 3:6]
            grip_idx = target[..., 6]
            coll_idx = target[..., 7]

        trans_loss = F.cross_entropy(pred["trans_logits"].reshape(-1, self.n_trans_bins), trans_idx.reshape(-1))
        rot_loss = F.cross_entropy(pred["rot_logits"].reshape(-1, self.n_rot_bins), rot_idx.reshape(-1))
        gripper_loss = F.cross_entropy(pred["gripper_logits"].reshape(-1, 2), grip_idx.reshape(-1))
        collision_loss = F.cross_entropy(pred["collision_logits"].reshape(-1, 2), coll_idx.reshape(-1))
        total_loss = (
            self.trans_loss_weight * trans_loss
            + self.rot_loss_weight * rot_loss
            + self.gripper_loss_weight * gripper_loss
            + self.collision_loss_weight * collision_loss
        )
        return total_loss

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

    def eval_forward(self, obs, goal):
        lang_text = goal["lang_text"]
        if isinstance(lang_text, str):
            lang_text = [lang_text]
        predictive_feature, latent_goal = self.extract_predictive_feature({**obs, "lang_text": lang_text})
        pred = self.bridge_head(predictive_feature, latent_goal)
        trans_idx = pred["trans_logits"].argmax(dim=-1)
        rot_idx = pred["rot_logits"].argmax(dim=-1)
        grip_idx = pred["gripper_logits"].argmax(dim=-1).float()
        coll_idx = pred["collision_logits"].argmax(dim=-1).float()

        pos = self._decode_translation(trans_idx)
        quat = self._decode_quaternion(rot_idx.reshape(-1, 3)).reshape(rot_idx.shape[0], rot_idx.shape[1], 4)
        gripper = torch.where(grip_idx > 0.5, torch.ones_like(grip_idx), -torch.ones_like(grip_idx)).unsqueeze(-1)
        collision = coll_idx.unsqueeze(-1)
        return torch.cat([pos, quat, gripper, collision], dim=-1)
