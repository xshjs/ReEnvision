import math
from typing import Dict

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from policy_models.VPP_policy import VPP_Policy
from policy_models.module.discrete_action_head import DiscreteActionHead
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


class RLBenchDiscretePolicy(VPP_Policy):
    """RLBench discrete BC policy with absolute pose supervision."""

    def __init__(
        self,
        *args,
        waypoint_hidden_dim: int = 512,
        n_trans_bins: int = 100,
        n_rot_bins: int = 72,
        scene_bounds: list[float] | None = None,
        **kwargs,
    ):
        kwargs["action_dim"] = 8
        kwargs["action_head_type"] = "discrete"
        super().__init__(*args, n_trans_bins=n_trans_bins, n_rot_bins=n_rot_bins, **kwargs)
        self.model = torch.nn.Identity()
        self.scene_bounds = scene_bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self.trans_min = torch.tensor(self.scene_bounds[:3], dtype=torch.float32)
        self.trans_max = torch.tensor(self.scene_bounds[3:], dtype=torch.float32)
        self.rot_min = torch.tensor([-math.pi, -math.pi, -math.pi], dtype=torch.float32)
        self.rot_max = torch.tensor([math.pi, math.pi, math.pi], dtype=torch.float32)
        self.discrete_head = DiscreteActionHead(
            obs_dim=self.latent_dim,
            goal_dim=512,
            hidden_dim=waypoint_hidden_dim,
            action_seq_len=self.act_window_size,
            n_trans_bins=n_trans_bins,
            n_rot_bins=n_rot_bins,
            use_latent_motion_tokens=True,
            latent_motion_dim=768,
        )

    def configure_optimizers(self):
        optim_groups = [
            {
                "params": self.lam.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
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
                "params": self.discrete_head.parameters(),
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
        return optimizer

    def _discretize_translation(self, pos: torch.Tensor) -> torch.Tensor:
        trans_min = self.trans_min.to(pos.device)
        trans_max = self.trans_max.to(pos.device)
        scaled = (pos - trans_min) / (trans_max - trans_min).clamp_min(1e-6)
        scaled = scaled.clamp(0.0, 1.0)
        return torch.round(scaled * (self.n_trans_bins - 1)).long()

    def _discretize_rotation(self, quat: torch.Tensor) -> torch.Tensor:
        quat_np = quat.detach().cpu().numpy()
        euler_np = R.from_quat(quat_np).as_euler("xyz", degrees=False)
        euler = torch.from_numpy(euler_np).to(quat.device, dtype=torch.float32)
        euler = _wrap_to_pi(euler)
        scaled = (euler - self.rot_min.to(quat.device)) / (
            self.rot_max.to(quat.device) - self.rot_min.to(quat.device)
        ).clamp_min(1e-6)
        scaled = scaled.clamp(0.0, 1.0)
        return torch.round(scaled * (self.n_rot_bins - 1)).long()

    def _decode_translation(self, trans_idx: torch.Tensor) -> torch.Tensor:
        trans_min = self.trans_min.to(trans_idx.device)
        trans_max = self.trans_max.to(trans_idx.device)
        scaled = trans_idx.float() / max(self.n_trans_bins - 1, 1)
        return trans_min + scaled * (trans_max - trans_min)

    def _decode_quaternion(self, rot_idx: torch.Tensor) -> torch.Tensor:
        scaled = rot_idx.float() / max(self.n_rot_bins - 1, 1)
        euler = self.rot_min.to(rot_idx.device) + scaled * (
            self.rot_max.to(rot_idx.device) - self.rot_min.to(rot_idx.device)
        )
        quat_np = R.from_euler("xyz", euler.detach().cpu().numpy(), degrees=False).as_quat().astype("float32")
        quat = torch.from_numpy(quat_np).to(rot_idx.device)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return quat

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)
        univla_out = self.lam(
            predictive_feature["state_images"],
            dataset_batch["lang_text"],
        )
        latent_motion_tokens_up = univla_out["video_action_patches"].squeeze(1)
        pred = self.discrete_head(predictive_feature, latent_goal, latent_motion_tokens_up)

        target = dataset_batch["actions"].to(self.device)
        target = target[:, : self.act_window_size]
        target_pos = target[..., :3]
        target_quat = target[..., 3:7]
        target_gripper = (target[..., 7] > 0).float()

        trans_idx = self._discretize_translation(target_pos)
        rot_idx = self._discretize_rotation(target_quat.reshape(-1, 4)).reshape(target_quat.shape[0], target_quat.shape[1], 3)

        trans_loss = F.cross_entropy(
            pred["trans_logits"].reshape(-1, self.n_trans_bins),
            trans_idx.reshape(-1),
        )
        rot_loss = F.cross_entropy(
            pred["rot_logits"].reshape(-1, self.n_rot_bins),
            rot_idx.reshape(-1),
        )
        gripper_loss = F.binary_cross_entropy_with_logits(pred["gripper_logits"], target_gripper)
        bc_loss = trans_loss + rot_loss + 0.1 * gripper_loss

        q_loss_u = ((univla_out["emb"].detach() - univla_out["z"]) ** 2).mean()
        commit_loss_u = ((univla_out["emb"] - univla_out["z"].detach()) ** 2).mean()
        total_loss = bc_loss + q_loss_u + 0.25 * commit_loss_u

        total_bs = target.shape[0]
        self.log("train/trans_loss", trans_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/rot_loss", rot_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/gripper_loss", gripper_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/action_loss", bc_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        return total_loss

    def eval_forward(self, obs, goal):
        lang_text = goal["lang_text"]
        if isinstance(lang_text, str):
            lang_text = [lang_text]
        predictive_feature, latent_goal = self.extract_predictive_feature({**obs, "lang_text": lang_text})
        univla_out = self.lam(
            predictive_feature["state_images"],
            lang_text,
        )
        latent_motion_tokens_up = univla_out["video_action_patches"].squeeze(1)
        pred = self.discrete_head(predictive_feature, latent_goal, latent_motion_tokens_up)

        trans_idx = pred["trans_logits"].argmax(dim=-1)
        rot_idx = pred["rot_logits"].argmax(dim=-1)
        gripper = torch.where(
            torch.sigmoid(pred["gripper_logits"]) > 0.5,
            torch.ones_like(pred["gripper_logits"]),
            -torch.ones_like(pred["gripper_logits"]),
        ).unsqueeze(-1)

        pos = self._decode_translation(trans_idx)
        quat = self._decode_quaternion(rot_idx.reshape(-1, 3)).reshape(rot_idx.shape[0], rot_idx.shape[1], 4)
        return torch.cat([pos, quat, gripper], dim=-1)
