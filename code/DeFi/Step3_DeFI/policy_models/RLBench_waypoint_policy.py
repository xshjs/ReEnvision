from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

from policy_models.VPP_policy import VPP_Policy
from policy_models.module.rlbench_waypoint_head import RLBenchWaypointHead, quaternion_geodesic_loss
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


class RLBenchWaypointPolicy(VPP_Policy):
    """RLBench-native absolute waypoint policy.

    This keeps the existing DeFi visual/language encoder stack but replaces the
    CALVIN-style continuous delta diffusion decoder with an absolute waypoint
    head that predicts xyz + quaternion + gripper.
    """

    def __init__(
        self,
        *args,
        waypoint_hidden_dim: int = 512,
        pos_loss_weight: float = 1.0,
        quat_loss_weight: float = 1.0,
        gripper_loss_weight: float = 0.1,
        stage_loss_weight: float = 0.5,
        num_stage_classes: int = 4,
        stage_gate_class: int = 2,
        use_stage_gate: bool = False,
        **kwargs,
    ):
        kwargs["action_dim"] = 8
        super().__init__(*args, **kwargs)
        self.pos_loss_weight = pos_loss_weight
        self.quat_loss_weight = quat_loss_weight
        self.gripper_loss_weight = gripper_loss_weight
        self.stage_loss_weight = stage_loss_weight
        self.num_stage_classes = num_stage_classes
        self.stage_gate_class = stage_gate_class
        self.use_stage_gate = use_stage_gate
        self.waypoint_head = RLBenchWaypointHead(
            obs_dim=self.latent_dim,
            goal_dim=512,
            hidden_dim=waypoint_hidden_dim,
            action_seq_len=self.act_window_size,
            num_stage_classes=num_stage_classes,
        )

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
                "params": self.waypoint_head.parameters(),
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

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)
        pred = self.waypoint_head(predictive_feature, latent_goal)

        target = dataset_batch["actions"].to(self.device)
        target_pos = target[..., :3]
        target_quat = target[..., 3:7]
        target_gripper = (target[..., 7] > 0).float()
        target_stage = dataset_batch.get("stage")
        if target_stage is not None:
            target_stage = target_stage.to(self.device).long()

        pos_loss = F.smooth_l1_loss(pred["pos"], target_pos)
        quat_loss = quaternion_geodesic_loss(pred["quat"], target_quat)
        gripper_loss = F.binary_cross_entropy_with_logits(pred["gripper_logits"], target_gripper)
        if target_stage is not None:
            stage_loss = F.cross_entropy(
                pred["stage_logits"].reshape(-1, self.num_stage_classes),
                target_stage.reshape(-1),
            )
        else:
            stage_loss = torch.tensor(0.0, device=self.device)
        total_loss = (
            self.pos_loss_weight * pos_loss
            + self.quat_loss_weight * quat_loss
            + self.gripper_loss_weight * gripper_loss
            + self.stage_loss_weight * stage_loss
        )
        total_bs = target.shape[0]
        self.log("train/pos_loss", pos_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/quat_loss", quat_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/gripper_loss", gripper_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/stage_loss", stage_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        return total_loss

    def eval_forward(self, obs, goal):
        lang_text = goal["lang_text"]
        if isinstance(lang_text, str):
            lang_text = [lang_text]
        if "lang_text" in goal:
            latent_goal = self.language_goal(lang_text)
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                obs["rgb_obs"]["rgb_static"].device
            )
        predictive_feature, _ = self.extract_predictive_feature({**obs, "lang_text": lang_text})
        pred = self.waypoint_head(predictive_feature, latent_goal)
        gripper_logits = pred["gripper_logits"]
        if getattr(self, "use_stage_gate", True):
            stage_pred = pred["stage_logits"].argmax(dim=-1)
            allow_close = stage_pred >= self.stage_gate_class
            gripper = torch.where(
                allow_close & (torch.sigmoid(gripper_logits) > 0.5),
                torch.ones_like(gripper_logits),
                -torch.ones_like(gripper_logits),
            ).unsqueeze(-1)
        else:
            gripper = torch.where(
                torch.sigmoid(gripper_logits) > 0.5,
                torch.ones_like(gripper_logits),
                -torch.ones_like(gripper_logits),
            ).unsqueeze(-1)
        return torch.cat([pred["pos"], pred["quat"], gripper], dim=-1)
