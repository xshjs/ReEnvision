from typing import Dict

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

try:
    import inspect as _inspect
    import triton as _triton

    if hasattr(_triton, "autotune"):
        _autotune_sig = _inspect.signature(_triton.autotune)
        if "use_cuda_graph" not in _autotune_sig.parameters:
            _orig_autotune = _triton.autotune

            def _compat_autotune(*args, **kwargs):
                kwargs.pop("use_cuda_graph", None)
                return _orig_autotune(*args, **kwargs)

            _triton.autotune = _compat_autotune
except Exception:
    pass

import einops
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from policy_models.module.Video_Former import Video_Former_3D
from policy_models.module.clip_lang_encoder import LangClip
from policy_models.module.rlbench_native_head import RLBenchNativeHead
from policy_models.module.rlbench_waypoint_head import quaternion_geodesic_loss
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


def load_primary_models(pretrained_model_path: str, eval_mode: bool = False, device: str | torch.device | None = None):
    from diffusers import StableVideoDiffusionPipeline

    use_fp16 = eval_mode
    if device is not None:
        device_str = str(device)
        if device_str == "cpu" or (device_str.startswith("cuda") and not torch.cuda.is_available()):
            use_fp16 = False
    elif not torch.cuda.is_available():
        use_fp16 = False

    if use_fp16:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            local_files_only=True,
        )
    return pipeline


class RLBenchNativePolicy(nn.Module):
    """Lightweight RLBench-native DeFi policy used for native 9D action training/eval."""

    def __init__(
        self,
        optimizer,
        lr_scheduler,
        latent_dim: int = 512,
        multistep: int = 10,
        use_lr_scheduler: bool = True,
        act_window_size: int = 10,
        use_text_not_embedding: bool = False,
        seed: int = 42,
        pretrained_model_path: str = "/ckpt/svd/checkpoint-100000",
        text_encoder_path: str = "/ckpt/clip-vit-base-patch32",
        t5_model_path: str = "/ckpt/t5_base",
        language_goal_path: str = "/ckpt/ViT-B-32.pt",
        use_position_encoding: bool = True,
        Former_depth: int = 3,
        Former_heads: int = 8,
        Former_dim_head: int = 64,
        Former_num_time_embeds: int = 1,
        num_latents: int = 3,
        use_Former: str = "3d",
        timestep: int = 20,
        max_length: int = 20,
        extract_layer_idx: int = 1,
        use_all_layer: bool = False,
        obs_seq_len: int = 1,
        action_dim: int = 9,
        action_seq_len: int = 10,
        action_head_type: str = "rlbench_native",
        n_trans_bins: int = 21,
        n_rot_bins: int = 21,
        first_steps_weight: float = 1.0,
        first_steps_count: int = 0,
        waypoint_hidden_dim: int = 512,
        pos_loss_weight: float = 1.0,
        quat_loss_weight: float = 1.0,
        gripper_loss_weight: float = 0.1,
        collision_loss_weight: float = 0.1,
        **unused_kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.multistep = multistep
        self.use_lr_scheduler = use_lr_scheduler
        self.act_window_size = act_window_size
        self.use_text_not_embedding = use_text_not_embedding
        self.seed = seed
        self.use_position_encoding = use_position_encoding
        self.t5_model_path = t5_model_path
        self.language_goal_path = language_goal_path
        self.Former_num_time_embeds = Former_num_time_embeds
        self.max_length = max_length
        self.extract_layer_idx = extract_layer_idx
        self.use_all_layer = use_all_layer
        self.use_Former = use_Former
        self.timestep = timestep
        self.action_dim = 9
        self.action_head_type = action_head_type
        self.n_trans_bins = n_trans_bins
        self.n_rot_bins = n_rot_bins
        self.first_steps_weight = first_steps_weight
        self.first_steps_count = first_steps_count
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.use_gripper = True
        self.use_univla = False
        self.pos_loss_weight = pos_loss_weight
        self.quat_loss_weight = quat_loss_weight
        self.gripper_loss_weight = gripper_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.rollout_step_counter = 0
        self.pred_action_seq = None

        condition_dim_list = [1280, 1280, 1280, 640]
        sum_dim = 0
        for i in range(extract_layer_idx + 1):
            sum_dim = sum_dim + condition_dim_list[i + 1]
        condition_dim = condition_dim_list[extract_layer_idx + 1] if not self.use_all_layer else sum_dim

        self.Video_Former = Video_Former_3D(
            dim=latent_dim,
            depth=Former_depth,
            dim_head=Former_dim_head,
            heads=Former_heads,
            num_time_embeds=Former_num_time_embeds,
            num_latents=num_latents,
            condition_dim=condition_dim,
            use_temporal=True,
        )

        self.language_goal = LangClip(model_name=self.language_goal_path)

        pipeline = load_primary_models(pretrained_model_path, eval_mode=True, device=self.device)
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_path,
            use_fast=False,
            local_files_only=True,
        )
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            text_encoder_path,
            local_files_only=True,
        ).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.parameters():
            param.requires_grad = False
        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False
        pipeline.unet.eval()

        from policy_models.module.diffusion_extract import Diffusion_feature_extractor

        self.TVP_encoder = Diffusion_feature_extractor(
            pipeline=pipeline,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            position_encoding=self.use_position_encoding,
        )

        self.goal_emb = nn.Sequential(
            nn.Linear(condition_dim, 768),
            nn.GELU(),
            nn.Linear(768, 768),
        )
        self.time_pos_emb = nn.Parameter(torch.randn(2, 1, 768))
        self.native_head = RLBenchNativeHead(
            obs_dim=self.latent_dim,
            goal_dim=512,
            hidden_dim=waypoint_hidden_dim,
            action_seq_len=self.act_window_size,
        )

        # Keep this name out of the checkpoint load path but preserve constructor compatibility.
        from policy_models.m_former_univla.latent_motion_tokenizer_univla import (
            UncontrolledDINOLatentActionModel,
        )

        self.lam = UncontrolledDINOLatentActionModel(t5_model_path=t5_model_path)
        for param in self.lam.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def log(self, *args, **kwargs):
        return None

    def process_device(self):
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
        self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)
        self.language_goal = self.language_goal.to(self.device)
        self.TVP_encoder = self.TVP_encoder.to(self.device)
        self.Video_Former = self.Video_Former.to(self.device)
        self.goal_emb = self.goal_emb.to(self.device)
        self.native_head = self.native_head.to(self.device)
        self.lam = self.lam.to(self.device)

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
                "params": self.native_head.parameters(),
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

    def on_train_start(self) -> None:
        dtype = next(self.parameters()).dtype
        self.Video_Former.to(dtype=dtype)
        self.language_goal.to(dtype=dtype)
        self.TVP_encoder.to(dtype=dtype)

    def extract_predictive_feature(self, dataset_batch):
        rgb_obs = dataset_batch["rgb_obs"]
        camera_keys = [k for k in rgb_obs.keys() if not k.endswith("_origin")]
        if not camera_keys:
            raise ValueError("No RGB observations found in dataset_batch['rgb_obs'].")
        camera_tensors = [rgb_obs[key].to(self.device) for key in camera_keys]
        base_dtype = camera_tensors[0].dtype
        modality = "lang"
        if self.use_text_not_embedding:
            latent_goal = self.language_goal(dataset_batch["lang_text"]).to(base_dtype)
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(base_dtype)

        language = dataset_batch["lang_text"]
        num_frames = self.Former_num_time_embeds
        batch = camera_tensors[0].shape[0]
        num_cams = len(camera_tensors)

        with torch.no_grad():
            input_rgb = torch.cat(camera_tensors, dim=0)
            language = language * num_cams
            perceptual_features = self.TVP_encoder(
                input_rgb,
                language,
                self.timestep,
                self.extract_layer_idx,
                all_layer=self.use_all_layer,
                step_time=1,
                max_length=self.max_length,
            )

        perceptual_features = einops.rearrange(perceptual_features, "b f c h w-> b f c (h w)")
        perceptual_features = einops.rearrange(perceptual_features, "b f c l-> b f l c")
        perceptual_features = perceptual_features[:, :num_frames, :, :]
        camera_features = list(torch.split(perceptual_features, [batch] * num_cams, dim=0))
        perceptual_features = torch.stack(camera_features, dim=0).mean(dim=0)
        perceptual_features_for_videoformer = torch.cat(camera_features, dim=2)

        perceptual_features_for_videoformer = perceptual_features_for_videoformer.to(torch.float32)
        perceptual_features = perceptual_features.to(torch.float32)

        frame_0 = perceptual_features[:, 0]
        frame_5 = perceptual_features[:, 5]
        perceptual_features = torch.stack([frame_0, frame_5], dim=1)
        perceptual_features = self.goal_emb(perceptual_features)
        time_pos_emb = self.time_pos_emb.unsqueeze(0).expand(perceptual_features.size(0), -1, -1, -1)
        perceptual_features = perceptual_features + time_pos_emb

        perceptual_features_for_videoformer = self.Video_Former(perceptual_features_for_videoformer)
        if self.use_Former == "linear":
            perceptual_features = rearrange(perceptual_features, "b T q d -> b (T q) d")
        predictive_feature = {
            "state_images": perceptual_features,
            "state_images_for_videoformer": perceptual_features_for_videoformer,
            "modality": modality,
        }
        return predictive_feature, latent_goal

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)
        pred = self.native_head(predictive_feature, latent_goal)

        target = dataset_batch["actions"].to(self.device)
        target_pos = target[..., :3]
        target_quat = target[..., 3:7]
        target_gripper = (target[..., 7] > 0).float()
        target_collision = (target[..., 8] > 0.5).float()

        pos_loss = F.smooth_l1_loss(pred["pos"], target_pos)
        quat_loss = quaternion_geodesic_loss(pred["quat"], target_quat)
        gripper_loss = F.binary_cross_entropy_with_logits(pred["gripper_logits"], target_gripper)
        collision_loss = F.binary_cross_entropy_with_logits(pred["collision_logits"], target_collision)
        total_loss = (
            self.pos_loss_weight * pos_loss
            + self.quat_loss_weight * quat_loss
            + self.gripper_loss_weight * gripper_loss
            + self.collision_loss_weight * collision_loss
        )
        return total_loss

    def forward(self, batch):
        return self.training_step(batch)

    def reset(self):
        self.rollout_step_counter = 0
        self.pred_action_seq = None

    def eval_forward(self, obs, goal):
        lang_text = goal["lang_text"]
        if isinstance(lang_text, str):
            lang_text = [lang_text]
        if "lang_text" in goal:
            latent_goal = self.language_goal(lang_text).to(torch.float32)
        else:
            latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                obs["rgb_obs"]["rgb_static"].device
            )
        predictive_feature, _ = self.extract_predictive_feature({**obs, "lang_text": lang_text})
        pred = self.native_head(predictive_feature, latent_goal)
        gripper = torch.where(
            torch.sigmoid(pred["gripper_logits"]) > 0.5,
            torch.ones_like(pred["gripper_logits"]),
            -torch.ones_like(pred["gripper_logits"]),
        ).unsqueeze(-1)
        collision = torch.where(
            torch.sigmoid(pred["collision_logits"]) > 0.5,
            torch.ones_like(pred["collision_logits"]),
            torch.zeros_like(pred["collision_logits"]),
        ).unsqueeze(-1)
        return torch.cat([pred["pos"], pred["quat"], gripper, collision], dim=-1)

    def step(self, obs, goal):
        # RLBench execution here is receding-horizon: predict from the current
        # observation every step, then execute only the first action.
        self.pred_action_seq = self.eval_forward(obs, goal)
        current_action = self.pred_action_seq[0, 0]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, "b d -> b 1 d")
        self.rollout_step_counter += 1
        return current_action
