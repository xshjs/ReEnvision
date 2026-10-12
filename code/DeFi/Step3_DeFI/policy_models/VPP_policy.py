import logging
from typing import Dict, Optional, Tuple
from functools import partial
import os

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
    import transformers.utils.import_utils as _tf_import_utils
    import transformers.modeling_utils as _tf_modeling_utils

    if not hasattr(_tf_utils, "FLAX_WEIGHTS_NAME"):
        _tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if hasattr(_tf_utils, "check_torch_load_is_safe"):
        _tf_utils.check_torch_load_is_safe = lambda: None
    if hasattr(_tf_import_utils, "check_torch_load_is_safe"):
        _tf_import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(_tf_modeling_utils, "check_torch_load_is_safe"):
        _tf_modeling_utils.check_torch_load_is_safe = lambda: None
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

import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_only
except Exception:
    class _LightningModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def save_hyperparameters(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    class _PLNamespace:
        LightningModule = _LightningModule

    pl = _PLNamespace()

    def rank_zero_only(fn):
        return fn
import einops
from policy_models.edm_diffusion.score_wrappers import GCDenoiser
import omegaconf
import hydra
from pathlib import Path
from policy_models.module.clip_lang_encoder import LangClip
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_3D


logger = logging.getLogger(__name__)

_GC_SAMPLING_IMPORTED = False
_GC_SAMPLING_IMPORT_ERROR = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        candidate = str(candidate).strip()
        if not candidate or candidate in {".", ".."}:
            continue
        if os.path.exists(candidate):
            return candidate
    return ""


def _resolve_model_path(user_value: str, *, env_key: str, fallback_candidates: list[str], label: str) -> str:
    chosen = _first_existing_path(
        user_value,
        os.environ.get(env_key, ""),
        *fallback_candidates,
    )
    if not chosen:
        raise FileNotFoundError(
            f"Could not resolve {label}. got={user_value!r} env[{env_key}]={os.environ.get(env_key, '')!r}"
        )
    print(f"[VPP_PATH] {label} -> {chosen}")
    return chosen


def _default_ckpt_candidates(*relative_paths: str) -> list[str]:
    root = _repo_root()
    candidates = []
    for rel in relative_paths:
        candidates.append(str(root / rel))
    return candidates


def _ensure_gc_sampling_imported():
    global _GC_SAMPLING_IMPORTED, _GC_SAMPLING_IMPORT_ERROR
    if _GC_SAMPLING_IMPORTED:
        return
    try:
        from policy_models.edm_diffusion.gc_sampling import (
            sample_lms,
            sample_heun,
            sample_euler,
            sample_dpm_2_ancestral,
            sample_euler_ancestral,
            sample_dpm_2,
            sample_dpm_adaptive,
            sample_dpm_fast,
            sample_dpmpp_2s_ancestral,
            sample_dpmpp_2m,
            sample_dpmpp_sde,
            sample_ddim,
            sample_dpmpp_2s,
            sample_dpmpp_2_with_lms,
            get_sigmas_karras,
            get_sigmas_exponential,
            get_sigmas_vp,
            get_sigmas_linear,
            cosine_beta_schedule,
            get_sigmas_ve,
            get_iddpm_sigmas,
            utils,
            math,
        )
        globals().update(locals())
        _GC_SAMPLING_IMPORTED = True
    except Exception as exc:
        _GC_SAMPLING_IMPORT_ERROR = exc
        raise


def load_primary_models(pretrained_model_path, eval=False, device=None):
    from diffusers import StableVideoDiffusionPipeline

    use_fp16 = eval
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
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.video_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


class VPP_Policy(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            use_text_not_embedding: bool = False,
            seed: int = 42,
            pretrained_model_path: str = '/ckpt/svd/checkpoint-100000',
            text_encoder_path: str = '/ckpt/clip-vit-base-patch32',
            t5_model_path: str = '/ckpt/t5_base',
            language_goal_path: str = '/ckpt/ViT-B-32.pt',
            use_position_encoding: bool = True,
            Former_depth: int = 3,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 1,
            num_latents: int = 3,
            use_Former: str = '3d',
            timestep: int = 20,
            max_length: int = 20,
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 7,
            action_seq_len: int = 10,
            action_head_type: str = "diffusion",
            n_trans_bins: int = 21,
            n_rot_bins: int = 21,
            first_steps_weight: float = 1.0,
            first_steps_count: int = 0,
            action_dim_weights: Optional[list[float]] = None,
            **unused_kwargs,
    ):
        super(VPP_Policy, self).__init__()
        init_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer
        self.use_position_encoding = use_position_encoding
        self.t5_model_path = t5_model_path
        self.language_goal_path = language_goal_path
        self.act_window_size = act_window_size
        self.action_dim = action_dim
        self.action_head_type = action_head_type
        self.n_trans_bins = n_trans_bins
        self.n_rot_bins = n_rot_bins
        self.first_steps_weight = first_steps_weight
        self.first_steps_count = first_steps_count
        self.action_dim_weights = action_dim_weights or [1.0] * action_dim
        self.timestep = timestep  # 20, 正确的改去噪步骤输出视频从这里改
        self.extract_layer_idx = extract_layer_idx  # 1
        self.use_Former = use_Former  # '3d'
        self.Former_num_time_embeds = Former_num_time_embeds  # 16
        self.max_length = max_length  # 20

        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim

        self.use_original_diffusion_policy = False
        
        if use_Former=='3d':  # True
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

        print('use_Former:', self.use_Former)
        print('use_all_layer',self.use_all_layer)

        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler

        # Whether to fuse the wrist/gripper camera view. Default keeps the
        # original multi-view behavior; set DEFI_USE_GRIPPER_VIEW=0 for a
        # third-person-only ablation.
        self.use_gripper = os.environ.get("DEFI_USE_GRIPPER_VIEW", "1").lower() not in {
            "0",
            "false",
            "no",
        }

        self.use_univla = True 

        pretrained_model_path = _resolve_model_path(
            pretrained_model_path,
            env_key="DEFI_VIDEO_MODEL_PATH",
            fallback_candidates=_default_ckpt_candidates("ckpts/_hf_defi/step1_gfdm"),
            label="pretrained_model_path",
        )
        text_encoder_path = _resolve_model_path(
            text_encoder_path,
            env_key="DEFI_CLIP_MODEL_PATH",
            fallback_candidates=_default_ckpt_candidates("ckpts/openai_clip_vit_base_patch32"),
            label="text_encoder_path",
        )
        t5_model_path = _resolve_model_path(
            t5_model_path,
            env_key="DEFI_T5_MODEL_PATH",
            fallback_candidates=_default_ckpt_candidates("ckpts/t5_base"),
            label="t5_model_path",
        )
        self.t5_model_path = t5_model_path
        self.language_goal_path = _resolve_model_path(
            self.language_goal_path,
            env_key="DEFI_LANGUAGE_GOAL_PATH",
            fallback_candidates=_default_ckpt_candidates("ckpts/ViT-B-32.pt"),
            label="language_goal_path",
        )

        # goal encoders
        # self.language_goal = LangClip(model_name='ViT-B/32').to(self.device)
        self.language_goal = LangClip(
            model_name=self.language_goal_path).to(
                init_device)

        pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(
            pretrained_model_path , eval = True, device=init_device)

        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            text_encoder_path,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_path,
            use_fast=False,
            local_files_only=True,
        )

        text_encoder = text_encoder.to(init_device).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.parameters():
            param.requires_grad = False

        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.unet.parameters():
            param.requires_grad = False

        pipeline = pipeline.to(init_device)
        pipeline.unet.eval()

        from policy_models.module.diffusion_extract import Diffusion_feature_extractor

        self.TVP_encoder = Diffusion_feature_extractor(pipeline=pipeline,
                                                        tokenizer=tokenizer,
                                                        text_encoder=text_encoder,
                                                        position_encoding = self.use_position_encoding)
        self.TVP_encoder = self.TVP_encoder.to(init_device)

        if not self.use_original_diffusion_policy and self.use_univla:
            self.goal_emb = nn.Sequential(
                nn.Linear(condition_dim, 768),
                nn.GELU(),
                nn.Linear(768, 768)
            )
            self.goal_emb = self.goal_emb.to(init_device)
            self.time_pos_emb = nn.Parameter(torch.randn(2, 1, 768))

        # policy network
        if self.use_univla:
            from policy_models.m_former_univla.latent_motion_tokenizer_univla import UncontrolledDINOLatentActionModel

            self.model = GCDenoiser(action_dim = action_dim,
                                    obs_dim=latent_dim,
                                    goal_dim=512,
                                    num_tokens=num_latents,
                                    goal_window_size = 1,
                                    obs_seq_len = obs_seq_len,
                                    act_seq_len = action_seq_len,
                                    device=init_device,
                                    use_original_diffusion_policy=False,
                                    sigma_data=0.5,
                                    action_dim_weights=self.action_dim_weights,
                                    first_steps_weight=self.first_steps_weight,
                                    first_steps_count=self.first_steps_count).to(init_device)
            self.lam = UncontrolledDINOLatentActionModel(t5_model_path=t5_model_path)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()

        # diffusion stuff
        self.sampler_type = sampler_type  # 'ddim'
        self.num_sampling_steps = num_sampling_steps  # 10
        self.noise_scheduler = noise_scheduler  # 'exponential'
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type

        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep  # 10
        self.latent_goal = None
        self.plan = None
        self.use_text_not_embedding = use_text_not_embedding  # True
        self.action_intent_dim = 16 * action_dim
        self.action_intent_proj = nn.Sequential(
            nn.LayerNorm(self.action_intent_dim),
            nn.Linear(self.action_intent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        ).to(init_device)
        self.override_action_intent = None

        # for clip loss ground truth plot
        self.ema_callback_idx = None

        for param in self.model.inner_model.proprio_emb.parameters():
            param.requires_grad = False
        for param in self.model.inner_model.goal_emb.parameters():
            param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def process_device(self):
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
        self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''
        if self.use_univla:  # True
            optim_groups = [
                {"params": self.model.inner_model.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.lam.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.Video_Former.parameters(), 
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.goal_emb.parameters(), 
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": [self.time_pos_emb], 
                "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.action_intent_proj.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay},
            ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # 如果不用 scheduler，那 AdamW 就直接用这个 lr。
        # 一旦用了 TriStageLRScheduler，实际 step 中的 lr 都来自 scheduler。
        
        # Optionally initialize the scheduler
        if self.use_lr_scheduler:  # True
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def training_step(self, dataset_batch: Dict[str, Dict],) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        total_loss, action_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)
        # predictive_feature['state_images'] torch.Size([28, 224, 384])
        # latent_goal torch.Size([28, 1, 512])

        if self.use_univla:
            univla_out = self.lam(
                predictive_feature['state_images'],
                dataset_batch['lang_text'],
            )
            latent_motion_tokens_up = univla_out['video_action_patches'].squeeze(1)
            act_loss, sigmas, noise = self.diffusion_loss(
                predictive_feature,
                latent_goal,
                dataset_batch["actions"],
                latent_motion_tokens_up,
            )

        action_loss += act_loss
        total_loss += act_loss

        total_bs = dataset_batch["actions"].shape[0]

        if self.use_univla:
            q_loss_u = ((univla_out["emb"].detach() - univla_out["z"]) ** 2).mean()
            commit_loss_u = ((univla_out["emb"] - univla_out["z"].detach()) ** 2).mean()

            total_loss = total_loss + q_loss_u + 0.25*commit_loss_u

        self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    def extract_predictive_feature(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'].to(self.device)  # torch.Size([28, 1, 3, 256, 256])
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'].to(self.device)  # torch.Size([28, 1, 3, 256, 256])
        if os.environ.get("DEFI_THIRD_ONLY_DUP_STATIC", "0").lower() in {"1", "true", "yes"}:
            rgb_gripper = rgb_static
        modality = "lang"
        if self.use_text_not_embedding:  # True
            latent_goal = self.language_goal(dataset_batch["lang_text"]).to(
                device=self.device,
                dtype=rgb_static.dtype,
            )  # torch.Size([28, 1, 512])
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(
                device=self.device,
                dtype=rgb_static.dtype,
            )
        latent_goal = self._condition_latent_goal_with_action_intent(latent_goal, dataset_batch)

        language = dataset_batch["lang_text"]

        num_frames = self.Former_num_time_embeds  # 16
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]  # 28

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
            language = language + language
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)  # torch.Size([56, 16, 2560, 16, 16])

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')  # torch.Size([56, 16, 2560, 256])
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')  # torch.Size([56, 16, 256, 2560])
        perceptual_features = perceptual_features[:, :num_frames, :, :]  # torch.Size([56, 16, 256, 2560])

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        
        #### use gripper (above) or donotuse gripper (below) ###
        if self.use_gripper:
            perceptual_features_for_videoformer = torch.cat([perceptual_features, gripper_feature], dim=2)
        else:
            perceptual_features_for_videoformer = perceptual_features

        perceptual_features_for_videoformer = perceptual_features_for_videoformer.to(torch.float32)

        perceptual_features = perceptual_features.to(torch.float32)  # torch.Size([28, 16, 512, 2560])

        # TODO 在这儿改frame, train过程
        frame_0 = perceptual_features[:, 0]      # shape: [28, 256, 2560]
        frame_5 = perceptual_features[:, 0+5]      # shape: [28, 256, 2560]
        perceptual_features = torch.stack([frame_0, frame_5], dim=1)  # shape [28, 2, 256, 2560]
        perceptual_features = self.goal_emb(perceptual_features)  # torch.Size([28, 2, 256, 1024])
        time_pos_emb = (
            self.time_pos_emb.unsqueeze(0).expand(perceptual_features.size(0), -1, -1, -1)
        )  # torch.Size([28, 2, 1, 384])
        perceptual_features = perceptual_features + time_pos_emb  # torch.Size([28, 2, 512, 384])

        perceptual_features_for_videoformer = self.Video_Former(perceptual_features_for_videoformer)  # torch.Size([28, 224(16*14), 384]) torch.Size([28, 28, 384])
        if self.use_Former=='linear':  # False
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')
        predictive_feature = {
            'state_images': perceptual_features,
            'state_images_for_videoformer': perceptual_features_for_videoformer,
        }
        predictive_feature['modality'] = modality
        return predictive_feature, latent_goal

    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def diffusion_loss(
            self,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,
            latent_motion_tokens_up: torch.Tensor = None,
            action_intent_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        _ensure_gc_sampling_imported()
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)  # torch.Size([28, 10, 7])
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas,
                                  latent_motion_tokens_up,
                                  action_intent_cond=action_intent_cond)
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            latent_motion_tokens_up: torch.Tensor = None,
            action_intent_cond: torch.Tensor = None,
            inference: Optional[bool] = False,
            extra_args={},
            initial_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps  # 10
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # torch.Size([28, 1, 512])
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)

        # 随机初始化action
        if initial_x is None:
            x = torch.randn((len(latent_goal), self.act_window_size, self.action_dim), device=self.device) * self.sigma_max
        else:
            x = initial_x.to(device=self.device, dtype=torch.float32)
        # torch.Size([28, 10, 7])
        self._last_denoise_initial_x = x.detach().clone()

        if action_intent_cond is not None:
            extra_args = dict(extra_args)
            extra_args["action_intent_cond"] = action_intent_cond
        actions = self.sample_loop(sigmas, latent_motion_tokens_up,
                                   x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions  # torch.Size([28, 10, 7])

    @torch.no_grad()
    def score_action_chunk(
            self,
            action_chunk: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            latent_motion_tokens_up: torch.Tensor = None,
            action_intent_cond: torch.Tensor = None,
            sigma: Optional[torch.Tensor] = None,
            return_velocity: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Score a provided action chunk under the current diffusion policy condition.

        This is the low-level interface needed by suffix transport: callers can
        evaluate the same A_remain under two different hypotheses and compare
        their diffusion velocities without resampling a fresh chunk.
        """
        _ensure_gc_sampling_imported()
        self.model.eval()

        if not torch.is_tensor(action_chunk):
            action_chunk = torch.as_tensor(action_chunk, dtype=torch.float32, device=self.device)
        else:
            action_chunk = action_chunk.to(device=self.device, dtype=torch.float32)
        if action_chunk.dim() == 2:
            action_chunk = action_chunk.unsqueeze(0)
        if action_chunk.dim() != 3:
            raise ValueError(f"expected action_chunk with shape [B,T,A] or [T,A], got {tuple(action_chunk.shape)}")

        batch = int(action_chunk.shape[0])
        if len(latent_goal.shape) < len(
                perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)
        latent_goal = latent_goal.to(device=self.device, dtype=action_chunk.dtype)

        if latent_motion_tokens_up is not None:
            latent_motion_tokens_up = latent_motion_tokens_up.to(device=self.device, dtype=action_chunk.dtype)
        if action_intent_cond is not None:
            action_intent_cond = action_intent_cond.to(device=self.device, dtype=action_chunk.dtype)

        if sigma is None:
            sigma = torch.full((batch,), float(self.sigma_data), device=self.device, dtype=action_chunk.dtype)
        elif not torch.is_tensor(sigma):
            sigma = torch.full((batch,), float(sigma), device=self.device, dtype=action_chunk.dtype)
        else:
            sigma = sigma.to(device=self.device, dtype=action_chunk.dtype).reshape(-1)
            if sigma.numel() == 1:
                sigma = sigma.expand(batch)
        if sigma.numel() != batch:
            raise ValueError(f"expected sigma batch {batch}, got {sigma.numel()}")
        sigma = sigma.clamp_min(1e-6)

        extra_args = {}
        if action_intent_cond is not None:
            extra_args["action_intent_cond"] = action_intent_cond

        denoised = self.model(
            latent_motion_tokens_up,
            perceptual_emb,
            action_chunk,
            latent_goal,
            sigma,
            **extra_args,
        )
        output = {
            "denoised": denoised,
            "score": denoised - action_chunk,
            "sigma": sigma,
        }
        if return_velocity:
            output["velocity"] = to_d(action_chunk, sigma, denoised)
        return output

    @torch.no_grad()
    def velocity_action_chunk(self, *args, **kwargs) -> torch.Tensor:
        """Return only the Karras ODE velocity for a provided action chunk."""
        return self.score_action_chunk(*args, **kwargs)["velocity"]

    @torch.no_grad()
    def counterfactual_action_velocity_delta(
            self,
            action_chunk: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal_expected: torch.Tensor,
            latent_goal_posterior: torch.Tensor,
            latent_motion_tokens_up: torch.Tensor = None,
            action_intent_expected: torch.Tensor = None,
            action_intent_posterior: torch.Tensor = None,
            sigma: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compare policy velocities for the same suffix under expected/posterior hypotheses.

        delta_velocity = v_theta(A_remain | h_post) - v_theta(A_remain | h_exp)
        is the counterfactual transport direction used for minimal suffix edits.
        """
        expected = self.score_action_chunk(
            action_chunk,
            perceptual_emb,
            latent_goal_expected,
            latent_motion_tokens_up=latent_motion_tokens_up,
            action_intent_cond=action_intent_expected,
            sigma=sigma,
        )
        posterior = self.score_action_chunk(
            action_chunk,
            perceptual_emb,
            latent_goal_posterior,
            latent_motion_tokens_up=latent_motion_tokens_up,
            action_intent_cond=action_intent_posterior,
            sigma=expected["sigma"],
        )
        return {
            "velocity_expected": expected["velocity"],
            "velocity_posterior": posterior["velocity"],
            "delta_velocity": posterior["velocity"] - expected["velocity"],
            "denoised_expected": expected["denoised"],
            "denoised_posterior": posterior["denoised"],
            "sigma": expected["sigma"],
        }

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        _ensure_gc_sampling_imported()
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':  # True
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            latent_motion_tokens_up,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        _ensure_gc_sampling_imported()
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys if x in extra_args}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None

        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':  # True
            x_0 = sample_ddim(self.model, latent_motion_tokens_up,
                              state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0  # torch.Size([28, 10, 7])

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        _ensure_gc_sampling_imported()
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)
        elif noise_schedule_type == 'exponential':  # True
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.pred_action_seq = None
        self.rollout_step_counter = 0
        self.override_action_intent = None
        self._last_suffix_transport_context = None
        self._last_action_intent_sensitivity = None
        self._last_denoise_initial_x = None

    def patch_remaining_action_seq(self, candidate_seq, mode: str = "replace", mix: float = 1.0):
        """
        Patch the not-yet-executed suffix of the cached action chunk in place.

        This is used by online correction logic when we want the next executor
        step to consume an updated action sequence immediately, instead of
        waiting for the next full chunk re-sampling boundary.
        """
        if self.pred_action_seq is None or candidate_seq is None:
            return {
                "patched": False,
                "reason": "missing_cached_or_candidate_seq",
            }

        if not torch.is_tensor(candidate_seq):
            candidate_seq = torch.as_tensor(
                candidate_seq,
                dtype=self.pred_action_seq.dtype,
                device=self.pred_action_seq.device,
            )
        else:
            candidate_seq = candidate_seq.to(
                device=self.pred_action_seq.device,
                dtype=self.pred_action_seq.dtype,
            )

        if candidate_seq.dim() == 2:
            candidate_seq = candidate_seq.unsqueeze(0)
        if candidate_seq.dim() != 3:
            return {
                "patched": False,
                "reason": f"bad_candidate_dim_{candidate_seq.dim()}",
            }

        batch = min(int(self.pred_action_seq.shape[0]), int(candidate_seq.shape[0]))
        start_idx = int(self.rollout_step_counter)
        total_len = int(self.pred_action_seq.shape[1])
        remaining = max(0, total_len - start_idx)
        if remaining <= 0:
            return {
                "patched": False,
                "reason": "no_remaining_actions",
            }

        overlap = min(remaining, int(candidate_seq.shape[1]))
        if overlap <= 0:
            return {
                "patched": False,
                "reason": "zero_overlap",
            }

        current_tail = self.pred_action_seq[:batch, start_idx : start_idx + overlap]
        candidate_tail = candidate_seq[:batch, :overlap]
        mode = str(mode or "replace").strip().lower()
        mix = float(mix)

        if mode == "delta":
            patched_tail = current_tail + mix * candidate_tail
        elif mode == "mix":
            patched_tail = (1.0 - mix) * current_tail + mix * candidate_tail
        else:
            patched_tail = candidate_tail

        self.pred_action_seq[:batch, start_idx : start_idx + overlap] = patched_tail
        return {
            "patched": True,
            "mode": mode,
            "mix": mix,
            "start_idx": start_idx,
            "overlap": overlap,
            "remaining_before_patch": remaining,
            "candidate_len": int(candidate_seq.shape[1]),
        }

    def get_remaining_action_seq(self, pad_to_full: bool = True):
        if self.pred_action_seq is None:
            return None
        start_idx = int(self.rollout_step_counter)
        total_len = int(self.pred_action_seq.shape[1])
        if start_idx >= total_len:
            return None
        remain = self.pred_action_seq[:, start_idx:].detach().clone()
        if not pad_to_full or remain.shape[1] == total_len:
            return remain
        padded = torch.zeros_like(self.pred_action_seq)
        padded[:, : remain.shape[1]] = remain
        return padded

    def request_full_replan_next_step(self):
        """Drop the current action buffer so the next executor step samples a fresh chunk."""
        self.pred_action_seq = None
        self.rollout_step_counter = 0
        self._last_action_intent_sensitivity = None
        self._last_suffix_transport_context = None
        return {"replan_requested": True}

    def _condition_latent_goal_with_action_intent(self, latent_goal, goal):
        action_intent = self._resolve_action_intent(goal, latent_goal.device, latent_goal.dtype)
        if action_intent is None:
            return latent_goal
        if action_intent.dim() == 2:
            action_intent = action_intent.unsqueeze(0)
        action_intent = action_intent.reshape(action_intent.shape[0], -1)
        if action_intent.is_floating_point() and not torch.isfinite(action_intent).all():
            return latent_goal
        if action_intent.shape[-1] != self.action_intent_dim:
            raise ValueError(f"expected flattened action_intent dim {self.action_intent_dim}, got {action_intent.shape[-1]}")
        action_intent = self.action_intent_proj(action_intent).unsqueeze(1)
        if action_intent.shape[0] == 1 and latent_goal.shape[0] != 1:
            action_intent = action_intent.expand(latent_goal.shape[0], -1, -1)
        elif action_intent.shape[0] != latent_goal.shape[0]:
            raise ValueError(f"expected action_intent batch {latent_goal.shape[0]}, got {action_intent.shape[0]}")
        return latent_goal + action_intent

    def _resolve_action_intent(self, goal, device, dtype):
        action_intent = None
        if isinstance(goal, dict):
            action_intent = goal.get("action_intent")
        if action_intent is None:
            action_intent = getattr(self, "override_action_intent", None)
        if action_intent is None:
            return None
        if not torch.is_tensor(action_intent):
            action_intent = torch.as_tensor(action_intent, dtype=dtype, device=device)
        else:
            action_intent = action_intent.to(device=device, dtype=dtype)
        return action_intent

    def _project_action_intent_condition(self, goal, device, dtype):
        action_intent = self._resolve_action_intent(goal, device, dtype)
        if action_intent is None:
            return None
        if action_intent.dim() == 2:
            action_intent = action_intent.unsqueeze(0)
        action_intent = action_intent.reshape(action_intent.shape[0], -1)
        if action_intent.is_floating_point() and not torch.isfinite(action_intent).all():
            return None
        if action_intent.shape[-1] != self.action_intent_dim:
            raise ValueError(f"expected flattened action_intent dim {self.action_intent_dim}, got {action_intent.shape[-1]}")
        action_intent = self.action_intent_proj(action_intent).unsqueeze(1)
        return action_intent

    def _make_action_intent_conditions(self, latent_goal_base, action_intent):
        if action_intent is None:
            return latent_goal_base, None
        goal = {"action_intent": action_intent}
        latent_goal = self._condition_latent_goal_with_action_intent(latent_goal_base, goal)
        action_intent_cond = self._project_action_intent_condition(goal, latent_goal.device, latent_goal.dtype)
        return latent_goal, action_intent_cond

    @torch.no_grad()
    def counterfactual_transport_remaining_action_seq(
            self,
            action_intent_expected: torch.Tensor = None,
            action_intent_posterior: torch.Tensor = None,
            sigma: Optional[torch.Tensor] = None,
            edit_steps: int = 1,
            edit_step_size: float = 0.05,
            max_abs_delta: float = 0.05,
    ) -> Dict[str, object]:
        """
        Locally edit the cached suffix using v(A|h_post)-v(A|h_exp).

        This preserves the existing action buffer and applies only a bounded
        transport update to the not-yet-executed suffix.
        """
        if self.pred_action_seq is None:
            return {"applied": False, "reason": "missing_cached_action_seq"}
        ctx = getattr(self, "_last_suffix_transport_context", None)
        if not ctx:
            return {"applied": False, "reason": "missing_denoise_context"}
        start_idx = int(self.rollout_step_counter)
        total_len = int(self.pred_action_seq.shape[1])
        if start_idx >= total_len:
            return {"applied": False, "reason": "no_remaining_actions"}

        action_tail = self.pred_action_seq[:, start_idx:].detach().clone()
        action_remain = torch.zeros_like(self.pred_action_seq)
        action_remain[:, : action_tail.shape[1]] = action_tail
        latent_goal_base = ctx["latent_goal_base"].to(device=self.device, dtype=action_remain.dtype)
        perceptual_emb = {
            key: value.to(device=self.device, dtype=action_remain.dtype) if torch.is_tensor(value) else value
            for key, value in ctx["perceptual_emb"].items()
        }
        latent_motion_tokens_up = ctx.get("latent_motion_tokens_up")
        if latent_motion_tokens_up is not None:
            latent_motion_tokens_up = latent_motion_tokens_up.to(device=self.device, dtype=action_remain.dtype)

        latent_goal_expected, cond_expected = self._make_action_intent_conditions(
            latent_goal_base,
            action_intent_expected,
        )
        latent_goal_posterior, cond_posterior = self._make_action_intent_conditions(
            latent_goal_base,
            action_intent_posterior,
        )
        if torch.equal(latent_goal_expected, latent_goal_posterior) and cond_expected is None and cond_posterior is None:
            return {"applied": False, "reason": "missing_counterfactual_hypothesis_delta"}

        edited = action_remain
        steps = max(1, int(edit_steps))
        velocity_norm = 0.0
        delta_norm = 0.0
        for _ in range(steps):
            delta = self.counterfactual_action_velocity_delta(
                edited,
                perceptual_emb,
                latent_goal_expected,
                latent_goal_posterior,
                latent_motion_tokens_up=latent_motion_tokens_up,
                action_intent_expected=cond_expected,
                action_intent_posterior=cond_posterior,
                sigma=sigma,
            )
            delta_v = delta["delta_velocity"]
            if not torch.isfinite(delta_v).all():
                return {"applied": False, "reason": "non_finite_delta_velocity"}
            velocity_norm = float(torch.norm(delta_v.reshape(-1), p=2).item())
            step_delta = float(edit_step_size) * delta_v
            if max_abs_delta > 0:
                step_delta = step_delta.clamp(min=-float(max_abs_delta), max=float(max_abs_delta))
            edited = edited + step_delta
            delta_norm = float(torch.norm((edited - action_remain).reshape(-1), p=2).item())

        patch_info = self.patch_remaining_action_seq(edited[:, start_idx:, :], mode="replace", mix=1.0)
        return {
            "applied": bool(patch_info.get("patched", False)),
            "reason": patch_info.get("reason", "transport_applied"),
            "transport_source": "counterfactual_velocity",
            "edit_steps": steps,
            "edit_step_size": float(edit_step_size),
            "max_abs_delta": float(max_abs_delta),
            "velocity_delta_norm": velocity_norm,
            "action_delta_norm": delta_norm,
            **{f"patch_{k}": v for k, v in patch_info.items()},
        }

    @torch.no_grad()
    def _cache_action_intent_sensitivity(
            self,
            base_action_seq: torch.Tensor,
            latent_goal_base: torch.Tensor,
            perceptual_emb: Dict[str, torch.Tensor],
            latent_motion_tokens_up: Optional[torch.Tensor],
            action_intent: Optional[torch.Tensor],
            rank: int = 8,
            eps: float = 1e-2,
            seed: int = 0,
    ) -> Dict[str, object]:
        """
        Precompute a low-rank local policy sensitivity dA/dh around the sampled chunk.

        Here h is the practical policy condition available at this level:
        [latent_goal/future-proxy, action-intent]. Runtime then applies
        J_h Delta h by cheap matrix multiplications and patches only the suffix.
        """
        self._last_action_intent_sensitivity = None
        latent_goal_base = latent_goal_base.to(device=self.device, dtype=torch.float32)
        goal_shape = tuple(latent_goal_base.shape)
        goal_flat = latent_goal_base.reshape(latent_goal_base.shape[0], -1)
        if not torch.isfinite(goal_flat).all():
            return {"cached": False, "reason": "non_finite_latent_goal"}

        action_shape = None
        action_flat = None
        if action_intent is not None:
            if not torch.is_tensor(action_intent):
                action_intent = torch.as_tensor(action_intent, dtype=torch.float32, device=self.device)
            else:
                action_intent = action_intent.to(device=self.device, dtype=torch.float32)
            if action_intent.dim() == 2:
                action_intent = action_intent.unsqueeze(0)
            action_shape = tuple(action_intent.shape)
            action_flat = action_intent.reshape(action_intent.shape[0], -1)
            if action_flat.shape[-1] != self.action_intent_dim:
                return {
                    "cached": False,
                    "reason": f"unexpected_action_intent_dim_{action_flat.shape[-1]}",
                }
            if not torch.isfinite(action_flat).all():
                return {"cached": False, "reason": "non_finite_action_intent"}

        h_flat = goal_flat if action_flat is None else torch.cat([goal_flat, action_flat], dim=-1)

        rank = max(1, min(int(rank), int(h_flat.shape[-1])))
        eps = max(float(eps), 1e-6)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        basis = torch.randn(
            (rank, h_flat.shape[-1]),
            generator=generator,
            device=self.device,
            dtype=h_flat.dtype,
        )
        basis, _ = torch.linalg.qr(basis.t(), mode="reduced")
        basis = basis.t().contiguous()

        init_x = getattr(self, "_last_denoise_initial_x", None)
        if init_x is None:
            return {"cached": False, "reason": "missing_initial_noise"}

        responses = []
        for direction in basis:
            goal_dim = int(goal_flat.shape[-1])
            goal_delta = direction[:goal_dim].reshape((1,) + goal_shape[1:])
            latent_goal_perturbed_base = latent_goal_base + eps * goal_delta
            perturbed = None
            if action_flat is not None and action_shape is not None:
                action_delta = direction[goal_dim:].unsqueeze(0)
                perturbed = (action_flat + eps * action_delta).reshape(action_shape)
            latent_goal_perturbed, cond_perturbed = self._make_action_intent_conditions(
                latent_goal_perturbed_base,
                perturbed,
            )
            perturbed_actions = self.denoise_actions(
                torch.zeros_like(latent_goal_perturbed).to(latent_goal_perturbed.device),
                perceptual_emb,
                latent_goal_perturbed,
                latent_motion_tokens_up,
                action_intent_cond=cond_perturbed,
                inference=True,
                initial_x=init_x,
            )
            responses.append(((perturbed_actions - base_action_seq) / eps).detach())

        j_basis = torch.stack(responses, dim=0)
        if not torch.isfinite(j_basis).all():
            self._last_action_intent_sensitivity = None
            return {"cached": False, "reason": "non_finite_sensitivity"}
        self._last_action_intent_sensitivity = {
            "basis": basis.detach(),
            "j_basis": j_basis.detach(),
            "goal_shape": goal_shape,
            "goal_dim": int(goal_flat.shape[-1]),
            "action_shape": action_shape,
            "action_dim": int(action_flat.shape[-1]) if action_flat is not None else 0,
            "h_flat": h_flat.detach(),
            "rank": rank,
            "eps": eps,
        }
        return {"cached": True, "rank": rank, "eps": eps, "h_dim": int(h_flat.shape[-1])}

    @torch.no_grad()
    def sensitivity_transport_remaining_action_seq(
            self,
            action_intent_expected: torch.Tensor,
            action_intent_posterior: torch.Tensor,
            latent_goal_expected: torch.Tensor = None,
            latent_goal_posterior: torch.Tensor = None,
            edit_step_size: float = 1.0,
            max_abs_delta: float = 0.03,
            min_step_score: float = 0.0,
            selective_step_threshold: float = 0.0,
            max_patch_fraction: float = 1.0,
            replan_fraction: float = 1.01,
            replan_delta_h_norm: float = 0.0,
            smooth_weight: float = 0.0,
            continuity_weight: float = 0.0,
            mpc_iters: int = 1,
    ) -> Dict[str, object]:
        """
        Patch the remaining suffix by the cached low-rank approximation J_h Delta h.

        Per-step scores s_k=||J_k Delta h|| form a temporal attribution mask.
        Only affected suffix actions are edited; unaffected actions remain fixed.
        """
        cache = getattr(self, "_last_action_intent_sensitivity", None)
        if self.pred_action_seq is None:
            return {"applied": False, "reason": "missing_cached_action_seq"}
        if not cache:
            return {"applied": False, "reason": "missing_sensitivity_cache"}

        goal_dim = int(cache.get("goal_dim", 0))
        action_dim = int(cache.get("action_dim", 0))

        def _flat_delta_or_pooled(before, after, dim: int):
            before_t = torch.as_tensor(before, device=self.device, dtype=torch.float32).reshape(1, -1)
            after_t = torch.as_tensor(after, device=self.device, dtype=torch.float32).reshape(1, -1)
            delta = after_t - before_t
            if delta.shape[-1] == dim:
                return delta, "native"
            if dim <= 0:
                return torch.zeros((1, 0), device=self.device, dtype=torch.float32), "empty"
            pooled = F.adaptive_avg_pool1d(delta.unsqueeze(1), dim).squeeze(1)
            return pooled, f"pooled_{delta.shape[-1]}_to_{dim}"

        delta_parts = []
        future_delta_available = False
        future_delta_mode = "missing"
        if latent_goal_expected is not None and latent_goal_posterior is not None and goal_dim > 0:
            goal_delta, future_delta_mode = _flat_delta_or_pooled(latent_goal_expected, latent_goal_posterior, goal_dim)
            delta_parts.append(goal_delta)
            future_delta_available = True
        elif goal_dim > 0:
            delta_parts.append(torch.zeros((1, goal_dim), device=self.device, dtype=torch.float32))

        action_delta_available = False
        action_delta_mode = "missing"
        if action_intent_expected is not None and action_intent_posterior is not None and action_dim > 0:
            action_delta, action_delta_mode = _flat_delta_or_pooled(action_intent_expected, action_intent_posterior, action_dim)
            if action_delta.shape[-1] != action_dim:
                return {"applied": False, "reason": "unexpected_action_intent_delta_dim"}
            delta_parts.append(action_delta)
            action_delta_available = True
        elif action_dim > 0:
            delta_parts.append(torch.zeros((1, action_dim), device=self.device, dtype=torch.float32))

        if not delta_parts or (not future_delta_available and not action_delta_available):
            return {"applied": False, "reason": "missing_joint_hypothesis_delta"}
        delta_h = torch.cat(delta_parts, dim=-1)
        if not torch.isfinite(delta_h).all():
            return {"applied": False, "reason": "non_finite_joint_hypothesis_delta"}
        delta_h_norm = float(torch.norm(delta_h.reshape(-1), p=2).item())
        if replan_delta_h_norm > 0 and delta_h_norm >= float(replan_delta_h_norm):
            return {
                "applied": False,
                "reason": "joint_hypothesis_shift_too_large_replan_preferred",
                "sensitivity_delta_h_norm": delta_h_norm,
                "sensitivity_replan_delta_h_norm": float(replan_delta_h_norm),
            }

        basis = cache["basis"].to(device=self.device, dtype=torch.float32)
        if delta_h.shape[-1] != basis.shape[-1]:
            return {
                "applied": False,
                "reason": f"joint_hypothesis_dim_mismatch_{delta_h.shape[-1]}_vs_{basis.shape[-1]}",
                "sensitivity_delta_h_norm": delta_h_norm,
            }
        j_basis = cache["j_basis"].to(device=self.device, dtype=self.pred_action_seq.dtype)
        coeff = torch.matmul(delta_h.to(dtype=basis.dtype), basis.t()).reshape(-1)
        delta_action = torch.einsum("r,rbtf->btf", coeff.to(j_basis.dtype), j_basis)
        delta_action = float(edit_step_size) * delta_action
        if max_abs_delta > 0:
            delta_action = delta_action.clamp(min=-float(max_abs_delta), max=float(max_abs_delta))
        if not torch.isfinite(delta_action).all():
            return {
                "applied": False,
                "reason": "non_finite_sensitivity_delta_action",
                "sensitivity_delta_h_norm": delta_h_norm,
            }

        batch = min(int(self.pred_action_seq.shape[0]), int(delta_action.shape[0]))
        start_idx = int(self.rollout_step_counter)
        total_len = int(self.pred_action_seq.shape[1])
        start_idx = max(0, min(start_idx, total_len))
        remaining = max(0, total_len - start_idx)
        if batch <= 0 or remaining <= 0:
            return {
                "applied": False,
                "reason": "no_remaining_actions",
                "sensitivity_delta_h_norm": delta_h_norm,
            }

        suffix_delta = delta_action[:batch, start_idx : start_idx + remaining, :]
        if suffix_delta.numel() == 0:
            return {
                "applied": False,
                "reason": "zero_suffix_delta",
                "sensitivity_delta_h_norm": delta_h_norm,
            }

        step_scores = torch.norm(suffix_delta.reshape(batch, remaining, -1), dim=-1)
        threshold = max(float(min_step_score), float(selective_step_threshold))
        selected = step_scores >= threshold if threshold > 0 else step_scores > 0
        max_steps = remaining
        if max_patch_fraction > 0:
            max_steps = max(1, int(float(remaining) * float(max_patch_fraction) + 0.999))
        max_steps = min(max_steps, remaining)
        if max_steps < remaining:
            topk_scores, topk_idx = torch.topk(step_scores, k=max_steps, dim=1)
            topk_mask = torch.zeros_like(selected)
            topk_mask.scatter_(1, topk_idx, topk_scores > 0)
            selected = selected & topk_mask
        if not bool(selected.any().item()):
            best_idx = torch.argmax(step_scores, dim=1, keepdim=True)
            best_scores = torch.gather(step_scores, 1, best_idx)
            selected = torch.zeros_like(selected)
            selected.scatter_(1, best_idx, best_scores > 0)

        selected_steps = int(selected.sum().item())
        selected_fraction = float(selected_steps) / float(max(1, batch * remaining))
        if replan_fraction > 0 and selected_fraction >= float(replan_fraction):
            return {
                "applied": False,
                "reason": "sensitivity_affects_too_much_replan_preferred",
                "sensitivity_delta_h_norm": delta_h_norm,
                "sensitivity_selected_steps": selected_steps,
                "sensitivity_remaining_steps": int(batch * remaining),
                "sensitivity_selected_fraction": selected_fraction,
            }
        if selected_steps <= 0:
            return {
                "applied": False,
                "reason": "no_sensitive_suffix_steps",
                "sensitivity_delta_h_norm": delta_h_norm,
                "sensitivity_step_score_max": float(step_scores.max().item()) if step_scores.numel() else 0.0,
            }

        patch = suffix_delta * selected.unsqueeze(-1).to(dtype=suffix_delta.dtype)
        if smooth_weight > 0 and remaining > 2:
            prev_patch = patch.clone()
            patch[:, 1:-1, :] = (
                (1.0 - float(smooth_weight)) * patch[:, 1:-1, :]
                + 0.5 * float(smooth_weight) * (prev_patch[:, :-2, :] + prev_patch[:, 2:, :])
            )
        if continuity_weight > 0 and start_idx > 0:
            prev_action = self.pred_action_seq[:batch, start_idx - 1 : start_idx, :]
            current_action = self.pred_action_seq[:batch, start_idx : start_idx + 1, :]
            continuity_delta = (prev_action - current_action) * float(continuity_weight)
            patch[:, :1, :] = patch[:, :1, :] + continuity_delta
        if mpc_iters > 1:
            patch = patch / float(max(1, int(mpc_iters)))

        self.pred_action_seq[:batch, start_idx : start_idx + remaining, :] = (
            self.pred_action_seq[:batch, start_idx : start_idx + remaining, :] + patch
        )
        return {
            "applied": True,
            "reason": "linear_suffix_sensitivity_transport_applied",
            "rank": int(cache.get("rank", 0)),
            "eps": float(cache.get("eps", 0.0)),
            "h_dim": int(basis.shape[-1]),
            "start_idx": start_idx,
            "remaining_steps": remaining,
            "selected_steps": selected_steps,
            "selected_fraction": selected_fraction,
            "future_delta_mode": future_delta_mode,
            "action_delta_mode": action_delta_mode,
            "sensitivity_delta_h_norm": delta_h_norm,
            "sensitivity_delta_action_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
            "sensitivity_suffix_delta_norm": float(torch.norm(suffix_delta.reshape(-1), p=2).item()),
            "sensitivity_patch_norm": float(torch.norm(patch.reshape(-1), p=2).item()),
            "sensitivity_step_score_max": float(step_scores.max().item()) if step_scores.numel() else 0.0,
            "sensitivity_step_score_mean": float(step_scores.mean().item()) if step_scores.numel() else 0.0,
            "sensitivity_threshold": threshold,
            "sensitivity_max_abs_delta": float(max_abs_delta),
            "edit_step_size": float(edit_step_size),
        }

    @torch.no_grad()
    def project_sensitivity_delta_action(
            self,
            action_intent_expected: torch.Tensor,
            action_intent_posterior: torch.Tensor,
            latent_goal_expected: torch.Tensor = None,
            latent_goal_posterior: torch.Tensor = None,
            joint_delta: torch.Tensor = None,
            edit_step_size: float = 1.0,
            max_abs_delta: float = 0.0,
    ) -> Dict[str, object]:
        """Return the cached finite-difference J_h Delta h teacher without editing the action buffer."""
        cache = getattr(self, "_last_action_intent_sensitivity", None)
        if self.pred_action_seq is None:
            return {"available": False, "reason": "missing_cached_action_seq"}
        if not cache:
            return {"available": False, "reason": "missing_sensitivity_cache"}

        goal_dim = int(cache.get("goal_dim", 0))
        action_dim = int(cache.get("action_dim", 0))

        def _flat_delta_or_pooled(before, after, dim: int):
            before_t = torch.as_tensor(before, device=self.device, dtype=torch.float32).reshape(1, -1)
            after_t = torch.as_tensor(after, device=self.device, dtype=torch.float32).reshape(1, -1)
            delta = after_t - before_t
            if delta.shape[-1] == dim:
                return delta, "native"
            if dim <= 0:
                return torch.zeros((1, 0), device=self.device, dtype=torch.float32), "empty"
            pooled = F.adaptive_avg_pool1d(delta.unsqueeze(1), dim).squeeze(1)
            return pooled, f"pooled_{delta.shape[-1]}_to_{dim}"

        basis = cache["basis"].to(device=self.device, dtype=torch.float32)
        future_delta_mode = "missing"
        action_delta_mode = "missing"
        if joint_delta is not None:
            delta_h = torch.as_tensor(joint_delta, device=self.device, dtype=torch.float32).reshape(1, -1)
            if delta_h.shape[-1] != basis.shape[-1]:
                delta_h = F.adaptive_avg_pool1d(delta_h.unsqueeze(1), basis.shape[-1]).squeeze(1)
                future_delta_mode = f"pooled_joint_delta_to_{basis.shape[-1]}"
            else:
                future_delta_mode = "joint_delta_native"
            action_delta_mode = "joint_delta"
        else:
            delta_parts = []
            if latent_goal_expected is not None and latent_goal_posterior is not None and goal_dim > 0:
                goal_delta, future_delta_mode = _flat_delta_or_pooled(latent_goal_expected, latent_goal_posterior, goal_dim)
                delta_parts.append(goal_delta)
            elif goal_dim > 0:
                delta_parts.append(torch.zeros((1, goal_dim), device=self.device, dtype=torch.float32))

            if action_intent_expected is not None and action_intent_posterior is not None and action_dim > 0:
                action_delta, action_delta_mode = _flat_delta_or_pooled(action_intent_expected, action_intent_posterior, action_dim)
                delta_parts.append(action_delta)
            elif action_dim > 0:
                delta_parts.append(torch.zeros((1, action_dim), device=self.device, dtype=torch.float32))

            if not delta_parts:
                return {"available": False, "reason": "missing_joint_hypothesis_delta"}
            delta_h = torch.cat(delta_parts, dim=-1)
        if not torch.isfinite(delta_h).all():
            return {"available": False, "reason": "non_finite_joint_hypothesis_delta"}

        basis = basis.to(dtype=delta_h.dtype)
        if delta_h.shape[-1] != basis.shape[-1]:
            return {
                "available": False,
                "reason": f"joint_hypothesis_dim_mismatch_{delta_h.shape[-1]}_vs_{basis.shape[-1]}",
            }
        j_basis = cache["j_basis"].to(device=self.device, dtype=self.pred_action_seq.dtype)
        coeff = torch.matmul(delta_h, basis.t()).reshape(-1)
        delta_action = torch.einsum("r,rbtf->btf", coeff.to(j_basis.dtype), j_basis)
        delta_action = float(edit_step_size) * delta_action
        if max_abs_delta > 0:
            delta_action = delta_action.clamp(min=-float(max_abs_delta), max=float(max_abs_delta))

        start_idx = int(self.rollout_step_counter)
        total_len = int(delta_action.shape[1])
        start_idx = max(0, min(start_idx, total_len))
        suffix_delta = delta_action[:, start_idx:, :]
        suffix_action = self.pred_action_seq[:, start_idx:, :].detach()
        step_scores = (
            torch.norm(suffix_delta.reshape(suffix_delta.shape[0], -1, self.action_dim), dim=-1)
            if suffix_delta.numel()
            else torch.zeros((delta_action.shape[0], 0), device=self.device, dtype=delta_action.dtype)
        )
        return {
            "available": True,
            "rank": int(cache.get("rank", 0)),
            "eps": float(cache.get("eps", 0.0)),
            "h_dim": int(basis.shape[-1]),
            "start_idx": start_idx,
            "delta_h": delta_h.detach(),
            "delta_action": delta_action.detach(),
            "suffix_delta_action": suffix_delta.detach(),
            "suffix_action": suffix_action.detach(),
            "step_scores": step_scores.detach(),
            "future_delta_mode": future_delta_mode,
            "action_delta_mode": action_delta_mode,
            "delta_h_norm": float(torch.norm(delta_h.reshape(-1), p=2).item()),
            "delta_action_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
            "suffix_delta_norm": float(torch.norm(suffix_delta.reshape(-1), p=2).item()) if suffix_delta.numel() else 0.0,
            "step_score_max": float(step_scores.max().item()) if step_scores.numel() else 0.0,
            "step_score_mean": float(step_scores.mean().item()) if step_scores.numel() else 0.0,
        }

    @torch.no_grad()
    def project_paired_counterfactual_delta_action(
            self,
            action_intent_expected: torch.Tensor,
            action_intent_posterior: torch.Tensor,
            latent_goal_expected: torch.Tensor = None,
            latent_goal_posterior: torch.Tensor = None,
            joint_delta: torch.Tensor = None,
            edit_step_size: float = 1.0,
            max_abs_delta: float = 0.0,
            joint_delta_to_action_intent_scale: float = 1.0,
            condition_ablation: str = "joint",
    ) -> Dict[str, object]:
        """
        Return a paired counterfactual teacher from the frozen policy itself.

        The two action chunks use the same observation, same video tokens and
        same diffusion initial noise. Only the hypothesis condition changes:
        A^- = pi(o, h_expected; xi), A^+ = pi(o, h_posterior; xi).
        The target is Delta A = A^+ - A^-.
        """
        if self.pred_action_seq is None:
            return {"available": False, "reason": "missing_cached_action_seq"}
        ctx = getattr(self, "_last_suffix_transport_context", None)
        if not ctx:
            return {"available": False, "reason": "missing_denoise_context"}
        init_x = getattr(self, "_last_denoise_initial_x", None)
        if init_x is None:
            return {"available": False, "reason": "missing_initial_noise"}

        latent_goal_base = ctx["latent_goal_base"].to(device=self.device, dtype=torch.float32)

        def _latent_goal_or_base(candidate, name: str):
            if candidate is None:
                return latent_goal_base, "base"
            candidate_t = torch.as_tensor(candidate, device=self.device, dtype=torch.float32)
            if not torch.isfinite(candidate_t).all():
                return latent_goal_base, f"{name}_non_finite_to_base"
            if tuple(candidate_t.shape) == tuple(latent_goal_base.shape):
                return candidate_t, "native"
            if candidate_t.numel() == latent_goal_base.numel():
                return candidate_t.reshape_as(latent_goal_base), "reshaped"
            # Runtime h stores richer future summaries than the diffusion decoder
            # condition. Pool them into the decoder latent shape instead of
            # silently dropping z_f from the counterfactual teacher.
            pooled = F.adaptive_avg_pool1d(
                candidate_t.reshape(1, 1, -1),
                int(latent_goal_base.numel()),
            ).reshape_as(latent_goal_base)
            if not torch.isfinite(pooled).all():
                return latent_goal_base, f"{name}_pooled_non_finite_to_base"
            return pooled, f"{name}_pooled_{tuple(candidate_t.shape)}_to_{tuple(latent_goal_base.shape)}"

        condition_ablation = str(condition_ablation or "joint")
        if condition_ablation not in {"joint", "future_only", "action_only"}:
            condition_ablation = "joint"
        if condition_ablation == "action_only":
            latent_goal_posterior = latent_goal_expected
        if condition_ablation == "future_only":
            action_intent_posterior = action_intent_expected

        latent_goal_expected_base, future_expected_mode = _latent_goal_or_base(latent_goal_expected, "expected")
        latent_goal_posterior_base, future_posterior_mode = _latent_goal_or_base(latent_goal_posterior, "posterior")
        action_delta_mode = "paired_action_intent_condition"
        if (
            condition_ablation != "future_only"
            and
            action_intent_expected is not None
            and joint_delta is not None
            and float(joint_delta_to_action_intent_scale) != 0.0
        ):
            expected_action_t = torch.as_tensor(action_intent_expected, device=self.device, dtype=torch.float32)
            posterior_action_t = (
                torch.as_tensor(action_intent_posterior, device=self.device, dtype=torch.float32)
                if action_intent_posterior is not None
                else expected_action_t
            )
            same_action_condition = (
                tuple(expected_action_t.shape) == tuple(posterior_action_t.shape)
                and torch.allclose(expected_action_t, posterior_action_t, atol=1e-7, rtol=1e-5)
            )
            if same_action_condition:
                delta_t = torch.as_tensor(joint_delta, device=self.device, dtype=torch.float32).reshape(1, -1)
                delta_pooled = F.adaptive_avg_pool1d(
                    delta_t.unsqueeze(1),
                    expected_action_t.numel(),
                ).squeeze(1).reshape_as(expected_action_t)
                action_intent_posterior = expected_action_t + float(joint_delta_to_action_intent_scale) * delta_pooled
                action_delta_mode = (
                    f"paired_action_intent_condition_from_joint_delta_scale_"
                    f"{float(joint_delta_to_action_intent_scale):.6g}"
                )
        latent_goal_expected_cond, cond_expected = self._make_action_intent_conditions(
            latent_goal_expected_base,
            action_intent_expected,
        )
        latent_goal_posterior_cond, cond_posterior = self._make_action_intent_conditions(
            latent_goal_posterior_base,
            action_intent_posterior,
        )

        perceptual_emb = {
            key: value.detach().clone().to(device=self.device, dtype=torch.float32)
            for key, value in ctx["perceptual_emb"].items()
            if torch.is_tensor(value)
        }
        latent_motion_tokens_up = ctx.get("latent_motion_tokens_up")
        if latent_motion_tokens_up is not None:
            latent_motion_tokens_up = latent_motion_tokens_up.detach().clone().to(device=self.device, dtype=torch.float32)

        prev_initial_x = getattr(self, "_last_denoise_initial_x", None)
        base_noise = init_x.detach().clone().to(device=self.device, dtype=torch.float32)
        try:
            action_expected = self.denoise_actions(
                torch.zeros_like(latent_goal_expected_cond).to(latent_goal_expected_cond.device),
                perceptual_emb,
                latent_goal_expected_cond,
                latent_motion_tokens_up,
                action_intent_cond=cond_expected,
                inference=True,
                initial_x=base_noise,
            )
            action_posterior = self.denoise_actions(
                torch.zeros_like(latent_goal_posterior_cond).to(latent_goal_posterior_cond.device),
                perceptual_emb,
                latent_goal_posterior_cond,
                latent_motion_tokens_up,
                action_intent_cond=cond_posterior,
                inference=True,
                initial_x=base_noise,
            )
        finally:
            if prev_initial_x is not None:
                self._last_denoise_initial_x = prev_initial_x

        if not torch.isfinite(action_expected).all() or not torch.isfinite(action_posterior).all():
            return {"available": False, "reason": "non_finite_paired_counterfactual_action"}
        delta_action = (action_posterior - action_expected).detach()
        delta_action = float(edit_step_size) * delta_action
        if max_abs_delta > 0:
            delta_action = delta_action.clamp(min=-float(max_abs_delta), max=float(max_abs_delta))

        start_idx = int(self.rollout_step_counter)
        total_len = int(delta_action.shape[1])
        start_idx = max(0, min(start_idx, total_len))
        suffix_delta = delta_action[:, start_idx:, :]
        suffix_action = action_expected[:, start_idx:, :].detach()
        step_scores = (
            torch.norm(suffix_delta.reshape(suffix_delta.shape[0], -1, self.action_dim), dim=-1)
            if suffix_delta.numel()
            else torch.zeros((delta_action.shape[0], 0), device=self.device, dtype=delta_action.dtype)
        )

        delta_h = None
        delta_h_norm = 0.0
        if joint_delta is not None:
            delta_h = torch.as_tensor(joint_delta, device=self.device, dtype=torch.float32).reshape(1, -1)
            delta_h_norm = float(torch.norm(delta_h.reshape(-1), p=2).item()) if torch.isfinite(delta_h).all() else 0.0
        return {
            "available": True,
            "teacher_mode": "paired_counterfactual_policy_response",
            "rank": 0,
            "eps": 0.0,
            "h_dim": int(delta_h.shape[-1]) if delta_h is not None else 0,
            "start_idx": start_idx,
            "delta_h": delta_h.detach() if delta_h is not None else torch.zeros((1, 0), device=self.device),
            "delta_action": delta_action.detach(),
            "suffix_delta_action": suffix_delta.detach(),
            "suffix_action": suffix_action.detach(),
            "step_scores": step_scores.detach(),
            "future_delta_mode": f"{future_expected_mode}_to_{future_posterior_mode}",
            "action_delta_mode": action_delta_mode,
            "condition_ablation": condition_ablation,
            "delta_h_norm": delta_h_norm,
            "delta_action_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
            "suffix_delta_norm": float(torch.norm(suffix_delta.reshape(-1), p=2).item()) if suffix_delta.numel() else 0.0,
            "step_score_max": float(step_scores.max().item()) if step_scores.numel() else 0.0,
            "step_score_mean": float(step_scores.mean().item()) if step_scores.numel() else 0.0,
        }

        basis = cache["basis"].to(device=self.device, dtype=delta_h.dtype)
        if delta_h.shape[-1] != basis.shape[-1]:
            return {
                "applied": False,
                "reason": f"joint_hypothesis_dim_mismatch_{delta_h.shape[-1]}_vs_{basis.shape[-1]}",
            }
        j_basis = cache["j_basis"].to(device=self.device, dtype=self.pred_action_seq.dtype)
        coeff = torch.matmul(delta_h, basis.t()).reshape(-1)
        delta_action = torch.einsum("r,rbtf->btf", coeff.to(j_basis.dtype), j_basis)
        delta_action = float(edit_step_size) * delta_action
        if max_abs_delta > 0:
            delta_action = delta_action.clamp(min=-float(max_abs_delta), max=float(max_abs_delta))

        start_idx = int(self.rollout_step_counter)
        total_len = int(self.pred_action_seq.shape[1])
        if start_idx >= total_len:
            return {"applied": False, "reason": "no_remaining_actions"}
        step_scores = torch.norm(delta_action[:, start_idx:, :].reshape(delta_action.shape[0], -1, self.action_dim), dim=-1)
        if float(step_scores.max().item()) < float(min_step_score):
            return {
                "applied": False,
                "reason": "below_min_step_score",
                "sensitivity_delta_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
                "sensitivity_delta_h_norm": delta_h_norm,
                "sensitivity_step_score_max": float(step_scores.max().item()),
            }

        suffix_len = int(step_scores.shape[1])
        if suffix_len <= 0:
            return {"applied": False, "reason": "no_remaining_actions"}
        threshold = float(selective_step_threshold)
        affected = step_scores > threshold
        if not bool(affected.any().item()):
            top_idx = torch.argmax(step_scores, dim=1, keepdim=True)
            affected = torch.zeros_like(step_scores, dtype=torch.bool).scatter(1, top_idx, True)

        max_fraction = float(max_patch_fraction)
        if 0.0 < max_fraction < 1.0:
            max_count = max(1, int(round(max_fraction * suffix_len)))
            top_scores, top_idx = torch.topk(step_scores, k=min(max_count, suffix_len), dim=1)
            capped = torch.zeros_like(affected)
            capped.scatter_(1, top_idx, True)
            affected = affected & capped
            if not bool(affected.any().item()):
                affected = capped

        affected_fraction = float(affected.float().mean().item())
        if affected_fraction >= float(replan_fraction):
            return {
                "applied": False,
                "reason": "sensitivity_affects_too_much_replan_preferred",
                "sensitivity_delta_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
                "sensitivity_delta_h_norm": delta_h_norm,
                "sensitivity_step_score_max": float(step_scores.max().item()),
                "sensitivity_step_score_mean": float(step_scores.mean().item()),
                "sensitivity_affected_fraction": affected_fraction,
            }

        full_mask = torch.zeros_like(delta_action[..., :1], dtype=torch.bool)
        full_mask[:, start_idx:, :] = affected.unsqueeze(-1)
        selective_delta = torch.where(full_mask, delta_action, torch.zeros_like(delta_action))
        patched = self.pred_action_seq.detach().clone()
        target = patched + selective_delta
        edited = patched.clone()
        iters = max(1, int(mpc_iters))
        for _ in range(iters):
            proposal = target.clone()
            if smooth_weight > 0 and total_len > 2:
                smoothed = proposal.clone()
                smoothed[:, 1:-1, :] = 0.5 * (proposal[:, :-2, :] + proposal[:, 2:, :])
                proposal = proposal + float(smooth_weight) * (smoothed - proposal)
            if continuity_weight > 0 and start_idx > 0:
                proposal[:, start_idx, :] = (
                    (1.0 - float(continuity_weight)) * proposal[:, start_idx, :]
                    + float(continuity_weight) * edited[:, start_idx - 1, :]
                )
            edited = torch.where(full_mask, proposal, self.pred_action_seq.detach())
            if max_abs_delta > 0:
                edited = torch.max(
                    torch.min(edited, self.pred_action_seq.detach() + float(max_abs_delta)),
                    self.pred_action_seq.detach() - float(max_abs_delta),
                )
        patch_info = self.patch_remaining_action_seq(edited[:, start_idx:, :], mode="replace", mix=1.0)
        return {
            "applied": bool(patch_info.get("patched", False)),
            "reason": patch_info.get("reason", "sensitivity_transport_applied"),
            "transport_source": "linear_sensitivity",
            "sensitivity_rank": int(cache.get("rank", 0)),
            "sensitivity_h_dim": int(basis.shape[-1]),
            "sensitivity_future_delta_available": bool(future_delta_available),
            "sensitivity_action_delta_available": bool(action_delta_available),
            "sensitivity_future_delta_mode": future_delta_mode,
            "sensitivity_action_delta_mode": action_delta_mode,
            "sensitivity_delta_norm": float(torch.norm(selective_delta.reshape(-1), p=2).item()),
            "sensitivity_raw_delta_norm": float(torch.norm(delta_action.reshape(-1), p=2).item()),
            "sensitivity_delta_h_norm": delta_h_norm,
            "sensitivity_step_score_max": float(step_scores.max().item()),
            "sensitivity_step_score_mean": float(step_scores.mean().item()),
            "sensitivity_selective_step_threshold": threshold,
            "sensitivity_affected_fraction": affected_fraction,
            "sensitivity_affected_count": int(affected.sum().item()),
            "sensitivity_suffix_len": suffix_len,
            "sensitivity_max_patch_fraction": max_fraction,
            "sensitivity_replan_fraction": float(replan_fraction),
            "sensitivity_replan_delta_h_norm": float(replan_delta_h_norm),
            "sensitivity_mpc_iters": iters,
            "sensitivity_smooth_weight": float(smooth_weight),
            "sensitivity_continuity_weight": float(continuity_weight),
            **{f"patch_{k}": v for k, v in patch_info.items()},
        }

    def forward(self,batch):  # This is used when training the model.
        return self.training_step(batch)

    def eval_forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        debug_nan = os.environ.get("DEFI_DEBUG_NAN", "0") == "1"

        def _debug_tensor(name, value):
            if not debug_nan:
                return
            if isinstance(value, dict):
                for key, tensor in value.items():
                    if torch.is_tensor(tensor):
                        _debug_tensor(f"{name}.{key}", tensor)
                return
            if not torch.is_tensor(value):
                return
            finite = torch.isfinite(value).all().item() if value.is_floating_point() else True
            print(f"[VPP_NAN] {name}: shape={tuple(value.shape)} dtype={value.dtype} finite={finite}", flush=True)
            if value.is_floating_point():
                detached = value.detach()
                finite_values = detached[torch.isfinite(detached)]
                if finite_values.numel() > 0:
                    min_value = float(finite_values.min())
                    max_value = float(finite_values.max())
                else:
                    min_value = float("nan")
                    max_value = float("nan")
                print(
                    f"[VPP_NAN] {name}: minmax={min_value} {max_value}",
                    flush=True,
                )
            if not finite:
                raise FloatingPointError(f"non-finite tensor in VPP eval_forward: {name}")

        def _finite_or_zero(name, value):
            if not torch.is_tensor(value) or not value.is_floating_point():
                return value
            if torch.isfinite(value).all():
                return value
            print(f"[VPP_NAN_GUARD] replacing non-finite tensor: {name}", flush=True)
            return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        lang_text = goal["lang_text"]
        if isinstance(lang_text, tuple):
            lang_text = list(lang_text)

        action_intent_raw = None
        if 'lang_text' in goal:
            if self.use_text_not_embedding:  # true
                latent_goal = self.language_goal(lang_text)  # torch.Size([28, 1, 512])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                    obs["rgb_obs"]['rgb_static'].device)
            latent_goal = _finite_or_zero("latent_goal.raw", latent_goal)
            latent_goal_base = latent_goal.detach().clone()
            latent_goal = self._condition_latent_goal_with_action_intent(latent_goal, goal)
            latent_goal = _finite_or_zero("latent_goal.conditioned", latent_goal)
            action_intent_cond = self._project_action_intent_condition(goal, latent_goal.device, latent_goal.dtype)
            action_intent_cond = _finite_or_zero("action_intent_cond", action_intent_cond)
            action_intent_raw = self._resolve_action_intent(goal, latent_goal.device, latent_goal.dtype)
            action_intent_raw = _finite_or_zero("action_intent_raw", action_intent_raw)
            _debug_tensor("latent_goal", latent_goal)
            _debug_tensor("action_intent_cond", action_intent_cond)

        rgb_static = obs["rgb_obs"]['rgb_static']  # torch.Size([28, 1, 3, 256, 256])
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']  # torch.Size([28, 1, 3, 256, 256])
        if os.environ.get("DEFI_THIRD_ONLY_DUP_STATIC", "0").lower() in {"1", "true", "yes"}:
            rgb_gripper = rgb_static

        language = lang_text

        num_frames = self.Former_num_time_embeds  # Video lenth 16
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]

        # 1. Predictive Visual Representations Learning
        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)  # torch.Size([56, 1, 3, 256, 256])
            if isinstance(language, list):
                language = language + language
            else:
                language = [language, language]
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,  # torch.Size([56, 16, 2560, 16, 16])
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)
            perceptual_features = _finite_or_zero("tvp_perceptual_features", perceptual_features)
            _debug_tensor("tvp_perceptual_features", perceptual_features)
        
        # 这里将两个视角的图像按batch维度拼接在一起, 过TVP, 然后再按channel维度拼接在一起

        # 2. Action Learning
        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')  # torch.Size([56, 16, 2560, 256])
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')  # torch.Size([56, 16, 256, 2560])
        perceptual_features = perceptual_features[:, :num_frames, :, :]  # torch.Size([56, 16, 256, 2560])

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        
        #### use gripper (above) or donotuse gripper (below) ###
        if self.use_gripper:
            perceptual_features_for_videoformer = torch.cat([perceptual_features, gripper_feature], dim=2)
        else:
            perceptual_features_for_videoformer = perceptual_features

        perceptual_features_for_videoformer = perceptual_features_for_videoformer.to(torch.float32)

        perceptual_features = perceptual_features.to(torch.float32)
        perceptual_features_for_videoformer = _finite_or_zero(
            "perceptual_features_for_videoformer",
            perceptual_features_for_videoformer,
        )
        perceptual_features = _finite_or_zero("perceptual_features", perceptual_features)

        # TODO 在这儿改frame, eval过程
        frame_0 = perceptual_features[:, 0]      # shape: [28, 256, 2560]
        frame_5 = perceptual_features[:, 0+5]      # shape: [28, 256, 2560]
        perceptual_features = torch.stack([frame_0, frame_5], dim=1)  # shape [28, 2, 256, 2560]
        perceptual_features = self.goal_emb(perceptual_features)  # torch.Size([28, 2, 256, 1024])
        perceptual_features = _finite_or_zero("goal_emb_features", perceptual_features)
        _debug_tensor("goal_emb_features", perceptual_features)
        time_pos_emb = (
            self.time_pos_emb.unsqueeze(0).expand(perceptual_features.size(0), -1, -1, -1)
        )  # torch.Size([28, 2, 1, 384])
        perceptual_features = perceptual_features + time_pos_emb  # torch.Size([28, 2, 512, 384])
        perceptual_features = _finite_or_zero("goal_emb_plus_time", perceptual_features)
        _debug_tensor("goal_emb_plus_time", perceptual_features)

        # 2.1 Video Former
        perceptual_features_for_videoformer = self.Video_Former(perceptual_features_for_videoformer)  # torch.Size([28, 224, 384])
        perceptual_features_for_videoformer = _finite_or_zero("video_former_features", perceptual_features_for_videoformer)
        _debug_tensor("video_former_features", perceptual_features_for_videoformer)
        if self.use_Former == 'linear':  # False
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')

        perceptual_emb = {
            'state_images': perceptual_features,
            'state_images_for_videoformer': perceptual_features_for_videoformer,
        }  # torch.Size([28, 224, 384])

        perceptual_emb['modality'] = "lang"

        # 2.2 DiT Diffusion Policy
        if self.use_univla:
            univla_out = self.lam(
                perceptual_emb['state_images'],
                lang_text,  # UniVLA stage 1 need to add, while stage 2 not need
            )
            if isinstance(univla_out, dict):
                univla_out = {
                    key: _finite_or_zero(f"lam_output.{key}", tensor)
                    for key, tensor in univla_out.items()
                }
            _debug_tensor("lam_output", univla_out)
            latent_motion_tokens_up = univla_out['video_action_patches'].squeeze(1)
            latent_motion_tokens_up = _finite_or_zero("latent_motion_tokens_up", latent_motion_tokens_up)
            _debug_tensor("latent_motion_tokens_up", latent_motion_tokens_up)
            perceptual_emb['state_images'] = perceptual_emb['state_images'].reshape(
                perceptual_emb['state_images'].size(0), -1, perceptual_emb['state_images'].size(-1)
            )
            act_seq = self.denoise_actions(
                torch.zeros_like(latent_goal).to(latent_goal.device),
                perceptual_emb,
                latent_goal,
                latent_motion_tokens_up,
                action_intent_cond=action_intent_cond,
                inference=True,
            )
            self._last_suffix_transport_context = {
                "perceptual_emb": {
                    key: value.detach().clone()
                    for key, value in perceptual_emb.items()
                    if torch.is_tensor(value)
                },
                "latent_goal_base": latent_goal_base.detach().clone(),
                "latent_motion_tokens_up": latent_motion_tokens_up.detach().clone()
                if torch.is_tensor(latent_motion_tokens_up)
                else None,
            }
            act_seq = _finite_or_zero("act_seq", act_seq)
            if os.environ.get("DEFI_SUFFIX_SENSITIVITY_CACHE", "0") == "1":
                sens_info = self._cache_action_intent_sensitivity(
                    act_seq,
                    latent_goal_base,
                    perceptual_emb,
                    latent_motion_tokens_up,
                    action_intent_raw,
                    rank=int(os.environ.get("DEFI_SUFFIX_SENSITIVITY_RANK", "8")),
                    eps=float(os.environ.get("DEFI_SUFFIX_SENSITIVITY_EPS", "0.01")),
                    seed=int(os.environ.get("DEFI_SUFFIX_SENSITIVITY_SEED", "0")),
                )
                if os.environ.get("DEFI_DEBUG_SUFFIX_SENSITIVITY", "0") == "1":
                    print(f"[VPP_SUFFIX_SENSITIVITY] {sens_info}", flush=True)
            _debug_tensor("act_seq", act_seq)
        return act_seq  # torch.Size([28, 10, 7])

    def step(self, obs, goal):  # This is used when rollouting in inference.
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        # 一次 rollout 10步, 机械臂执行10步之后再 rollout 下一次
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self.eval_forward(obs, goal)  # torch.Size([28, 10, 7])

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]  # torch.Size([7])
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:
        if not self.use_original_diffusion_policy and self.use_univla:
            self.lam.to(dtype=self.dtype)
        self.Video_Former.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        self.TVP_encoder.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)
