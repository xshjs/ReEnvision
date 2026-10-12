from typing import Dict, List
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from policy_models.m_former_univla.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _debug_tensor(name: str, value: Tensor) -> None:
    if os.environ.get("DEFI_DEBUG_NAN", "0") != "1" or not torch.is_tensor(value):
        return
    finite = torch.isfinite(value).all().item() if value.is_floating_point() else True
    print(f"[LAM_NAN] {name}: shape={tuple(value.shape)} dtype={value.dtype} finite={finite}", flush=True)
    if value.is_floating_point():
        finite_values = value.detach()[torch.isfinite(value.detach())]
        if finite_values.numel() > 0:
            min_value = float(finite_values.min())
            max_value = float(finite_values.max())
        else:
            min_value = float("nan")
            max_value = float("nan")
        print(f"[LAM_NAN] {name}: minmax={min_value} {max_value}", flush=True)
    if not finite:
        raise FloatingPointError(f"non-finite tensor in LAM: {name}")


def _finite_or_zero(name: str, value):
    if not torch.is_tensor(value) or not value.is_floating_point():
        return value
    if torch.isfinite(value).all():
        return value
    print(f"[LAM_NAN_GUARD] replacing non-finite tensor: {name}", flush=True)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class UncontrolledDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int = 3,
            model_dim: int = 768,
            latent_dim: int = 128,
            num_latents: int = 16,
            patch_size: int = 14,
            enc_blocks: int = 12,
            dec_blocks: int = 12,
            num_heads: int = 12,
            dropout: float = 0.0,
            t5_model_path: str = '/ckpt/t5_base',
    ) -> None:
        super(UncontrolledDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        # self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        # self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        # self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        # self.decoder = SpatioTransformer(
        #     in_dim=model_dim,
        #     model_dim=model_dim,
        #     out_dim=dino_dim,        # Dim of DINOv2-Base
        #     num_blocks=dec_blocks,
        #     num_heads=num_heads,
        #     dropout=dropout,
        # )

        # Official DeFI loads T5 directly. Keep compatibility sanitizing opt-in
        # so baseline runs can match the upstream code path.
        self.text_encoder = T5EncoderModel.from_pretrained(t5_model_path)
        if os.environ.get("DEFI_T5_SANITIZE", "0") == "1":
            self._sanitize_text_encoder_parameters()
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)

        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_path)

    def _sanitize_text_encoder_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self.text_encoder.named_parameters():
                if name.endswith("layer_norm.bias"):
                    print(f"[LAM] zeroing T5 layer-norm bias for compatibility: {name}", flush=True)
                    param.data.zero_()
                elif param.is_floating_point() and not torch.isfinite(param).all():
                    print(f"[LAM] replacing non-finite T5 parameter with zeros: {name}", flush=True)
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)

    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']  # torch.Size([3, 14])
        attention_mask = encoding['attention_mask']  # torch.Size([3, 14])

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state  # torch.Size([3, 14, 768])
        last_hidden_states = _finite_or_zero("text_last_hidden_states", last_hidden_states)
        _debug_tensor("text_last_hidden_states", last_hidden_states)

        return last_hidden_states, attention_mask

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        dion_features = videos
        dion_features = _finite_or_zero("vq_encode.input_videos", dion_features)
        _debug_tensor("vq_encode.input_videos", videos)
        # videos = rearrange(videos, "b T c h w -> (b T) c h w")  # torch.Size([6, 3, 224, 224])
        # videos = self.dino_transform(videos)  # torch.Size([6, 3, 224, 224])
        # dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']  # torch.Size([6, 256, 768])
        # dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)  # torch.Size([3, 2, 256, 768])

        action_pad = self.action_latent.expand(B, T, -1, -1)  # torch.Size([3, 2, 4, 768])
        action_pad = _finite_or_zero("vq_encode.action_latent", action_pad)
        _debug_tensor("vq_encode.action_latent", action_pad)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)  # torch.Size([3, 2, 260, 768])
        padded_patches = _finite_or_zero("vq_encode.padded_patches", padded_patches)
        _debug_tensor("vq_encode.padded_patches", padded_patches)

        # Encode, torch.Size([3, 2, 260, 768]), torch.Size([3, 2, 14, 768]), torch.Size([6, 274])
        z = self.encoder(padded_patches, lang_embed, attention_mask)  # torch.Size([3, 2, 274, 768])
        z = _finite_or_zero("vq_encode.encoder_out", z)
        _debug_tensor("vq_encode.encoder_out", z)

        # Get language embedding
        lang_emb = z[:, 1:, self.num_codes + dion_features.shape[2]:]  # torch.Size([3, 1, 14, 768])

        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # torch.Size([3, 1, 4, 128])
        z = _finite_or_zero("vq_encode.to_codebook", z)
        _debug_tensor("vq_encode.to_codebook", z)

        

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)  # torch.Size([3, 4, 128])
        z = _finite_or_zero("vq_encode.pre_vq_z", z)
        _debug_tensor("vq_encode.pre_vq_z", z)
        z_q, z, emb, indices = self.vq(z)  # torch.Size([3, 4, 128]), torch.Size([3, 4, 128]), torch.Size([3, 4, 128]), torch.Size([3, 4])
        z_q = _finite_or_zero("vq_encode.post_vq_z_q", z_q)
        z = _finite_or_zero("vq_encode.post_vq_z", z)
        emb = _finite_or_zero("vq_encode.post_vq_emb", emb)
        _debug_tensor("vq_encode.post_vq_z_q", z_q)
        _debug_tensor("vq_encode.post_vq_z", z)
        _debug_tensor("vq_encode.post_vq_emb", emb)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)  # torch.Size([3, 1, 4, 128])
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
            "lang_emb": lang_emb
        }

    def forward(self, batch, text):
        # Encode + VQ
        B, T, H, W = batch.shape  # batch["videos"] torch.Size([3, 2, 3, 224, 224])
        # H, W = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(text)
        lang_embed = _finite_or_zero("forward.lang_embed_raw", lang_embed)
        _debug_tensor("forward.lang_embed_raw", lang_embed)
        lang_embed = self.lang_proj(lang_embed)  # torch.Size([3, 14, 768])
        lang_embed = _finite_or_zero("forward.lang_embed_projected", lang_embed)
        _debug_tensor("forward.lang_embed_projected", lang_embed)
        attention_mask = torch.cat([torch.ones((B, self.num_codes + (224 // self.patch_size)**2)).to(self.device),
                                    attention_mask],
                                    dim = -1)  # torch.Size([3, 274])

        outputs = self.vq_encode(batch, repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        # video_patches = self.patch_up(outputs["patches"][:, :-1])  # torch.Size([3, 1, 256, 768])
        action_patches = self.action_up(outputs["z_q"])  # torch.Size([3, 1, 4, 768])
        action_patches = _finite_or_zero("forward.action_patches", action_patches)
        video_action_patches = action_patches 
        video_action_patches = _finite_or_zero("forward.video_action_patches", video_action_patches)
        outputs.update(
            {
                "video_action_patches": video_action_patches,  # torch.Size([3, 1, 264, 768])
            }
        )
        # video_action_patches = torch.cat([action_patches, video_patches], dim=2)  # torch.Size([3, 1, 260, 768])

        # Decode
        # video_recon = self.decoder(video_action_patches, outputs['lang_emb'], attention_mask)  # torch.Size([3, 1, 274, 768])
        # video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]]  # torch.Size([3, 1, 256, 768])

        # outputs.update(
        #     {
        #         "recon": video_recon,  # torch.Size([3, 1, 256, 768])
        #         "target": outputs["patches"][:, [-1]]  # torch.Size([3, 1, 256, 768])
        #     }
        # )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device




class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int = 3,
            model_dim: int = 768,
            latent_dim: int = 128,
            num_latents: int = 16,
            patch_size: int = 14,
            enc_blocks: int = 12,
            dec_blocks: int = 12,
            num_heads: int = 12,
            dropout: float = 0.0,
    ) -> None:
        super(ControllableDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        # self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        # self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        # self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        # self.decoder = SpatioTransformer(
        #     in_dim=model_dim,
        #     model_dim=model_dim,
        #     out_dim=dino_dim,        # Dim of DINOv2-Base
        #     num_blocks=dec_blocks,
        #     num_heads=num_heads,
        #     dropout=dropout,
        # )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        # self.vq.requires_grad_(False)
        # self.vq_action.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        # videos = rearrange(videos, "b T c h w -> (b T) c h w")  # torch.Size([6, 3, 224, 224])
        # videos = self.dino_transform(videos)  # torch.Size([6, 3, 224, 224])
        # dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']  # torch.Size([6, 256, 768])
        # dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)  # torch.Size([3, 2, 256, 768])

        dion_features = _finite_or_zero("mv_vq_encode.input_videos", videos)  # torch.Size([3, 2, 256, 768])
        
        action_pad = self.action_latent.expand(B, T, -1, -1)  # torch.Size([3, 2, 4, 768])
        action_pad = _finite_or_zero("mv_vq_encode.action_latent", action_pad)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)  # torch.Size([3, 2, 260, 768])
        padded_patches = _finite_or_zero("mv_vq_encode.padded_patches_uncontrol", padded_patches)

        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)  # torch.Size([3, 2, 4, 768])
        action_pad_controllable = _finite_or_zero("mv_vq_encode.action_latent_controllable", action_pad_controllable)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)  # torch.Size([3, 2, 264, 768])
        padded_patches = _finite_or_zero("mv_vq_encode.padded_patches", padded_patches)

        # Encode
        z = self.encoder(padded_patches)  # torch.Size([3, 2, 264, 768])
        z = _finite_or_zero("mv_vq_encode.encoder_out", z)


        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])  # torch.Size([3, 1, 4, 128])
        z_uncontrol = _finite_or_zero("mv_vq_encode.to_codebook_uncontrol", z_uncontrol)

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)  # torch.Size([3, 4, 128])
        z_uncontrol = _finite_or_zero("mv_vq_encode.pre_vq_uncontrol", z_uncontrol)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)  # torch.Size([3, 4, 128])*3, torch.Size([3, 4])
        z_q_uncontrol = _finite_or_zero("mv_vq_encode.post_vq_z_q_uncontrol", z_q_uncontrol)
        z_uncontrol = _finite_or_zero("mv_vq_encode.post_vq_z_uncontrol", z_uncontrol)
        emb_uncontrol = _finite_or_zero("mv_vq_encode.post_vq_emb_uncontrol", emb_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)  # torch.Size([3, 1, 4, 128])


        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # torch.Size([3, 1, 4, 128])
        z_action = _finite_or_zero("mv_vq_encode.to_codebook_action", z_action)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)  # torch.Size([3, 4, 128])
        z_action = _finite_or_zero("mv_vq_encode.pre_vq_action", z_action)
        z_q, z, emb, indices = self.vq_action(z_action)  # torch.Size([3, 4, 128])*3, torch.Size([3, 4])
        z_q = _finite_or_zero("mv_vq_encode.post_vq_z_q", z_q)
        z = _finite_or_zero("mv_vq_encode.post_vq_z", z)
        emb = _finite_or_zero("mv_vq_encode.post_vq_emb", emb)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)  # torch.Size([3, 1, 4, 128])



        return {
            "patches": dion_features,

            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,

            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, videos) -> Dict:
        # Encode + VQ
        # B, T = batch["videos"].shape[:2]
        # H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(videos) 
        # video_patches = self.patch_up(outputs["patches"][:, :-1])  # torch.Size([3, 1, 256, 768])

        # Decode
        action_patches = _finite_or_zero("mv_forward.action_patches", self.action_up(outputs["z_q"]))
        action_patches_uncontrol = _finite_or_zero(
            "mv_forward.action_patches_uncontrol",
            self.action_up_uncontrol(outputs["z_q_uncontrol"]),
        )
        video_action_patches = torch.cat([
                action_patches, 
                action_patches_uncontrol, 
                # video_patches,
            ],
            dim=2
        )  # torch.Size([3, 1, 264, 768])
        video_action_patches = _finite_or_zero("mv_forward.video_action_patches", video_action_patches)

        outputs.update(
            {
                "video_action_patches": video_action_patches,  # torch.Size([3, 1, 264, 768])
            }
        )

        # video_recon = self.decoder(video_action_patches)  # torch.Size([3, 1, 264, 768])
        # video_recon = video_recon[:, :, -video_patches.shape[2]:]  # torch.Size([3, 1, 256, 768])

        # outputs.update(
        #     {
        #         "recon": video_recon,  # torch.Size([3, 1, 256, 768])
        #         "target": outputs["patches"][:, [-1]]  # torch.Size([3, 1, 256, 768])
        #     }
        # )

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
