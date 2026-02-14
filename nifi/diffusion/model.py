from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from nifi.diffusion.lora import LowRankSpatialAdapter
from nifi.diffusion.schedule import SigmaSchedule


@dataclass
class DiffusionConfig:
    model_name_or_path: str
    num_train_timesteps: int = 1000
    lora_rank: int = 64
    guidance_scale: float = 3.0
    prompt_dropout: float = 0.1
    max_token_length: int = 77
    vae_scaling_factor: float = 0.18215


class FrozenLDMWithNiFiAdapters(nn.Module):
    """
    Frozen latent diffusion backbone + two trainable LoRA-style adapters:
      phi_minus: restoration adapter
      phi_plus: critic adapter
    """

    def __init__(self, cfg: DiffusionConfig, device: torch.device, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.cfg = cfg
        self.device_obj = device
        self.dtype = dtype

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(cfg.model_name_or_path, subfolder="unet")

        for m in [self.text_encoder, self.vae, self.unet]:
            m.requires_grad_(False)
            m.eval()

        text_dim = int(self.text_encoder.config.hidden_size)
        in_ch = int(self.unet.config.in_channels)

        self.phi_minus = LowRankSpatialAdapter(
            in_channels=in_ch,
            text_dim=text_dim,
            rank=cfg.lora_rank,
            max_t=cfg.num_train_timesteps,
        )
        self.phi_plus = LowRankSpatialAdapter(
            in_channels=in_ch,
            text_dim=text_dim,
            rank=cfg.lora_rank,
            max_t=cfg.num_train_timesteps,
        )

        self.schedule = SigmaSchedule(num_train_timesteps=cfg.num_train_timesteps)

        self.to(device=device)
        self.text_encoder.to(device=device, dtype=dtype)
        self.vae.to(device=device, dtype=dtype)
        self.unet.to(device=device, dtype=dtype)
        # Keep trainable adapters in fp32 for stable optimizer/scaler behavior.
        self.phi_minus.to(device=device, dtype=torch.float32)
        self.phi_plus.to(device=device, dtype=torch.float32)
        self.schedule.to(device=device, dtype=dtype)

    @property
    def vae_scale(self) -> float:
        return float(self.cfg.vae_scaling_factor)

    def maybe_dropout_prompts(self, prompts: List[str], force_dropout: bool = False) -> List[str]:
        if force_dropout:
            return ["" for _ in prompts]
        if self.cfg.prompt_dropout <= 0:
            return prompts

        out = []
        for p in prompts:
            if torch.rand(1).item() < self.cfg.prompt_dropout:
                out.append("")
            else:
                out.append(p)
        return out

    @torch.no_grad()
    def encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_token_length,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device_obj) for k, v in toks.items()}
        hidden = self.text_encoder(**toks).last_hidden_state
        return hidden

    def get_cfg_embeddings(self, prompts: List[str], train_mode: bool = True):
        prompts = self.maybe_dropout_prompts(prompts, force_dropout=not train_mode)
        cond = self.encode_prompt(prompts)
        uncond = self.encode_prompt(["" for _ in prompts])
        return cond, uncond

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = self.vae.encode(images.to(dtype=vae_dtype)).latent_dist.sample()
        latents = latents * self.vae_scale
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = (latents / self.vae_scale).to(dtype=vae_dtype)
        imgs = self.vae.decode(latents).sample
        return imgs.clamp(-1.0, 1.0)

    def _unet_eps(self, latents: torch.Tensor, timesteps: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        unet_dtype = next(self.unet.parameters()).dtype
        return self.unet(latents.to(dtype=unet_dtype), timesteps, encoder_hidden_states=enc.to(dtype=unet_dtype)).sample

    def _adapter_delta(
        self,
        adapter: Optional[nn.Module],
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        enc: torch.Tensor,
    ) -> torch.Tensor:
        if adapter is None:
            return torch.zeros_like(latents)
        adapter_dtype = next(adapter.parameters()).dtype
        delta = adapter(latents.to(dtype=adapter_dtype), timesteps, enc.to(dtype=adapter_dtype))
        return delta.to(dtype=latents.dtype)

    def predict_eps(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: List[str],
        adapter_type: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        train_mode: bool = True,
    ) -> torch.Tensor:
        g = float(self.cfg.guidance_scale if guidance_scale is None else guidance_scale)
        cond_emb, uncond_emb = self.get_cfg_embeddings(prompts, train_mode=train_mode)

        eps_uncond = self._unet_eps(latents, timesteps, uncond_emb)
        eps_cond = self._unet_eps(latents, timesteps, cond_emb)

        if adapter_type == "minus":
            adapter = self.phi_minus
        elif adapter_type == "plus":
            adapter = self.phi_plus
        else:
            adapter = None

        delta_u = self._adapter_delta(adapter, latents, timesteps, uncond_emb)
        delta_c = self._adapter_delta(adapter, latents, timesteps, cond_emb)

        eps_u = eps_uncond + delta_u
        eps_c = eps_cond + delta_c
        eps = eps_u + g * (eps_c - eps_u)
        return eps

    def freeze_backbone(self) -> None:
        for m in [self.text_encoder, self.vae, self.unet]:
            m.eval()
            m.requires_grad_(False)

    def adapter_parameters(self, which: str):
        if which == "minus":
            return self.phi_minus.parameters()
        if which == "plus":
            return self.phi_plus.parameters()
        raise ValueError(f"Unknown adapter: {which}")

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.schedule.sigma(timesteps)

    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        return self.schedule.q_sample(x, t, noise=noise)

    @staticmethod
    def score_from_eps(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return SigmaSchedule.score_from_eps(eps, sigma)
