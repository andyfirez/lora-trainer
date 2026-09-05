"""CFG-aware SDXL UNet wrapper exposing the k-diffusion (x, sigma) → denoised API."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.trainer.sdxl.latent_sampling.kdiffusion import append_dims
from src.trainer.sdxl.latent_sampling.model_sampling import DiscreteEpsSampling
from src.trainer.sdxl.sampling import SamplePromptEmbeds


class SDXLKDiffusionModel:
    def __init__(
        self,
        unet: torch.nn.Module,
        model_sampling: DiscreteEpsSampling,
        embeds: SamplePromptEmbeds,
        add_time_ids: Tensor,
        guidance_scale: float,
        autocast_dtype: torch.dtype,
    ) -> None:
        self.unet = unet
        self.model_sampling = model_sampling
        self.embeds = embeds
        self.add_time_ids = add_time_ids
        self.guidance_scale = guidance_scale
        self.autocast_dtype = autocast_dtype
        self.combined_prompt_embeds = torch.cat(
            [embeds.negative_prompt_embeds, embeds.prompt_embeds],
            dim=0,
        ).to(dtype=autocast_dtype)
        self.added_cond_kwargs = {
            "text_embeds": torch.cat(
                [embeds.negative_pooled_prompt_embeds, embeds.pooled_prompt_embeds],
                dim=0,
            ).to(dtype=autocast_dtype),
            "time_ids": torch.cat([add_time_ids, add_time_ids], dim=0),
        }

    def __call__(self, x: Tensor, sigma: Tensor, **_: Any) -> Tensor:
        sigma = sigma.to(device=x.device, dtype=x.dtype)
        c_in = append_dims(1.0 / (sigma**2 + 1.0).sqrt(), x.ndim)
        scaled = x * c_in
        latent_in = torch.cat([scaled, scaled], dim=0)
        timestep = self.model_sampling.timestep(sigma.reshape(-1)[:1]).to(device=x.device)
        timestep = timestep.expand(latent_in.shape[0])
        noise_pred = self.unet(
            sample=latent_in,
            timestep=timestep,
            encoder_hidden_states=self.combined_prompt_embeds,
            added_cond_kwargs=self.added_cond_kwargs,
            return_dict=False,
        )[0]
        noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
        noise_pred = noise_pred_negative + self.guidance_scale * (
            noise_pred_positive - noise_pred_negative
        )
        return self.model_sampling.calculate_denoised(sigma, noise_pred, x)
