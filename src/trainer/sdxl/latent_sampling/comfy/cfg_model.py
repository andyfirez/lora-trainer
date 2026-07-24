"""Diffusers UNet wrapper exposing Comfy k-diffusion model(x, sigma) interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from src.trainer.sdxl.latent_sampling.comfy.model_sampling import EpsModelSampling
from src.trainer.sdxl.sampling import SamplePromptEmbeds


@dataclass(frozen=True)
class DiffusersCfgContext:
    unet: torch.nn.Module
    model_sampling: EpsModelSampling
    embeds: SamplePromptEmbeds
    guidance_scale: float
    add_time_ids: Tensor
    autocast_dtype: torch.dtype
    device: torch.device


class DiffusersCfgModel:
    """Returns denoised latents for a given sigma, with batched CFG."""

    def __init__(self, context: DiffusersCfgContext) -> None:
        self._context = context

    def __call__(self, x: Tensor, sigma: Tensor, **_extra_args: Any) -> Tensor:
        context = self._context
        sigma_scalar = sigma.reshape(-1)[0]
        model_input = context.model_sampling.calculate_input(sigma_scalar, x)
        timestep = context.model_sampling.diffusers_timestep(sigma_scalar)

        combined_prompt_embeds = torch.cat(
            [context.embeds.negative_prompt_embeds, context.embeds.prompt_embeds],
            dim=0,
        ).to(dtype=context.autocast_dtype)
        added_cond_kwargs = {
            "text_embeds": torch.cat(
                [context.embeds.negative_pooled_prompt_embeds, context.embeds.pooled_prompt_embeds],
                dim=0,
            ).to(dtype=context.autocast_dtype),
            "time_ids": torch.cat([context.add_time_ids, context.add_time_ids], dim=0),
        }

        latent_model_input = torch.cat([model_input, model_input], dim=0)
        timestep_batch = timestep.expand(latent_model_input.shape[0])

        with torch.autocast(device_type=context.device.type, dtype=context.autocast_dtype):
            noise_pred = context.unet(
                sample=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=combined_prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        noise_pred_negative, noise_pred_positive = noise_pred.chunk(2)
        noise_pred = noise_pred_negative + context.guidance_scale * (
            noise_pred_positive - noise_pred_negative
        )
        return context.model_sampling.calculate_denoised(sigma_scalar, noise_pred, model_input)
