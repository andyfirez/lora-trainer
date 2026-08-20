"""Shared SDXL inference sampling helpers."""

from dataclasses import dataclass, field
from typing import Any

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from torch import Tensor

from src.trainer.config import SampleScheduler
from src.trainer.sdxl.prompt_encoding import encode_sdxl_prompt

_SCHEDULER_MAP = {
    SampleScheduler.EULER: EulerDiscreteScheduler,
    SampleScheduler.EULER_A: EulerAncestralDiscreteScheduler,
    SampleScheduler.DDIM: DDIMScheduler,
    SampleScheduler.DPM_PP: DPMSolverMultistepScheduler,
}


def build_inference_scheduler(
    sample_scheduler: SampleScheduler,
    noise_scheduler: DDPMScheduler,
) -> object:
    return _SCHEDULER_MAP[sample_scheduler].from_config(noise_scheduler.config)


@dataclass
class PromptEmbedCache:
    _positive_entries: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    _negative: tuple[str, Tensor, Tensor] | None = None

    def get_positive(
        self,
        *,
        prompt: str,
        tokenizer_1: Any,
        tokenizer_2: Any,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        clip_skip: int,
    ) -> tuple[Tensor, Tensor]:
        cached = self._positive_entries.get(prompt)
        if cached is not None:
            return cached

        value = encode_sdxl_prompt(
            [prompt],
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            device,
            dtype,
            clip_skip,
        )
        self._positive_entries[prompt] = value
        return value

    def get_negative(
        self,
        *,
        negative_prompt: str,
        tokenizer_1: Any,
        tokenizer_2: Any,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        clip_skip: int,
    ) -> tuple[Tensor, Tensor]:
        if self._negative is not None and self._negative[0] == negative_prompt:
            return self._negative[1], self._negative[2]

        negative_prompt_embeds, negative_pooled_prompt_embeds = encode_sdxl_prompt(
            [negative_prompt],
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            device,
            dtype,
            clip_skip,
        )
        self._negative = (negative_prompt, negative_prompt_embeds, negative_pooled_prompt_embeds)
        return negative_prompt_embeds, negative_pooled_prompt_embeds

    def clear(self) -> None:
        self._positive_entries.clear()
        self._negative = None


@dataclass(frozen=True)
class SamplePromptEmbeds:
    prompt_embeds: Tensor
    pooled_prompt_embeds: Tensor
    negative_prompt_embeds: Tensor
    negative_pooled_prompt_embeds: Tensor


def precompute_all_sample_embeds(
    *,
    sample_prompts: list[str],
    negative_prompt: str,
    tokenizer_1: Any,
    tokenizer_2: Any,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    clip_skip: int,
    cache: PromptEmbedCache | None = None,
) -> list[SamplePromptEmbeds]:
    embed_cache = cache if cache is not None else PromptEmbedCache()
    negative_prompt_embeds, negative_pooled_prompt_embeds = embed_cache.get_negative(
        negative_prompt=negative_prompt,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        device=device,
        dtype=dtype,
        clip_skip=clip_skip,
    )
    results: list[SamplePromptEmbeds] = []
    for prompt in sample_prompts:
        prompt_embeds, pooled_prompt_embeds = embed_cache.get_positive(
            prompt=prompt,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            device=device,
            dtype=dtype,
            clip_skip=clip_skip,
        )
        results.append(
            SamplePromptEmbeds(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )
        )
    return results


def prepare_vae_for_decode(vae: torch.nn.Module, *, use_fp32: bool = True) -> bool:
    """Keep VAE in float32 during sampling to avoid per-image upcast in diffusers."""
    needs_fp32 = (
        use_fp32
        and vae.dtype == torch.float16
        and getattr(vae.config, "force_upcast", False)
    )
    if needs_fp32:
        vae.to(dtype=torch.float32)
    return needs_fp32
