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


def encode_sdxl_prompt(
    captions: list[str],
    tokenizer_1: Any,
    tokenizer_2: Any,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    tokens_1 = tokenizer_1(
        captions,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens_2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        enc1_out = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_1 = enc1_out.hidden_states[-2].to(dtype=dtype)
        enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_2 = enc2_out.hidden_states[-2].to(dtype=dtype)
        pooled_prompt_embeds = enc2_out[0].to(dtype=dtype)
    return torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1), pooled_prompt_embeds


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


def prepare_vae_for_decode(vae: torch.nn.Module) -> bool:
    """Keep VAE in float32 during sampling to avoid per-image upcast in diffusers."""
    needs_fp32 = (
        vae.dtype == torch.float16
        and getattr(vae.config, "force_upcast", False)
    )
    if needs_fp32:
        vae.to(dtype=torch.float32)
    return needs_fp32
