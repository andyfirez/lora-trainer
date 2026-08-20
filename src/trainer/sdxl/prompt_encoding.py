"""Shared CLIP hidden-state selection and SDXL prompt encoding."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import Tensor


def select_clip_hidden_state(hidden_states: tuple[Tensor, ...], clip_skip: int) -> Tensor:
    return hidden_states[-clip_skip]


def encode_sdxl_prompt(
    captions: list[str],
    tokenizer_1: Any,
    tokenizer_2: Any,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    clip_skip: int,
    *,
    train_te1: bool = False,
    train_te2: bool = False,
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

    te1_ctx = contextlib.nullcontext() if train_te1 else torch.no_grad()
    te2_ctx = contextlib.nullcontext() if train_te2 else torch.no_grad()

    with te1_ctx:
        enc1_out = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_1 = select_clip_hidden_state(enc1_out.hidden_states, clip_skip).to(dtype=dtype)

    with te2_ctx:
        enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_2 = select_clip_hidden_state(enc2_out.hidden_states, clip_skip).to(dtype=dtype)
        pooled_prompt_embeds = enc2_out[0].to(dtype=dtype)

    return torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1), pooled_prompt_embeds
