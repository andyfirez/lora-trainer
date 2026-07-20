"""Merged PEFT adapter helpers and shared sampling orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers import DDPMScheduler

from src.trainer.config import TrainConfig, WeightDtype
from src.trainer.sdxl.latent_sampling import SDXLSamplingSession, run_sdxl_sampling_pass
from src.trainer.sdxl.sampling import (
    PromptEmbedCache,
    build_inference_scheduler,
    precompute_all_sample_embeds,
)

_DTYPE_MAP = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}


@dataclass(frozen=True)
class MergedInferenceModels:
    unet: torch.nn.Module
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module


@dataclass
class AdapterMergeState:
    unet_merged: bool
    te1_merged: bool
    te2_merged: bool
    models: MergedInferenceModels


def apply_lora_weight_to_model(model: Any, weight: float) -> None:
    """Scale active LoRA adapters before merge."""
    if abs(weight - 1.0) < 1e-6:
        return
    if hasattr(model, "set_adapters") and hasattr(model, "peft_config"):
        adapter_names = list(getattr(model, "peft_config", {}).keys())
        if adapter_names:
            model.set_adapters(adapter_names, weights=[weight] * len(adapter_names))
            return
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(weight)


def merge_adapters_for_inference(
    *,
    unet: Any,
    text_encoder_1: Any,
    text_encoder_2: Any,
    lora_config: TrainConfig,
    merge_unet: bool = True,
    lora_weight: float = 1.0,
) -> AdapterMergeState:
    apply_lora_weight_to_model(unet, lora_weight)
    if lora_config.text_encoder_1.train:
        apply_lora_weight_to_model(text_encoder_1, lora_weight)
    if lora_config.text_encoder_2.train:
        apply_lora_weight_to_model(text_encoder_2, lora_weight)

    unet_merged = False
    te1_merged = False
    te2_merged = False

    if merge_unet:
        unet.merge_adapter()
        unet_merged = True
        inference_unet = unet.base_model.model
    else:
        inference_unet = unet

    if lora_config.text_encoder_1.train:
        text_encoder_1.merge_adapter()
        te1_merged = True
        inference_te1 = text_encoder_1.base_model.model
    else:
        inference_te1 = text_encoder_1

    if lora_config.text_encoder_2.train:
        text_encoder_2.merge_adapter()
        te2_merged = True
        inference_te2 = text_encoder_2.base_model.model
    else:
        inference_te2 = text_encoder_2

    return AdapterMergeState(
        unet_merged=unet_merged,
        te1_merged=te1_merged,
        te2_merged=te2_merged,
        models=MergedInferenceModels(
            unet=inference_unet,
            text_encoder_1=inference_te1,
            text_encoder_2=inference_te2,
        ),
    )


def unmerge_adapters(
    *,
    unet: Any,
    text_encoder_1: Any,
    text_encoder_2: Any,
    state: AdapterMergeState,
) -> None:
    if state.unet_merged:
        unet.unmerge_adapter()
    if state.te1_merged:
        text_encoder_1.unmerge_adapter()
    if state.te2_merged:
        text_encoder_2.unmerge_adapter()


def run_sampling_pass_with_embeds(
    *,
    inference_unet: torch.nn.Module,
    inference_te1: torch.nn.Module,
    inference_te2: torch.nn.Module,
    vae: torch.nn.Module,
    tokenizer_1: Any,
    tokenizer_2: Any,
    noise_scheduler: DDPMScheduler,
    sampling_config: TrainConfig,
    device: torch.device,
    sample_prompts: list[str],
    output_dir: Path,
    output_stem: str,
    log: Logger,
    embed_cache: PromptEmbedCache | None = None,
    reference_add_time_ids: tuple[int, ...] | None = None,
    on_status: Callable[[int, int], None] | None = None,
    on_step: Callable[[int, int, int], None] | None = None,
    log_step_context: str = "[sample {prompt_index}/{n_prompts}]",
    output_filenames: list[str] | None = None,
    clear_embed_cache_on_te_train: bool = False,
) -> None:
    """Precompute embeds, create session, and run one sampling pass."""
    inference_unet.eval()
    inference_te1.eval()
    inference_te2.eval()

    width = sampling_config.sample_width or sampling_config.resolution
    height = sampling_config.sample_height or sampling_config.resolution
    autocast_dtype = _DTYPE_MAP[sampling_config.mixed_precision]

    cache = embed_cache if embed_cache is not None else PromptEmbedCache()
    if clear_embed_cache_on_te_train:
        cache.clear()

    negative_prompt = sampling_config.sample_negative_prompt or ""
    inference_te1 = inference_te1.to(
        device=device,
        dtype=_DTYPE_MAP[sampling_config.text_encoder_1.weight_dtype],
    )
    inference_te2 = inference_te2.to(
        device=device,
        dtype=_DTYPE_MAP[sampling_config.text_encoder_2.weight_dtype],
    )
    log.info(
        "Precomputing prompt embeddings (%d prompt(s), %dx%d)...",
        len(sample_prompts),
        width,
        height,
    )
    all_embeds = precompute_all_sample_embeds(
        sample_prompts=sample_prompts,
        negative_prompt=negative_prompt,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=inference_te1,
        text_encoder_2=inference_te2,
        device=device,
        dtype=autocast_dtype,
        clip_skip=sampling_config.clip_skip,
        cache=cache,
    )
    inference_te1.to("cpu")
    inference_te2.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log.info("Starting diffusion (%d steps, scheduler=%s)...", sampling_config.sample_steps, sampling_config.sample_scheduler)
    session = SDXLSamplingSession.create(
        unet=inference_unet,
        vae=vae,
        scheduler=build_inference_scheduler(sampling_config.sample_scheduler, noise_scheduler),
        device=device,
        width=width,
        height=height,
        sample_steps=sampling_config.sample_steps,
        autocast_dtype=autocast_dtype,
        config=sampling_config,
        reference_add_time_ids=reference_add_time_ids,
    )

    run_sdxl_sampling_pass(
        session=session,
        embeds_list=all_embeds,
        config=sampling_config,
        output_dir=output_dir,
        output_stem=output_stem,
        log=log,
        on_status=on_status,
        on_step=on_step,
        log_step_context=log_step_context,
        output_filenames=output_filenames,
    )


def run_merged_adapter_sampling(
    *,
    unet: Any,
    text_encoder_1: Any,
    text_encoder_2: Any,
    vae: torch.nn.Module,
    tokenizer_1: Any,
    tokenizer_2: Any,
    noise_scheduler: DDPMScheduler,
    lora_config: TrainConfig,
    sampling_config: TrainConfig,
    device: torch.device,
    sample_prompts: list[str],
    output_dir: Path,
    output_stem: str,
    log: Logger,
    merge_unet: bool = True,
    embed_cache: PromptEmbedCache | None = None,
    reference_add_time_ids: tuple[int, ...] | None = None,
    on_status: Callable[[int, int], None] | None = None,
    on_step: Callable[[int, int, int], None] | None = None,
    log_step_context: str = "[sample {prompt_index}/{n_prompts}]",
    clear_embed_cache_on_te_train: bool = True,
    lora_weight: float = 1.0,
    output_filenames: list[str] | None = None,
) -> None:
    """Merge adapters, precompute embeds, run one sampling pass, then unmerge."""
    log.info("Merging LoRA adapters for inference (weight=%.2f)...", lora_weight)
    merge_state = merge_adapters_for_inference(
        unet=unet,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        lora_config=lora_config,
        merge_unet=merge_unet,
        lora_weight=lora_weight,
    )
    inference_unet = merge_state.models.unet
    inference_te1 = merge_state.models.text_encoder_1
    inference_te2 = merge_state.models.text_encoder_2

    try:
        run_sampling_pass_with_embeds(
            inference_unet=inference_unet,
            inference_te1=inference_te1,
            inference_te2=inference_te2,
            vae=vae,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            sampling_config=sampling_config,
            device=device,
            sample_prompts=sample_prompts,
            output_dir=output_dir,
            output_stem=output_stem,
            log=log,
            embed_cache=embed_cache,
            reference_add_time_ids=reference_add_time_ids,
            on_status=on_status,
            on_step=on_step,
            log_step_context=log_step_context,
            output_filenames=output_filenames,
            clear_embed_cache_on_te_train=clear_embed_cache_on_te_train
            and (lora_config.text_encoder_1.train or lora_config.text_encoder_2.train),
        )
    finally:
        unmerge_adapters(
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            state=merge_state,
        )
