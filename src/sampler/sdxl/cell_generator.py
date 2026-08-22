"""Single-image SDXL sampling cell generation for sweep runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.services.png_info.writer import build_a1111_infotext
from src.trainer.concept_training_metadata import (
    ConceptTrainingMetadata,
    resolve_reference_add_time_ids,
)
from src.trainer.inference_config import SDXLInferenceConfig
from src.trainer.sdxl.inference_context import run_merged_adapter_sampling
from src.trainer.sdxl.pipeline_loader import SamplingStack
from src.trainer.sdxl.sampling import PromptEmbedCache

ProgressStepCallback = Callable[[int, int], None]


def generate_sampling_cell(
    *,
    stack: SamplingStack,
    lora_config: SDXLInferenceConfig,
    sampling_config: SDXLInferenceConfig,
    merge_unet: bool,
    prompt: str,
    lora_weight: float,
    output_dir: Path,
    output_filename: str,
    completed_images: int,
    total_steps: int,
    concept_metadata: dict[int, ConceptTrainingMetadata],
    prompt_embed_cache: PromptEmbedCache,
    log: logging.Logger,
    on_progress: ProgressStepCallback | None = None,
    lora_path: str | None = None,
    lora_stack: list[dict[str, object]] | None = None,
    preview_path: Path | None = None,
) -> None:
    config = sampling_config
    width = config.sample_width or config.resolution
    height = config.sample_height or config.resolution
    log.info(
        "Generating image: %dx%d, %d steps, prompt=%r",
        width,
        height,
        config.sample_steps,
        prompt[:120],
    )
    reference_add_time_ids = resolve_reference_add_time_ids(
        concept_metadata,
        dataset_ids=_reference_dataset_ids(concept_metadata),
        width=config.sample_width or config.resolution,
        height=config.sample_height or config.resolution,
    )

    def on_step(_prompt_index: int, completed: int, _total: int) -> None:
        if on_progress is not None:
            image_offset = completed_images * config.sample_steps
            on_progress(image_offset + completed, total_steps)

    infotext = build_a1111_infotext(
        prompt=prompt,
        negative_prompt=config.sample_negative_prompt,
        steps=config.sample_steps,
        sampler=config.sample_scheduler,
        cfg_scale=config.sample_cfg_scale,
        seed=config.seed,
        width=width,
        height=height,
        model_name=config.base_model_name,
        lora_path=lora_path,
        lora_weight=lora_weight if lora_path else None,
        loras=_loras_for_infotext(lora_stack),
    )

    run_merged_adapter_sampling(
        unet=stack.unet,
        text_encoder_1=stack.text_encoder_1,
        text_encoder_2=stack.text_encoder_2,
        vae=stack.vae,
        tokenizer_1=stack.tokenizer_1,
        tokenizer_2=stack.tokenizer_2,
        noise_scheduler=stack.noise_scheduler,
        lora_config=lora_config,
        sampling_config=config,
        device=stack.device,
        sample_prompts=[prompt],
        output_dir=output_dir,
        output_stem="cell",
        log=log,
        merge_unet=merge_unet,
        embed_cache=prompt_embed_cache,
        reference_add_time_ids=reference_add_time_ids,
        on_step=on_step,
        lora_weight=lora_weight,
        output_filenames=[output_filename],
        clear_embed_cache_on_te_train=True,
        png_infotext=infotext,
        preview_path=preview_path,
    )


def _loras_for_infotext(
    lora_stack: list[dict[str, object]] | None,
) -> list[tuple[str, float | None]] | None:
    if not lora_stack:
        return None
    items: list[tuple[str, float | None]] = []
    for item in lora_stack:
        path = item.get("path")
        if not path:
            continue
        raw_weight = item.get("weight", 1.0)
        try:
            weight = float(raw_weight) if raw_weight is not None else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        items.append((str(path), weight))
    return items or None


def _reference_dataset_ids(
    concept_metadata: dict[int, ConceptTrainingMetadata],
) -> list[int]:
    return list(concept_metadata.keys())
