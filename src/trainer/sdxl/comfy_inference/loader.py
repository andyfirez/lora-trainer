"""Load SDXL models through vendored ComfyUI logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.trainer.config import TrainConfig
from src.trainer.sdxl.comfy_inference.bootstrap import ensure_vendored_comfy
from src.trainer.sdxl.comfy_inference.types import ComfyInferenceStack
from src.trainer.sdxl.model_loader import is_checkpoint_file

logger = logging.getLogger(__name__)


def _is_diffusers_folder(path: Path) -> bool:
    return (path / "model_index.json").is_file() or (
        (path / "unet").is_dir() and (path / "vae").is_dir()
    )


def _configure_clip_skip(clip: Any, clip_skip: int) -> None:
    if clip_skip != 2:
        clip.clip_layer(-clip_skip)


def load_comfy_sdxl_stack(
    base_model_path: str,
    *,
    config: TrainConfig,
    lora_state: dict[str, Any] | None = None,
    lora_apply_te1: bool = False,
    lora_apply_te2: bool = False,
) -> ComfyInferenceStack:
    ensure_vendored_comfy()
    import comfy.diffusers_load as comfy_diffusers_load
    import comfy.sd as comfy_sd

    path = Path(base_model_path)
    disable_dynamic = True

    if is_checkpoint_file(path):
        logger.info("Loading SDXL checkpoint via vendored Comfy: %s", path)
        model, clip, vae, _clipvision = comfy_sd.load_checkpoint_guess_config(
            str(path),
            disable_dynamic=disable_dynamic,
        )
    elif _is_diffusers_folder(path):
        logger.info("Loading SDXL diffusers folder via vendored Comfy: %s", path)
        model, clip, vae = comfy_diffusers_load.load_diffusers(str(path))
    else:
        raise ValueError(f"Unsupported base model path for Comfy sampling: {base_model_path}")

    if clip is not None:
        _configure_clip_skip(clip, config.clip_skip)

    return ComfyInferenceStack(
        model=model,
        clip=clip,
        vae=vae,
        base_model_path=str(path),
        lora_state=lora_state,
        lora_apply_te1=lora_apply_te1,
        lora_apply_te2=lora_apply_te2,
        _base_model=model,
        _base_clip=clip,
    )
