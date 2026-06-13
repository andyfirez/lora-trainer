"""Configure UNet attention backends."""

import logging
from typing import Literal

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

AttentionMechanism = Literal["default", "sdpa", "xformers"]


def configure_unet_attention(
    unet: UNet2DConditionModel,
    mechanism: AttentionMechanism,
    log: logging.Logger | logging.LoggerAdapter,
) -> None:
    """Apply the requested attention backend."""
    if mechanism in ("default", "sdpa"):
        log.info("Using PyTorch SDPA attention (diffusers default on torch 2.x).")
        return

    if not is_xformers_available():
        raise RuntimeError(
            "attention_mechanism='xformers' requires xformers, but it is not installed. "
            "Install it with `uv add xformers` and re-run training."
        )

    try:
        unet.enable_xformers_memory_efficient_attention()
        log.info("xformers memory-efficient attention enabled.")
    except Exception as exc:
        raise RuntimeError(
            "attention_mechanism='xformers' was requested, but enabling xformers "
            f"memory-efficient attention failed: {exc}"
        ) from exc
