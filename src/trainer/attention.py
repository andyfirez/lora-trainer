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
    """Apply the requested attention backend, falling back to SDPA when needed."""
    if mechanism in ("default", "sdpa"):
        log.info("Using PyTorch SDPA attention (diffusers default on torch 2.x).")
        return

    if not is_xformers_available():
        log.warning(
            "xformers is not installed; falling back to PyTorch SDPA. "
            "See https://github.com/facebookresearch/xformers#installing-xformers"
        )
        return

    try:
        unet.enable_xformers_memory_efficient_attention()
        log.info("xformers memory-efficient attention enabled.")
    except Exception as exc:
        log.warning(
            "Failed to enable xformers (%s); falling back to PyTorch SDPA.",
            exc,
        )
