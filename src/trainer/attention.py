"""Configure UNet attention backends."""

import logging
from typing import Literal

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

AttentionMechanism = Literal["default", "sdpa", "xformers"]


def _enable_pytorch_sdp_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def configure_unet_attention(
    unet: UNet2DConditionModel,
    mechanism: AttentionMechanism,
    log: logging.Logger | logging.LoggerAdapter,
) -> None:
    """Apply the requested attention backend."""
    if mechanism in ("default", "sdpa"):
        _enable_pytorch_sdp_backends()
        log.info("Using PyTorch SDPA attention with flash/mem-efficient backends enabled.")
        return

    if not is_xformers_available():
        raise RuntimeError(
            "attention_mechanism='xformers' requires xformers, but it is not installed. "
            "Install it with `uv add xformers` and re-run."
        )

    try:
        unet.enable_xformers_memory_efficient_attention()
        log.info("xformers memory-efficient attention enabled.")
    except Exception as exc:
        raise RuntimeError(
            "attention_mechanism='xformers' was requested, but enabling xformers "
            f"memory-efficient attention failed: {exc}"
        ) from exc
