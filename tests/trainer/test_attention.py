import logging
from unittest.mock import MagicMock, patch

from src.trainer.attention import configure_unet_attention


def test_configure_unet_attention_sdpa_skips_xformers() -> None:
    unet = MagicMock()
    log = logging.getLogger("test")

    configure_unet_attention(unet, "sdpa", log)

    unet.enable_xformers_memory_efficient_attention.assert_not_called()


@patch("src.trainer.attention.is_xformers_available", return_value=False)
def test_configure_unet_attention_xformers_missing_falls_back(
    _is_xformers_available: MagicMock,
) -> None:
    unet = MagicMock()
    log = logging.getLogger("test")

    configure_unet_attention(unet, "xformers", log)

    unet.enable_xformers_memory_efficient_attention.assert_not_called()


@patch("src.trainer.attention.is_xformers_available", return_value=True)
def test_configure_unet_attention_xformers_runtime_error_falls_back(
    _is_xformers_available: MagicMock,
) -> None:
    unet = MagicMock()
    unet.enable_xformers_memory_efficient_attention.side_effect = RuntimeError("CUDA mismatch")
    log = logging.getLogger("test")

    configure_unet_attention(unet, "xformers", log)

    unet.enable_xformers_memory_efficient_attention.assert_called_once()


@patch("src.trainer.attention.is_xformers_available", return_value=True)
def test_configure_unet_attention_xformers_success(
    _is_xformers_available: MagicMock,
) -> None:
    unet = MagicMock()
    log = logging.getLogger("test")

    configure_unet_attention(unet, "xformers", log)

    unet.enable_xformers_memory_efficient_attention.assert_called_once()
