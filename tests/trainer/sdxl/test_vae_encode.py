from unittest.mock import MagicMock, patch

import torch

from src.trainer.sdxl.vae_encode import encode_pixel_values_to_latents


def _build_vae(*, dtype: torch.dtype) -> MagicMock:
    vae = MagicMock()
    vae.dtype = dtype
    vae.config.scaling_factor = 0.13025
    sample = MagicMock()
    sample.sample.return_value = torch.zeros(1, 4, 8, 8)
    encode_out = MagicMock()
    encode_out.latent_dist = sample
    vae.encode.return_value = encode_out
    return vae


def test_encode_pixel_values_uses_fp32_input_for_fp32_vae() -> None:
    vae = _build_vae(dtype=torch.float32)
    pixels = torch.zeros(1, 3, 64, 64)

    latents = encode_pixel_values_to_latents(vae, pixels, device=torch.device("cpu"))

    vae.encode.assert_called_once()
    assert vae.encode.call_args[0][0].dtype == torch.float32
    assert torch.allclose(latents, torch.zeros(1, 4, 8, 8) * 0.13025)


def test_encode_pixel_values_uses_autocast_for_bfloat16_vae() -> None:
    vae = _build_vae(dtype=torch.bfloat16)
    pixels = torch.zeros(1, 3, 64, 64)

    with patch("src.trainer.sdxl.vae_encode.torch.autocast") as mock_autocast:
        mock_autocast.return_value.__enter__ = MagicMock(return_value=None)
        mock_autocast.return_value.__exit__ = MagicMock(return_value=False)
        encode_pixel_values_to_latents(vae, pixels, device=torch.device("cuda"))

    mock_autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)
    assert vae.encode.call_args[0][0].dtype == torch.bfloat16
