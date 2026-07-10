"""VAE decode step (Comfy VAEDecode equivalent)."""

import loggingimport timeimport numpy as npimport torchfrom PIL import Imagefrom torch import Tensorfrom src.trainer.sdxl.latent_sampling.session import SDXLSamplingSessiondef _tensor_to_pil(image: Tensor) -> Image.Image:
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype(np.uint8)
    return Image.fromarray(image)


def _decode_full_vae(session: SDXLSamplingSession, latent: Tensor) -> Tensor:
    vae = session.vae
    scaled_latent = latent / vae.config.scaling_factor
    decode_dtype = torch.float32 if vae.dtype == torch.float32 else session.autocast_dtype
    scaled_latent = scaled_latent.to(dtype=decode_dtype)

    if decode_dtype == torch.float32:
        return vae.decode(scaled_latent, return_dict=False)[0]

    with torch.autocast(device_type=session.device.type, dtype=session.autocast_dtype):
        return vae.decode(scaled_latent, return_dict=False)[0]


def decode_sdxl_latent(
    session: SDXLSamplingSession,
    latent: Tensor,
    *,
    log: logging.Logger | None = None,
    log_prefix: str = "",
) -> Image.Image:
    gpu_decode_started_at = time.perf_counter()
    decoded = _decode_full_vae(session, latent)

    if log is not None:
        if session.device.type == "cuda":
            torch.cuda.synchronize()
        log.info(
            "%s vae.decode (full GPU): %.3fs",
            log_prefix,
            time.perf_counter() - gpu_decode_started_at,
        )

    cpu_convert_started_at = time.perf_counter()
    pil_image = _tensor_to_pil(decoded)

    if log is not None:
        log.info(
            "%s tensor→PIL (CPU): %.3fs",
            log_prefix,
            time.perf_counter() - cpu_convert_started_at,
        )

    return pil_image
