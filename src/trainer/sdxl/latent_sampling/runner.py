"""Orchestrates encode → KSampler → VAEDecode for multiple prompts."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import torch

from src.trainer.config import TrainConfig
from src.trainer.sdxl.latent_sampling.ksample import ksample_sdxl_latent
from src.trainer.sdxl.latent_sampling.session import SDXLSamplingSession
from src.trainer.sdxl.latent_sampling.vae_decode import decode_sdxl_latent
from src.trainer.sdxl.sampling import SamplePromptEmbeds

StatusCallback = Callable[[int, int], None]
StepProgressCallback = Callable[[int, int, int], None]

_SLOW_CALLBACK_THRESHOLD_S = 0.05


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _log_phase(
    log: logging.Logger,
    *,
    prefix: str,
    phase: str,
    started_at: float,
    device: torch.device,
    sync_cuda: bool = False,
) -> float:
    if sync_cuda:
        _cuda_sync(device)
    elapsed = time.perf_counter() - started_at
    log.info("%s %s: %.3fs", prefix, phase, elapsed)
    return elapsed


def _run_timed_callback(
    log: logging.Logger,
    *,
    prefix: str,
    phase: str,
    callback: Callable[[], None] | None,
) -> float:
    if callback is None:
        return 0.0
    started_at = time.perf_counter()
    callback()
    elapsed = time.perf_counter() - started_at
    if elapsed >= _SLOW_CALLBACK_THRESHOLD_S:
        log.info("%s slow %s callback: %.3fs", prefix, phase, elapsed)
    return elapsed


def _unet_device(unet: torch.nn.Module) -> torch.device:
    return next(unet.parameters()).device


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    if actual.type != "cuda":
        return actual == expected
    actual_index = actual.index if actual.index is not None else 0
    expected_index = expected.index if expected.index is not None else 0
    return actual_index == expected_index


def _ensure_unet_on_device(
    session: SDXLSamplingSession,
    log: logging.Logger,
    *,
    prefix: str,
) -> None:
    if _device_matches(_unet_device(session.unet), session.device):
        return
    restore_started_at = time.perf_counter()
    session.unet.to(session.device)
    log.info(
        "%s unet restore before ksample: %.3fs",
        prefix,
        time.perf_counter() - restore_started_at,
    )


def _restore_unet_after_decode(
    session: SDXLSamplingSession,
    log: logging.Logger,
    *,
    prefix: str,
) -> None:
    if _device_matches(_unet_device(session.unet), session.device):
        return
    restore_started_at = time.perf_counter()
    session.unet.to(session.device)
    log.info(
        "%s unet restore after decode: %.3fs",
        prefix,
        time.perf_counter() - restore_started_at,
    )


def run_sdxl_sampling_pass(
    *,
    session: SDXLSamplingSession,
    embeds_list: list[SamplePromptEmbeds],
    config: TrainConfig,
    output_dir: Path,
    output_stem: str,
    log: logging.Logger,
    on_status: StatusCallback | None = None,
    on_step: StepProgressCallback | None = None,
    log_step_context: str = "",
    output_filenames: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = len(embeds_list)
    log_interval = max(1, config.sample_steps // 5)
    pass_started_at = time.perf_counter()
    previous_image_finished_at: float | None = None

    width = config.sample_width or config.resolution
    height = config.sample_height or config.resolution
    log.info(
        "Starting sampling pass: %d prompt(s), %dx%d, %d steps, vae_tiling=%s, stem=%s",
        n_prompts,
        width,
        height,
        config.sample_steps,
        config.sample_vae_tiling,
        output_stem,
    )

    with torch.no_grad():
        for prompt_index, embeds in enumerate(embeds_list):
            prefix = f"[sample {prompt_index + 1}/{n_prompts}]"
            iteration_started_at = time.perf_counter()

            if previous_image_finished_at is not None:
                gap = iteration_started_at - previous_image_finished_at
                log.info("%s gap since previous image finished: %.3fs", prefix, gap)

            status_callback_s = _run_timed_callback(
                log,
                prefix=prefix,
                phase="on_status",
                callback=(
                    lambda: on_status(prompt_index, n_prompts)
                    if on_status is not None
                    else None
                ),
            )
            if status_callback_s > 0.0 and status_callback_s < _SLOW_CALLBACK_THRESHOLD_S:
                log.debug("%s on_status callback: %.3fs", prefix, status_callback_s)

            step_callback_total_s = 0.0

            def _on_step_end(completed: int, total: int) -> None:
                nonlocal step_callback_total_s
                if on_step is not None:
                    callback_started_at = time.perf_counter()
                    on_step(prompt_index, completed, total)
                    step_callback_total_s += time.perf_counter() - callback_started_at
                if log_step_context and (completed % log_interval == 0 or completed == total):
                    log.info(
                        "%s step %d/%d",
                        log_step_context.format(
                            prompt_index=prompt_index + 1,
                            n_prompts=n_prompts,
                        ),
                        completed,
                        total,
                    )

            progress_reset_s = _run_timed_callback(
                log,
                prefix=prefix,
                phase="on_step(reset)",
                callback=(
                    lambda: on_step(prompt_index, 0, config.sample_steps)
                    if on_step is not None
                    else None
                ),
            )
            step_callback_total_s += progress_reset_s

            if config.sample_offload_unet_before_decode:
                _ensure_unet_on_device(session, log, prefix=prefix)

            ksample_started_at = time.perf_counter()
            latent = ksample_sdxl_latent(
                session,
                embeds,
                width=width,
                height=height,
                guidance_scale=config.sample_cfg_scale,
                seed=config.seed,
                prompt_index=prompt_index,
                on_step_end=_on_step_end,
                log=log,
                log_prefix=prefix,
            )
            ksample_s = _log_phase(
                log,
                prefix=prefix,
                phase="ksample (GPU)",
                started_at=ksample_started_at,
                device=session.device,
                sync_cuda=True,
            )

            offload_s = 0.0
            if config.sample_offload_unet_before_decode:
                offload_started_at = time.perf_counter()
                session.unet.to("cpu")
                if session.device.type == "cuda":
                    torch.cuda.empty_cache()
                offload_s = time.perf_counter() - offload_started_at
                log.info("%s unet offload before decode: %.3fs", prefix, offload_s)

            log.info("%s decoding started", prefix)
            decode_started_at = time.perf_counter()
            image = decode_sdxl_latent(
                session,
                latent,
                log=log,
                log_prefix=prefix,
            )
            decode_total_s = time.perf_counter() - decode_started_at
            log.info("%s decode (total): %.3fs", prefix, decode_total_s)

            if config.sample_offload_unet_before_decode:
                _restore_unet_after_decode(session, log, prefix=prefix)

            filename = (
                output_filenames[prompt_index]
                if output_filenames is not None and prompt_index < len(output_filenames)
                else f"{output_stem}_{prompt_index:02d}.png"
            )
            output_path = output_dir / filename
            save_started_at = time.perf_counter()
            image.save(output_path)
            save_s = time.perf_counter() - save_started_at
            log.info("%s save PNG: %.3fs -> %s", prefix, save_s, output_path)

            image_total_s = time.perf_counter() - iteration_started_at
            previous_image_finished_at = time.perf_counter()
            log.info(
                "%s image total: %.3fs (ksample=%.3fs, offload=%.3fs, decode=%.3fs, save=%.3fs, step_callbacks=%.3fs)",
                prefix,
                image_total_s,
                ksample_s,
                offload_s,
                decode_total_s,
                save_s,
                step_callback_total_s,
            )
            if step_callback_total_s >= _SLOW_CALLBACK_THRESHOLD_S:
                log.warning(
                    "%s progress callbacks consumed %.3fs across %d step updates",
                    prefix,
                    step_callback_total_s,
                    config.sample_steps + 1,
                )

    log.info(
        "Sampling pass finished: %d image(s) in %.3fs (stem=%s)",
        n_prompts,
        time.perf_counter() - pass_started_at,
        output_stem,
    )
