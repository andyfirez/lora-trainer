"""End-to-end SDXL sampling via vendored ComfyUI stack."""

from __future__ import annotations

import time
from collections.abc import Callable
from logging import Logger
from pathlib import Path

import torch

from src.trainer.config import TrainConfig
from src.trainer.sdxl.comfy_inference.bootstrap import ensure_vendored_comfy
from src.trainer.sdxl.comfy_inference.decode import comfy_tensor_to_pil
from src.trainer.sdxl.comfy_inference.encoding import encode_sdxl_conditioning
from src.trainer.sdxl.comfy_inference.lora import apply_comfy_lora
from src.trainer.sdxl.comfy_inference.types import ComfyInferenceStack

StatusCallback = Callable[[int, int], None]
StepProgressCallback = Callable[[int, int, int], None]
_SLOW_CALLBACK_THRESHOLD_S = 0.05


def _resolve_seed(seed: int | None) -> int:
    return 0 if seed is None else int(seed)


def run_comfy_inference_sampling(
    *,
    stack: ComfyInferenceStack,
    sampling_config: TrainConfig,
    sample_prompts: list[str],
    output_dir: Path,
    output_stem: str,
    log: Logger,
    lora_weight: float = 1.0,
    reference_add_time_ids: tuple[float, ...] | None = None,
    on_status: StatusCallback | None = None,
    on_step: StepProgressCallback | None = None,
    log_step_context: str = "[sample {prompt_index}/{n_prompts}]",
    output_filenames: list[str] | None = None,
) -> None:
    ensure_vendored_comfy()
    import comfy.model_management as model_management
    import comfy.sample as comfy_sample

    output_dir.mkdir(parents=True, exist_ok=True)
    width = sampling_config.sample_width or sampling_config.resolution
    height = sampling_config.sample_height or sampling_config.resolution
    negative_prompt = sampling_config.sample_negative_prompt or ""
    n_prompts = len(sample_prompts)
    log_interval = max(1, sampling_config.sample_steps // 5)
    pass_started_at = time.perf_counter()

    log.info(
        "Starting vendored Comfy sampling: %d prompt(s), %dx%d, %d steps, %s+%s",
        n_prompts,
        width,
        height,
        sampling_config.sample_steps,
        sampling_config.sample_sampler_name,
        sampling_config.sample_scheduler,
    )

    model, clip = apply_comfy_lora(stack, lora_weight=lora_weight)
    latent_h = height // 8
    latent_w = width // 8
    intermediate = model_management.intermediate_device()
    intermediate_dtype = model_management.intermediate_dtype()

    with torch.no_grad():
        for prompt_index, prompt in enumerate(sample_prompts):
            prefix = f"[sample {prompt_index + 1}/{n_prompts}]"
            if on_status is not None:
                on_status(prompt_index, n_prompts)

            positive, negative = encode_sdxl_conditioning(
                clip,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                reference_add_time_ids=reference_add_time_ids,
            )

            seed = _resolve_seed(sampling_config.seed)
            empty_latent = torch.zeros(
                (1, 4, latent_h, latent_w),
                device=intermediate,
                dtype=intermediate_dtype,
            )
            noise = comfy_sample.prepare_noise(empty_latent, seed)
            total_steps = sampling_config.sample_steps

            def _callback(step: int, _denoised, _x, total: int) -> None:
                completed = step + 1
                if on_step is not None:
                    on_step(prompt_index, completed, total)
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

            if on_step is not None:
                on_step(prompt_index, 0, total_steps)

            ksample_started_at = time.perf_counter()
            samples = comfy_sample.sample(
                model,
                noise,
                sampling_config.sample_steps,
                sampling_config.sample_cfg_scale,
                sampling_config.sample_sampler_name,
                sampling_config.sample_scheduler,
                positive,
                negative,
                empty_latent,
                denoise=1.0,
                seed=seed,
                callback=_callback,
                disable_pbar=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log.info(
                "%s vendored ksample: %.3fs (%d steps, %s+%s)",
                prefix,
                time.perf_counter() - ksample_started_at,
                total_steps,
                sampling_config.sample_sampler_name,
                sampling_config.sample_scheduler,
            )

            decode_started_at = time.perf_counter()
            images = stack.vae.decode(samples)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log.info("%s vae.decode: %.3fs", prefix, time.perf_counter() - decode_started_at)

            image = comfy_tensor_to_pil(images)
            filename = (
                output_filenames[prompt_index]
                if output_filenames is not None and prompt_index < len(output_filenames)
                else f"{output_stem}_{prompt_index:02d}.png"
            )
            output_path = output_dir / filename
            image.save(output_path)
            log.info("%s saved -> %s", prefix, output_path)

    log.info(
        "Vendored Comfy sampling finished: %d image(s) in %.3fs",
        n_prompts,
        time.perf_counter() - pass_started_at,
    )
