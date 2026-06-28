"""Standalone SDXL LoRA sampling service."""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model

from src.trainer.attention import configure_unet_attention
from src.trainer.config import TrainConfig, WeightDtype
from src.trainer.sdxl.caption import apply_trigger_words_to_sample_prompts
from src.trainer.sdxl.lora_io import apply_lora_state_dict, load_lora_file
from src.trainer.sdxl.model_loader import load_sdxl_components, resolve_vae_dtype
from src.trainer.sdxl.sampling import (
    PromptEmbedCache,
    SamplePromptEmbeds,
    build_embed_only_sdxl_pipeline,
    build_inference_scheduler,
    precompute_all_sample_embeds,
    prepare_vae_for_decode,
)

logger = logging.getLogger(__name__)

ProgressStatusCallback = Callable[[str | None], None]
ProgressCallback = Callable[[int, int], None]

_DTYPE_MAP = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}


@dataclass
class _SamplingStack:
    device: torch.device
    tokenizer_1: Any
    tokenizer_2: Any
    noise_scheduler: Any
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module
    vae: torch.nn.Module
    unet: torch.nn.Module


class SDXLLoRASampler:
    def __init__(
        self,
        config: TrainConfig,
        *,
        lora_paths: list[Path],
        output_dir: Path,
        trigger_words: list[str] | None = None,
        progress_status_callback: ProgressStatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._lora_paths = lora_paths
        self._output_dir = output_dir
        self._trigger_words = trigger_words or []
        self._progress_status_callback = progress_status_callback
        self._progress_callback = progress_callback
        self._log = log or logger
        self._prompt_embed_cache = PromptEmbedCache()

    def run(self) -> None:
        config = self._config
        if not self._effective_sample_prompts():
            raise ValueError("No sample prompts configured")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available (torch {torch.__version__})")

        self._prompt_embed_cache.clear()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        config.validate_gpu()

        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._log.info("Loading SDXL pipeline from %s", config.base_model_name)
        stack = self._load_stack(config, enable_lora=bool(self._lora_paths))

        total_diffusion_steps = self._total_diffusion_steps()
        self._set_progress(0, total_diffusion_steps)
        completed_images = 0
        if self._lora_paths:
            for lora_index, lora_path in enumerate(self._lora_paths):
                status_prefix = f"Sampling {lora_path.name} ({lora_index + 1}/{len(self._lora_paths)})"
                self._set_status(f"{status_prefix} — loading LoRA")
                state_dict = load_lora_file(lora_path)
                apply_lora_state_dict(
                    state_dict,
                    unet=stack.unet,
                    text_encoder_1=stack.text_encoder_1,
                    text_encoder_2=stack.text_encoder_2,
                    config=config,
                )
                self._sample_pass(
                    output_stem=self._safe_stem(lora_path),
                    status_prefix=status_prefix,
                    completed_images=completed_images,
                    stack=stack,
                    merge_unet_adapter=True,
                )
                completed_images += len(self._effective_sample_prompts())
        else:
            status_prefix = "Sampling base model"
            self._set_status(status_prefix)
            self._sample_pass(
                output_stem="base",
                status_prefix=status_prefix,
                completed_images=0,
                stack=stack,
                merge_unet_adapter=False,
            )
        self._set_status(None)

    def _load_stack(self, config: TrainConfig, *, enable_lora: bool) -> _SamplingStack:
        device = torch.device("cuda")
        vae_dtype = resolve_vae_dtype(config.vae_dtype)
        components = load_sdxl_components(
            config.base_model_name,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
            vae_dtype=config.vae_dtype,
        )

        vae = components.vae
        text_encoder_1 = components.text_encoder_1
        text_encoder_2 = components.text_encoder_2
        unet = components.unet

        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)

        if enable_lora:
            unet = get_peft_model(
                unet,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                ),
            )
        if enable_lora and config.text_encoder_1.train:
            text_encoder_1 = get_peft_model(
                text_encoder_1,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                ),
            )
        if enable_lora and config.text_encoder_2.train:
            text_encoder_2 = get_peft_model(
                text_encoder_2,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                ),
            )

        unet = unet.to(device=device, dtype=_DTYPE_MAP[config.unet.weight_dtype])
        text_encoder_1 = text_encoder_1.to(device=device, dtype=_DTYPE_MAP[config.text_encoder_1.weight_dtype])
        text_encoder_2 = text_encoder_2.to(device=device, dtype=_DTYPE_MAP[config.text_encoder_2.weight_dtype])
        vae = vae.to(device=device, dtype=vae_dtype)
        configure_unet_attention(unet, config.attention_mechanism, self._log)

        return _SamplingStack(
            device=device,
            tokenizer_1=components.tokenizer_1,
            tokenizer_2=components.tokenizer_2,
            noise_scheduler=components.noise_scheduler,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            unet=unet,
        )

    def _sample_pass(
        self,
        *,
        output_stem: str,
        status_prefix: str,
        completed_images: int,
        stack: _SamplingStack,
        merge_unet_adapter: bool,
    ) -> None:
        config = self._config
        sample_prompts = self._effective_sample_prompts()
        device = stack.device
        started_at = time.perf_counter()
        unet = stack.unet
        text_encoder_1 = stack.text_encoder_1
        text_encoder_2 = stack.text_encoder_2
        unet_merged = False
        te1_merged = False
        te2_merged = False
        try:
            if merge_unet_adapter:
                unet.merge_adapter()
                unet_merged = True
                inference_unet = unet.base_model.model
            else:
                inference_unet = unet
            if config.text_encoder_1.train:
                text_encoder_1.merge_adapter()
                te1_merged = True
                inference_te1 = text_encoder_1.base_model.model
            else:
                inference_te1 = text_encoder_1
            if config.text_encoder_2.train:
                text_encoder_2.merge_adapter()
                te2_merged = True
                inference_te2 = text_encoder_2.base_model.model
            else:
                inference_te2 = text_encoder_2

            inference_unet.eval()
            inference_te1.eval()
            inference_te2.eval()
            width = config.sample_width or config.resolution
            height = config.sample_height or config.resolution
            autocast_dtype = _DTYPE_MAP[config.mixed_precision]

            if config.text_encoder_1.train or config.text_encoder_2.train:
                self._prompt_embed_cache.clear()

            negative_prompt = config.sample_negative_prompt or ""
            all_embeds = precompute_all_sample_embeds(
                sample_prompts=sample_prompts,
                negative_prompt=negative_prompt,
                tokenizer_1=stack.tokenizer_1,
                tokenizer_2=stack.tokenizer_2,
                text_encoder_1=inference_te1,
                text_encoder_2=inference_te2,
                device=device,
                dtype=autocast_dtype,
                cache=self._prompt_embed_cache,
            )
            shared_negative_embeds = all_embeds[0].negative_prompt_embeds
            shared_negative_pooled = all_embeds[0].negative_pooled_prompt_embeds
            inference_te1.to("cpu")
            inference_te2.to("cpu")
            torch.cuda.empty_cache()
            prepare_vae_for_decode(stack.vae)

            pipe = build_embed_only_sdxl_pipeline(
                vae=stack.vae,
                unet=inference_unet,
                tokenizer_1=stack.tokenizer_1,
                tokenizer_2=stack.tokenizer_2,
                scheduler=build_inference_scheduler(config.sample_scheduler, stack.noise_scheduler),
            )
            pipe.set_progress_bar_config(disable=True)

            with torch.no_grad():
                for prompt_index, embeds in enumerate(all_embeds):
                    self._set_status(f"{status_prefix} — image {prompt_index + 1}/{len(sample_prompts)}")
                    self._report_diffusion_progress(completed_images, prompt_index, 0)
                    generator = torch.Generator(device=device)
                    if config.seed is not None:
                        generator.manual_seed(config.seed + prompt_index)
                    image = self._generate_image(
                        pipe=pipe,
                        embeds=embeds,
                        shared_negative_embeds=shared_negative_embeds,
                        shared_negative_pooled=shared_negative_pooled,
                        prompt_index=prompt_index,
                        completed_images=completed_images,
                        generator=generator,
                        width=width,
                        height=height,
                        autocast_dtype=autocast_dtype,
                        device=device,
                    )
                    filename = f"{output_stem}_{prompt_index:02d}.png"
                    image.save(self._output_dir / filename)
                    self._report_diffusion_progress(completed_images, prompt_index, config.sample_steps)
                    self._log.info("Sample saved: %s", filename)
        finally:
            if unet_merged:
                unet.unmerge_adapter()
            if te1_merged:
                text_encoder_1.unmerge_adapter()
            if te2_merged:
                text_encoder_2.unmerge_adapter()
            self._log.info("Sampling %s completed in %.2fs", output_stem, time.perf_counter() - started_at)

    def _generate_image(
        self,
        *,
        pipe: StableDiffusionXLPipeline,
        embeds: SamplePromptEmbeds,
        shared_negative_embeds: torch.Tensor,
        shared_negative_pooled: torch.Tensor,
        prompt_index: int,
        completed_images: int,
        generator: torch.Generator,
        width: int,
        height: int,
        autocast_dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        config = self._config
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            return pipe(
                prompt_embeds=embeds.prompt_embeds,
                negative_prompt_embeds=shared_negative_embeds,
                pooled_prompt_embeds=embeds.pooled_prompt_embeds,
                negative_pooled_prompt_embeds=shared_negative_pooled,
                width=width,
                height=height,
                num_inference_steps=config.sample_steps,
                guidance_scale=config.sample_cfg_scale,
                generator=generator,
                callback_on_step_end=self._make_diffusers_progress_callback(prompt_index, completed_images),
                callback_on_step_end_tensor_inputs=[],
            ).images[0]

    def _effective_sample_prompts(self) -> list[str]:
        if not self._lora_paths:
            return self._config.sample_prompts
        return apply_trigger_words_to_sample_prompts(self._config.sample_prompts, self._trigger_words)

    def _total_diffusion_steps(self) -> int:
        lora_count = len(self._lora_paths) if self._lora_paths else 1
        return lora_count * len(self._effective_sample_prompts()) * self._config.sample_steps

    def _report_diffusion_progress(
        self,
        completed_images: int,
        prompt_index: int,
        diffusion_step: int,
    ) -> None:
        image_offset = (completed_images + prompt_index) * self._config.sample_steps
        self._set_progress(image_offset + diffusion_step, self._total_diffusion_steps())

    def _make_diffusers_progress_callback(
        self,
        prompt_index: int,
        completed_images: int,
    ) -> Callable[..., dict[str, Any]]:
        def _on_step_end(_pipeline: Any, step_index: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            completed = step_index + 1
            self._report_diffusion_progress(completed_images, prompt_index, completed)
            self._log_sample_step(prompt_index, completed, self._config.sample_steps)
            return callback_kwargs

        return _on_step_end

    def _log_sample_step(self, prompt_index: int, completed: int, total: int) -> None:
        log_interval = max(1, total // 5)
        if completed % log_interval == 0 or completed == total:
            self._log.info(
                "[sample %d/%d] step %d/%d",
                prompt_index + 1,
                len(self._effective_sample_prompts()),
                completed,
                total,
            )

    def _set_status(self, status: str | None) -> None:
        if self._progress_status_callback is not None:
            self._progress_status_callback(status)

    def _set_progress(self, step: int, total: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(step, total)

    def _safe_stem(self, lora_path: Path) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", lora_path.stem)
