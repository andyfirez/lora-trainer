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
from torch import Tensor

from src.trainer.attention import configure_unet_attention
from src.trainer.config import TrainConfig
from src.trainer.sdxl.lora_io import apply_lora_state_dict, load_lora_file
from src.trainer.sdxl.model_loader import load_sdxl_components, resolve_vae_dtype
from src.trainer.sdxl.reforge_sampler import generate_preview_image
from src.trainer.sdxl.trainer import _DTYPE_MAP, _build_inference_scheduler

logger = logging.getLogger(__name__)

ProgressStatusCallback = Callable[[str | None], None]
ProgressCallback = Callable[[int, int], None]

PromptEmbedCacheKey = tuple[str, str]
PromptEmbedCacheValue = tuple[Tensor, Tensor, Tensor, Tensor]


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
        progress_status_callback: ProgressStatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._lora_paths = lora_paths
        self._output_dir = output_dir
        self._progress_status_callback = progress_status_callback
        self._progress_callback = progress_callback
        self._log = log or logger
        self._prompt_embed_cache: dict[PromptEmbedCacheKey, PromptEmbedCacheValue] = {}

    def run(self) -> None:
        config = self._config
        if not config.sample_prompts:
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
        stack = self._load_stack(config)

        total_diffusion_steps = self._total_diffusion_steps()
        self._set_progress(0, total_diffusion_steps)
        completed_images = 0
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
            self._sample_lora(
                lora_path=lora_path,
                status_prefix=status_prefix,
                completed_images=completed_images,
                stack=stack,
            )
            completed_images += len(config.sample_prompts)
        self._set_status(None)

    def _load_stack(self, config: TrainConfig) -> _SamplingStack:
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
        if config.text_encoder_1.train:
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
        if config.text_encoder_2.train:
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

    def _sample_lora(
        self,
        *,
        lora_path: Path,
        status_prefix: str,
        completed_images: int,
        stack: _SamplingStack,
    ) -> None:
        config = self._config
        device = stack.device
        started_at = time.perf_counter()
        unet = stack.unet
        text_encoder_1 = stack.text_encoder_1
        text_encoder_2 = stack.text_encoder_2
        unet_merged = False
        te1_merged = False
        te2_merged = False
        pipe: StableDiffusionXLPipeline | None = None
        try:
            unet.merge_adapter()
            unet_merged = True
            inference_unet = unet.base_model.model
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
            if not config.use_reforge_sampler:
                pipe = StableDiffusionXLPipeline(
                    vae=stack.vae,
                    text_encoder=inference_te1,
                    text_encoder_2=inference_te2,
                    tokenizer=stack.tokenizer_1,
                    tokenizer_2=stack.tokenizer_2,
                    unet=inference_unet,
                    scheduler=_build_inference_scheduler(config.sample_scheduler, stack.noise_scheduler),
                )
                pipe.set_progress_bar_config(disable=True)

            with torch.no_grad():
                for prompt_index, prompt in enumerate(config.sample_prompts):
                    self._set_status(f"{status_prefix} — image {prompt_index + 1}/{len(config.sample_prompts)}")
                    self._report_diffusion_progress(completed_images, prompt_index, 0)
                    generator = torch.Generator(device=device)
                    if config.seed is not None:
                        generator.manual_seed(config.seed + prompt_index)
                    image = self._generate_image(
                        prompt=prompt,
                        prompt_index=prompt_index,
                        completed_images=completed_images,
                        pipe=pipe,
                        unet=inference_unet,
                        vae=stack.vae,
                        text_encoder_1=inference_te1,
                        text_encoder_2=inference_te2,
                        tokenizer_1=stack.tokenizer_1,
                        tokenizer_2=stack.tokenizer_2,
                        noise_scheduler=stack.noise_scheduler,
                        generator=generator,
                        width=width,
                        height=height,
                        autocast_dtype=autocast_dtype,
                        device=device,
                    )
                    filename = f"{self._safe_stem(lora_path)}_{prompt_index:02d}.png"
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
            self._log.info("Sampling %s completed in %.2fs", lora_path.name, time.perf_counter() - started_at)

    def _generate_image(
        self,
        *,
        prompt: str,
        prompt_index: int,
        completed_images: int,
        pipe: StableDiffusionXLPipeline | None,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        tokenizer_1: Any,
        tokenizer_2: Any,
        noise_scheduler: Any,
        generator: torch.Generator,
        width: int,
        height: int,
        autocast_dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        config = self._config
        negative_prompt = config.sample_negative_prompt or ""
        if config.use_reforge_sampler:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self._get_cached_prompt_embeds(
                prompt=prompt,
                negative_prompt=negative_prompt,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            add_time_ids = self._get_add_time_ids(
                original_size=(height, width),
                crops_coords_top_left=(0, 0),
                target_size=(height, width),
                dtype=autocast_dtype,
                device=device,
                batch_size=1,
            )
            return generate_preview_image(
                unet=unet,
                vae=vae,
                noise_scheduler_config=noise_scheduler.config,
                sampler_name=config.sample_sampler.value,
                scheduler_mode=config.sample_scheduler_mode.value,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                add_time_ids=add_time_ids,
                width=width,
                height=height,
                num_inference_steps=config.sample_steps,
                guidance_scale=config.sample_cfg_scale,
                generator=generator,
                autocast_dtype=autocast_dtype,
                device=device,
                on_step_end=self._make_reforge_progress_callback(prompt_index, completed_images),
            )
        assert pipe is not None
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            return pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=config.sample_steps,
                guidance_scale=config.sample_cfg_scale,
                generator=generator,
                callback_on_step_end=self._make_diffusers_progress_callback(prompt_index, completed_images),
                callback_on_step_end_tensor_inputs=[],
            ).images[0]

    def _get_cached_prompt_embeds(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        tokenizer_1: Any,
        tokenizer_2: Any,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> PromptEmbedCacheValue:
        cache_key = (prompt, negative_prompt)
        cached = self._prompt_embed_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
            [prompt],
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            device,
            autocast_dtype,
        )
        negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
            [negative_prompt],
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            device,
            autocast_dtype,
        )
        value = (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        self._prompt_embed_cache[cache_key] = value
        return value

    def _encode_prompt(
        self,
        captions: list[str],
        tokenizer_1: Any,
        tokenizer_2: Any,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        tokens_1 = tokenizer_1(
            captions,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            enc1_out = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_1 = enc1_out.hidden_states[-2].to(dtype=dtype)
            enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_2 = enc2_out.hidden_states[-2].to(dtype=dtype)
            pooled_prompt_embeds = enc2_out[0].to(dtype=dtype)
        return torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1), pooled_prompt_embeds

    def _get_add_time_ids(
        self,
        original_size: tuple[int, int],
        crops_coords_top_left: tuple[int, int],
        target_size: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> Tensor:
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids] * batch_size, dtype=dtype, device=device)

    def _total_diffusion_steps(self) -> int:
        return len(self._lora_paths) * len(self._config.sample_prompts) * self._config.sample_steps

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

    def _make_reforge_progress_callback(
        self,
        prompt_index: int,
        completed_images: int,
    ) -> Callable[[int, int], None]:
        def _on_step_end(completed: int, total: int) -> None:
            self._report_diffusion_progress(completed_images, prompt_index, completed)
            self._log_sample_step(prompt_index, completed, total)

        return _on_step_end

    def _log_sample_step(self, prompt_index: int, completed: int, total: int) -> None:
        log_interval = max(1, total // 5)
        if completed % log_interval == 0 or completed == total:
            self._log.info(
                "[sample %d/%d] step %d/%d",
                prompt_index + 1,
                len(self._config.sample_prompts),
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
