"""SDXL LoRA trainer using diffusers + peft."""

import contextlib
import logging
import math
import random
from pathlib import Path
from typing import Callable, Optional

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.trainer.attention import configure_unet_attention
from src.trainer.config import Optimizer, SampleScheduler, TrainConfig, WeightDtype
from src.trainer.progress import TrainProgress
from src.trainer.sdxl.dataset import (
    ConceptDataset,
    collect_all_image_paths_and_captions,
    count_latent_cache_items,
    count_te_cache_items,
)
from src.trainer.sdxl.latent_cache import build_latent_cache
from src.trainer.sdxl.model_loader import load_sdxl_components
from src.trainer.sdxl.te_cache import build_te_cache
from src.trainer.training_log import JobTrainingLogger

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}

_SCHEDULER_MAP = {
    SampleScheduler.EULER: EulerDiscreteScheduler,
    SampleScheduler.EULER_A: EulerAncestralDiscreteScheduler,
    SampleScheduler.DDIM: DDIMScheduler,
    SampleScheduler.DPM_PP: DPMSolverMultistepScheduler,
}


def _build_inference_scheduler(
    sample_scheduler: SampleScheduler,
    noise_scheduler: DDPMScheduler,
) -> object:
    cls = _SCHEDULER_MAP[sample_scheduler]
    return cls.from_config(noise_scheduler.config)


class SDXLLoRATrainer:
    def __init__(
        self,
        config: TrainConfig,
        progress_callback: Optional[Callable[..., None]] = None,
        cache_progress_callback: Optional[Callable[[int, int], None]] = None,
        sampling_status_callback: Optional[Callable[[Optional[str]], None]] = None,
        sampling_progress_callback: Optional[Callable[[int, int], None]] = None,
        training_logger: Optional[JobTrainingLogger] = None,
    ) -> None:
        self._config = config
        self._progress_callback = progress_callback
        self._cache_progress_callback = cache_progress_callback
        self._sampling_status_callback = sampling_status_callback
        self._sampling_progress_callback = sampling_progress_callback
        self._training_logger = training_logger
        self._progress = TrainProgress()
        self._total_steps: int = 0

    def train(self) -> None:
        config = self._config
        self._validate_config(config)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is not available (torch {torch.__version__}). "
                "Install GPU-enabled PyTorch: run `uv sync` in the project root "
                "after configuring the pytorch-cu130 index in pyproject.toml."
            )
        device = torch.device("cuda")
        weight_dtype = _DTYPE_MAP[config.mixed_precision]

        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        log = self._training_logger.logger if self._training_logger is not None else logger
        log.info("Loading SDXL pipeline from %s", config.base_model_name)

        components = load_sdxl_components(
            config.base_model_name,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
        )
        tokenizer_1 = components.tokenizer_1
        tokenizer_2 = components.tokenizer_2
        noise_scheduler = components.noise_scheduler
        text_encoder_1 = components.text_encoder_1
        text_encoder_2 = components.text_encoder_2
        vae = components.vae
        unet = components.unet

        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, unet_lora_config)
        trainable_params: list = list(unet.parameters())

        if config.text_encoder_1.train:
            te1_lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            text_encoder_1 = get_peft_model(text_encoder_1, te1_lora_config)
            trainable_params += list(text_encoder_1.parameters())

        if config.text_encoder_2.train:
            te2_lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            text_encoder_2 = get_peft_model(text_encoder_2, te2_lora_config)
            trainable_params += list(text_encoder_2.parameters())

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        configure_unet_attention(unet, config.attention_mechanism, log)

        # UNet always on GPU.
        unet = unet.to(device)

        # VAE: stays on CPU when latents will be cached (build_latent_cache manages device).
        if not config.cache_latents:
            vae = vae.to(device)

        # TEs: to GPU only if they will be called during training (i.e., not cached).
        if not config.cache_text_encoder_outputs:
            text_encoder_1 = text_encoder_1.to(device)
            text_encoder_2 = text_encoder_2.to(device)
        # If training TEs (impossible when cache_text_encoder_outputs=True due to validation),
        # they need to be on GPU — already handled by the condition above.

        if config.torch_compile:
            log.info("Compiling UNet with torch.compile (inductor)...")
            unet = torch.compile(unet, backend="inductor")

        optimizer = self._build_optimizer(trainable_params, config)
        cache_mode = config.cache_latents or config.cache_text_encoder_outputs
        train_dataset = self._build_dataset(config, cache_mode=cache_mode)
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_dataloader_workers,
            pin_memory=config.dataloader_pin_memory and config.num_dataloader_workers > 0,
        )

        num_update_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
        total_steps = config.epochs * num_update_steps_per_epoch
        self._progress.epoch_total_steps = num_update_steps_per_epoch
        self._total_steps = total_steps

        all_paths: list[Path] = []
        all_pairs: list[tuple[Path, str]] = []
        cache_steps = 0
        if config.cache_latents or config.cache_text_encoder_outputs:
            all_paths, all_pairs = collect_all_image_paths_and_captions(config)
            if config.cache_latents:
                cache_steps += count_latent_cache_items(all_paths)
            if config.cache_text_encoder_outputs:
                cache_steps += count_te_cache_items(all_pairs)

        if self._training_logger is not None:
            self._training_logger.log_training_start(
                config,
                epochs=config.epochs,
                steps_per_epoch=num_update_steps_per_epoch,
                total_steps=total_steps,
            )
        elif cache_steps > 0:
            log.info("Caching phase: %d items before training", cache_steps)
        else:
            log.info("Starting training: %d epochs, %d steps/epoch", config.epochs, num_update_steps_per_epoch)

        cache_progress = 0

        def _on_cache_progress(phase_current: int, phase_total: int, phase: str) -> None:
            nonlocal cache_progress
            cache_progress += 1
            if self._training_logger is not None:
                if cache_progress == 1:
                    self._training_logger.create_progress_bar(cache_steps, desc=f"cache {phase}")
                self._training_logger.log_cache_progress(phase, phase_current, phase_total)
                self._training_logger.advance_progress(1, desc=f"cache {phase}")
            if self._cache_progress_callback is not None:
                self._cache_progress_callback(cache_progress, cache_steps)

        # --- Build caches (before the training loop) ---
        latent_cache: Optional[dict[str, Tensor]] = None
        te_cache: Optional[dict[str, tuple[Tensor, Tensor]]] = None

        if config.cache_latents:
            log.info("Building latent cache...")
            latent_cache = build_latent_cache(
                all_paths,
                config.resolution,
                vae,
                device,
                config.cache_latents_to_disk,
                on_progress=_on_cache_progress if cache_steps > 0 else None,
            )

        if config.cache_text_encoder_outputs:
            log.info("Building text encoder cache...")
            te_cache = build_te_cache(
                all_pairs,
                tokenizer_1,
                tokenizer_2,
                text_encoder_1,
                text_encoder_2,
                device,
                weight_dtype,
                config.cache_text_encoder_outputs_to_disk,
                on_progress=_on_cache_progress if cache_steps > 0 else None,
            )

        if self._training_logger is not None:
            if cache_steps > 0:
                self._training_logger.close_progress_bar()
                self._training_logger.logger.info("Caching complete, starting training")
            self._training_logger.create_progress_bar(total_steps, desc="steps")

        self._save_config(config)

        from diffusers.optimization import get_scheduler
        lr_scheduler = get_scheduler(
            config.lr_scheduler.value,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
            num_training_steps=total_steps * config.gradient_accumulation_steps,
        )

        try:
            if config.sample_before_training:
                log.info("Running pre-training samples (epoch 0)...")
                self._run_sampling(
                    0, unet, text_encoder_1, text_encoder_2, vae,
                    tokenizer_1, tokenizer_2, noise_scheduler, config, device,
                    self._training_logger, self._sampling_status_callback,
                    self._sampling_progress_callback,
                )

            for epoch in range(config.epochs):
                self._progress.next_epoch()
                if self._training_logger is not None:
                    self._training_logger.log_epoch(epoch + 1, config.epochs)
                unet.train()
                if config.text_encoder_1.train:
                    text_encoder_1.train()
                if config.text_encoder_2.train:
                    text_encoder_2.train()

                accumulated_loss = 0.0
                optimizer.zero_grad()

                for step, batch in enumerate(dataloader):
                    captions: list[str] = batch["caption"]

                    # --- Latents ---
                    if latent_cache is not None:
                        image_paths: list[str] = batch["image_path"]
                        latents = torch.stack(
                            [latent_cache[p].to(device, dtype=weight_dtype) for p in image_paths]
                        )
                    else:
                        pixel_values = batch["pixel_values"].to(device)
                        with torch.no_grad():
                            latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor
                            latents = latents.to(dtype=weight_dtype)

                    # --- Noise + timesteps ---
                    with torch.no_grad():
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                        ).long()
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        add_time_ids = self._get_add_time_ids(
                            original_size=(config.resolution, config.resolution),
                            crops_coords_top_left=(0, 0),
                            target_size=(config.resolution, config.resolution),
                            dtype=weight_dtype,
                            device=device,
                            batch_size=bsz,
                        )

                    # --- Text embeddings ---
                    if te_cache is not None:
                        prompt_embeds = torch.cat(
                            [te_cache[c][0].to(device, dtype=weight_dtype) for c in captions], dim=0
                        )
                        pooled_prompt_embeds = torch.cat(
                            [te_cache[c][1].to(device, dtype=weight_dtype) for c in captions], dim=0
                        )
                    else:
                        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
                            captions,
                            tokenizer_1,
                            tokenizer_2,
                            text_encoder_1,
                            text_encoder_2,
                            device,
                            weight_dtype,
                            train_te1=config.text_encoder_1.train,
                            train_te2=config.text_encoder_2.train,
                        )

                    # --- UNet forward + loss ---
                    with torch.autocast(device_type=device.type, dtype=weight_dtype):
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                        ).sample

                    target = (
                        noise
                        if noise_scheduler.config.prediction_type == "epsilon"
                        else noise_scheduler.get_velocity(latents, noise, timesteps)
                    )
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                    accumulated_loss += loss.item()

                    if (step + 1) % config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        self._progress.next_step(accumulated_loss)
                        current_lr = lr_scheduler.get_last_lr()[0]
                        avr_loss = accumulated_loss
                        if self._training_logger is not None:
                            avr_loss = self._training_logger.log_step(
                                step=self._progress.global_step,
                                total_steps=total_steps,
                                loss=accumulated_loss,
                                lr=current_lr,
                                epoch=epoch + 1,
                                epoch_total=config.epochs,
                                epoch_step=self._progress.epoch_step,
                            )
                        accumulated_loss = 0.0
                        if self._progress_callback is not None:
                            self._progress_callback(
                                self._progress.global_step,
                                total_steps,
                                self._progress.loss,
                                avr_loss,
                                epoch + 1,
                                config.epochs,
                                current_lr,
                            )

                if (epoch + 1) % config.save_every_n_epochs == 0:
                    self._save_checkpoint(unet, text_encoder_1, text_encoder_2, config, epoch + 1, log)

                if (
                    config.sample_every_n_epochs is not None
                    and (epoch + 1) % config.sample_every_n_epochs == 0
                ):
                    self._run_sampling(
                        epoch + 1, unet, text_encoder_1, text_encoder_2, vae,
                        tokenizer_1, tokenizer_2, noise_scheduler, config, device,
                        self._training_logger, self._sampling_status_callback,
                        self._sampling_progress_callback,
                    )

            self._save_final(unet, text_encoder_1, text_encoder_2, config)
            log.info("Training complete. Output dir: %s", self._work_dir(config))
        finally:
            if self._training_logger is not None:
                self._training_logger.close_progress_bar()

    @staticmethod
    def _validate_config(config: TrainConfig) -> None:
        if config.cache_text_encoder_outputs and (
            config.text_encoder_1.train or config.text_encoder_2.train
        ):
            raise ValueError(
                "cache_text_encoder_outputs=True is incompatible with training text encoders. "
                "Set cache_text_encoder_outputs=False or disable text encoder training."
            )

    def _encode_prompt(
        self,
        captions: list[str],
        tokenizer_1,
        tokenizer_2,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        train_te1: bool,
        train_te2: bool,
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

        # Use no_grad for frozen encoders; enable_grad for trained ones so LoRA receives gradients.
        te1_ctx = contextlib.nullcontext() if train_te1 else torch.no_grad()
        te2_ctx = contextlib.nullcontext() if train_te2 else torch.no_grad()

        with te1_ctx:
            enc1_out = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_1 = enc1_out.hidden_states[-2].to(dtype=dtype)

        with te2_ctx:
            enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_2 = enc2_out.hidden_states[-2].to(dtype=dtype)
            pooled_prompt_embeds = enc2_out[0].to(dtype=dtype)

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

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

    def _build_optimizer(self, params: list, config: TrainConfig):
        lr = config.learning_rate
        if config.optimizer == Optimizer.ADAMW:
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
        if config.optimizer == Optimizer.ADAMW_8BIT:
            from bitsandbytes.optim import AdamW8bit
            return AdamW8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
        if config.optimizer == Optimizer.ADAFACTOR:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False)
        if config.optimizer == Optimizer.PRODIGY:
            from prodigyopt import Prodigy
            return Prodigy(params, lr=1.0, weight_decay=1e-2)
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)

    def _build_dataset(self, config: TrainConfig, cache_mode: bool = False) -> Dataset:
        datasets = [ConceptDataset(c, config.resolution, cache_mode=cache_mode) for c in config.concepts]
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _work_dir(self, config: TrainConfig) -> Path:
        return Path(config.output_dir) / config.lora_name

    def _save_config(self, config: TrainConfig) -> None:
        work_dir = self._work_dir(config)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = work_dir / f"{config.lora_name}_config.yaml"
        config_path.write_text(config.to_yaml(), encoding="utf-8")

    def _save_checkpoint(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
        epoch: int,
        log: logging.Logger,
    ) -> None:
        ext = f".{config.output_format.value}"
        checkpoint_path = self._work_dir(config) / f"{config.lora_name}_epoch{epoch}{ext}"
        self._export_lora(unet, text_encoder_1, text_encoder_2, config, checkpoint_path)
        log.info("Checkpoint saved to %s", checkpoint_path)

    def _save_final(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
    ) -> None:
        ext = f".{config.output_format.value}"
        final_path = self._work_dir(config) / f"{config.lora_name}{ext}"
        self._export_lora(unet, text_encoder_1, text_encoder_2, config, final_path)

    def _run_sampling(
        self,
        epoch: int,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        vae: torch.nn.Module,
        tokenizer_1: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        noise_scheduler: DDPMScheduler,
        config: TrainConfig,
        device: torch.device,
        training_logger: Optional[JobTrainingLogger],
        sampling_status_callback: Optional[Callable[[Optional[str]], None]] = None,
        sampling_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if not config.sample_prompts:
            return

        log = training_logger.logger if training_logger is not None else logger

        sample_dir = self._work_dir(config) / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        te1_device = next(text_encoder_1.parameters()).device
        te2_device = next(text_encoder_2.parameters()).device
        vae_device = next(vae.parameters()).device

        if te1_device.type == "cpu":
            text_encoder_1 = text_encoder_1.to(device)
        if te2_device.type == "cpu":
            text_encoder_2 = text_encoder_2.to(device)
        if vae_device.type == "cpu":
            vae = vae.to(device)

        # Merge LoRA adapter weights into the base model so the pipeline receives
        # a plain UNet2DConditionModel with no PEFT wrapping. Passing a PeftModel
        # to StableDiffusionXLPipeline deadlocks on PyTorch 2.12 / Windows.
        unet.merge_adapter()
        inference_unet = unet.base_model.model

        if config.text_encoder_1.train:
            text_encoder_1.merge_adapter()
            inference_te1 = text_encoder_1.base_model.model
        else:
            inference_te1 = text_encoder_1

        if config.text_encoder_2.train:
            text_encoder_2.merge_adapter()
            inference_te2 = text_encoder_2.base_model.model
        else:
            inference_te2 = text_encoder_2

        inference_unet_dtype = _DTYPE_MAP[config.unet.weight_dtype]
        inference_te1_dtype = _DTYPE_MAP[config.text_encoder_1.weight_dtype]
        inference_te2_dtype = _DTYPE_MAP[config.text_encoder_2.weight_dtype]
        autocast_dtype = _DTYPE_MAP[config.mixed_precision]

        inference_unet = inference_unet.to(device=device, dtype=inference_unet_dtype)
        inference_te1 = inference_te1.to(device=device, dtype=inference_te1_dtype)
        inference_te2 = inference_te2.to(device=device, dtype=inference_te2_dtype)
        vae = vae.to(device=device, dtype=torch.float32)

        inference_scheduler = _build_inference_scheduler(config.sample_scheduler, noise_scheduler)
        pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=inference_te1,
            text_encoder_2=inference_te2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=inference_unet,
            scheduler=inference_scheduler,
        )
        pipe.to(device)
        unet_training = unet.training
        te1_training = text_encoder_1.training
        te2_training = text_encoder_2.training
        inference_unet.eval()
        inference_te1.eval()
        inference_te2.eval()

        width = config.sample_width or config.resolution
        height = config.sample_height or config.resolution

        n_prompts = len(config.sample_prompts)
        log.info("Sampling %d image(s) for epoch %d...", n_prompts, epoch)

        if training_logger is not None:
            training_logger.close_progress_bar()

        torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                for i, prompt in enumerate(config.sample_prompts):
                    if sampling_status_callback is not None:
                        sampling_status_callback(f"Sampling epoch {epoch} — image {i + 1}/{n_prompts}")
                    if sampling_progress_callback is not None:
                        sampling_progress_callback(0, config.sample_steps)

                    pipe.set_progress_bar_config(disable=True)
                    generator = torch.Generator(device=device)
                    if config.seed is not None:
                        generator.manual_seed(config.seed + i)

                    log_interval = max(1, config.sample_steps // 5)

                    def _on_step_end(pipeline, step_index: int, timestep, callback_kwargs: dict) -> dict:
                        completed = step_index + 1
                        if sampling_progress_callback is not None:
                            sampling_progress_callback(completed, config.sample_steps)
                        if completed % log_interval == 0 or completed == config.sample_steps:
                            print(
                                f"[sample {i + 1}/{n_prompts} e{epoch}] step {completed}/{config.sample_steps}",
                                flush=True,
                            )
                        return callback_kwargs

                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        image = pipe(
                            prompt=prompt,
                            negative_prompt=config.sample_negative_prompt or None,
                            width=width,
                            height=height,
                            num_inference_steps=config.sample_steps,
                            guidance_scale=config.sample_cfg_scale,
                            generator=generator,
                            callback_on_step_end=_on_step_end,
                            callback_on_step_end_tensor_inputs=[],
                        ).images[0]
                    filename = f"{config.lora_name}_epoch{epoch}_{i:02d}.png"
                    image.save(sample_dir / filename)
                    log.info("Sample saved: %s", filename)
        finally:
            if sampling_status_callback is not None:
                sampling_status_callback(None)
            if training_logger is not None:
                training_logger.create_progress_bar(
                    self._total_steps,
                    initial=self._progress.global_step,
                    desc="steps",
                )
            # Restore LoRA adapters to separate state (unmerge from base weights).
            unet.unmerge_adapter()
            if config.text_encoder_1.train:
                text_encoder_1.unmerge_adapter()
            if config.text_encoder_2.train:
                text_encoder_2.unmerge_adapter()
            # Restore training modes on the PEFT wrappers (not the extracted base models).
            if unet_training:
                unet.train()
            if te1_training:
                text_encoder_1.train()
            if te2_training:
                text_encoder_2.train()
            if te1_device.type == "cpu":
                text_encoder_1.to("cpu")
            if te2_device.type == "cpu":
                text_encoder_2.to("cpu")
            if vae_device.type == "cpu":
                vae.to("cpu")

    def _export_lora(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file  # noqa: PLC0415

        state_dict: dict[str, Tensor] = {}
        for name, param in unet.named_parameters():
            if "lora_" in name and param.requires_grad:
                state_dict[f"lora_unet_{name.replace('.', '_')}"] = param.data.cpu()

        if config.text_encoder_1.train:
            for name, param in text_encoder_1.named_parameters():
                if "lora_" in name and param.requires_grad:
                    state_dict[f"lora_te1_{name.replace('.', '_')}"] = param.data.cpu()

        if config.text_encoder_2.train:
            for name, param in text_encoder_2.named_parameters():
                if "lora_" in name and param.requires_grad:
                    state_dict[f"lora_te2_{name.replace('.', '_')}"] = param.data.cpu()

        if config.output_format.value == "safetensors":
            save_file(state_dict, str(path))
        else:
            torch.save(state_dict, str(path))
