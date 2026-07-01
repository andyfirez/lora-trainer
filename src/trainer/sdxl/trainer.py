"""SDXL LoRA trainer using diffusers + peft."""

import contextlib
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from diffusers import DDPMScheduler
from peft import get_peft_model
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.trainer.attention import configure_unet_attention
from src.trainer.config import TrainConfig, WeightDtype
from src.trainer.optimizer_config import Optimizer, build_optimizer
from src.trainer.progress import TrainProgress
from src.trainer.concept_training_metadata import ConceptTrainingMetadata
from src.trainer.sdxl.bucket_batch_sampler import build_bucket_batch_sampler
from src.trainer.sdxl.dataset import (
    build_training_dataset,
    collect_all_image_paths_and_captions,
    collect_bucket_keys,
    count_latent_cache_items,
    count_te_cache_items,
)
from src.trainer.sdxl.checkpoint_state import load_resume_state, save_resume_state
from src.trainer.sdxl.latent_cache import build_latent_cache
from src.trainer.sdxl.loss import apply_noise_offset, min_snr_weight
from src.trainer.sdxl.lora_export import export_kohya_state_dict
from src.trainer.sdxl.lora_io import apply_lora_state_dict, apply_lora_state_to_module
from src.trainer.sdxl.mixed_precision import cast_trainable_params_to_fp32, create_grad_scaler
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config
from src.trainer.sdxl.lora_targets import SDXL_TE_LORA_TARGET_MODULES, SDXL_UNET_LORA_TARGET_MODULES
from src.trainer.sdxl.model_loader import load_sdxl_components, resolve_vae_dtype
from src.trainer.sdxl.prompt_encoding import select_clip_hidden_state
from src.trainer.sdxl.latent_sampling import SDXLSamplingSession, run_sdxl_sampling_pass
from src.trainer.sdxl.sampling import (
    PromptEmbedCache,
    build_inference_scheduler,
    precompute_all_sample_embeds,
)
from src.trainer.sdxl.te_cache import build_te_cache
from src.trainer.training_log import JobTrainingLogger

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}


class TrainingCancelledAfterSave(Exception):
    """Raised when cancellation with save-checkpoint was requested."""


class SDXLLoRATrainer:
    def __init__(
        self,
        config: TrainConfig,
        progress_callback: Optional[Callable[..., None]] = None,
        sampling_status_callback: Optional[Callable[[Optional[str]], None]] = None,
        sampling_progress_callback: Optional[Callable[[int, int], None]] = None,
        training_logger: Optional[JobTrainingLogger] = None,
        checkpoint_callback: Optional[Callable[[str, int, int], None]] = None,
        save_checkpoint_requested_callback: Optional[Callable[[], bool]] = None,
        concept_metadata: Optional[dict[int, ConceptTrainingMetadata]] = None,
    ) -> None:
        self._config = config
        self._concept_metadata = concept_metadata or {}
        self._progress_callback = progress_callback
        self._sampling_status_callback = sampling_status_callback
        self._sampling_progress_callback = sampling_progress_callback
        self._training_logger = training_logger
        self._checkpoint_callback = checkpoint_callback
        self._save_checkpoint_requested_callback = save_checkpoint_requested_callback
        self._progress = TrainProgress()
        self._total_steps: int = 0
        self._optimizer: Optional[object] = None
        self._device: Optional[torch.device] = None

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
        grad_scaler = create_grad_scaler(config.mixed_precision)

        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        log = self._training_logger.logger if self._training_logger is not None else logger
        log.info("Loading SDXL pipeline from %s", config.base_model_name)
        resume_state = None
        start_epoch = 0
        start_step = 0
        if config.resume_from_checkpoint:
            resume_state = load_resume_state(Path(config.resume_from_checkpoint))
            start_epoch = resume_state.epoch
            start_step = resume_state.global_step
            log.info(
                "Resuming from checkpoint %s (epoch_index=%d, global_step=%d)",
                config.resume_from_checkpoint,
                start_epoch,
                start_step,
            )

        components = load_sdxl_components(
            config.base_model_name,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
            vae_dtype=config.vae_dtype,
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

        unet_lora_config = build_sdxl_lora_config(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=SDXL_UNET_LORA_TARGET_MODULES,
        )
        unet = get_peft_model(unet, unet_lora_config)
        trainable_params: list = list(unet.parameters())

        if config.text_encoder_1.train:
            te1_lora_config = build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_TE_LORA_TARGET_MODULES,
            )
            text_encoder_1 = get_peft_model(text_encoder_1, te1_lora_config)
            trainable_params += list(text_encoder_1.parameters())

        if config.text_encoder_2.train:
            te2_lora_config = build_sdxl_lora_config(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=SDXL_TE_LORA_TARGET_MODULES,
            )
            text_encoder_2 = get_peft_model(text_encoder_2, te2_lora_config)
            trainable_params += list(text_encoder_2.parameters())

        if resume_state is not None:
            self._load_lora_state_dict(
                resume_state.lora_state_dict,
                unet=unet,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                config=config,
            )

        fp32_cast_count = cast_trainable_params_to_fp32(unet, text_encoder_1, text_encoder_2)
        log.info("LoRA trainable params cast to fp32 (N=%d)", fp32_cast_count)

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

        optimizer = build_optimizer(trainable_params, config)
        self._optimizer = optimizer
        self._device = device
        cache_mode = config.cache_latents or config.cache_text_encoder_outputs
        train_dataset = self._build_dataset(config, cache_mode=cache_mode)
        if config.enable_bucket:
            bucket_sampler = build_bucket_batch_sampler(
                train_dataset,
                collect_bucket_keys(train_dataset),
                config.batch_size,
            )
            dataloader = DataLoader(
                train_dataset,
                batch_sampler=bucket_sampler,
                num_workers=config.num_dataloader_workers,
                pin_memory=config.dataloader_pin_memory and config.num_dataloader_workers > 0,
            )
        else:
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

        # --- Build caches (before the training loop) ---
        latent_cache: Optional[dict[str, Tensor]] = None
        te_cache: Optional[dict[str, tuple[Tensor, Tensor]]] = None

        if config.cache_latents:
            log.info("Building latent cache...")
            latent_cache = build_latent_cache(
                all_paths,
                vae,
                device,
                config.cache_latents_to_disk,
                on_progress=_on_cache_progress if cache_steps > 0 else None,
                log=log,
            )

        if config.cache_text_encoder_outputs_to_disk:
            log.warning(
                "Text encoder disk cache is enabled. Delete *_te.npz files next to images "
                "after changing trigger_words, captions, or clip_skip."
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
                config.clip_skip,
                config.cache_text_encoder_outputs_to_disk,
                on_progress=_on_cache_progress if cache_steps > 0 else None,
                log=log,
            )

        if self._training_logger is not None:
            if cache_steps > 0:
                self._training_logger.close_progress_bar()
                self._training_logger.logger.info("Caching complete, starting training")
            self._training_logger.create_progress_bar(total_steps, desc="steps")

        if self._progress_callback is not None:
            self._progress_callback(
                start_step,
                total_steps,
                0.0,
                0.0,
                start_epoch,
                config.epochs,
                0.0,
            )

        self._save_config(config)

        from diffusers.optimization import get_scheduler
        lr_scheduler = get_scheduler(
            config.lr_scheduler.value,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
            num_training_steps=total_steps * config.gradient_accumulation_steps,
        )
        if resume_state is not None:
            optimizer.load_state_dict(resume_state.optimizer_state_dict)
            lr_scheduler.load_state_dict(resume_state.lr_scheduler_state_dict)
            if grad_scaler is not None and resume_state.grad_scaler_state_dict is not None:
                grad_scaler.load_state_dict(resume_state.grad_scaler_state_dict)
            self._progress.global_step = resume_state.global_step
            self._progress.epoch_step = resume_state.epoch_step

        try:
            for epoch in range(start_epoch, config.epochs):
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

                skip_batches = 0
                if (
                    resume_state is not None
                    and epoch == start_epoch
                    and resume_state.epoch_step > 0
                ):
                    skip_batches = resume_state.epoch_step * config.gradient_accumulation_steps

                for step, batch in enumerate(dataloader):
                    if step < skip_batches:
                        continue
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
                        noise = apply_noise_offset(latents, noise, config.noise_offset)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                        ).long()
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        add_time_ids = batch["add_time_ids"].to(dtype=weight_dtype, device=device)

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
                            config.clip_skip,
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
                    per_sample_loss = torch.nn.functional.mse_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="none",
                    ).mean(dim=(1, 2, 3))
                    if config.min_snr_gamma > 0:
                        v_prediction = noise_scheduler.config.prediction_type == "v_prediction"
                        snr_weights = min_snr_weight(
                            timesteps,
                            noise_scheduler.alphas_cumprod,
                            config.min_snr_gamma,
                            v_prediction=v_prediction,
                        )
                        per_sample_loss = per_sample_loss * snr_weights.to(device=per_sample_loss.device)
                    loss = per_sample_loss.mean() / config.gradient_accumulation_steps
                    if grad_scaler is not None:
                        grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    accumulated_loss += loss.item()

                    if (step + 1) % config.gradient_accumulation_steps == 0:
                        if grad_scaler is not None:
                            grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        if grad_scaler is not None:
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                        else:
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
                        if (
                            self._save_checkpoint_requested_callback is not None
                            and self._save_checkpoint_requested_callback()
                        ):
                            checkpoint_path = self._save_checkpoint(
                                unet,
                                text_encoder_1,
                                text_encoder_2,
                                optimizer,
                                lr_scheduler,
                                config,
                                epoch=epoch + 1,
                                resume_epoch_index=epoch,
                                checkpoint_step=self._progress.global_step,
                                epoch_step=self._progress.epoch_step,
                                checkpoint_name=f"{config.lora_name}_step{self._progress.global_step}",
                                log=log,
                                grad_scaler=grad_scaler,
                            )
                            log.info("Cancellation requested with checkpoint save: %s", checkpoint_path)
                            raise TrainingCancelledAfterSave()

                if config.checkpointing_enabled and (epoch + 1) % config.save_every_n_epochs == 0:
                    self._save_checkpoint(
                        unet,
                        text_encoder_1,
                        text_encoder_2,
                        optimizer,
                        lr_scheduler,
                        config,
                        epoch=epoch + 1,
                        resume_epoch_index=epoch + 1,
                        checkpoint_step=self._progress.global_step,
                        epoch_step=0,
                        checkpoint_name=f"{config.lora_name}_epoch{epoch + 1}",
                        log=log,
                        grad_scaler=grad_scaler,
                    )

            self._save_final(unet, text_encoder_1, text_encoder_2, config)
            log.info("Training complete. Output dir: %s", self._work_dir(config))
        finally:
            if self._training_logger is not None:
                self._training_logger.close_progress_bar()

    @staticmethod
    def _validate_config(config: TrainConfig) -> None:
        config.validate_gpu()
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
        clip_skip: int,
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
            prompt_embeds_1 = select_clip_hidden_state(enc1_out.hidden_states, clip_skip).to(dtype=dtype)

        with te2_ctx:
            enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_2 = select_clip_hidden_state(enc2_out.hidden_states, clip_skip).to(dtype=dtype)
            pooled_prompt_embeds = enc2_out[0].to(dtype=dtype)

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def _build_dataset(self, config: TrainConfig, cache_mode: bool = False) -> Dataset:
        return build_training_dataset(
            config,
            cache_mode=cache_mode,
            concept_metadata=self._concept_metadata,
        )

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
        optimizer: Any,
        lr_scheduler: Any,
        config: TrainConfig,
        epoch: int,
        resume_epoch_index: int,
        checkpoint_step: int,
        epoch_step: int,
        checkpoint_name: str,
        log: logging.Logger,
        grad_scaler: Any | None = None,
    ) -> Path:
        ext = f".{config.output_format.value}"
        checkpoint_path = self._work_dir(config) / f"{checkpoint_name}{ext}"
        self._export_lora(unet, text_encoder_1, text_encoder_2, config, checkpoint_path)
        lora_state_dict = self._collect_lora_state_dict(unet, text_encoder_1, text_encoder_2, config)
        grad_scaler_state_dict = grad_scaler.state_dict() if grad_scaler is not None else None
        save_resume_state(
            checkpoint_path=checkpoint_path,
            lora_state_dict=lora_state_dict,
            optimizer_state_dict=optimizer.state_dict(),
            lr_scheduler_state_dict=lr_scheduler.state_dict(),
            epoch=resume_epoch_index,
            global_step=checkpoint_step,
            epoch_step=epoch_step,
            grad_scaler_state_dict=grad_scaler_state_dict,
        )
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(str(checkpoint_path), epoch, checkpoint_step)
        log.info("Checkpoint saved to %s", checkpoint_path)
        return checkpoint_path

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

    def _offload_to_cpu(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
        log: logging.Logger,
    ) -> None:
        log.info("Offloading training state to CPU for sampling...")
        unet.to("cpu")
        if not config.cache_text_encoder_outputs:
            text_encoder_1.to("cpu")
            text_encoder_2.to("cpu")
        if config.optimizer.type != Optimizer.ADAMW_8BIT and self._optimizer is not None:
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        log.info("VRAM freed. Available: %.1f GB", free_gb)

    def _restore_to_gpu(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
        log: logging.Logger,
    ) -> None:
        assert self._device is not None
        log.info("Restoring training state to GPU...")
        unet.to(self._device)
        if not config.cache_text_encoder_outputs:
            text_encoder_1.to(self._device)
            text_encoder_2.to(self._device)
        if config.optimizer.type != Optimizer.ADAMW_8BIT and self._optimizer is not None:
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._device)
        log.info("Training state restored to GPU.")

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
        sampling_started_at = time.perf_counter()

        if training_logger is not None:
            training_logger.close_progress_bar()

        sample_dir = self._work_dir(config) / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_prompts = config.sample_prompts
        n_prompts = len(sample_prompts)
        log.info("Sampling %d image(s) for epoch %d...", n_prompts, epoch)

        if sampling_status_callback is not None:
            sampling_status_callback(f"Sampling epoch {epoch} — preparing")

        vae_device = next(vae.parameters()).device
        te1_device = next(text_encoder_1.parameters()).device
        te2_device = next(text_encoder_2.parameters()).device

        offload_started_at = time.perf_counter()
        self._offload_to_cpu(unet, text_encoder_1, text_encoder_2, config, log)
        log.info("[sampling e%d] offload: %.2fs", epoch, time.perf_counter() - offload_started_at)

        inference_unet: Optional[torch.nn.Module] = None
        inference_te1: Optional[torch.nn.Module] = None
        inference_te2: Optional[torch.nn.Module] = None
        unet_merged = False
        te1_merged = False
        te2_merged = False

        try:
            merge_started_at = time.perf_counter()
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
            log.info("[sampling e%d] merge adapters: %.2fs", epoch, time.perf_counter() - merge_started_at)

            build_started_at = time.perf_counter()
            inference_unet = inference_unet.to(device=device, dtype=_DTYPE_MAP[config.unet.weight_dtype])
            inference_te1 = inference_te1.to(device=device, dtype=_DTYPE_MAP[config.text_encoder_1.weight_dtype])
            inference_te2 = inference_te2.to(device=device, dtype=_DTYPE_MAP[config.text_encoder_2.weight_dtype])
            vae = vae.to(device=device, dtype=resolve_vae_dtype(config.vae_dtype))

            width = config.sample_width or config.resolution
            height = config.sample_height or config.resolution
            autocast_dtype = _DTYPE_MAP[config.mixed_precision]

            inference_scheduler = build_inference_scheduler(config.sample_scheduler, noise_scheduler)
            inference_unet.eval()
            inference_te1.eval()
            inference_te2.eval()
            log.info("[sampling e%d] move inference models: %.2fs", epoch, time.perf_counter() - build_started_at)

            embed_started_at = time.perf_counter()
            negative_prompt = config.sample_negative_prompt or ""
            prompt_embed_cache = PromptEmbedCache()
            all_embeds = precompute_all_sample_embeds(
                sample_prompts=sample_prompts,
                negative_prompt=negative_prompt,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                text_encoder_1=inference_te1,
                text_encoder_2=inference_te2,
                device=device,
                dtype=autocast_dtype,
                clip_skip=config.clip_skip,
                cache=prompt_embed_cache,
            )
            inference_te1.to("cpu")
            inference_te2.to("cpu")
            torch.cuda.empty_cache()
            log.info("[sampling e%d] precompute embeds + TE offload: %.2fs", epoch, time.perf_counter() - embed_started_at)

            session = SDXLSamplingSession.create(
                unet=inference_unet,
                vae=vae,
                scheduler=inference_scheduler,
                device=device,
                width=width,
                height=height,
                sample_steps=config.sample_steps,
                autocast_dtype=autocast_dtype,
                config=config,
            )

            def _on_status(prompt_index: int, total_prompts: int) -> None:
                if sampling_status_callback is not None:
                    sampling_status_callback(
                        f"Sampling epoch {epoch} — image {prompt_index + 1}/{total_prompts}",
                    )

            def _on_step(_prompt_index: int, completed: int, total: int) -> None:
                if sampling_progress_callback is not None:
                    sampling_progress_callback(completed, total)

            run_sdxl_sampling_pass(
                session=session,
                embeds_list=all_embeds,
                config=config,
                output_dir=sample_dir,
                output_stem=f"{config.lora_name}_epoch{epoch}",
                log=log,
                on_status=_on_status,
                on_step=_on_step,
                log_step_context=f"[sample {{prompt_index}}/{{n_prompts}} e{epoch}]",
            )
        finally:
            restore_started_at = time.perf_counter()
            if inference_unet is not None:
                inference_unet.to("cpu")
            if inference_te1 is not None:
                inference_te1.to(te1_device)
            if inference_te2 is not None:
                inference_te2.to(te2_device)
            vae.to(vae_device)
            if unet_merged:
                unet.unmerge_adapter()
            if te1_merged:
                text_encoder_1.unmerge_adapter()
            if te2_merged:
                text_encoder_2.unmerge_adapter()
            self._restore_to_gpu(unet, text_encoder_1, text_encoder_2, config, log)
            log.info("[sampling e%d] restore: %.2fs", epoch, time.perf_counter() - restore_started_at)
            log.info("[sampling e%d] total: %.2fs", epoch, time.perf_counter() - sampling_started_at)
            if sampling_status_callback is not None:
                sampling_status_callback(None)
            if training_logger is not None:
                training_logger.create_progress_bar(
                    self._total_steps,
                    initial=self._progress.global_step,
                    desc="steps",
                )

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

        state_dict = export_kohya_state_dict(unet, text_encoder_1, text_encoder_2, config)
        if config.output_format.value == "safetensors":
            save_file(state_dict, str(path))
        else:
            torch.save(state_dict, str(path))

    def _collect_lora_state_dict(
        self,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
    ) -> dict[str, Tensor]:
        return export_kohya_state_dict(unet, text_encoder_1, text_encoder_2, config)

    def _load_lora_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        unet: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        config: TrainConfig,
    ) -> None:
        apply_lora_state_dict(
            state_dict,
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            config=config,
        )

    def _apply_lora_state_to_module(
        self,
        module: torch.nn.Module,
        state_dict: dict[str, Any],
        *,
        prefix: str,
    ) -> None:
        apply_lora_state_to_module(module, state_dict, prefix=prefix)
