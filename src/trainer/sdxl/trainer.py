"""SDXL LoRA trainer using diffusers + peft."""

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.storage.config_paths import resolve_config_base_model
from src.trainer.attention import configure_unet_attention
from src.trainer.concept_training_metadata import ConceptTrainingMetadata
from src.trainer.config import TrainConfig
from src.gpu.runtime import CudaRuntime, setup_cuda_runtime
from src.trainer.optimizer_config import build_optimizer
from src.trainer.progress import TrainProgress
from src.trainer.sdxl.bucket_batch_sampler import build_bucket_batch_sampler
from src.trainer.sdxl.checkpoint_state import (
    delete_all_resume_states,
    load_resume_state,
    prune_stale_resume_states,
    save_resume_state,
)
from src.trainer.sdxl.dataset import (
    build_training_dataset,
    collect_all_image_paths_and_captions,
    collect_bucket_keys,
    count_latent_cache_items,
    count_te_cache_items,
)
from src.trainer.sdxl.latent_cache import build_latent_cache
from src.trainer.sdxl.lora_peft import attach_sdxl_lora_adapters
from src.trainer.sdxl.lora_persistence import (
    collect_lora_state_dict,
    export_lora_weights,
    load_lora_state_dict,
)
from src.trainer.sdxl.loss import apply_noise_offset, min_snr_weight
from src.trainer.sdxl.mixed_precision import (
    cast_trainable_params_to_fp32,
    create_grad_scaler,
)
from src.trainer.sdxl.model_loader import load_sdxl_components
from src.trainer.sdxl.prompt_encoding import encode_sdxl_prompt
from src.trainer.sdxl.te_cache import build_te_cache
from src.trainer.training_log import JobTrainingLogger, resolve_part_learning_rates

logger = logging.getLogger(__name__)


class TrainingCancelledAfterSave(Exception):
    """Raised when cancellation with save-checkpoint was requested."""


class TrainingCancelledDuringCache(Exception):
    """Raised when stop was requested during cache/setup before the training loop."""


@dataclass
class _TrainingContext:
    config: TrainConfig
    runtime: CudaRuntime
    log: logging.Logger
    grad_scaler: Any | None
    tokenizer_1: Any
    tokenizer_2: Any
    noise_scheduler: Any
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module
    vae: torch.nn.Module
    unet: torch.nn.Module
    optimizer: Any
    trainable_params: list
    dataloader: DataLoader
    total_steps: int
    num_update_steps_per_epoch: int
    start_epoch: int
    start_step: int
    resume_state: Any | None
    latent_cache: Optional[dict[str, Tensor]] = None
    te_cache: Optional[dict[str, tuple[Tensor, Tensor]]] = None
    cache_steps: int = 0
    all_paths: list[Path] = field(default_factory=list)
    all_pairs: list[tuple[Path, str]] = field(default_factory=list)


class SDXLLoRATrainer:
    def __init__(
        self,
        config: TrainConfig,
        progress_callback: Optional[Callable[..., None]] = None,
        training_logger: Optional[JobTrainingLogger] = None,
        checkpoint_callback: Optional[Callable[[str, int, int], None]] = None,
        save_checkpoint_requested_callback: Optional[Callable[[], bool]] = None,
        stop_requested_callback: Optional[Callable[[], bool]] = None,
        concept_metadata: Optional[dict[int, ConceptTrainingMetadata]] = None,
    ) -> None:
        self._config = config
        self._concept_metadata = concept_metadata or {}
        self._progress_callback = progress_callback
        self._training_logger = training_logger
        self._checkpoint_callback = checkpoint_callback
        self._save_checkpoint_requested_callback = save_checkpoint_requested_callback
        self._stop_requested_callback = stop_requested_callback
        self._progress = TrainProgress()
        self._total_steps: int = 0
        self._optimizer: Optional[object] = None
        self._device: Optional[torch.device] = None

    def train(self) -> None:
        ctx = self._setup_training_context()
        try:
            self._build_caches(ctx)
            self._run_training_loop(ctx)
            self._save_final(ctx.unet, ctx.text_encoder_1, ctx.text_encoder_2, ctx.config)
            delete_all_resume_states(self._work_dir(ctx.config))
            ctx.log.info("Training complete. Output dir: %s", self._work_dir(ctx.config))
        finally:
            if self._training_logger is not None:
                self._training_logger.close_progress_bar()

    def _setup_training_context(self) -> _TrainingContext:
        config = self._config
        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)

        runtime = setup_cuda_runtime(config)
        device = runtime.device
        gpu = runtime.gpu
        grad_scaler = create_grad_scaler(gpu.mixed_precision)

        log = self._training_logger.logger if self._training_logger is not None else logger
        resolved_base_model = resolve_config_base_model(config.base_model_name)
        log.info("Loading SDXL pipeline from %s", resolved_base_model)

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
            resolved_base_model,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
            vae_dtype=gpu.vae_dtype,
        )
        vae = components.vae
        text_encoder_1 = components.text_encoder_1
        text_encoder_2 = components.text_encoder_2
        unet = components.unet

        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)

        attachment = attach_sdxl_lora_adapters(
            unet,
            text_encoder_1,
            text_encoder_2,
            config,
            enable_lora=True,
            for_training=True,
        )
        unet = attachment.unet
        text_encoder_1 = attachment.text_encoder_1
        text_encoder_2 = attachment.text_encoder_2
        param_groups = attachment.param_groups or []
        trainable_params: list = [param for group in param_groups for param in group["params"]]

        if resume_state is not None:
            load_lora_state_dict(
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

        configure_unet_attention(unet, gpu.attention_mechanism, log)

        unet = unet.to(device)
        if not config.cache_latents:
            vae = vae.to(device)
        if not config.cache_text_encoder_outputs:
            text_encoder_1 = text_encoder_1.to(device)
            text_encoder_2 = text_encoder_2.to(device)

        if config.torch_compile:
            log.info("Compiling UNet with torch.compile (inductor)...")
            unet = torch.compile(unet, backend="inductor")

        optimizer = build_optimizer(param_groups, config)
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

        return _TrainingContext(
            config=config,
            runtime=runtime,
            log=log,
            grad_scaler=grad_scaler,
            tokenizer_1=components.tokenizer_1,
            tokenizer_2=components.tokenizer_2,
            noise_scheduler=components.noise_scheduler,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            unet=unet,
            optimizer=optimizer,
            trainable_params=trainable_params,
            dataloader=dataloader,
            total_steps=total_steps,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            start_epoch=start_epoch,
            start_step=start_step,
            resume_state=resume_state,
            cache_steps=cache_steps,
            all_paths=all_paths,
            all_pairs=all_pairs,
        )

    def _build_caches(self, ctx: _TrainingContext) -> None:
        config = ctx.config
        device = ctx.runtime.device
        weight_dtype = ctx.runtime.weight_dtype
        log = ctx.log
        cache_progress = 0

        def _check_stop_during_cache() -> None:
            if self._stop_requested_callback is not None and self._stop_requested_callback():
                log.info("Stop requested during cache phase, cancelling")
                raise TrainingCancelledDuringCache()

        def _on_cache_progress(phase_current: int, phase_total: int, phase: str) -> None:
            nonlocal cache_progress
            _check_stop_during_cache()
            cache_progress += 1
            if self._training_logger is not None:
                if cache_progress == 1:
                    self._training_logger.create_progress_bar(ctx.cache_steps, desc=f"cache {phase}")
                self._training_logger.log_cache_progress(phase, phase_current, phase_total)
                self._training_logger.advance_progress(1, desc=f"cache {phase}")

        _check_stop_during_cache()
        if config.cache_latents:
            log.info("Building latent cache...")
            ctx.latent_cache = build_latent_cache(
                ctx.all_paths,
                ctx.vae,
                device,
                config.cache_latents_to_disk,
                on_progress=_on_cache_progress if ctx.cache_steps > 0 else None,
                log=log,
            )
            _check_stop_during_cache()

        if config.cache_text_encoder_outputs_to_disk:
            log.warning(
                "Text encoder disk cache is enabled. Delete *_te.npz files next to images "
                "after changing trigger_words, captions, or clip_skip."
            )

        if config.cache_text_encoder_outputs:
            log.info("Building text encoder cache...")
            ctx.te_cache = build_te_cache(
                ctx.all_pairs,
                ctx.tokenizer_1,
                ctx.tokenizer_2,
                ctx.text_encoder_1,
                ctx.text_encoder_2,
                device,
                weight_dtype,
                config.clip_skip,
                config.cache_text_encoder_outputs_to_disk,
                on_progress=_on_cache_progress if ctx.cache_steps > 0 else None,
                log=log,
            )
            _check_stop_during_cache()

        if self._training_logger is not None:
            if ctx.cache_steps > 0:
                self._training_logger.close_progress_bar()
                self._training_logger.logger.info("Caching complete, starting training")
            self._training_logger.create_progress_bar(ctx.total_steps, desc="steps")

        if self._progress_callback is not None:
            self._progress_callback(
                ctx.start_step,
                ctx.total_steps,
                0.0,
                0.0,
                ctx.start_epoch,
                config.epochs,
                0.0,
            )

        self._save_config(config)

    def _run_training_loop(self, ctx: _TrainingContext) -> None:
        config = ctx.config
        device = ctx.runtime.device
        weight_dtype = ctx.runtime.weight_dtype
        log = ctx.log
        grad_scaler = ctx.grad_scaler
        optimizer = ctx.optimizer
        unet = ctx.unet
        text_encoder_1 = ctx.text_encoder_1
        text_encoder_2 = ctx.text_encoder_2
        vae = ctx.vae
        noise_scheduler = ctx.noise_scheduler
        latent_cache = ctx.latent_cache
        te_cache = ctx.te_cache
        trainable_params = ctx.trainable_params
        dataloader = ctx.dataloader
        total_steps = ctx.total_steps
        resume_state = ctx.resume_state
        start_epoch = ctx.start_epoch

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
            if resume_state is not None and epoch == start_epoch and resume_state.epoch_step > 0:
                skip_batches = resume_state.epoch_step * config.gradient_accumulation_steps

            for step, batch in enumerate(dataloader):
                if step < skip_batches:
                    continue
                captions: list[str] = batch["caption"]

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

                with torch.no_grad():
                    noise = torch.randn_like(latents)
                    noise = apply_noise_offset(latents, noise, config.noise_offset)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    add_time_ids = batch["add_time_ids"].to(dtype=weight_dtype, device=device)

                if te_cache is not None:
                    prompt_embeds = torch.cat(
                        [te_cache[c][0].to(device, dtype=weight_dtype) for c in captions], dim=0
                    )
                    pooled_prompt_embeds = torch.cat(
                        [te_cache[c][1].to(device, dtype=weight_dtype) for c in captions], dim=0
                    )
                else:
                    prompt_embeds, pooled_prompt_embeds = encode_sdxl_prompt(
                        captions,
                        ctx.tokenizer_1,
                        ctx.tokenizer_2,
                        text_encoder_1,
                        text_encoder_2,
                        device,
                        weight_dtype,
                        config.clip_skip,
                        train_te1=config.text_encoder_1.train,
                        train_te2=config.text_encoder_2.train,
                    )

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
                    part_lrs = resolve_part_learning_rates(config, lr_scheduler.get_last_lr())
                    current_lr = part_lrs["unet"]
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
                            part_lrs=part_lrs if len(part_lrs) > 1 else None,
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
        export_lora_weights(unet, text_encoder_1, text_encoder_2, config, checkpoint_path)
        lora_state_dict = collect_lora_state_dict(unet, text_encoder_1, text_encoder_2, config)
        grad_scaler_state_dict = grad_scaler.state_dict() if grad_scaler is not None else None
        state_path = save_resume_state(
            checkpoint_path=checkpoint_path,
            lora_state_dict=lora_state_dict,
            optimizer_state_dict=optimizer.state_dict(),
            lr_scheduler_state_dict=lr_scheduler.state_dict(),
            epoch=resume_epoch_index,
            global_step=checkpoint_step,
            epoch_step=epoch_step,
            grad_scaler_state_dict=grad_scaler_state_dict,
        )
        prune_stale_resume_states(self._work_dir(config), keep_state_path=state_path)
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
        export_lora_weights(unet, text_encoder_1, text_encoder_2, config, final_path)
