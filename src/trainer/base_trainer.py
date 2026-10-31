"""Shared LoRA trainer skeleton.

``BaseLoRATrainer`` owns everything that does not depend on the model family:
config validation, seeding, GPU resolution, dataloader construction, cache phase
progress, optimizer / LR scheduler, the epoch/step loop, checkpointing, resume
state and final export.

Subclasses (SDXL, Flux.1, FLUX.2 Klein) implement the model-specific hooks:
``_setup_model``, ``_set_train_mode``, ``_build_caches``, ``_compute_loss``,
``_export_weights`` and ``_collect_lora_state``.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.services.loras.paths import resolve_training_work_dir_from_config
from src.settings.app_settings import settings
from src.trainer.concept_training_metadata import ConceptTrainingMetadata
from src.trainer.config import TrainConfig, WeightDtype
from src.trainer.optimizer_config import build_optimizer
from src.trainer.progress import TrainProgress
from src.trainer.sdxl.bucket_batch_sampler import build_bucket_batch_sampler
from src.trainer.sdxl.checkpoint_state import (
    ResumeState,
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
from src.trainer.sdxl.mixed_precision import create_grad_scaler
from src.trainer.training_log import JobTrainingLogger, resolve_part_learning_rates

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}

CacheProgress = Callable[[int, int, str], None]


class TrainingCancelledAfterSave(Exception):
    """Raised when cancellation with save-checkpoint was requested."""


class TrainingCancelledDuringCache(Exception):
    """Raised when stop was requested during cache/setup before the training loop."""


@dataclass
class TrainContext:
    """Runtime values shared between the base loop and subclass hooks."""

    config: TrainConfig
    device: torch.device
    weight_dtype: torch.dtype
    vae_dtype: Any
    attention_mechanism: str
    resume_state: Optional[ResumeState]
    log: logging.Logger


class BaseLoRATrainer(ABC):
    def __init__(
        self,
        config: TrainConfig,
        progress_callback: Optional[Callable[..., None]] = None,
        training_logger: Optional[JobTrainingLogger] = None,
        checkpoint_callback: Optional[Callable[[str, int, int], None]] = None,
        save_checkpoint_requested_callback: Optional[Callable[[], bool]] = None,
        stop_requested_callback: Optional[Callable[[], bool]] = None,
        concept_metadata: Optional[dict[int, ConceptTrainingMetadata]] = None,
        work_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._work_dir_override = work_dir.expanduser().resolve() if work_dir is not None else None
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

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def _setup_model(self, ctx: TrainContext) -> list[dict]:
        """Load the model, attach LoRA, place on device; return optimizer param groups."""

    @abstractmethod
    def _set_train_mode(self) -> None:
        """Put trainable modules into train mode (called at the start of every epoch)."""

    @abstractmethod
    def _build_caches(
        self,
        ctx: TrainContext,
        all_paths: list[Path],
        all_pairs: list[tuple[Path, str]],
        on_progress: Optional[CacheProgress],
    ) -> None:
        """Build latent / text-encoder caches according to ``ctx.config`` flags."""

    @abstractmethod
    def _compute_loss(self, batch: dict, ctx: TrainContext) -> Tensor:
        """Run one forward pass and return the mean loss for the batch."""

    @abstractmethod
    def _export_weights(self, path: Path) -> None:
        """Write the final LoRA file (inference-compatible format)."""

    @abstractmethod
    def _collect_lora_state(self) -> dict[str, Tensor]:
        """Return the LoRA state used for resume checkpoints."""

    @property
    def _model_label(self) -> str:
        return "LoRA"

    # --------------------------------------------------------------- template

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
        gpu = config.resolve_gpu(settings.gpu_defaults)
        weight_dtype = DTYPE_MAP[gpu.mixed_precision]
        grad_scaler = create_grad_scaler(gpu.mixed_precision)

        if gpu.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        log = self._training_logger.logger if self._training_logger is not None else logger
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

        ctx = TrainContext(
            config=config,
            device=device,
            weight_dtype=weight_dtype,
            vae_dtype=gpu.vae_dtype,
            attention_mechanism=gpu.attention_mechanism,
            resume_state=resume_state,
            log=log,
        )
        param_groups = self._setup_model(ctx)
        trainable_params: list = [param for group in param_groups for param in group["params"]]

        optimizer = build_optimizer(param_groups, config)
        self._optimizer = optimizer
        self._device = device
        cache_mode = config.cache_latents or config.cache_text_encoder_outputs
        train_dataset = self._build_dataset(config, cache_mode=cache_mode)
        dataloader = self._build_dataloader(config, train_dataset)

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
                    self._training_logger.create_progress_bar(cache_steps, desc=f"cache {phase}")
                self._training_logger.log_cache_progress(phase, phase_current, phase_total)
                self._training_logger.advance_progress(1, desc=f"cache {phase}")

        _check_stop_during_cache()
        self._build_caches(ctx, all_paths, all_pairs, _on_cache_progress if cache_steps > 0 else None)
        _check_stop_during_cache()

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
                self._set_train_mode()

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

                    loss = self._compute_loss(batch, ctx) / config.gradient_accumulation_steps
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

            self._save_final(config)
            delete_all_resume_states(self._work_dir(config))
            log.info("Training complete. Output dir: %s", self._work_dir(config))
        finally:
            if self._training_logger is not None:
                self._training_logger.close_progress_bar()

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _validate_config(config: TrainConfig) -> None:
        config.validate_gpu()

    def _build_dataset(self, config: TrainConfig, cache_mode: bool = False) -> Dataset:
        return build_training_dataset(
            config,
            cache_mode=cache_mode,
            concept_metadata=self._concept_metadata,
        )

    @staticmethod
    def _build_dataloader(config: TrainConfig, train_dataset: Dataset) -> DataLoader:
        if config.enable_bucket:
            bucket_sampler = build_bucket_batch_sampler(
                train_dataset,
                collect_bucket_keys(train_dataset),
                config.batch_size,
            )
            return DataLoader(
                train_dataset,
                batch_sampler=bucket_sampler,
                num_workers=config.num_dataloader_workers,
                pin_memory=config.dataloader_pin_memory and config.num_dataloader_workers > 0,
            )
        return DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_dataloader_workers,
            pin_memory=config.dataloader_pin_memory and config.num_dataloader_workers > 0,
        )

    def _work_dir(self, config: TrainConfig) -> Path:
        if self._work_dir_override is not None:
            return self._work_dir_override
        return resolve_training_work_dir_from_config(config)

    def _save_config(self, config: TrainConfig) -> None:
        work_dir = self._work_dir(config)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = work_dir / f"{config.lora_name}_config.yaml"
        config_path.write_text(config.to_yaml(), encoding="utf-8")

    def _save_checkpoint(
        self,
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
        self._export_weights(checkpoint_path)
        lora_state_dict = self._collect_lora_state()
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

    def _save_final(self, config: TrainConfig) -> None:
        ext = f".{config.output_format.value}"
        final_path = self._work_dir(config) / f"{config.lora_name}{ext}"
        self._export_weights(final_path)
