"""Standalone SDXL LoRA sampling service."""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from peft import get_peft_model
from src.sampler.config import SamplingConfig
from src.trainer.attention import configure_unet_attention
from src.trainer.concept_training_metadata import (
    ConceptTrainingMetadata,
    resolve_reference_add_time_ids,
)
from src.trainer.config import TrainConfig, WeightDtype
from src.trainer.sdxl.inference_context import run_merged_adapter_sampling
from src.trainer.sdxl.lora_export import apply_lora_metadata_to_config
from src.trainer.sdxl.lora_io import apply_lora_state_dict, load_lora_file
from src.trainer.sdxl.lora_peft import build_sdxl_lora_config
from src.trainer.sdxl.lora_targets import (
    SDXL_TE_LORA_TARGET_MODULES,
    SDXL_UNET_LORA_TARGET_MODULES,
)
from src.storage.config_paths import resolve_config_base_model
from src.trainer.sdxl.model_loader import load_sdxl_components, resolve_vae_dtype
from src.trainer.sdxl.sampling import PromptEmbedCache

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
        sampling_config: SamplingConfig | None = None,
        lora_paths: list[Path] | None = None,
        output_dir: Path,
        progress_status_callback: ProgressStatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        log: logging.Logger | None = None,
        concept_metadata: dict[int, ConceptTrainingMetadata] | None = None,
        compose_grids: bool = True,
        job_id: int | None = None,
    ) -> None:
        self._config = config
        self._sampling_config = sampling_config
        self._lora_paths = lora_paths or []
        self._output_dir = output_dir
        self._progress_status_callback = progress_status_callback
        self._progress_callback = progress_callback
        self._log = log or logger
        self._prompt_embed_cache = PromptEmbedCache()
        self._concept_metadata = concept_metadata or {}
        self._compose_grids = compose_grids
        self._job_id = job_id

    def run(self) -> None:
        if self._sampling_config is not None:
            from src.sampler.sweep.combinations import build_combinations
            from src.sampler.sweep.engine import SweepEngine

            combos = build_combinations(self._sampling_config.parameters)
            if not combos:
                raise ValueError("No sample prompts configured")
            self._log.info("Sweep mode: %d cell(s) planned", len(combos))
            engine = SweepEngine(
                self._sampling_config,
                base_train_config=self._config,
                output_dir=self._output_dir,
                job_id=self._job_id,
                progress_status_callback=self._progress_status_callback,
                progress_callback=self._progress_callback,
                log=self._log,
                concept_metadata=self._concept_metadata,
                compose_grids=self._compose_grids,
            )
            engine.run()
            return

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

        total_diffusion_steps = self._total_diffusion_steps()
        self._set_progress(0, total_diffusion_steps)
        completed_images = 0
        if self._lora_paths:
            for lora_index, lora_path in enumerate(self._lora_paths):
                status_prefix = f"Sampling {lora_path.name} ({lora_index + 1}/{len(self._lora_paths)})"
                self._set_status(f"{status_prefix} — loading LoRA")
                stack, lora_config, merge_unet = self.load_stack_for_combo(
                    base_model=config.base_model_name,
                    lora_path=lora_path,
                    combo_params={"lora_weight": 1.0},
                )
                try:
                    self._sample_pass(
                        output_stem=self._safe_stem(lora_path),
                        status_prefix=status_prefix,
                        completed_images=completed_images,
                        stack=stack,
                        lora_config=lora_config,
                        merge_unet_adapter=merge_unet,
                    )
                finally:
                    del stack
                    torch.cuda.empty_cache()
                completed_images += len(self._effective_sample_prompts())
        else:
            stack, lora_config, merge_unet = self.load_stack_for_combo(
                base_model=config.base_model_name,
                lora_path=None,
                combo_params={},
            )
            status_prefix = "Sampling base model"
            self._set_status(status_prefix)
            try:
                self._sample_pass(
                    output_stem="base",
                    status_prefix=status_prefix,
                    completed_images=0,
                    stack=stack,
                    lora_config=lora_config,
                    merge_unet_adapter=merge_unet,
                )
            finally:
                del stack
                torch.cuda.empty_cache()
        self._set_status(None)

    def load_stack_for_combo(
        self,
        *,
        base_model: str,
        lora_path: Path | None,
        combo_params: dict[str, Any],
    ) -> tuple[_SamplingStack, TrainConfig, bool]:
        config = self._config.model_copy(update={"base_model_name": base_model})
        enable_lora = lora_path is not None
        if enable_lora and lora_path is not None:
            self._log.info("Reading LoRA file: %s", lora_path)
            load_started = time.perf_counter()
            state_dict = load_lora_file(lora_path)
            self._log.info("LoRA file read in %.1fs", time.perf_counter() - load_started)
            lora_config = apply_lora_metadata_to_config(config, state_dict)
            self._log.info(
                "LoRA metadata: rank=%d alpha=%.1f te1=%s te2=%s",
                lora_config.lora_rank,
                lora_config.lora_alpha,
                lora_config.text_encoder_1.train,
                lora_config.text_encoder_2.train,
            )
            stack = self._load_stack(lora_config, enable_lora=True)
            self._log.info("Applying LoRA weights to pipeline...")
            apply_started = time.perf_counter()
            apply_lora_state_dict(
                state_dict,
                unet=stack.unet,
                text_encoder_1=stack.text_encoder_1,
                text_encoder_2=stack.text_encoder_2,
                config=lora_config,
            )
            self._log.info("LoRA weights applied in %.1fs", time.perf_counter() - apply_started)
            return stack, lora_config, True
        self._log.info("Loading base model pipeline (no LoRA)")
        stack = self._load_stack(config, enable_lora=False)
        return stack, config, False

    def _load_stack(self, config: TrainConfig, *, enable_lora: bool) -> _SamplingStack:
        device = torch.device("cuda")
        vae_dtype = resolve_vae_dtype(config.vae_dtype)
        resolved_base_model = resolve_config_base_model(config.base_model_name)
        self._log.info(
            "Loading SDXL components from %s (lora=%s, attention=%s)...",
            resolved_base_model,
            enable_lora,
            config.attention_mechanism,
        )
        load_started = time.perf_counter()
        components = load_sdxl_components(
            resolved_base_model,
            unet_dtype=config.unet.weight_dtype,
            text_encoder_1_dtype=config.text_encoder_1.weight_dtype,
            text_encoder_2_dtype=config.text_encoder_2.weight_dtype,
            vae_dtype=config.vae_dtype,
        )
        self._log.info("SDXL components loaded from disk in %.1fs", time.perf_counter() - load_started)

        vae = components.vae
        text_encoder_1 = components.text_encoder_1
        text_encoder_2 = components.text_encoder_2
        unet = components.unet

        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)

        if enable_lora:
            self._log.info("Attaching LoRA adapters (rank=%d)...", config.lora_rank)
            unet = get_peft_model(
                unet,
                build_sdxl_lora_config(
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                    target_modules=SDXL_UNET_LORA_TARGET_MODULES,
                ),
            )
        if enable_lora and config.text_encoder_1.train:
            text_encoder_1 = get_peft_model(
                text_encoder_1,
                build_sdxl_lora_config(
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                    target_modules=SDXL_TE_LORA_TARGET_MODULES,
                ),
            )
        if enable_lora and config.text_encoder_2.train:
            text_encoder_2 = get_peft_model(
                text_encoder_2,
                build_sdxl_lora_config(
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                    target_modules=SDXL_TE_LORA_TARGET_MODULES,
                ),
            )

        self._log.info("Moving SDXL models to GPU...")
        gpu_started = time.perf_counter()
        unet = unet.to(device=device, dtype=_DTYPE_MAP[config.unet.weight_dtype])
        text_encoder_1 = text_encoder_1.to(device=device, dtype=_DTYPE_MAP[config.text_encoder_1.weight_dtype])
        text_encoder_2 = text_encoder_2.to(device=device, dtype=_DTYPE_MAP[config.text_encoder_2.weight_dtype])
        vae = vae.to(device=device, dtype=vae_dtype)
        self._log.info("GPU transfer finished in %.1fs", time.perf_counter() - gpu_started)
        configure_unet_attention(unet, config.attention_mechanism, self._log)
        self._log.info("Pipeline ready for sampling")

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

    def generate_single_cell(
        self,
        *,
        stack: _SamplingStack,
        lora_config: TrainConfig,
        sampling_config: TrainConfig,
        merge_unet: bool,
        prompt: str,
        lora_weight: float,
        output_dir: Path,
        output_filename: str,
        completed_images: int,
        total_steps: int,
    ) -> None:
        config = sampling_config
        device = stack.device
        width = config.sample_width or config.resolution
        height = config.sample_height or config.resolution
        self._log.info(
            "Generating image: %dx%d, %d steps, prompt=%r",
            width,
            height,
            config.sample_steps,
            prompt[:120],
        )
        reference_add_time_ids = resolve_reference_add_time_ids(
            self._concept_metadata,
            dataset_ids=self._reference_dataset_ids(config),
            width=config.sample_width or config.resolution,
            height=config.sample_height or config.resolution,
        )

        def on_step(_prompt_index: int, completed: int, _total: int) -> None:
            image_offset = completed_images * config.sample_steps
            self._set_progress(image_offset + completed, total_steps)

        run_merged_adapter_sampling(
            unet=stack.unet,
            text_encoder_1=stack.text_encoder_1,
            text_encoder_2=stack.text_encoder_2,
            vae=stack.vae,
            tokenizer_1=stack.tokenizer_1,
            tokenizer_2=stack.tokenizer_2,
            noise_scheduler=stack.noise_scheduler,
            lora_config=lora_config,
            sampling_config=config,
            device=device,
            sample_prompts=[prompt],
            output_dir=output_dir,
            output_stem="cell",
            log=self._log,
            merge_unet=merge_unet,
            embed_cache=self._prompt_embed_cache,
            reference_add_time_ids=reference_add_time_ids,
            on_step=on_step,
            lora_weight=lora_weight,
            output_filenames=[output_filename],
            clear_embed_cache_on_te_train=True,
        )

    def _sample_pass(
        self,
        *,
        output_stem: str,
        status_prefix: str,
        completed_images: int,
        stack: _SamplingStack,
        lora_config: TrainConfig,
        merge_unet_adapter: bool,
    ) -> None:
        config = self._config
        sample_prompts = self._effective_sample_prompts()
        device = stack.device
        started_at = time.perf_counter()

        reference_add_time_ids = resolve_reference_add_time_ids(
            self._concept_metadata,
            dataset_ids=self._reference_dataset_ids(config),
            width=config.sample_width or config.resolution,
            height=config.sample_height or config.resolution,
        )
        if reference_add_time_ids is not None:
            self._log.info(
                "Sampling %s: using aligned add_time_ids %s (bucket match)",
                output_stem,
                reference_add_time_ids,
            )

        run_merged_adapter_sampling(
            unet=stack.unet,
            text_encoder_1=stack.text_encoder_1,
            text_encoder_2=stack.text_encoder_2,
            vae=stack.vae,
            tokenizer_1=stack.tokenizer_1,
            tokenizer_2=stack.tokenizer_2,
            noise_scheduler=stack.noise_scheduler,
            lora_config=lora_config,
            sampling_config=config,
            device=device,
            sample_prompts=sample_prompts,
            output_dir=self._output_dir,
            output_stem=output_stem,
            log=self._log,
            merge_unet=merge_unet_adapter,
            embed_cache=self._prompt_embed_cache,
            reference_add_time_ids=reference_add_time_ids,
            on_status=lambda prompt_index, n: self._set_status(
                f"{status_prefix} — image {prompt_index + 1}/{n}",
            ),
            on_step=lambda prompt_index, completed, total: self._on_sampling_step(
                prompt_index,
                completed,
                total,
                completed_images,
            ),
            log_step_context="[sample {prompt_index}/{n_prompts}]",
            clear_embed_cache_on_te_train=True,
        )
        self._log.info("Sampling %s completed in %.2fs", output_stem, time.perf_counter() - started_at)

    def _on_sampling_step(
        self,
        prompt_index: int,
        completed: int,
        total: int,
        completed_images: int,
    ) -> None:
        self._report_diffusion_progress(completed_images, prompt_index, completed)
        self._log_sample_step(prompt_index, completed, total)

    def _effective_sample_prompts(self) -> list[str]:
        return self._config.sample_prompts

    def _reference_dataset_ids(self, config: TrainConfig) -> list[int]:
        if config.concepts:
            return [c.dataset_id for c in config.concepts]
        return list(self._concept_metadata.keys())

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
