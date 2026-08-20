"""Sampling configuration — Pydantic model, serialized as YAML."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field
from src.sampler.sweep.models import (
    SWEEP_PARAM_ORDER,
    GridLayout,
    LoraEntry,
    SweepMode,
    SweepParameters,
)
from src.gpu import (
    FORBIDDEN_GLOBAL_GPU_KEYS,
    ResolvedGpuConfig,
    YamlGpuConfigMixin,
    resolve_gpu_config,
    strip_gpu_overrides_matching_defaults,
)
from src.trainer.config import SampleScheduler, VaeDtype, WeightDtype

if TYPE_CHECKING:
    from src.settings.models import GpuDefaultsSettings
    from src.trainer.inference_config import SDXLInferenceConfig

DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

LEGACY_FLAT_SAMPLING_KEYS: frozenset[str] = frozenset(
    {
        "base_model_name",
        "sample_prompts",
        "sample_negative_prompt",
        "sample_steps",
        "sample_cfg_scale",
        "sample_width",
        "sample_height",
        "sample_scheduler",
    }
)

FORBIDDEN_DEPRECATED_SAMPLING_KEYS: frozenset[str] = frozenset(
    {
        "source_type",
        "lora_name",
        "use_reforge_sampler",
        "sample_sampler",
        "sample_scheduler_mode",
        "post_training_sampling_config_id",
    }
)

FORBIDDEN_LEGACY_SAMPLING_KEYS: frozenset[str] = LEGACY_FLAT_SAMPLING_KEYS | FORBIDDEN_DEPRECATED_SAMPLING_KEYS

FORBIDDEN_ENTITY_GPU_KEYS: frozenset[str] = FORBIDDEN_GLOBAL_GPU_KEYS


class SamplingConfig(YamlGpuConfigMixin, BaseModel):
    """SDXL LoRA sampling configuration with unified parameter sweep support."""

    output_dir: str = ""
    sample_vae_tiling: bool | None = None
    sample_vae_fp32: bool = False
    sample_offload_unet_before_decode: bool = True
    mixed_precision: WeightDtype | None = None
    vae_dtype: VaeDtype | None = None

    # Snapshot/runtime GPU fields (explicit in job YAML; omitted from entity YAML)
    tf32: bool | None = None
    attention_mechanism: str | None = None

    lora_paths: list[str] = Field(default_factory=list)
    include_base_model_sample: bool = False
    grid: GridLayout = Field(default_factory=GridLayout)
    parameters: SweepParameters = Field(default_factory=SweepParameters)

    @staticmethod
    def _prompts_from_parameters(params: SweepParameters) -> list[str]:
        prompts = params.prompt.effective_values()
        return [str(p) for p in prompts if p is not None and str(p).strip()]

    def resolve_gpu(self, defaults: "GpuDefaultsSettings") -> "ResolvedGpuConfig":
        from src.gpu import ResolvedGpuConfig

        if self.tf32 is not None and self.attention_mechanism is not None:
            return ResolvedGpuConfig(
                tf32=self.tf32,
                attention_mechanism=self.attention_mechanism,  # type: ignore[arg-type]
                mixed_precision=self.mixed_precision or defaults.mixed_precision,
                vae_dtype=self.vae_dtype or defaults.vae_dtype,
                sample_vae_tiling=(
                    self.sample_vae_tiling
                    if self.sample_vae_tiling is not None
                    else defaults.sample_vae_tiling
                ),
            )
        return resolve_gpu_config(
            defaults=defaults,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
            sample_vae_tiling=self.sample_vae_tiling,
        )

    def _entity_yaml_data(self) -> dict[str, object]:
        from src.settings.app_settings import settings

        data = self.model_dump(mode="json", exclude_none=True)
        for field in FORBIDDEN_ENTITY_GPU_KEYS:
            data.pop(field, None)
        return strip_gpu_overrides_matching_defaults(data, settings.gpu_defaults)

    def to_snapshot_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    def effective_prompts(self) -> list[str]:
        return self._prompts_from_parameters(self.parameters)

    def effective_base_model_name(self) -> str:
        return str(self.parameters.base_model_name.first_value() or DEFAULT_BASE_MODEL)

    def effective_negative_prompt(self) -> str:
        return str(self.parameters.negative_prompt.first_value() or "")

    def effective_steps(self) -> int:
        return int(self.parameters.steps.first_value() or 30)

    def effective_cfg_scale(self) -> float:
        return float(self.parameters.cfg_scale.first_value() or 7.5)

    def effective_width(self) -> int | None:
        value = self.parameters.width.first_value()
        return int(value) if value is not None else None

    def effective_height(self) -> int | None:
        value = self.parameters.height.first_value()
        return int(value) if value is not None else None

    def effective_scheduler(self) -> SampleScheduler:
        scheduler = self.parameters.scheduler.first_value()
        if scheduler is None:
            return SampleScheduler.EULER
        return SampleScheduler(str(scheduler))

    def has_varying_params_except_prompt(self) -> bool:
        for key in SWEEP_PARAM_ORDER:
            if key == "prompt":
                continue
            param = self.parameters.get_param(key)
            if param.mode == SweepMode.VARY and len(param.values) > 1:
                return True
        return False

    def inference_config_field_updates(self) -> dict[str, object]:
        from src.settings.app_settings import settings

        params = self.parameters
        resolved = self.resolve_gpu(settings.gpu_defaults)
        return {
            "sample_prompts": self.effective_prompts(),
            "sample_negative_prompt": self.effective_negative_prompt(),
            "sample_steps": self.effective_steps(),
            "sample_cfg_scale": self.effective_cfg_scale(),
            "sample_width": params.width.first_value(),
            "sample_height": params.height.first_value(),
            "sample_scheduler": self.effective_scheduler(),
            "sample_vae_tiling": resolved.sample_vae_tiling,
            "sample_vae_fp32": self.sample_vae_fp32,
            "sample_offload_unet_before_decode": self.sample_offload_unet_before_decode,
        }

    def to_inference_config(self) -> "SDXLInferenceConfig":
        from src.settings.app_settings import settings
        from src.trainer.config import ModelPartConfig, WeightDtype
        from src.trainer.inference_config import SDXLInferenceConfig

        resolved = self.resolve_gpu(settings.gpu_defaults)
        return SDXLInferenceConfig(
            base_model_name=self.effective_base_model_name(),
            output_dir=self.output_dir,
            mixed_precision=resolved.mixed_precision,
            vae_dtype=resolved.vae_dtype,
            tf32=resolved.tf32,
            attention_mechanism=resolved.attention_mechanism,
            unet=ModelPartConfig(train=True, weight_dtype=WeightDtype.FLOAT_16),
            text_encoder_1=ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16),
            text_encoder_2=ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16),
            **self.inference_config_field_updates(),
        )

    def with_resolved_lora_sweep(
        self,
        entries: list[LoraEntry],
        file_paths: list[str],
    ) -> "SamplingConfig":
        if not entries:
            return self
        updated_params = self.parameters.set_resolved_lora_sweep_values(entries)
        return self.model_copy(update={"parameters": updated_params, "lora_paths": file_paths})

    def with_resolved_lora_paths(self, paths: list[str]) -> "SamplingConfig":
        if not paths:
            return self
        return self.with_resolved_lora_sweep(
            [LoraEntry(path=path, trigger="") for path in paths],
            paths,
        )

    def sweep_enabled(self) -> bool:
        return len(self.parameters.vary_keys_with_values()) > 0 or len(self.effective_prompts()) > 1
