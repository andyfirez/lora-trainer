"""SDXL inference configuration — model loading and sampling without training hyperparameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from src.gpu import (
    FORBIDDEN_GLOBAL_GPU_KEYS,
    YamlGpuConfigMixin,
    resolve_gpu_config,
    strip_gpu_overrides_matching_defaults,
)
from src.trainer.config import ModelPartConfig, SampleScheduler, VaeDtype, WeightDtype

if TYPE_CHECKING:
    from src.gpu import ResolvedGpuConfig
    from src.settings.models import GpuDefaultsSettings

FORBIDDEN_ENTITY_GPU_KEYS: frozenset[str] = FORBIDDEN_GLOBAL_GPU_KEYS


class SDXLInferenceConfig(YamlGpuConfigMixin, BaseModel):
    """Runtime config for SDXL pipeline loading and image generation."""

    base_model_name: str = ""
    output_dir: str = ""

    lora_rank: int = Field(default=32, ge=1, le=256)
    lora_alpha: float = Field(default=32.0, gt=0.0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    unet: ModelPartConfig = Field(
        default_factory=lambda: ModelPartConfig(train=True, weight_dtype=WeightDtype.FLOAT_16)
    )
    text_encoder_1: ModelPartConfig = Field(
        default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16)
    )
    text_encoder_2: ModelPartConfig = Field(
        default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16)
    )

    mixed_precision: WeightDtype | None = None
    vae_dtype: VaeDtype | None = None
    tf32: bool | None = None
    attention_mechanism: Literal["default", "sdpa", "xformers"] | None = None

    clip_skip: int = Field(default=2, ge=1)
    seed: Optional[int] = None
    resolution: int = Field(default=1024, ge=64, le=2048)

    sample_prompts: list[str] = Field(default_factory=list)
    sample_negative_prompt: str = ""
    sample_steps: int = Field(default=30, ge=1)
    sample_cfg_scale: float = Field(default=7.5, gt=0.0)
    sample_width: Optional[int] = Field(default=None, ge=64, le=2048)
    sample_height: Optional[int] = Field(default=None, ge=64, le=2048)
    sample_scheduler: SampleScheduler = SampleScheduler.EULER
    sample_vae_tiling: bool = True
    sample_vae_fp32: bool = False
    sample_offload_unet_before_decode: bool = True

    def resolve_gpu(self, defaults: GpuDefaultsSettings) -> ResolvedGpuConfig:
        from src.gpu import ResolvedGpuConfig

        if self.tf32 is not None and self.attention_mechanism is not None:
            return ResolvedGpuConfig(
                tf32=self.tf32,
                attention_mechanism=self.attention_mechanism,
                mixed_precision=self.mixed_precision or defaults.mixed_precision,
                vae_dtype=self.vae_dtype or defaults.vae_dtype,
                sample_vae_tiling=self.sample_vae_tiling,
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
