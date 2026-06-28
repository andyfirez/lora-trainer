"""Sampling configuration — Pydantic model, serialized as YAML."""

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from src.trainer.config import SampleScheduler, VaeDtype, WeightDtype
from src.trainer.gpu_config_validation import validate_gpu_config


class SamplingConfig(BaseModel):
    """SDXL LoRA sampling configuration. Serialized to/from YAML."""

    base_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir: str = "output"
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
    attention_mechanism: Literal["default", "sdpa", "xformers"] = "sdpa"
    mixed_precision: WeightDtype = WeightDtype.FLOAT_16
    vae_dtype: VaeDtype = VaeDtype.AUTO
    tf32: bool = True

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SamplingConfig":
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False)

    @classmethod
    def default_yaml(cls) -> str:
        return cls().to_yaml()

    def validate_gpu(self) -> None:
        validate_gpu_config(
            attention_mechanism=self.attention_mechanism,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
        )

    def build_sampling_field_updates(self) -> dict[str, object]:
        return {
            "sample_prompts": self.sample_prompts,
            "sample_negative_prompt": self.sample_negative_prompt,
            "sample_steps": self.sample_steps,
            "sample_cfg_scale": self.sample_cfg_scale,
            "sample_width": self.sample_width,
            "sample_height": self.sample_height,
            "sample_scheduler": self.sample_scheduler,
            "sample_vae_tiling": self.sample_vae_tiling,
            "sample_vae_fp32": self.sample_vae_fp32,
            "sample_offload_unet_before_decode": self.sample_offload_unet_before_decode,
        }

    def to_train_config(self) -> "TrainConfig":
        from src.trainer.config import ModelPartConfig, TrainConfig

        base = TrainConfig(
            base_model_name=self.base_model_name,
            output_dir=self.output_dir,
            attention_mechanism=self.attention_mechanism,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
            tf32=self.tf32,
            unet=ModelPartConfig(train=True, weight_dtype=WeightDtype.FLOAT_16),
            text_encoder_1=ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16),
            text_encoder_2=ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16),
        )
        return base.resolve_sampling(self)
