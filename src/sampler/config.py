"""Sampling configuration — Pydantic model, serialized as YAML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator
from src.sampler.sweep.models import (
    SWEEP_PARAM_ORDER,
    GridLayout,
    SourceType,
    SweepMode,
    SweepParameter,
    SweepParameters,
)
from src.trainer.config import SampleScheduler, VaeDtype, WeightDtype
from src.trainer.gpu_config_validation import validate_gpu_config

if TYPE_CHECKING:
    from src.trainer.config import TrainConfig

_LEGACY_FIELD_MAP: dict[str, tuple[str, str]] = {
    "base_model_name": ("base_model_name", "base_model_name"),
    "sample_negative_prompt": ("negative_prompt", "sample_negative_prompt"),
    "sample_steps": ("steps", "sample_steps"),
    "sample_cfg_scale": ("cfg_scale", "sample_cfg_scale"),
    "sample_width": ("width", "sample_width"),
    "sample_height": ("height", "sample_height"),
    "sample_scheduler": ("scheduler", "sample_scheduler"),
}


class SamplingConfig(BaseModel):
    """SDXL LoRA sampling configuration with unified parameter sweep support."""

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

    source_type: SourceType = "manual"
    source_job_id: Optional[int] = None
    lora_paths: list[str] = Field(default_factory=list)
    include_final_checkpoint: bool = True
    grid: GridLayout = Field(default_factory=GridLayout)
    parameters: SweepParameters = Field(default_factory=SweepParameters)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_yaml(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        params_data = data.get("parameters")
        if isinstance(params_data, SweepParameters):
            migrated = params_data
        elif isinstance(params_data, dict):
            migrated = SweepParameters.model_validate(params_data)
        else:
            migrated = SweepParameters()

        if "sample_prompts" in data and not isinstance(data.get("parameters"), dict):
            prompts = data.get("sample_prompts") or []
            if isinstance(prompts, list):
                if len(prompts) > 1:
                    migrated = migrated.model_copy(
                        update={"prompt": SweepParameter(mode=SweepMode.VARY, values=list(prompts))}
                    )
                elif len(prompts) == 1:
                    migrated = migrated.model_copy(
                        update={"prompt": SweepParameter(mode=SweepMode.FIXED, value=prompts[0])}
                    )

        for legacy_key, (param_key, _) in _LEGACY_FIELD_MAP.items():
            if legacy_key in data and legacy_key != "base_model_name":
                migrated = migrated.model_copy(
                    update={
                        param_key: SweepParameter(mode=SweepMode.FIXED, value=data[legacy_key]),
                    }
                )
            elif legacy_key == "base_model_name" and "base_model_name" in data:
                migrated = migrated.model_copy(
                    update={
                        "base_model_name": SweepParameter(
                            mode=SweepMode.FIXED,
                            value=data["base_model_name"],
                        )
                    }
                )

        if data.get("lora_paths"):
            paths = data["lora_paths"]
            if isinstance(paths, list) and paths and not migrated.lora_path.effective_values():
                migrated = migrated.model_copy(
                    update={"lora_path": SweepParameter(mode=SweepMode.VARY, values=list(paths))}
                )

        result = {**data, "parameters": migrated.model_dump(mode="json")}
        result["sample_prompts"] = cls._prompts_from_parameters(migrated)
        result["sample_negative_prompt"] = str(migrated.negative_prompt.first_value() or "")
        result["sample_steps"] = int(migrated.steps.first_value() or 30)
        result["sample_cfg_scale"] = float(migrated.cfg_scale.first_value() or 7.5)
        width = migrated.width.first_value()
        height = migrated.height.first_value()
        result["sample_width"] = width
        result["sample_height"] = height
        scheduler = migrated.scheduler.first_value()
        result["sample_scheduler"] = scheduler or "euler"
        result["base_model_name"] = str(migrated.base_model_name.first_value() or data.get("base_model_name", ""))
        return result

    @staticmethod
    def _prompts_from_parameters(params: SweepParameters) -> list[str]:
        prompts = params.prompt.effective_values()
        return [str(p) for p in prompts if p is not None and str(p).strip()]

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SamplingConfig":
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        payload = self.model_dump(mode="json")
        for legacy_key in (
            "sample_prompts",
            "sample_negative_prompt",
            "sample_steps",
            "sample_cfg_scale",
            "sample_width",
            "sample_height",
            "sample_scheduler",
        ):
            payload.pop(legacy_key, None)
        payload.pop("base_model_name", None)
        return yaml.dump(payload, allow_unicode=True, sort_keys=False)

    @classmethod
    def default_yaml(cls) -> str:
        return cls().to_yaml()

    def validate_gpu(self) -> None:
        validate_gpu_config(
            attention_mechanism=self.attention_mechanism,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
        )

    def effective_prompts(self) -> list[str]:
        return self._prompts_from_parameters(self.parameters)

    def has_varying_params_except_prompt(self) -> bool:
        for key in SWEEP_PARAM_ORDER:
            if key == "prompt":
                continue
            param = self.parameters.get_param(key)
            if param.mode == SweepMode.VARY and len(param.values) > 1:
                return True
        return False

    def mid_training_prompts(self) -> list[str]:
        return self.effective_prompts()

    def mid_training_field_updates(self) -> dict[str, object]:
        params = self.parameters
        return {
            "sample_prompts": self.mid_training_prompts(),
            "sample_negative_prompt": str(params.negative_prompt.first_value() or ""),
            "sample_steps": int(params.steps.first_value() or 30),
            "sample_cfg_scale": float(params.cfg_scale.first_value() or 7.5),
            "sample_width": params.width.first_value(),
            "sample_height": params.height.first_value(),
            "sample_scheduler": params.scheduler.first_value() or SampleScheduler.EULER,
            "sample_vae_tiling": self.sample_vae_tiling,
            "sample_vae_fp32": self.sample_vae_fp32,
            "sample_offload_unet_before_decode": self.sample_offload_unet_before_decode,
        }

    def build_sampling_field_updates(self) -> dict[str, object]:
        return self.mid_training_field_updates()

    def to_train_config(self) -> "TrainConfig":
        from src.trainer.config import ModelPartConfig, TrainConfig

        params = self.parameters
        base = TrainConfig(
            base_model_name=str(params.base_model_name.first_value() or self.base_model_name),
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

    def with_resolved_lora_paths(self, paths: list[str]) -> "SamplingConfig":
        if not paths:
            return self
        updated_params = self.parameters.set_resolved_lora_paths(paths)
        return self.model_copy(update={"parameters": updated_params, "lora_paths": paths})

    def sweep_enabled(self) -> bool:
        return len(self.parameters.vary_keys_with_values()) > 0 or len(self.effective_prompts()) > 1
