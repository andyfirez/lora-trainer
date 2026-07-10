"""SDXL LoRA training configuration — Pydantic model, serialized as YAML.

Persisted YAML omits runtime-only fields:
- Concept ``image_dir`` / ``prepared_dir`` are populated by ``resolve_concepts`` and
  stripped by ``to_yaml`` (see ``ResolvedConceptPaths``).
- Sampling prompt/size fields on ``TrainConfig`` are a runtime overlay applied via
  ``resolve_sampling`` from a persisted ``SamplingConfig`` entity (``sampling_config_id``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from src.trainer.optimizer_config import OptimizerConfig

if TYPE_CHECKING:
    from src.sampler.config import SamplingConfig
    from src.trainer.concept_resolution import ResolvedConceptPaths


class OutputFormat(StrEnum):
    SAFETENSORS = "safetensors"
    PT = "pt"


class LRScheduler(StrEnum):
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"


class WeightDtype(StrEnum):
    FLOAT_32 = "float32"
    FLOAT_16 = "float16"
    BFLOAT_16 = "bfloat16"


class VaeDtype(StrEnum):
    AUTO = "auto"
    FLOAT_32 = "float32"
    FLOAT_16 = "float16"
    BFLOAT_16 = "bfloat16"


class SampleScheduler(StrEnum):
    EULER = "euler"
    EULER_A = "euler_a"
    DDIM = "ddim"
    DPM_PP = "dpm++"


RUNTIME_SAMPLING_FIELDS: tuple[str, ...] = (
    "sample_prompts",
    "sample_negative_prompt",
    "sample_steps",
    "sample_cfg_scale",
    "sample_width",
    "sample_height",
    "sample_scheduler",
    "sample_vae_tiling",
    "sample_vae_fp32",
    "sample_offload_unet_before_decode",
)

FORBIDDEN_INLINE_SAMPLING_KEYS: frozenset[str] = frozenset(
    {
        *RUNTIME_SAMPLING_FIELDS,
        "post_training_sampling_config_id",
    }
)

FORBIDDEN_DEPRECATED_TRAIN_KEYS: frozenset[str] = frozenset({"sample_after_training"})

FORBIDDEN_DEPRECATED_CONCEPT_KEYS: frozenset[str] = frozenset({"image_dir", "prepared_dir"})


class ConceptConfig(BaseModel):
    """Dataset concept for training. ``image_dir``/``prepared_dir`` are runtime-resolved paths."""

    dataset_id: int
    image_dir: str | None = None  # runtime: set by resolve_concepts, not persisted
    prepared_dir: str | None = None  # runtime: set by resolve_concepts, not persisted
    caption_extension: str = ".txt"
    trigger_words: list[str] = Field(default_factory=list)
    caption_suffix: str = ""
    repeats: int = Field(default=3, ge=1)


class ModelPartConfig(BaseModel):
    train: bool = True
    weight_dtype: WeightDtype = WeightDtype.FLOAT_16


class LoggingConfig(BaseModel):
    use_ui_logger: bool = True
    log_every: int = Field(default=1, ge=1)
    log_dir: Optional[str] = None


class TrainConfig(BaseModel):
    """SDXL LoRA training configuration. Serialized to/from YAML."""

    # Model
    base_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir: str = "output"
    lora_name: str = "lora"
    output_format: OutputFormat = OutputFormat.SAFETENSORS

    # LoRA
    lora_rank: int = Field(default=32, ge=1, le=256)
    lora_alpha: float = Field(default=32.0, gt=0.0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    # Training targets
    unet: ModelPartConfig = Field(default_factory=lambda: ModelPartConfig(train=True, weight_dtype=WeightDtype.FLOAT_16))
    text_encoder_1: ModelPartConfig = Field(default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16))
    text_encoder_2: ModelPartConfig = Field(default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16))

    # Training hyperparameters
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0.0)
    lr_scheduler: LRScheduler = LRScheduler.CONSTANT
    lr_warmup_steps: int = Field(default=0, ge=0)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig.defaults)
    min_snr_gamma: float = Field(default=5.0, ge=0.0)
    noise_offset: float = Field(default=0.0357, ge=0.0)
    clip_skip: int = Field(default=2, ge=1)

    # Data
    resolution: int = Field(default=1024, ge=64, le=2048)
    enable_bucket: bool = False
    bucket_reso_steps: int = Field(default=64, ge=8, le=512)
    min_bucket_reso: int = Field(default=512, ge=64, le=2048)
    max_bucket_reso: int = Field(default=2048, ge=64, le=2048)
    bucket_no_upscale: bool = True
    concepts: list[ConceptConfig] = Field(default_factory=list)

    # Optimization
    gradient_checkpointing: bool = True
    mixed_precision: WeightDtype = WeightDtype.FLOAT_16
    seed: Optional[int] = None

    # Caching (latents + text encoder outputs)
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = False

    # Attention backend
    attention_mechanism: Literal["default", "sdpa", "xformers"] = "sdpa"

    # Precision
    vae_dtype: VaeDtype = VaeDtype.AUTO
    tf32: bool = True

    # DataLoader
    num_dataloader_workers: int = Field(default=0, ge=0)
    dataloader_pin_memory: bool = True

    # Compile
    torch_compile: bool = False

    # Checkpointing
    checkpointing_enabled: bool = True
    save_every_n_epochs: int = Field(default=1, ge=1)
    resume_from_checkpoint: Optional[str] = None

    # Sampling
    sampling_enabled: bool = False
    sampling_config_id: Optional[int] = None
    sample_every_n_epochs: Optional[int] = None
    sample_before_training: bool = False
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

    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TrainConfig":
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    def resolve_concepts(self, paths: dict[int, ResolvedConceptPaths]) -> TrainConfig:
        from src.trainer.concept_resolution import ResolvedConceptPaths

        resolved: list[ConceptConfig] = []
        for concept in self.concepts:
            entry = paths.get(concept.dataset_id)
            if entry is None:
                raise ValueError(f"Dataset with id={concept.dataset_id} not found")
            if not isinstance(entry, ResolvedConceptPaths):
                raise TypeError("paths values must be ResolvedConceptPaths")
            resolved.append(
                concept.model_copy(
                    update={
                        "image_dir": entry.image_dir,
                        "prepared_dir": entry.prepared_dir,
                    }
                )
            )
        return self.model_copy(update={"concepts": resolved})

    def resolve_sampling(self, sampling: SamplingConfig) -> TrainConfig:
        from src.sampler.config import SamplingConfig
        from src.trainer.sdxl.caption import (
            apply_trigger_words_to_sample_prompts,
            collect_trigger_words,
        )

        if not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig instance")
        merged = self.model_copy(update=sampling.build_sampling_field_updates())
        return merged.model_copy(
            update={
                "sample_prompts": apply_trigger_words_to_sample_prompts(
                    merged.sample_prompts,
                    collect_trigger_words(self.concepts),
                )
            }
        )

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True)
        for concept in data.get("concepts", []):
            concept.pop("image_dir", None)
            concept.pop("prepared_dir", None)
        for field in RUNTIME_SAMPLING_FIELDS:
            data.pop(field, None)
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    @classmethod
    def default_yaml(cls) -> str:
        return cls().to_yaml()

    def validate_gpu(self) -> None:
        from src.trainer.gpu_config_validation import validate_gpu_config

        validate_gpu_config(
            attention_mechanism=self.attention_mechanism,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
        )
