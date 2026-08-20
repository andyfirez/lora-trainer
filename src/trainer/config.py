"""SDXL LoRA training configuration — Pydantic model, serialized as YAML.

Persisted YAML omits runtime-only fields:
- Concept ``image_dir`` / ``prepared_dir`` are populated by ``resolve_concepts`` and
  stripped by ``to_yaml`` (see ``ResolvedConceptPaths``).
- Sampling uses ``SDXLInferenceConfig`` (``src.trainer.inference_config``), built from
  ``SamplingConfig.to_inference_config()``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Optional, Self, TypeAlias

import yaml
from pydantic import BaseModel, Field, model_validator

from src.gpu import (
    FORBIDDEN_GLOBAL_GPU_KEYS,
    YamlGpuConfigMixin,
    resolve_gpu_config,
    strip_gpu_overrides_matching_defaults,
)
from src.trainer.optimizer_config import OptimizerConfig

if TYPE_CHECKING:
    from src.settings.models import GpuDefaultsSettings
    from src.trainer.concept_resolution import ResolvedConceptPaths
    from src.gpu import ResolvedGpuConfig


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


FORBIDDEN_DEPRECATED_TRAIN_KEYS: frozenset[str] = frozenset({
    "sample_after_training",
    "learning_rate",
    "sampling_enabled",
    "sampling_config_id",
})

# Historical Alembic migrations strip these keys from persisted training YAML.
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

FORBIDDEN_DEPRECATED_CONCEPT_KEYS: frozenset[str] = frozenset({"image_dir", "prepared_dir"})

FORBIDDEN_ENTITY_GPU_KEYS: frozenset[str] = FORBIDDEN_GLOBAL_GPU_KEYS

TrainablePart: TypeAlias = Literal["unet", "text_encoder_1", "text_encoder_2"]


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
    learning_rate: float = Field(default=5e-5, gt=0.0)


class LoggingConfig(BaseModel):
    use_ui_logger: bool = True
    log_every: int = Field(default=1, ge=1)
    log_dir: Optional[str] = None


class TrainConfig(YamlGpuConfigMixin, BaseModel):
    """SDXL LoRA training configuration. Serialized to/from YAML."""

    # Model
    base_model_name: str = ""
    output_dir: str = ""
    lora_name: str = "lora"
    output_format: OutputFormat = OutputFormat.SAFETENSORS

    # LoRA
    lora_rank: int = Field(default=32, ge=1, le=256)
    lora_alpha: float = Field(default=32.0, gt=0.0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    # Training targets
    unet: ModelPartConfig = Field(
        default_factory=lambda: ModelPartConfig(train=True, weight_dtype=WeightDtype.FLOAT_16, learning_rate=5e-5)
    )
    text_encoder_1: ModelPartConfig = Field(
        default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16, learning_rate=5e-5)
    )
    text_encoder_2: ModelPartConfig = Field(
        default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16, learning_rate=5e-5)
    )

    # Training hyperparameters
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
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
    mixed_precision: WeightDtype | None = None
    seed: Optional[int] = None

    # Caching (latents + text encoder outputs)
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = False

    # GPU overrides (sparse in entity YAML; resolved values in job snapshots)
    vae_dtype: VaeDtype | None = None

    # Snapshot/runtime GPU fields (explicit in job YAML; omitted from entity YAML)
    tf32: bool | None = None
    attention_mechanism: Literal["default", "sdpa", "xformers"] | None = None

    # DataLoader
    num_dataloader_workers: int = Field(default=0, ge=0)
    dataloader_pin_memory: bool = True

    # Compile
    torch_compile: bool = False

    # Checkpointing
    checkpointing_enabled: bool = True
    save_every_n_epochs: int = Field(default=1, ge=1)
    resume_from_checkpoint: Optional[str] = None

    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def resolve_gpu(self, defaults: "GpuDefaultsSettings") -> "ResolvedGpuConfig":
        from src.gpu import ResolvedGpuConfig

        if self.tf32 is not None and self.attention_mechanism is not None:
            return ResolvedGpuConfig(
                tf32=self.tf32,
                attention_mechanism=self.attention_mechanism,
                mixed_precision=self.mixed_precision or defaults.mixed_precision,
                vae_dtype=self.vae_dtype or defaults.vae_dtype,
                sample_vae_tiling=defaults.sample_vae_tiling,
            )
        return resolve_gpu_config(
            defaults=defaults,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
        )

    def _entity_yaml_data(self) -> dict[str, object]:
        from src.settings.app_settings import settings

        data = self.model_dump(mode="json", exclude_none=True)
        for concept in data.get("concepts", []):
            concept.pop("image_dir", None)
            concept.pop("prepared_dir", None)
        for field in FORBIDDEN_ENTITY_GPU_KEYS:
            data.pop(field, None)
        return strip_gpu_overrides_matching_defaults(data, settings.gpu_defaults)

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

    def to_snapshot_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True)
        for concept in data.get("concepts", []):
            concept.pop("image_dir", None)
            concept.pop("prepared_dir", None)
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    @model_validator(mode="after")
    def sync_te_cache_with_training(self) -> Self:
        if self.text_encoder_1.train or self.text_encoder_2.train:
            self.cache_text_encoder_outputs = False
            self.cache_text_encoder_outputs_to_disk = False
        return self
