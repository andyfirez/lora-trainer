"""SDXL LoRA training configuration — Pydantic model, serialized as YAML."""

from enum import StrEnum
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


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


class Optimizer(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"
    ADAFACTOR = "adafactor"
    PRODIGY = "prodigy"


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


class ReforgeSampleSampler(StrEnum):
    EULER_A = "euler_a"
    DPMPP_2M = "dpmpp_2m"


class ReforgeSampleSchedulerMode(StrEnum):
    NORMAL = "normal"
    KARRAS = "karras"


class ConceptConfig(BaseModel):
    image_dir: str
    caption_extension: str = ".txt"
    caption_prefix: str = ""
    caption_suffix: str = ""
    repeats: int = Field(default=1, ge=1)


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
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: float = Field(default=16.0, gt=0.0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    # Training targets
    unet: ModelPartConfig = Field(default_factory=lambda: ModelPartConfig(train=True, weight_dtype=WeightDtype.FLOAT_16))
    text_encoder_1: ModelPartConfig = Field(default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16))
    text_encoder_2: ModelPartConfig = Field(default_factory=lambda: ModelPartConfig(train=False, weight_dtype=WeightDtype.FLOAT_16))

    # Training hyperparameters
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    lr_scheduler: LRScheduler = LRScheduler.CONSTANT
    lr_warmup_steps: int = Field(default=0, ge=0)
    optimizer: Optimizer = Optimizer.ADAMW

    # Data
    resolution: int = Field(default=1024, ge=64, le=2048)
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

    # Checkpointing / sampling
    save_every_n_epochs: int = Field(default=1, ge=1)
    resume_from_checkpoint: Optional[str] = None
    sample_every_n_epochs: Optional[int] = None
    sample_before_training: bool = False
    sample_after_training: bool = False
    sample_prompts: list[str] = Field(default_factory=list)
    sample_negative_prompt: str = ""
    sample_steps: int = Field(default=30, ge=1)
    sample_cfg_scale: float = Field(default=7.5, gt=0.0)
    sample_width: Optional[int] = Field(default=None, ge=64, le=2048)
    sample_height: Optional[int] = Field(default=None, ge=64, le=2048)
    sample_scheduler: SampleScheduler = SampleScheduler.EULER
    use_reforge_sampler: bool = False
    sample_sampler: ReforgeSampleSampler = ReforgeSampleSampler.EULER_A
    sample_scheduler_mode: ReforgeSampleSchedulerMode = ReforgeSampleSchedulerMode.NORMAL
    post_training_sampling_config_id: Optional[int] = None

    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TrainConfig":
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False)

    @classmethod
    def default_yaml(cls) -> str:
        return cls().to_yaml()

    def validate_gpu(self) -> None:
        from src.trainer.gpu_config_validation import validate_gpu_config

        validate_gpu_config(
            attention_mechanism=self.attention_mechanism,
            mixed_precision=self.mixed_precision,
            vae_dtype=self.vae_dtype,
            use_reforge_sampler=self.use_reforge_sampler,
        )
