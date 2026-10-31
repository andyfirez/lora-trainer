"""Select the LoRA trainer class for a config's base model type."""

from __future__ import annotations

from typing import Any

from src.trainer.base_trainer import BaseLoRATrainer
from src.trainer.config import ModelType, TrainConfig


def trainer_class_for(model_type: ModelType) -> type[BaseLoRATrainer]:
    # Imports are local so SDXL runs never import Flux-only dependencies and vice versa.
    if model_type == ModelType.SDXL:
        from src.trainer.sdxl.trainer import SDXLLoRATrainer

        return SDXLLoRATrainer
    if model_type == ModelType.FLUX1:
        from src.trainer.flux1.trainer import Flux1LoRATrainer

        return Flux1LoRATrainer
    if model_type == ModelType.FLUX2_KLEIN:
        from src.trainer.flux2.trainer import Flux2KleinLoRATrainer

        return Flux2KleinLoRATrainer
    raise ValueError(f"Unsupported model type: {model_type}")


def build_trainer(config: TrainConfig, **kwargs: Any) -> BaseLoRATrainer:
    return trainer_class_for(config.model_type)(config, **kwargs)
