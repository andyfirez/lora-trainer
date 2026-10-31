"""Trainer dispatch, base-class contract and Flux-LoRA guard for the SDXL sampler."""

import pytest
import torch
from safetensors.torch import save_file

from src.services.runnable.exceptions import RunnableValidationError
from src.services.sampling.lora_paths import validate_lora_paths
from src.trainer.base_trainer import BaseLoRATrainer
from src.trainer.config import ModelType, TrainConfig
from src.trainer.factory import build_trainer, trainer_class_for
from src.trainer.flux.base import FluxBaseTrainer
from src.trainer.flux1.trainer import Flux1LoRATrainer
from src.trainer.flux2.trainer import Flux2KleinLoRATrainer, format_qwen3_prompt
from src.trainer.sdxl.trainer import SDXLLoRATrainer


def test_factory_returns_trainer_for_each_model_type():
    assert trainer_class_for(ModelType.SDXL) is SDXLLoRATrainer
    assert trainer_class_for(ModelType.FLUX1) is Flux1LoRATrainer
    assert trainer_class_for(ModelType.FLUX2_KLEIN) is Flux2KleinLoRATrainer
    for cls in (SDXLLoRATrainer, Flux1LoRATrainer, Flux2KleinLoRATrainer):
        assert issubclass(cls, BaseLoRATrainer)
    assert issubclass(Flux1LoRATrainer, FluxBaseTrainer)


def test_build_trainer_uses_config_model_type():
    flux = TrainConfig.model_validate({"model_type": "flux2_klein"})
    assert isinstance(build_trainer(flux), Flux2KleinLoRATrainer)
    sdxl = TrainConfig.model_validate({"base_model_name": "a.safetensors"})
    assert isinstance(build_trainer(sdxl), SDXLLoRATrainer)


def test_qwen3_prompt_fallback_format():
    class NoTemplate:
        chat_template = None

    text = format_qwen3_prompt(NoTemplate(), "a cat")
    assert text.startswith("<|im_start|>user\na cat<|im_end|>")
    assert text.endswith("<think>\n\n</think>\n\n")


def test_sampler_rejects_flux_lora(tmp_path, storage_roots):
    lora_dir = storage_roots["lora"]
    flux_file = lora_dir / "flux.safetensors"
    save_file({"transformer.x.lora_A.weight": torch.zeros(1)}, str(flux_file))
    with pytest.raises(RunnableValidationError, match="Flux LoRA"):
        validate_lora_paths([str(flux_file)])

    sdxl_file = lora_dir / "sdxl.safetensors"
    save_file({"lora_unet_x.lora_down.weight": torch.zeros(1)}, str(sdxl_file))
    validate_lora_paths([str(sdxl_file)])
