"""Load SDXL model components from HuggingFace repos, diffusers folders, or single-file checkpoints."""

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.trainer.config import VaeDtype, WeightDtype
from src.trainer.sdxl.dtypes import weight_dtype_to_torch

_CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt"}
_SDXL_ORIGINAL_CONFIG = Path(__file__).resolve().parent / "resources" / "sd_xl_base.yaml"


@dataclass(frozen=True)
class SDXLComponents:
    tokenizer_1: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    vae: AutoencoderKL
    unet: UNet2DConditionModel


def is_checkpoint_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _CHECKPOINT_EXTENSIONS


def _build_training_noise_scheduler(source: object) -> DDPMScheduler:
    """Build variance-preserving DDPM scheduler for the train forward process."""
    return DDPMScheduler.from_config(source.config)


def resolve_vae_dtype(vae_dtype: VaeDtype) -> torch.dtype:
    if vae_dtype != VaeDtype.AUTO:
        return weight_dtype_to_torch(vae_dtype)
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float32


def load_sdxl_components(
    base_model_name: str,
    *,
    unet_dtype: WeightDtype,
    text_encoder_1_dtype: WeightDtype,
    text_encoder_2_dtype: WeightDtype,
    vae_dtype: VaeDtype = VaeDtype.AUTO,
) -> SDXLComponents:
    resolved_vae_dtype = resolve_vae_dtype(vae_dtype)
    path = Path(base_model_name)
    if is_checkpoint_file(path):
        return _load_from_checkpoint(
            path,
            unet_dtype=unet_dtype,
            text_encoder_1_dtype=text_encoder_1_dtype,
            text_encoder_2_dtype=text_encoder_2_dtype,
            vae_dtype=resolved_vae_dtype,
        )
    return _load_from_pretrained(
        base_model_name,
        unet_dtype=unet_dtype,
        text_encoder_1_dtype=text_encoder_1_dtype,
        text_encoder_2_dtype=text_encoder_2_dtype,
        vae_dtype=resolved_vae_dtype,
    )


def _load_from_pretrained(
    base_model_name: str,
    *,
    unet_dtype: WeightDtype,
    text_encoder_1_dtype: WeightDtype,
    text_encoder_2_dtype: WeightDtype,
    vae_dtype: torch.dtype,
) -> SDXLComponents:
    return SDXLComponents(
        tokenizer_1=CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer"),
        tokenizer_2=CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer_2"),
        noise_scheduler=_build_training_noise_scheduler(
            DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler")
        ),
        text_encoder_1=CLIPTextModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=weight_dtype_to_torch(text_encoder_1_dtype),
        ),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
            base_model_name,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype_to_torch(text_encoder_2_dtype),
        ),
        vae=AutoencoderKL.from_pretrained(base_model_name, subfolder="vae", torch_dtype=vae_dtype),
        unet=UNet2DConditionModel.from_pretrained(
            base_model_name,
            subfolder="unet",
            torch_dtype=weight_dtype_to_torch(unet_dtype),
        ),
    )


def _load_from_checkpoint(
    checkpoint_path: Path,
    *,
    unet_dtype: WeightDtype,
    text_encoder_1_dtype: WeightDtype,
    text_encoder_2_dtype: WeightDtype,
    vae_dtype: torch.dtype,
) -> SDXLComponents:
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(checkpoint_path),
        original_config=str(_SDXL_ORIGINAL_CONFIG),
        use_safetensors=checkpoint_path.suffix.lower() == ".safetensors",
    )
    return SDXLComponents(
        tokenizer_1=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        noise_scheduler=_build_training_noise_scheduler(pipeline.scheduler),
        text_encoder_1=pipeline.text_encoder.to(dtype=weight_dtype_to_torch(text_encoder_1_dtype)),
        text_encoder_2=pipeline.text_encoder_2.to(dtype=weight_dtype_to_torch(text_encoder_2_dtype)),
        vae=pipeline.vae.to(dtype=vae_dtype),
        unet=pipeline.unet.to(dtype=weight_dtype_to_torch(unet_dtype)),
    )
