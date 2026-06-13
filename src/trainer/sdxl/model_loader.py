"""Load SDXL model components from HuggingFace repos, diffusers folders, or single-file checkpoints."""

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.trainer.config import WeightDtype

_CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt"}
_SDXL_ORIGINAL_CONFIG = Path(__file__).resolve().parent / "resources" / "sd_xl_base.yaml"

_DTYPE_MAP = {
    WeightDtype.FLOAT_32: torch.float32,
    WeightDtype.FLOAT_16: torch.float16,
    WeightDtype.BFLOAT_16: torch.bfloat16,
}


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


def load_sdxl_components(
    base_model_name: str,
    *,
    unet_dtype: WeightDtype,
    text_encoder_1_dtype: WeightDtype,
    text_encoder_2_dtype: WeightDtype,
) -> SDXLComponents:
    path = Path(base_model_name)
    if is_checkpoint_file(path):
        return _load_from_checkpoint(
            path,
            unet_dtype=unet_dtype,
            text_encoder_1_dtype=text_encoder_1_dtype,
            text_encoder_2_dtype=text_encoder_2_dtype,
        )
    return _load_from_pretrained(
        base_model_name,
        unet_dtype=unet_dtype,
        text_encoder_1_dtype=text_encoder_1_dtype,
        text_encoder_2_dtype=text_encoder_2_dtype,
    )


def _load_from_pretrained(
    base_model_name: str,
    *,
    unet_dtype: WeightDtype,
    text_encoder_1_dtype: WeightDtype,
    text_encoder_2_dtype: WeightDtype,
) -> SDXLComponents:
    return SDXLComponents(
        tokenizer_1=CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer"),
        tokenizer_2=CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer_2"),
        noise_scheduler=DDPMScheduler.from_pretrained(base_model_name, subfolder="scheduler"),
        text_encoder_1=CLIPTextModel.from_pretrained(
            base_model_name,
            subfolder="text_encoder",
            torch_dtype=_DTYPE_MAP[text_encoder_1_dtype],
        ),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
            base_model_name,
            subfolder="text_encoder_2",
            torch_dtype=_DTYPE_MAP[text_encoder_2_dtype],
        ),
        vae=AutoencoderKL.from_pretrained(base_model_name, subfolder="vae", torch_dtype=torch.float32),
        unet=UNet2DConditionModel.from_pretrained(
            base_model_name,
            subfolder="unet",
            torch_dtype=_DTYPE_MAP[unet_dtype],
        ),
    )


def _load_from_checkpoint(
    checkpoint_path: Path,
    *,
    unet_dtype: WeightDtype,
    text_encoder_1_dtype: WeightDtype,
    text_encoder_2_dtype: WeightDtype,
) -> SDXLComponents:
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(checkpoint_path),
        original_config=str(_SDXL_ORIGINAL_CONFIG),
        use_safetensors=checkpoint_path.suffix.lower() == ".safetensors",
    )
    return SDXLComponents(
        tokenizer_1=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        noise_scheduler=pipeline.scheduler,
        text_encoder_1=pipeline.text_encoder.to(dtype=_DTYPE_MAP[text_encoder_1_dtype]),
        text_encoder_2=pipeline.text_encoder_2.to(dtype=_DTYPE_MAP[text_encoder_2_dtype]),
        vae=pipeline.vae.to(dtype=torch.float32),
        unet=pipeline.unet.to(dtype=_DTYPE_MAP[unet_dtype]),
    )
