"""Resolve and validate local model files for Flux.1 / FLUX.2 Klein training.

Everything is local: no Hugging Face ids and no downloads. Every component is a single
``.safetensors`` file; text encoders additionally need a directory with ``config.json``
and tokenizer files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.storage.paths import StoragePaths

if TYPE_CHECKING:
    from src.trainer.config import TextEncoderFiles, TrainConfig

_TOKENIZER_MARKERS = ("tokenizer.json", "spiece.model", "vocab.json")


def resolve_model_path(raw: str) -> Path:
    """Absolute path as-is; relative path resolves against the base models directory."""
    value = raw.strip()
    if Path(value).is_absolute():
        return Path(value).expanduser().resolve()
    return StoragePaths.resolve_base_model(value)


def resolve_weights_file(label: str, raw: str) -> Path:
    if not raw or not raw.strip():
        raise ValueError(f"{label}: path is required")
    path = resolve_model_path(raw)
    if not path.is_file():
        raise ValueError(f"{label}: file not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"{label}: must be a single .safetensors file, got: {path.name}")
    return path


def resolve_tokenizer_dir(label: str, raw: str) -> Path:
    if not raw or not raw.strip():
        raise ValueError(f"{label}: tokenizer directory is required")
    path = resolve_model_path(raw)
    if not path.is_dir():
        raise ValueError(f"{label}: tokenizer directory not found: {path}")
    if not (path / "config.json").is_file():
        raise ValueError(f"{label}: config.json is missing in {path}")
    if not any((path / marker).is_file() for marker in _TOKENIZER_MARKERS):
        raise ValueError(
            f"{label}: no tokenizer files in {path} (expected one of {', '.join(_TOKENIZER_MARKERS)})"
        )
    return path


def resolve_text_encoder_files(label: str, files: TextEncoderFiles) -> tuple[Path, Path]:
    return (
        resolve_weights_file(f"{label} weights", files.weights_path),
        resolve_tokenizer_dir(label, files.tokenizer_dir),
    )


def validate_flux_model_files(config: TrainConfig) -> None:
    from src.trainer.config import ModelType

    if config.model_type == ModelType.FLUX1:
        flux1 = config.flux1
        assert flux1 is not None
        resolve_weights_file("Flux.1 transformer", flux1.transformer_path)
        resolve_weights_file("Flux.1 VAE", flux1.vae_path)
        resolve_text_encoder_files("CLIP-L", flux1.clip_l)
        resolve_text_encoder_files("T5-XXL", flux1.t5)
    elif config.model_type == ModelType.FLUX2_KLEIN:
        klein = config.flux2_klein
        assert klein is not None
        resolve_weights_file("FLUX.2 Klein transformer", klein.transformer_path)
        resolve_weights_file("FLUX.2 VAE", klein.vae_path)
        resolve_text_encoder_files("Qwen3", klein.qwen3)
