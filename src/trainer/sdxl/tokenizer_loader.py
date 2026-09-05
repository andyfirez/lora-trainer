"""Load SDXL CLIP tokenizers without pulling in UNet/VAE weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import CLIPTokenizer

from src.storage.config_paths import resolve_config_base_model
from src.trainer.sdxl.model_loader import is_checkpoint_file

_TOKENIZER_CACHE: dict[str, tuple[Any, Any]] = {}
_CHECKPOINT_TOKENIZER_1 = "openai/clip-vit-large-patch14"
_CHECKPOINT_TOKENIZER_2 = ("stabilityai/stable-diffusion-xl-base-1.0", "tokenizer_2")


def load_sdxl_tokenizers(base_model_name: str) -> tuple[Any, Any]:
    cached = _TOKENIZER_CACHE.get(base_model_name)
    if cached is not None:
        return cached

    errors: list[str] = []
    for source in _tokenizer_sources(base_model_name):
        try:
            pair = _load_tokenizer_pair(source)
            _TOKENIZER_CACHE[base_model_name] = pair
            return pair
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    raise ValueError(f"Could not load CLIP tokenizers for {base_model_name}: {'; '.join(errors)}")


def _tokenizer_sources(base_model_name: str) -> list[str]:
    sources: list[str] = []
    raw = Path(base_model_name)
    if raw.is_absolute() and (raw.is_file() or raw.is_dir()):
        sources.append(str(raw))
    try:
        resolved = resolve_config_base_model(base_model_name)
        if Path(resolved).exists():
            sources.append(resolved)
    except ValueError:
        pass
    if base_model_name not in sources:
        sources.append(base_model_name)
    return sources


def _load_tokenizer_pair(source: str) -> tuple[Any, Any]:
    path = Path(source)
    if is_checkpoint_file(path):
        tokenizer_1 = CLIPTokenizer.from_pretrained(_CHECKPOINT_TOKENIZER_1)
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            _CHECKPOINT_TOKENIZER_2[0],
            subfolder=_CHECKPOINT_TOKENIZER_2[1],
        )
        return tokenizer_1, tokenizer_2
    return (
        CLIPTokenizer.from_pretrained(source, subfolder="tokenizer"),
        CLIPTokenizer.from_pretrained(source, subfolder="tokenizer_2"),
    )
