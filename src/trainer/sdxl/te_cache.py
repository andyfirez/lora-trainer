"""Pre-compute and cache SDXL text encoder outputs for LoRA training."""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_TE_NPZ_SUFFIX = "_te.npz"
CacheProgressCallback = Callable[[int, int, str], None]


def _te_npz_path(image_path: Path) -> Path:
    return image_path.parent / (image_path.stem + _TE_NPZ_SUFFIX)


def _disk_cache_valid(image_path: Path, npz: Path) -> bool:
    return npz.is_file() and npz.stat().st_mtime >= image_path.stat().st_mtime


def build_te_cache(
    path_caption_pairs: list[tuple[Path, str]],
    tokenizer_1,
    tokenizer_2,
    text_encoder_1: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    to_disk: bool,
    on_progress: Optional[CacheProgressCallback] = None,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Encode all captions through both SDXL text encoders once.

    Returns a mapping of caption_string → (prompt_embeds, pooled_prompt_embeds),
    both as CPU tensors in the requested dtype.

    prompt_embeds shape:        [1, seq_len, 2048]  (TE1 768 + TE2 1280 concatenated)
    pooled_prompt_embeds shape: [1, 1280]

    If to_disk=True, outputs are saved as .npz files beside each image (as float32
    for numpy compatibility) and reused on subsequent runs.
    Both text encoders are moved to CPU and CUDA cache is cleared after encoding.
    """
    text_encoder_1.eval()
    text_encoder_2.eval()
    text_encoder_1.to(device)
    text_encoder_2.to(device)

    cache: dict[str, tuple[Tensor, Tensor]] = {}
    loaded_from_disk = 0
    encoded = 0
    processed = 0
    total_unique = len({caption for _, caption in path_caption_pairs})

    logger.info(
        "Caching text encoder outputs for %d unique captions (to_disk=%s)...",
        total_unique,
        to_disk,
    )

    for image_path, caption in path_caption_pairs:
        if caption in cache:
            continue

        processed += 1

        npz = _te_npz_path(image_path)
        if to_disk and _disk_cache_valid(image_path, npz):
            data = np.load(npz)
            prompt_embeds = torch.from_numpy(data["prompt_embeds"].copy()).to(dtype)
            pooled_prompt_embeds = torch.from_numpy(data["pooled_prompt_embeds"].copy()).to(dtype)
            cache[caption] = (prompt_embeds, pooled_prompt_embeds)
            loaded_from_disk += 1
            if on_progress is not None:
                on_progress(processed, total_unique, "text_encoder")
            continue

        tokens_1 = tokenizer_1(
            [caption],
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = tokenizer_2(
            [caption],
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            enc1_out = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_1 = enc1_out.hidden_states[-2].to(dtype=dtype).cpu()

            enc2_out = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
            prompt_embeds_2 = enc2_out.hidden_states[-2].to(dtype=dtype).cpu()
            pooled_prompt_embeds = enc2_out[0].to(dtype=dtype).cpu()

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        if to_disk:
            # Store as float32 for numpy compatibility (loaded and cast to dtype on retrieval)
            np.savez(
                npz,
                prompt_embeds=prompt_embeds.float().numpy(),
                pooled_prompt_embeds=pooled_prompt_embeds.float().numpy(),
            )

        cache[caption] = (prompt_embeds, pooled_prompt_embeds)
        encoded += 1
        if on_progress is not None:
            on_progress(processed, total_unique, "text_encoder")

    logger.info(
        "TE cache ready: %d encoded, %d loaded from disk. Moving text encoders to CPU.",
        encoded,
        loaded_from_disk,
    )
    text_encoder_1.to("cpu")
    text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    return cache
