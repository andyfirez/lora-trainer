"""Write A1111/webui-compatible PNG generation metadata."""

from __future__ import annotations

from pathlib import Path

from PIL import PngImagePlugin

from src.sampler.registry.legacy import LEGACY_BUNDLED_SCHEDULERS
from src.sampler.registry.profiles import SamplingProfile
from src.services.png_info.names import infotext_sampler_label, infotext_scheduler_label
from src.trainer.config import SampleScheduler


def a1111_sampler_name(scheduler: str | SampleScheduler) -> str:
    key = scheduler.value if isinstance(scheduler, SampleScheduler) else str(scheduler)
    mapped = LEGACY_BUNDLED_SCHEDULERS.get(key)
    sampler_id = mapped[0] if mapped else key
    return infotext_sampler_label(sampler_id)


def build_a1111_infotext(
    *,
    prompt: str,
    negative_prompt: str = "",
    steps: int,
    sampler: str | SampleScheduler,
    cfg_scale: float,
    seed: int | None,
    width: int,
    height: int,
    model_name: str,
    lora_path: str | None = None,
    lora_weight: float | None = None,
    loras: list[tuple[str, float | None]] | None = None,
    scheduler: str | None = None,
    sampling_profile: SamplingProfile = SamplingProfile.REFORGE,
) -> str:
    """Format generation info so A1111/reForge PNG Info can parse it."""
    sampler_key = sampler.value if isinstance(sampler, SampleScheduler) else str(sampler)
    mapped = LEGACY_BUNDLED_SCHEDULERS.get(sampler_key)
    if mapped is not None:
        sampler_id, scheduler_id = mapped
        if scheduler:
            scheduler_id = scheduler
    else:
        sampler_id = sampler_key
        scheduler_id = scheduler or "normal"
    lines = [prompt.strip()]
    negative = negative_prompt.strip()
    if negative:
        lines.append(f"Negative prompt: {negative}")
    fields = [
        f"Steps: {steps}",
        f"Sampler: {infotext_sampler_label(sampler_id, sampling_profile)}",
        f"Schedule type: {infotext_scheduler_label(scheduler_id, sampling_profile)}",
        f"CFG scale: {cfg_scale}",
        f"Seed: {-1 if seed is None else seed}",
        f"Size: {width}x{height}",
        f"Model: {Path(model_name).name}",
    ]
    stack = list(loras) if loras else []
    if not stack and lora_path:
        stack.append((lora_path, lora_weight))
    if len(stack) == 1:
        path, weight = stack[0]
        fields.append(f"Lora: {Path(path).stem}")
        if weight is not None:
            fields.append(f"Lora weight: {weight}")
    elif len(stack) > 1:
        formatted = ", ".join(
            f"{Path(path).stem} ({1.0 if weight is None else weight})" for path, weight in stack
        )
        fields.append(f'Lora: "{formatted}"')
    lines.append(", ".join(fields))
    return "\n".join(lines)


def pnginfo_with_parameters(infotext: str) -> PngImagePlugin.PngInfo:
    pnginfo = PngImagePlugin.PngInfo()
    if infotext:
        pnginfo.add_text("parameters", infotext)
    return pnginfo
