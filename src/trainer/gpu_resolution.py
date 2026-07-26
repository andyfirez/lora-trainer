"""Resolve GPU settings from global defaults and per-config overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.trainer.attention import AttentionMechanism

if TYPE_CHECKING:
    from src.settings.models import GpuDefaultsSettings
    from src.trainer.config import VaeDtype, WeightDtype

FORBIDDEN_GLOBAL_GPU_KEYS: frozenset[str] = frozenset({"tf32", "attention_mechanism"})

GPU_OVERRIDE_KEYS: tuple[str, ...] = ("mixed_precision", "vae_dtype", "sample_vae_tiling")


@dataclass(frozen=True)
class ResolvedGpuConfig:
    tf32: bool
    attention_mechanism: AttentionMechanism
    mixed_precision: WeightDtype
    vae_dtype: VaeDtype
    sample_vae_tiling: bool

    def as_train_fields(self) -> dict[str, object]:
        return {
            "tf32": self.tf32,
            "attention_mechanism": self.attention_mechanism,
            "mixed_precision": self.mixed_precision,
            "vae_dtype": self.vae_dtype,
            "sample_vae_tiling": self.sample_vae_tiling,
        }


def resolve_gpu_config(
    *,
    defaults: GpuDefaultsSettings,
    mixed_precision: WeightDtype | None = None,
    vae_dtype: VaeDtype | None = None,
    sample_vae_tiling: bool | None = None,
) -> ResolvedGpuConfig:
    return ResolvedGpuConfig(
        tf32=defaults.tf32,
        attention_mechanism=defaults.attention_mechanism,
        mixed_precision=mixed_precision if mixed_precision is not None else defaults.mixed_precision,
        vae_dtype=vae_dtype if vae_dtype is not None else defaults.vae_dtype,
        sample_vae_tiling=(
            sample_vae_tiling if sample_vae_tiling is not None else defaults.sample_vae_tiling
        ),
    )


def strip_global_gpu_keys(data: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in data.items() if key not in FORBIDDEN_GLOBAL_GPU_KEYS}


def strip_gpu_overrides_matching_defaults(
    data: dict[str, object],
    defaults: GpuDefaultsSettings,
) -> dict[str, object]:
    cleaned = strip_global_gpu_keys(data)
    if cleaned.get("mixed_precision") == defaults.mixed_precision.value:
        cleaned.pop("mixed_precision", None)
    if cleaned.get("vae_dtype") == defaults.vae_dtype.value:
        cleaned.pop("vae_dtype", None)
    if cleaned.get("sample_vae_tiling") is defaults.sample_vae_tiling:
        cleaned.pop("sample_vae_tiling", None)
    return cleaned


def merge_proposed_gpu_overrides(
    config: dict[str, object],
    *,
    defaults: GpuDefaultsSettings,
) -> dict[str, object]:
    """Drop override keys from config when they match global defaults (sparse YAML)."""
    merged = dict(config)
    for key in GPU_OVERRIDE_KEYS:
        if key not in merged:
            continue
        value = merged[key]
        if key == "mixed_precision" and value == defaults.mixed_precision.value:
            merged.pop(key)
        elif key == "vae_dtype" and value == defaults.vae_dtype.value:
            merged.pop(key)
        elif key == "sample_vae_tiling" and value is defaults.sample_vae_tiling:
            merged.pop(key)
    return strip_global_gpu_keys(merged)
