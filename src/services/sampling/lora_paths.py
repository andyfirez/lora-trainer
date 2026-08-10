"""Sampling LoRA path resolution and validation helpers."""

from __future__ import annotations

from pathlib import Path

from src.sampler.config import SamplingConfig
from src.sampler.output_paths import resolve_sampling_output_path
from src.sampler.sweep.models import (
    LoraEntry,
    SweepMode,
    SweepParameter,
    dedupe_lora_entries,
    lora_entry_path,
    normalize_lora_path_value,
    parse_lora_entry,
)
from src.services.sampling.exceptions import (
    SamplingLoRAPathNotFoundError,
    SamplingPromptsNotConfiguredError,
)
from src.storage.config_paths import resolve_config_lora_file


def lora_path_sweep_values(param: SweepParameter) -> list[LoraEntry]:
    """Explicit sweep LoRA entries including empty paths."""
    if param.mode == SweepMode.VARY and param.values:
        entries = [parse_lora_entry(value) for value in param.values]
        return dedupe_lora_entries(entries)
    if param.value is not None:
        return dedupe_lora_entries([parse_lora_entry(param.value)])
    return []


def validate_sample_prompts(config: SamplingConfig) -> None:
    if not config.effective_prompts():
        raise SamplingPromptsNotConfiguredError()


def validate_lora_paths(lora_paths: list[str]) -> None:
    for lora_path in lora_paths:
        path = resolve_config_lora_file(lora_path)
        if not path.is_file():
            raise SamplingLoRAPathNotFoundError(lora_path)


def resolve_lora_paths_from_sampling_config(sampling_config: SamplingConfig) -> list[str]:
    """Collect file LoRA paths stored in a sampling config (manual source)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(path: object | None) -> None:
        normalized = normalize_lora_path_value(path)
        if normalized is None or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for path in sampling_config.lora_paths:
        add(path)

    for entry in lora_path_sweep_values(sampling_config.parameters.lora_path):
        add(lora_entry_path(entry))

    return ordered


def _merge_lora_sweep_entries(
    explicit: list[LoraEntry],
    extra_file_paths: list[str],
    *,
    include_base_model_sample: bool,
    default_trigger: str | None,
) -> list[LoraEntry]:
    trigger_by_path = {path: entry.trigger for entry in explicit if (path := lora_entry_path(entry))}
    empty_trigger = next((entry.trigger for entry in explicit if lora_entry_path(entry) is None), "")

    sweep_entries: list[LoraEntry] = list(explicit)
    seen_files = set(trigger_by_path.keys())

    for path in extra_file_paths:
        if path in seen_files:
            continue
        seen_files.add(path)
        trigger = trigger_by_path.get(path) or (default_trigger or "")
        sweep_entries.append(LoraEntry(path=path, trigger=trigger))

    if include_base_model_sample and not any(lora_entry_path(entry) is None for entry in sweep_entries):
        sweep_entries.insert(0, LoraEntry(path=None, trigger=empty_trigger))

    return dedupe_lora_entries(sweep_entries)


def prepare_sampling_config_lora_paths(
    sampling_config: SamplingConfig,
    entity_lora_paths: list[str] | None = None,
    *,
    default_trigger: str | None = None,
) -> tuple[SamplingConfig, list[str]]:
    """Merge entity-level and config-level LoRA paths; preserve triggers and empty entries."""
    explicit = lora_path_sweep_values(sampling_config.parameters.lora_path)
    extra_file_paths: list[str] = []
    seen: set[str] = set()
    for candidate in (entity_lora_paths or []) + resolve_lora_paths_from_sampling_config(sampling_config):
        normalized = normalize_lora_path_value(candidate)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        extra_file_paths.append(normalized)

    sweep_entries = _merge_lora_sweep_entries(
        explicit,
        extra_file_paths,
        include_base_model_sample=sampling_config.include_base_model_sample,
        default_trigger=default_trigger,
    )
    file_paths = [path for entry in sweep_entries if (path := lora_entry_path(entry))]

    if not sweep_entries:
        return sampling_config, []

    return sampling_config.with_resolved_lora_sweep(sweep_entries, file_paths), file_paths


def resolve_sampling_output_dir(sampling_config: SamplingConfig, sampling_id: int) -> Path:
    return resolve_sampling_output_path(sampling_config, sampling_id)
