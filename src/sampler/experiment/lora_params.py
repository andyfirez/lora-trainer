"""Apply resolved LoRA entries to experiment combination params."""

from __future__ import annotations

from typing import Any

from src.sampler.experiment.models import (
    LoraEntry,
    lora_entry_path,
    lora_entry_to_param_value,
    parse_trigger_words,
)


def stack_trigger(entries: list[LoraEntry]) -> str:
    words: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for word in parse_trigger_words(entry.trigger):
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(word)
    return ", ".join(words)


def apply_lora_params(params: dict[str, Any], entries: list[LoraEntry]) -> None:
    file_entries = [entry for entry in entries if lora_entry_path(entry) and not entry.bypassed]
    if len(file_entries) > 1:
        params["lora_path"] = lora_entry_path(file_entries[0])
        params["lora_stack"] = [lora_entry_to_param_value(entry) for entry in file_entries]
        params["lora_trigger"] = stack_trigger(file_entries)
        return
    entry = file_entries[0] if file_entries else (entries[0] if entries else LoraEntry())
    params["lora_path"] = lora_entry_path(entry)
    params["lora_trigger"] = entry.trigger
    if lora_entry_path(entry):
        params["lora_weight"] = entry.weight
