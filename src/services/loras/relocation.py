"""Match relocated LoRA work directories to stale catalog records."""

from __future__ import annotations

from pathlib import Path

from src.db.tables.trained_lora import TrainedLora
from src.services.loras.discovery import DiscoveredLora
from src.services.storage.relocation import match_by_basename, unique_match


def find_relocated_lora(stale_loras: list[TrainedLora], item: DiscoveredLora) -> TrainedLora | None:
    by_basename = match_by_basename(
        stale_loras,
        get_relative_path=lambda lora: lora.relative_path,
        discovered_basename=item.name,
    )
    match = unique_match(by_basename)
    if match is not None:
        return match

    weights_name = Path(item.weights_relpath).name
    by_weights = [
        lora for lora in stale_loras if Path(lora.weights_relpath).name == weights_name
    ]
    return unique_match(by_weights)
