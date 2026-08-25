"""Discover LoRA weight files under the inference root for sampling picklists."""

from __future__ import annotations

from dataclasses import dataclass

from src.storage.paths import StoragePaths

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}


@dataclass(frozen=True)
class InferenceLoraEntry:
    name: str
    relative_path: str


def discover_inference_loras() -> list[InferenceLoraEntry]:
    """Recursively scan inference_lora_root; name is filename stem (last wins on collision)."""
    root = StoragePaths.inference_lora_root()
    if not root.is_dir():
        return []

    by_name: dict[str, InferenceLoraEntry] = {}
    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in LORA_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        by_name[path.stem] = InferenceLoraEntry(name=path.stem, relative_path=relative)

    return sorted(by_name.values(), key=lambda entry: entry.name.lower())


def inference_lora_name_index() -> dict[str, str]:
    """Map display name (filename stem) → relative_path under inference root."""
    return {entry.name: entry.relative_path for entry in discover_inference_loras()}


def resolve_inference_lora_name(name: str) -> str:
    """Resolve a LoRA tag name to a relative path; raise ValueError if unknown."""
    key = name.strip()
    if not key:
        raise ValueError("LoRA name is empty")
    relative = inference_lora_name_index().get(key)
    if relative is None:
        raise ValueError(f"LoRA not found in inference catalog: {key}")
    return relative
