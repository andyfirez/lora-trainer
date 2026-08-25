"""Parse and resolve <lora:name:weight> tags in sampling prompts (A1111/reForge style)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.sampler.experiment.lora_params import apply_lora_params
from src.sampler.experiment.models import LoraEntry
from src.services.storage.inference_loras import resolve_inference_lora_name
from src.storage.config_paths import resolve_sampling_lora_file

LORA_TAG_RE = re.compile(r"<lora:([^>]+)>", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedLoraTag:
    name: str
    weight: float = 1.0


def _parse_tag_args(args: str) -> ParsedLoraTag:
    parts = [part.strip() for part in args.split(":") if part.strip()]
    if not parts:
        raise ValueError("LoRA tag is missing a name")
    name = parts[0]
    weight = 1.0
    if len(parts) > 1:
        try:
            weight = float(parts[1])
        except ValueError:
            weight = 1.0
    return ParsedLoraTag(name=name, weight=weight)


def parse_lora_tags(prompt: str) -> tuple[str, list[ParsedLoraTag]]:
    """Strip LoRA tags from prompt text; return cleaned prompt and ordered tag list."""
    tags: list[ParsedLoraTag] = []

    def replace(match: re.Match[str]) -> str:
        tags.append(_parse_tag_args(match.group(1)))
        return ""

    cleaned = re.sub(LORA_TAG_RE, replace, prompt)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, tags


def resolve_lora_tags(tags: list[ParsedLoraTag]) -> list[LoraEntry]:
    """Map tag names to inference catalog entries."""
    entries: list[LoraEntry] = []
    for tag in tags:
        relative_path = resolve_inference_lora_name(tag.name)
        entries.append(LoraEntry(path=relative_path, trigger="", weight=tag.weight))
    return entries


def validate_lora_tag_names(tags: list[ParsedLoraTag]) -> None:
    for tag in tags:
        relative_path = resolve_inference_lora_name(tag.name)
        path = resolve_sampling_lora_file(relative_path)
        if not path.is_file():
            raise ValueError(f"LoRA file not found for tag <lora:{tag.name}>: {relative_path}")


def apply_prompt_lora_to_params(params: dict[str, Any]) -> None:
    """If prompt contains LoRA tags, strip them and override lora_path/lora_stack params."""
    raw_prompt = str(params.get("prompt") or "")
    cleaned, tags = parse_lora_tags(raw_prompt)
    if not tags:
        return
    entries = resolve_lora_tags(tags)
    params["prompt"] = cleaned
    params["infotext_prompt"] = raw_prompt
    apply_lora_params(params, entries)


def collect_lora_tags_from_prompts(prompts: list[str]) -> list[ParsedLoraTag]:
    seen: set[tuple[str, float]] = set()
    ordered: list[ParsedLoraTag] = []
    for prompt in prompts:
        _, tags = parse_lora_tags(prompt)
        for tag in tags:
            key = (tag.name, tag.weight)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(tag)
    return ordered
