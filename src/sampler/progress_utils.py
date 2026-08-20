"""Sampling progress helpers shared by sweep and legacy paths."""

from __future__ import annotations


def total_diffusion_steps(*, lora_count: int, prompt_count: int, sample_steps: int) -> int:
    return lora_count * prompt_count * sample_steps


def diffusion_progress_step(
    *,
    completed_images: int,
    prompt_index: int,
    diffusion_step: int,
    sample_steps: int,
) -> int:
    return (completed_images + prompt_index) * sample_steps + diffusion_step
