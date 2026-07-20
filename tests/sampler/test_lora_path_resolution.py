"""Tests for LoRA path resolution from sampling configs."""

from pathlib import Path

from src.sampler.config import SamplingConfig
from src.sampler.sweep.models import SweepMode, SweepParameter, SweepParameters
from src.services.jobs.sampling_jobs import (
    prepare_sampling_config_lora_paths,
    resolve_lora_paths_from_sampling_config,
)


def test_resolve_lora_paths_from_flat_list() -> None:
    config = SamplingConfig(lora_paths=[r"D:\loras\a.safetensors", r"D:\loras\b.safetensors"])
    assert resolve_lora_paths_from_sampling_config(config) == [
        r"D:\loras\a.safetensors",
        r"D:\loras\b.safetensors",
    ]


def test_resolve_lora_paths_from_parameters() -> None:
    config = SamplingConfig(
        parameters=SweepParameters(
            lora_path=SweepParameter(
                mode=SweepMode.VARY,
                values=[r"D:\loras\a.safetensors", r"D:\loras\b.safetensors"],
            )
        )
    )
    assert len(resolve_lora_paths_from_sampling_config(config)) == 2


def test_prepare_merges_job_and_config_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    config = SamplingConfig(lora_paths=[str(a)])
    updated, paths = prepare_sampling_config_lora_paths(config, [str(b)])
    assert paths == [str(b), str(a)]
    assert updated.parameters.lora_path.mode == SweepMode.VARY
    assert updated.parameters.lora_path.values == [str(b), str(a)]
