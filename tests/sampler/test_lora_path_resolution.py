"""Tests for LoRA path resolution from sampling configs."""

from pathlib import Path

from src.sampler.config import SamplingConfig
from src.sampler.sweep.models import LoraEntry, SweepMode, SweepParameter, SweepParameters, parse_lora_entry
from src.services.jobs.sampling_jobs import (
    lora_path_sweep_values,
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
                values=[
                    {"path": r"D:\loras\a.safetensors", "trigger": "a"},
                    {"path": r"D:\loras\b.safetensors", "trigger": "b"},
                ],
            )
        )
    )
    assert len(resolve_lora_paths_from_sampling_config(config)) == 2


def test_resolve_lora_paths_skips_null_entries() -> None:
    config = SamplingConfig(
        parameters=SweepParameters(
            lora_path=SweepParameter(
                mode=SweepMode.VARY,
                values=[
                    {"path": None, "trigger": ""},
                    {"path": r"D:\loras\a.safetensors", "trigger": "a"},
                ],
            )
        )
    )
    assert resolve_lora_paths_from_sampling_config(config) == [r"D:\loras\a.safetensors"]


def test_lora_path_sweep_values_parses_entries() -> None:
    param = SweepParameter(
        mode=SweepMode.VARY,
        values=[{"path": None, "trigger": "x"}, {"path": "pathA", "trigger": "ohwx"}],
    )
    entries = lora_path_sweep_values(param)
    assert len(entries) == 2
    assert entries[0].path is None
    assert entries[1].trigger == "ohwx"


def test_prepare_merges_job_and_config_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    config = SamplingConfig(lora_paths=[str(a)])
    updated, paths = prepare_sampling_config_lora_paths(config, [str(b)])
    assert paths == [str(a), str(b)]
    assert updated.parameters.lora_path.mode == SweepMode.VARY
    values = updated.parameters.lora_path.values
    assert values[0]["path"] == str(a)
    assert values[1]["path"] == str(b)


def test_prepare_preserves_triggers_when_merging_job_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    config = SamplingConfig(
        parameters=SweepParameters(
            lora_path=SweepParameter(
                mode=SweepMode.VARY,
                values=[
                    {"path": None, "trigger": ""},
                    {"path": str(a), "trigger": "custom"},
                ],
            )
        )
    )
    updated, paths = prepare_sampling_config_lora_paths(config, [str(b)], default_trigger="ohwx")
    assert paths == [str(a), str(b)]
    values = updated.parameters.lora_path.values
    assert values[0]["path"] is None
    assert values[1]["trigger"] == "custom"
    assert values[2]["path"] == str(b)
    assert values[2]["trigger"] == "ohwx"


def test_prepare_include_base_model_sample_prepends_null(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch1.safetensors"
    checkpoint.write_bytes(b"x")
    config = SamplingConfig(include_base_model_sample=True)
    updated, paths = prepare_sampling_config_lora_paths(
        config,
        [str(checkpoint)],
        default_trigger="ohwx, person",
    )
    assert paths == [str(checkpoint)]
    values = updated.parameters.lora_path.values
    assert values[0]["path"] is None
    assert values[1]["path"] == str(checkpoint)
    assert values[1]["trigger"] == "ohwx, person"


def test_parse_lora_entry_legacy_string() -> None:
    entry = parse_lora_entry(r"D:\loras\demo.safetensors")
    assert entry == LoraEntry(path=r"D:\loras\demo.safetensors", trigger="")
