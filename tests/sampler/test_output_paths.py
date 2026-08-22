"""Tests for sampling output path resolution."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from src.sampler.config import SamplingConfig
from src.sampler.output_paths import (
    DEFAULT_SAMPLING_OUTPUT_DIR,
    effective_sampling_output_dir,
    flat_grid_filename,
    flat_sample_filename,
    resolve_sampling_config_output_dir,
    resolve_sampling_output_path,
)


def test_effective_sampling_output_dir_defaults_to_output() -> None:
    config = SamplingConfig(output_dir="")
    assert effective_sampling_output_dir(config) == DEFAULT_SAMPLING_OUTPUT_DIR


def test_effective_sampling_output_dir_preserves_explicit_value() -> None:
    config = SamplingConfig(output_dir="  custom/out  ")
    assert effective_sampling_output_dir(config) == "custom/out"


def test_flat_grid_filename_uses_sampling_id_prefix() -> None:
    assert flat_grid_filename(6, 0) == "6_grid_000.png"


def test_flat_sample_filename_uses_sampling_id_and_seed() -> None:
    assert flat_sample_filename(6, 12345, 0) == "6_12345.png"


def test_resolve_sampling_config_output_dir_relative_to_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_sampling_config_output_dir("output")
    assert resolved == (tmp_path / "output").resolve()


def test_resolve_sampling_config_output_dir_rejects_escape(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    with pytest.raises(ValueError, match="escapes project directory"):
        resolve_sampling_config_output_dir("../../outside")


def test_resolve_sampling_output_path_uses_date_only(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed_date = date(2026, 8, 22)
    config = SamplingConfig(output_dir=str(tmp_path / "custom"))
    with patch("src.sampler.output_paths.date") as mock_date:
        mock_date.today.return_value = fixed_date
        path = resolve_sampling_output_path(config, 42)
    assert path == (tmp_path / "custom").resolve() / "2026-08-22"


def test_resolve_sampling_output_path_defaults_empty_to_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed_date = date(2026, 8, 22)
    config = SamplingConfig(output_dir="")
    with patch("src.sampler.output_paths.date") as mock_date:
        mock_date.today.return_value = fixed_date
        path = resolve_sampling_output_path(config, 7)
    assert path == (tmp_path / "output").resolve() / "2026-08-22"
