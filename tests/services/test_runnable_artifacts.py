"""Tests for shared runnable log/sample artifact helpers."""

from pathlib import Path
from types import SimpleNamespace

from src.services.runnable.artifacts import list_runnable_samples, read_runnable_logs


def test_read_runnable_logs_empty_without_path() -> None:
    entity = SimpleNamespace(log_path=None, output_path=None)
    assert read_runnable_logs(entity) == []


def test_read_runnable_logs_reads_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    log_path.write_text("a\nb\nc\n", encoding="utf-8")
    entity = SimpleNamespace(log_path=str(log_path), output_path=None)
    assert read_runnable_logs(entity, tail=2) == ["b", "c"]


def test_list_runnable_samples_empty_without_output() -> None:
    entity = SimpleNamespace(log_path=None, output_path=None)
    assert list_runnable_samples(entity) == []


def test_list_runnable_samples_from_output_dir(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "preview.png").write_bytes(b"png")
    entity = SimpleNamespace(log_path=None, output_path=str(tmp_path))
    found = list_runnable_samples(entity)
    assert len(found) == 1
    assert found[0][0].name == "preview.png"
