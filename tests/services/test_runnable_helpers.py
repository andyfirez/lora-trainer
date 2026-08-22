"""Tests for shared runnable log/sample artifact helpers."""

from pathlib import Path
from types import SimpleNamespace

from src.services.runnable.artifacts import list_runnable_samples, read_runnable_logs
from src.services.runnable.samples import resolve_safe_sample_file


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


def test_list_runnable_samples_from_flat_date_dir(tmp_path: Path) -> None:
    (tmp_path / "6_12345.png").write_bytes(b"png")
    (tmp_path / "6_grid_000.png").write_bytes(b"png")
    (tmp_path / "7_99999.png").write_bytes(b"png")
    entity = SimpleNamespace(id=6, log_path=None, output_path=str(tmp_path))
    found = list_runnable_samples(entity)
    assert len(found) == 2
    assert found[0][0].name == "6_12345.png"
    assert found[1][0].name == "6_grid_000.png"
    assert found[1][1] == "grid"


def test_list_runnable_samples_from_images_dir_before_manifest(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "cell_0000.png").write_bytes(b"png")
    entity = SimpleNamespace(log_path=None, output_path=str(tmp_path))
    found = list_runnable_samples(entity)
    assert len(found) == 1
    assert found[0][1] == "cell"


def test_resolve_safe_sample_file_accepts_nested_file(tmp_path: Path) -> None:
    nested = tmp_path / "samples"
    nested.mkdir()
    sample = nested / "a.png"
    sample.write_bytes(b"ok")

    assert resolve_safe_sample_file(tmp_path, "samples/a.png") == sample.resolve()


def test_resolve_safe_sample_file_rejects_missing(tmp_path: Path) -> None:
    assert resolve_safe_sample_file(tmp_path, "missing.png") is None


def test_resolve_safe_sample_file_rejects_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.png"
    outside.write_bytes(b"no")
    assert resolve_safe_sample_file(tmp_path, "../secret.png") is None


def test_resolve_safe_sample_file_rejects_prefix_sibling(tmp_path: Path) -> None:
    base = tmp_path / "foo"
    sibling = tmp_path / "foobar"
    base.mkdir()
    sibling.mkdir()
    (sibling / "x.png").write_bytes(b"x")

    assert resolve_safe_sample_file(base, "../foobar/x.png") is None
