"""Tests for sample-file path safety and job-loss schema ownership."""

from pathlib import Path

from src.api.schemas.job_loss import JobLossResponse as ApiJobLossResponse
from src.api.schemas.job_loss import LossPoint as ApiLossPoint
from src.services.runnable.samples import resolve_safe_sample_file
from src.services.runnable.schemas import JobLossResponse, LossPoint


def test_job_loss_api_schemas_reexport_service_schemas() -> None:
    assert ApiJobLossResponse is JobLossResponse
    assert ApiLossPoint is LossPoint


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
